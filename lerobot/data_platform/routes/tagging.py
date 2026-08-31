import json
import logging
import re
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
from flask import abort, jsonify, redirect, render_template, request, send_file

from lerobot.data_platform.precompute.dataset_io import read_episode_table
from lerobot.data_platform.precompute.labeling import sample_episodes_by_task_type
from lerobot.data_platform.precompute.tagging import (
    DEFAULT_SELECTED_TAG_NAMES,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MODEL,
    load_tags_jsonl,
    merge_tags_to_metadata,
    remove_reviewed_tag,
    run_tagging,
    save_reviewed_tag,
    tags_path,
)
from lerobot.data_platform.precompute.tagging import (
    get_capabilities as get_tagging_capabilities,
)
from lerobot.data_platform.precompute.tagging import (
    get_schema as get_tagging_schema,
)
from lerobot.data_platform.precompute.tagging import (
    load_episode_record as load_tagging_episode_record,
)
from lerobot.data_platform.precompute.tagging import (
    resolved_reviewed_path as tagging_resolved_reviewed_path,
)
from lerobot.data_platform.precompute.tagging import (
    resolved_tags_path as tagging_resolved_tags_path,
)
from lerobot.data_platform.precompute.tagging import (
    source_path as tagging_source_path,
)
from lerobot.data_platform.precompute.tagging.geometric_backend import trajectory_xy_details_from_table
from lerobot.data_platform.routes.context import RouteContext


def register_tagging_routes(app, ctx: RouteContext) -> None:
    def _static_context(dataset_namespace: str, dataset_name: str) -> tuple[object | None, Path, bool]:
        dataset_key = (dataset_namespace, dataset_name)
        try:
            dataset_obj, ds_static = ctx.get_ctx(dataset_namespace, dataset_name)
            return dataset_obj, Path(ds_static), False
        except Exception:
            entry = ctx.datasets_index.get(dataset_key)
            if entry is not None:
                ds_static = Path(entry["output_dir"]).expanduser() / "static"
            elif ctx.static_dir_for_key is not None:
                ds_static = ctx.static_dir_for_key(dataset_key)
                if ds_static is None:
                    raise KeyError(ctx.repo_id_from_key(dataset_key)) from None
                ds_static = Path(ds_static).expanduser()
            else:
                raise KeyError(ctx.repo_id_from_key(dataset_key)) from None
            if not ds_static.exists():
                raise
            return None, ds_static, True

    def _tagging_image_key(ds_static: Path, variant: str | None = None) -> str:
        source_file = tagging_source_path(ds_static / "tagging", variant)
        if not source_file.is_file():
            source_file = tagging_source_path(ds_static / "tagging")
        if source_file.is_file():
            with suppress(Exception):
                value = json.loads(source_file.read_text()).get("image_key")
                if value:
                    return str(value)
        videos_dir = Path(ds_static) / "videos"
        if videos_dir.is_dir():
            image_dirs = sorted(path.name for path in videos_dir.iterdir() if path.is_dir())
            if image_dirs:
                return image_dirs[0]
        return ""

    @app.route("/api/tagging/capabilities")
    def api_tagging_capabilities():
        caps = get_tagging_capabilities()
        return jsonify(
            {
                **caps,
                "supported_backends": ["rule", "geometric"] + (["vlm"] if caps.get("available") else []),
                "default_model": caps.get("default_model", DEFAULT_VLM_MODEL),
            }
        )

    @app.route("/api/tagging/schema")
    def api_tagging_schema():
        return jsonify({"tags": get_tagging_schema(), "default_selected_tags": DEFAULT_SELECTED_TAG_NAMES})

    @app.route("/api/tagging/start", methods=["POST"])
    def api_start_tagging():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        except Exception as exc:
            logging.exception("Failed to load dataset for auto-tagging")
            return jsonify({"error": str(exc)}), 400

        options = body.get("options") or body
        selected_tags = ctx.parse_str_list(options.get("tag_names"))
        selected_episodes = ctx.parse_int_list(options.get("episodes"))
        vlm_backend = str(options.get("vlm_backend") or DEFAULT_VLM_BACKEND).strip() or DEFAULT_VLM_BACKEND
        vlm_model = str(options.get("vlm_model") or DEFAULT_VLM_MODEL)
        vlm_endpoint = str(options.get("vlm_endpoint") or "").strip() or None
        vlm_token = str(options.get("vlm_token") or "").strip() or None
        workers = max(1, int(options.get("workers") or 8))
        overwrite = ctx.bool_option(options, "overwrite", False)
        trial = ctx.bool_option(options, "trial", False)
        trial_per_type = max(1, int(options.get("trial_per_type") or 20))
        trial_seed_value = options.get("trial_seed")
        trial_seed = int(trial_seed_value) if trial_seed_value not in (None, "") else int(time.time())
        repo_id = ctx.repo_id_from_key(dataset_key)
        sample_result = None
        output_variant = None
        tag_defs = get_tagging_schema()
        selected_tag_set = set(selected_tags or [tag["name"] for tag in tag_defs])
        selected_vlm_tags = [
            tag for tag in tag_defs if tag.get("backend") == "vlm" and tag["name"] in selected_tag_set
        ]
        if trial:
            sample_result = sample_episodes_by_task_type(
                dataset_obj.root,
                dataset_obj.meta,
                per_type=trial_per_type,
                episodes=selected_episodes,
                seed=trial_seed,
            )
            selected_episodes = sample_result.episodes
            output_variant = "trial"
            if not selected_episodes:
                return jsonify({"error": "No supported episodes found for trial tagging."}), 400
        tagging_dir = ds_static / "tagging"
        tagging_dir.mkdir(parents=True, exist_ok=True)
        tags_path(tagging_dir, output_variant).touch(exist_ok=True)
        candidate_episodes = (
            selected_episodes
            if selected_episodes is not None
            else ctx.dataset_episode_ids(dataset_obj, dataset_key)
        )
        existing_records = load_tags_jsonl(tags_path(tagging_dir, output_variant))

        def _has_selected_tags(record: dict | None) -> bool:
            tags = (record or {}).get("tags") or {}
            if not isinstance(tags, dict):
                return False
            return all(name in tags for name in selected_tag_set)

        missing_episodes = (
            [int(episode) for episode in candidate_episodes]
            if overwrite
            else [
                int(episode)
                for episode in candidate_episodes
                if not _has_selected_tags(existing_records.get(int(episode)))
            ]
        )
        vlm_capabilities = get_tagging_capabilities(vlm_model, backend=vlm_backend)
        if selected_vlm_tags and missing_episodes and not vlm_capabilities.get("available"):
            return jsonify(
                {"error": vlm_capabilities.get("error") or "Selected VLM backend is unavailable."}
            ), 400
        if (
            selected_vlm_tags
            and missing_episodes
            and vlm_capabilities.get("requires_token")
            and not vlm_token
            and not vlm_capabilities.get("token_configured")
        ):
            token_env = (vlm_capabilities.get("token_env_vars") or ["DASHSCOPE_API_KEY"])[0]
            return jsonify(
                {"error": f"{vlm_backend} requires a token. Set {token_env} or paste it in the token field."}
            ), 400
        ctx.invalidate_tagging_status(dataset_key, ds_static)
        total = len(missing_episodes)
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        job = {
            "id": job_id,
            "job_type": "tagging",
            "dataset_key": repo_id,
            "status": "queued",
            "progress": 0,
            "current": 0,
            "total": total,
            "message": (
                f"Queued trial auto-tagging ({total} episodes, seed={trial_seed})" if trial else "Queued"
            ),
            "error": None,
            "review_url": f"/{repo_id}/tagging" + (f"?variant={output_variant}" if output_variant else ""),
            "viewer_url": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "logs": [],
        }
        with ctx.jobs_lock:
            ctx.jobs_registry[job_id] = job

        def _run_job() -> None:
            try:
                if sample_result is not None:
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "message": (
                                f"Trial auto-tagging seed={sample_result.seed}, "
                                f"counts={sample_result.counts}, available={sample_result.available_counts}"
                            ),
                            "current": 0,
                            "total": len(selected_episodes or []),
                        },
                    )
                kept = len(candidate_episodes) - len(missing_episodes) if not overwrite else 0
                message = f"Starting auto-tagging with {workers} worker(s)"
                if kept:
                    message += f" ({kept} existing episodes skipped)"
                ctx.update_job(job, {"status": "running", "message": message})
                result = run_tagging(
                    root=dataset_obj.root,
                    meta=dataset_obj.meta,
                    episodes=selected_episodes,
                    static_dir=ds_static,
                    selected_tags=selected_tags,
                    vlm_backend=vlm_backend,
                    vlm_model=vlm_model,
                    vlm_endpoint=vlm_endpoint,
                    vlm_token=vlm_token,
                    output_variant=output_variant,
                    workers=workers,
                    overwrite=overwrite,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                ctx.finish_job(
                    job,
                    "Trial auto-tagging complete" if output_variant else "Auto-tagging complete",
                    current=len(result.episodes),
                    total=len(result.episodes),
                    review_url=f"/{result.repo_id}/tagging"
                    + (f"?variant={output_variant}" if output_variant else ""),
                )
            except Exception as exc:
                logging.exception("Auto-tagging job failed")
                ctx.fail_job(job, "Auto-tagging failed", exc)

        threading.Thread(target=_run_job, name=f"tagging-{job_id}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/tagging")
    def show_tagging(dataset_namespace, dataset_name):
        dataset_key = (dataset_namespace, dataset_name)
        dataset_obj, ds_static, cache_only = _static_context(dataset_namespace, dataset_name)
        repo_id = ctx.repo_id_from_key(dataset_key)
        active_variant = ctx.active_tag_variant(ds_static / "tagging", request.args.get("variant"))
        if request.args.get("variant") is None and active_variant:
            return redirect(f"/{repo_id}/tagging?variant={quote(active_variant, safe='')}")
        if dataset_obj is not None:
            episode_ids = ctx.dataset_episode_ids(dataset_obj, dataset_key)
            image_key = (ctx.dataset_image_keys(dataset_obj) or [""])[0]
        else:
            records = load_tags_jsonl(tagging_resolved_tags_path(ds_static / "tagging", active_variant))
            if not records:
                records = load_tags_jsonl(
                    tagging_resolved_reviewed_path(ds_static / "tagging", active_variant)
                )
            episode_ids = sorted(records)
            image_key = _tagging_image_key(ds_static, active_variant)
        first_episode = episode_ids[0] if episode_ids else 0
        source_protected = bool(
            ctx.dataset_is_protected and ctx.dataset_is_protected(dataset_key)
        )
        return render_template(
            "visualize_dataset_tagging.html",
            dataset_namespace=dataset_namespace,
            dataset_name=dataset_name,
            dataset_key=repo_id,
            image_key=image_key,
            cache_only=cache_only,
            legacy_mutations_enabled=(
                bool(getattr(ctx, "legacy_mutations_enabled", False)) and not source_protected
            ),
            viewer_url=f"/{repo_id}/episode_{first_episode}",
            **ctx.dataset_nav(
                repo_id, first_episode, "tagging", dataset_obj, ds_static, cache_only=cache_only
            ),
        )

    @app.route("/api/tagging/<string:dataset_namespace>/<string:dataset_name>/episodes")
    def api_tagging_episodes(dataset_namespace, dataset_name):
        _, ds_static, _ = _static_context(dataset_namespace, dataset_name)
        tagging_dir = ds_static / "tagging"
        variant = ctx.active_tag_variant(tagging_dir, request.args.get("variant"))
        originals = load_tags_jsonl(tagging_resolved_tags_path(tagging_dir, variant))
        reviewed = load_tags_jsonl(tagging_resolved_reviewed_path(tagging_dir, variant))
        out = []
        for episode_index, original in sorted(originals.items()):
            current = reviewed.get(episode_index, original)
            out.append(
                {
                    "episode_index": episode_index,
                    "task": original.get("task"),
                    "tags": current.get("tags", {}),
                    "reviewed": episode_index in reviewed,
                    "manual": bool(current.get("manual")),
                }
            )
        return jsonify(out)

    def _trajectory_parquet_columns(parquet_path: Path) -> list[str]:
        try:
            schema_names = set(pq.read_schema(parquet_path).names)
        except Exception:
            return []
        preferred = [
            "observation.ee_pose",
            "observation.eef_pose",
            "observation.tcp_pose",
            "ee_pose",
            "eef_pose",
            "tcp_pose",
            "observation.state",
            "state",
            "action",
            "observation.action",
        ]
        columns = [key for key in preferred if key in schema_names]
        axis_suffix = re.compile(r"(^|[._/-])(x|y)$", re.IGNORECASE)
        columns.extend(key for key in schema_names if axis_suffix.search(key))
        return list(dict.fromkeys(columns))

    @app.route("/api/tagging/<string:dataset_namespace>/<string:dataset_name>/episode/<int:episode_index>")
    def api_tagging_episode(dataset_namespace, dataset_name, episode_index):
        dataset_obj, ds_static, cache_only = _static_context(dataset_namespace, dataset_name)
        variant = ctx.active_tag_variant(ds_static / "tagging", request.args.get("variant"))
        record = load_tagging_episode_record(ds_static / "tagging", episode_index, variant=variant)
        if record is None:
            return jsonify({"error": "not found"}), 404
        if cache_only or dataset_obj is None:
            record["trajectory_xy"] = []
            record["trajectory_source"] = ""
            record["trajectory_projection"] = "unavailable"
            record["trajectory_reason"] = "Source parquet is unavailable in cache-only mode."
            record["trajectory_active_arm"] = None
            record["trajectory_gripper_source"] = None
            record["trajectory_grasp_index"] = None
            record["trajectory_grasp_point"] = None
            record["trajectory_bounds"] = None
            return jsonify(record)
        try:
            parquet_path = dataset_obj.root / dataset_obj.meta.get_data_file_path(episode_index)
            trajectory_columns = _trajectory_parquet_columns(parquet_path)
            table = (
                read_episode_table(
                    dataset_obj.root,
                    dataset_obj.meta,
                    episode_index,
                    columns=trajectory_columns,
                )
                if trajectory_columns
                else pa.table({})
            )
            details = trajectory_xy_details_from_table(
                table,
                getattr(dataset_obj.meta, "features", {}),
                robot_type=getattr(dataset_obj.meta, "robot_type", ""),
            )
            record["trajectory_xy"] = details["points"]
            record["trajectory_source"] = details["source"]
            record["trajectory_projection"] = details["projection"]
            record["trajectory_reason"] = details["reason"]
            record["trajectory_active_arm"] = details.get("active_arm")
            record["trajectory_gripper_source"] = details.get("gripper_source")
            record["trajectory_grasp_index"] = details.get("grasp_index")
            record["trajectory_grasp_point"] = details.get("grasp_point")
            record["trajectory_bounds"] = details.get("bounds")
        except Exception:
            logging.exception("Failed to compute tagging trajectory for episode %s", episode_index)
            record["trajectory_xy"] = []
            record["trajectory_source"] = ""
            record["trajectory_projection"] = "unavailable"
            record["trajectory_reason"] = "Failed to compute trajectory."
            record["trajectory_active_arm"] = None
            record["trajectory_gripper_source"] = None
            record["trajectory_grasp_index"] = None
            record["trajectory_grasp_point"] = None
            record["trajectory_bounds"] = None
        return jsonify(record)

    @app.route(
        "/api/tagging/<string:dataset_namespace>/<string:dataset_name>/save/<int:episode_index>",
        methods=["POST"],
    )
    def api_tagging_save(dataset_namespace, dataset_name, episode_index):
        _, ds_static, _ = _static_context(dataset_namespace, dataset_name)
        body = request.get_json(force=True)
        variant = ctx.active_tag_variant(
            ds_static / "tagging", request.args.get("variant") or body.get("variant")
        )
        record = save_reviewed_tag(
            ds_static / "tagging", episode_index, body.get("tags") or {}, manual=True, variant=variant
        )
        ctx.invalidate_tagging_status((dataset_namespace, dataset_name), ds_static)
        return jsonify({"record": record})

    @app.route(
        "/api/tagging/<string:dataset_namespace>/<string:dataset_name>/reset/<int:episode_index>",
        methods=["POST"],
    )
    def api_tagging_reset(dataset_namespace, dataset_name, episode_index):
        _, ds_static, _ = _static_context(dataset_namespace, dataset_name)
        body = request.get_json(silent=True) or {}
        variant = ctx.active_tag_variant(
            ds_static / "tagging", request.args.get("variant") or body.get("variant")
        )
        remove_reviewed_tag(ds_static / "tagging", episode_index, variant=variant)
        ctx.invalidate_tagging_status((dataset_namespace, dataset_name), ds_static)
        return jsonify({"ok": True})

    @app.route("/api/tagging/<string:dataset_namespace>/<string:dataset_name>/merge", methods=["POST"])
    def api_tagging_merge(dataset_namespace, dataset_name):
        dataset_key = (dataset_namespace, dataset_name)
        dataset_obj, ds_static, cache_only = _static_context(dataset_namespace, dataset_name)
        if cache_only or dataset_obj is None:
            return jsonify({"error": "Merge requires the original dataset, not cache-only files."}), 400
        variant = ctx.active_tag_variant(ds_static / "tagging", request.args.get("variant"))
        try:
            result = merge_tags_to_metadata(dataset_obj.root, ds_static / "tagging", variant=variant)
            refreshed = ctx.meta_only_dataset_cls(ctx.repo_id_from_key(dataset_key), root=dataset_obj.root)
            ctx.register_dataset(refreshed, ds_static.parent)
        except Exception as exc:
            logging.exception("Failed to merge tags")
            return jsonify({"error": str(exc)}), 500
        return jsonify({"status": "ok", **result})

    @app.route("/api/tagging/<string:dataset_namespace>/<string:dataset_name>/heatmap")
    def api_tagging_heatmap(dataset_namespace, dataset_name):
        _, ds_static, _ = _static_context(dataset_namespace, dataset_name)
        path = ds_static / "tagging" / "grasp_heatmap.png"
        source_file = tagging_source_path(ds_static / "tagging")
        source = {}
        if source_file.is_file():
            try:
                source = json.loads(source_file.read_text())
            except Exception:
                source = {}
        if not path.is_file() or not source.get("grasp_xy_coordinate_source"):
            abort(404)
        return send_file(path, mimetype="image/png")
