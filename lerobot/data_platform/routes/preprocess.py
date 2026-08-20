import json
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from flask import jsonify, render_template, request

from lerobot.data_platform.cli import get_default_output_dir, run_precompute
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    load_episode_records,
    read_episode_table,
)
from lerobot.data_platform.precompute.preprocess import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_PROMPT_PATTERN,
    DEFAULT_PROMPT_REPLACEMENT,
    DEFAULT_V3_CONVERT_WORKERS,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    FLAG_FIX_DELETE_ALL_FLAGGED,
    FLAG_FIX_STATE_GRIPPER_TRANSITION_ACTION,
    FLAG_FIX_STUCK_CLOSED_ACTION,
    FLAG_FIX_TRIM_EARLY_GRIPPER,
    IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL,
    clear_all_flags,
    default_standardize_path,
    default_v3_path,
    delete_episodes_inplace,
    load_flagged_episode_ids,
    normalize_image_video_mode,
    repair_v3_video_timestamps,
    run_convert_action,
    run_convert_v3,
    run_drop_field,
    run_fix_prompt_prepositions,
    run_flag_fix,
    run_lowercase_prompts,
    run_merge,
    run_quality_flag_detection,
    run_rewrite_prompts,
    run_smooth_action,
    run_split,
    run_standardize_dataset,
    run_subtract,
    run_value_edits,
)
from lerobot.data_platform.precompute.preprocess import (
    get_capabilities as get_preprocess_capabilities,
)
from lerobot.data_platform.precompute.preprocess.common import format_data_path, load_json
from lerobot.data_platform.precompute.preprocess.quality_flags import apply_task_assignment_choice
from lerobot.data_platform.precompute.preprocess.smooth_action import SMOOTH_ACTION_META
from lerobot.data_platform.precompute.timeseries import (
    DATA_VERSION_DVT1,
    DATA_VERSION_DVT2,
    infer_data_version_from_features,
)
from lerobot.data_platform.routes.context import RouteContext


def register_preprocess_routes(app, ctx: RouteContext) -> None:
    @app.route("/api/preprocess/capabilities")
    def api_preprocess_capabilities():
        return jsonify(get_preprocess_capabilities())

    def _new_job(
        job_type: str,
        dataset_key: str | None,
        total: int = 1,
        output_root: str | None = None,
        related_dataset_keys: list[str] | None = None,
    ) -> dict:
        now = time.time()
        job = {
            "id": uuid.uuid4().hex[:12],
            "job_type": job_type,
            "dataset_key": dataset_key,
            "related_dataset_keys": related_dataset_keys or ([] if dataset_key is None else [dataset_key]),
            "status": "queued",
            "progress": 0,
            "current": 0,
            "total": total,
            "message": "Queued",
            "error": None,
            "review_url": None,
            "viewer_url": None,
            "output_root": output_root,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "logs": [],
        }
        with ctx.jobs_lock:
            ctx.jobs_registry[job["id"]] = job
        return job

    def _register_output_dataset(result, job: dict) -> None:
        if result.dry_run:
            ctx.finish_job(
                job,
                "Preprocess dry run complete",
                current=job.get("total") or 1,
                output_root=str(result.out_root),
            )
            return
        dataset = ctx.meta_only_dataset_cls(result.repo_id, root=result.out_root)
        dataset_key = ctx.register_dataset(dataset, get_default_output_dir(result.out_root))
        repo_id = ctx.repo_id_from_key(dataset_key)
        if ctx.append_operation_log:
            ctx.append_operation_log(
                get_default_output_dir(result.out_root) / "static",
                job.get("job_type") or "preprocess_output_dataset",
                dataset_key=dataset_key,
                dataset_root=result.out_root,
                details={
                    "source_job_id": job.get("id"),
                    "summary": getattr(result, "summary", {}),
                    "total_episodes": getattr(result, "total_episodes", None),
                    "total_frames": getattr(result, "total_frames", None),
                },
            )
        ctx.finish_job(
            job,
            "Preprocess complete",
            current=result.total_episodes or job.get("total") or 1,
            total=result.total_episodes or job.get("total") or 1,
            output_root=str(result.out_root),
            output_dataset_key=repo_id,
            viewer_url=f"/{repo_id}/episode_0",
            review_url=f"/{repo_id}/smoothing" if str(result.op).startswith("smooth_action") else None,
        )
        with ctx.jobs_lock:
            ctx.append_job_log(job, f"Summary: {result.summary}")

    def _positive_int_option(options: dict, key: str, default: int = 1) -> int:
        value = options.get(key)
        if value in (None, ""):
            return default
        return max(1, int(value))

    def _out_root(options: dict) -> Path | None:
        value = str(options.get("out_root") or "").strip()
        return Path(value).expanduser() if value else None

    def _invalidate_value_edit_cache(static_dir: Path, episode_ids: list[int] | None) -> int:
        csv_dir = Path(static_dir) / "csv"
        if not csv_dir.is_dir():
            return 0
        paths = (
            list(csv_dir.glob("episode_*_ds*.csv"))
            if episode_ids is None
            else [
                path
                for episode_id in set(episode_ids)
                for path in csv_dir.glob(f"episode_{episode_id:06d}_ds*.csv")
            ]
        )
        for path in paths:
            path.unlink(missing_ok=True)
        return len(paths)

    def _merge_out_root(options: dict) -> Path | None:
        out_name = str(options.get("out_name") or "").strip()
        if not out_name:
            return _out_root(options)
        out_path = Path(out_name)
        if out_path.is_absolute() or len(out_path.parts) != 1 or out_name in {".", ".."}:
            raise ValueError("output must be a folder name, not a path")
        root_dir_value = str(options.get("root_dir") or "").strip()
        if not root_dir_value:
            raise ValueError("root_dir is required when output folder name is set")
        root_dir = Path(root_dir_value).expanduser()
        if not root_dir.exists():
            raise ValueError(f"root_dir does not exist: {root_dir}")
        return root_dir / out_name

    def _data_version_option(options: dict, default: str) -> str:
        value = str(options.get("data_version") or default).upper()
        if value not in {DATA_VERSION_DVT1, DATA_VERSION_DVT2}:
            raise ValueError(f"Unsupported data_version: {value}")
        return value

    def _step_progress(job: dict, start: int, end: int, label: str):
        span = max(0, int(end) - int(start))

        def _callback(payload: dict) -> None:
            payload = dict(payload or {})
            sub_total = payload.get("total")
            sub_current = payload.get("current")
            if sub_total:
                fraction = max(0.0, min(1.0, float(sub_current or 0) / float(sub_total)))
            elif payload.get("status") == "done":
                fraction = 1.0
            else:
                fraction = 0.0
            payload["status"] = "running"
            payload["current"] = int(round(start + span * fraction))
            payload["total"] = 100
            if payload.get("message"):
                payload["message"] = f"{label}: {payload['message']}"
            ctx.update_job(job, payload)

        return _callback

    def _smooth_meta(root: Path) -> dict:
        path = Path(root) / "meta" / SMOOTH_ACTION_META
        if not path.is_file():
            raise FileNotFoundError(f"Missing smoothing metadata: {path}")
        return load_json(path)

    def _pending_prompt_assignments_path(static_dir: Path) -> Path:
        return Path(static_dir) / "prompt_assignments_pending.json"

    def _load_pending_prompt_assignments(static_dir: Path) -> dict[int, dict]:
        path = _pending_prompt_assignments_path(static_dir)
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        raw_items = payload.get("assignments") if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            return {}
        assignments: dict[int, dict] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                episode_index = int(item.get("episode_index"))
            except (TypeError, ValueError):
                continue
            selected_task = str(item.get("selected_task") or "").strip()
            if not selected_task:
                continue
            assignments[episode_index] = {
                "episode_index": episode_index,
                "selected_task": selected_task,
                "updated_at": item.get("updated_at") or time.time(),
                "source": item.get("source") or "viewer_cache_only",
            }
        return dict(sorted(assignments.items()))

    def _save_pending_prompt_assignments(static_dir: Path, assignments: dict[int, dict]) -> None:
        items = [
            {
                "episode_index": int(item["episode_index"]),
                "selected_task": str(item["selected_task"]),
                "updated_at": item.get("updated_at") or time.time(),
                "source": item.get("source") or "viewer_cache_only",
            }
            for _, item in sorted(assignments.items())
        ]
        _pending_prompt_assignments_path(static_dir).write_text(
            json.dumps({"version": 1, "assignments": items}, indent=2, ensure_ascii=False)
        )

    def _series_array(root: Path, episode_index: int, field: str) -> np.ndarray:
        root = Path(root).expanduser()
        info = load_json(root / "meta" / "info.json")
        meta = V3DatasetMetadata(f"local/{root.name}", root) if is_v3_dataset(root) else None
        path = (
            root / meta.get_data_file_path(episode_index)
            if meta is not None
            else root / format_data_path(info, episode_index)
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing episode parquet: {path}")
        schema = pq.read_schema(path)
        if field not in schema.names:
            raise ValueError(f"Missing {field} column in {path}")
        table = (
            read_episode_table(root, meta, episode_index, columns=[field])
            if meta is not None
            else pq.read_table(path, columns=[field])
        )
        arr = np.asarray(table[field].to_pylist(), dtype=np.float32)
        if arr.ndim < 2:
            arr = arr.reshape((-1, 1))
        return arr

    @app.route("/<ns>/<name>/smoothing")
    def smoothing_report_page(ns: str, name: str):
        repo_id = f"{ns}/{name}"
        try:
            dataset_obj, ds_static = ctx.get_ctx(ns, name)
            meta = _smooth_meta(Path(dataset_obj.root))
        except Exception as exc:
            return str(exc), 404
        episodes = load_episode_records(Path(dataset_obj.root))
        first_episode = int(episodes[0]["episode_index"]) if episodes else 0
        return render_template(
            "visualize_dataset_smoothing.html",
            dataset_key=repo_id,
            dataset_namespace=ns,
            dataset_name=name,
            source_root=meta.get("source_root", ""),
            window=meta.get("window"),
            **ctx.dataset_nav(repo_id, first_episode, "smoothing", dataset_obj, ds_static),
        )

    @app.route("/api/preprocess/smooth_action/<ns>/<name>/episodes")
    def api_smooth_action_episodes(ns: str, name: str):
        try:
            dataset_obj, _ = ctx.get_ctx(ns, name)
            meta = _smooth_meta(Path(dataset_obj.root))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 404
        episodes = load_episode_records(Path(dataset_obj.root))
        return jsonify(
            {
                "meta": meta,
                "fields": meta.get("fields") or [meta.get("field") or "action"],
                "episodes": [
                    {
                        "episode_index": int(row["episode_index"]),
                        "length": int(row.get("length") or 0),
                        "tasks": row.get("tasks", []),
                    }
                    for row in episodes
                ],
            }
        )

    @app.route("/api/preprocess/smooth_action/<ns>/<name>/episode/<int:episode_index>")
    def api_smooth_action_episode(ns: str, name: str, episode_index: int):
        try:
            dataset_obj, _ = ctx.get_ctx(ns, name)
            meta = _smooth_meta(Path(dataset_obj.root))
            available_fields = meta.get("fields") or [meta.get("field") or "action"]
            field = str(request.args.get("field") or "action")
            if field not in available_fields:
                raise ValueError(f"Field {field!r} was not smoothed. Available fields: {available_fields}")
            before = _series_array(Path(meta["source_root"]), episode_index, field)
            after = _series_array(Path(dataset_obj.root), episode_index, field)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 404

        frames = min(before.shape[0], after.shape[0])
        dims = min(before.shape[1], after.shape[1])
        before = before[:frames, :dims]
        after = after[:frames, :dims]
        delta = after - before
        rms = np.sqrt(np.mean(np.square(delta), axis=0)) if frames else np.zeros((dims,), dtype=np.float32)
        max_abs = np.max(np.abs(delta), axis=0) if frames else np.zeros((dims,), dtype=np.float32)

        max_points = max(1, int(request.args.get("max_points", 1500)))
        stride = max(1, int(np.ceil(frames / max_points))) if frames else 1
        frame_index = np.arange(frames, dtype=np.int64)[::stride]
        return jsonify(
            {
                "episode_index": episode_index,
                "field": field,
                "frames": int(frames),
                "dims": int(dims),
                "stride": int(stride),
                "frame_index": frame_index.tolist(),
                "before": np.round(before[::stride], 6).tolist(),
                "after": np.round(after[::stride], 6).tolist(),
                "rms_by_dim": np.round(rms, 6).tolist(),
                "max_abs_by_dim": np.round(max_abs, 6).tolist(),
                "meta": meta,
            }
        )

    @app.route("/api/preprocess/convert_action/start", methods=["POST"])
    def api_preprocess_convert_action_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, _ = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        target_dim = int(options.get("target_dim") or 16)
        dry_run = ctx.bool_option(options, "dry_run", False)
        total = int(getattr(dataset_obj, "total_episodes", 1) or 1)
        job = _new_job(
            "preprocess_convert_action",
            ctx.repo_id_from_key(dataset_key),
            total,
            str(_out_root(options) or ""),
        )

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job, {"status": "running", "message": "Starting action/state dimension conversion"}
                )
                result = run_convert_action(
                    dataset_obj.root,
                    out_root=_out_root(options),
                    target_dim=target_dim,
                    dry_run=dry_run,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                _register_output_dataset(result, job)
            except Exception as exc:
                logging.exception("Preprocess convert_action failed")
                ctx.fail_job(job, "Preprocess convert_action failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-convert-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/convert_v3/start", methods=["POST"])
    def api_preprocess_convert_v3_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        index_entry = ctx.datasets_index.get(dataset_key)
        if index_entry is None:
            return jsonify({"error": "dataset is not registered"}), 404

        src_root = Path(index_entry["root"]).expanduser()
        info = load_json(src_root / "meta" / "info.json")
        raw_version = str(info.get("codebase_version") or "").lower().removeprefix("v")
        source_is_v3 = raw_version == "3" or raw_version.startswith("3.")
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
        workers = DEFAULT_V3_CONVERT_WORKERS
        image_video_mode = IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL
        if not source_is_v3:
            try:
                data_file_size_value = options.get("data_file_size_in_mb")
                video_file_size_value = options.get("video_file_size_in_mb")
                data_file_size_in_mb = int(
                    DEFAULT_DATA_FILE_SIZE_IN_MB
                    if data_file_size_value in (None, "")
                    else data_file_size_value
                )
                video_file_size_in_mb = int(
                    DEFAULT_VIDEO_FILE_SIZE_IN_MB
                    if video_file_size_value in (None, "")
                    else video_file_size_value
                )
            except (TypeError, ValueError):
                return jsonify({"error": "data/video file size limits must be integers"}), 400
            if data_file_size_in_mb <= 0 or video_file_size_in_mb <= 0:
                return jsonify({"error": "data/video file size limits must be positive"}), 400
            try:
                workers_value = options.get("workers")
                workers = int(DEFAULT_V3_CONVERT_WORKERS if workers_value in (None, "") else workers_value)
            except (TypeError, ValueError):
                return jsonify({"error": "workers must be an integer"}), 400
            if workers <= 0:
                return jsonify({"error": "workers must be positive"}), 400
            try:
                image_video_mode = normalize_image_video_mode(
                    options.get("image_video_mode") or IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL
                )
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        total = int(info.get("total_episodes") or 1)
        dry_run = ctx.bool_option(options, "dry_run", False)
        overwrite = ctx.bool_option(options, "overwrite", False)
        target_root = None if source_is_v3 else (_out_root(options) or default_v3_path(src_root))
        if target_root is not None and target_root.exists() and not dry_run and not overwrite:
            return (
                jsonify(
                    {
                        "error": f"Output dataset already exists: {target_root}",
                        "requires_overwrite_confirmation": True,
                        "output_root": str(target_root),
                    }
                ),
                409,
            )
        job = _new_job(
            "preprocess_convert_v3",
            ctx.repo_id_from_key(dataset_key),
            total,
            str(src_root if source_is_v3 else target_root or ""),
        )

        def _run_job() -> None:
            try:
                ctx.update_job(job, {"status": "running", "message": "Starting LeRobot v3.0 conversion"})
                result = run_convert_v3(
                    src_root,
                    out_root=target_root,
                    data_file_size_in_mb=data_file_size_in_mb,
                    video_file_size_in_mb=video_file_size_in_mb,
                    workers=workers,
                    image_video_mode=image_video_mode,
                    overwrite=overwrite,
                    dry_run=dry_run,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                output_root = str(result.out_root)
                if result.summary.get("already_v3"):
                    message = f"Dataset is already v3.0; no conversion is needed: {output_root}"
                elif result.dry_run:
                    message = f"v3.0 conversion preview complete: {output_root}"
                else:
                    action = str(result.summary.get("action") or "convert")
                    message = f"v3.0 {action} complete: {output_root}"
                ctx.finish_job(
                    job,
                    message,
                    current=result.total_episodes or total,
                    total=result.total_episodes or total,
                    output_root=output_root,
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {result.summary}")
                    if not result.summary.get("already_v3"):
                        ctx.append_job_log(job, "v3.0 outputs are not auto-registered in the legacy viewer.")
            except Exception as exc:
                logging.exception("Preprocess convert_v3 failed")
                ctx.fail_job(job, "Preprocess convert_v3 failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-v3-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/repair_v3_video_timestamps/start", methods=["POST"])
    def api_preprocess_repair_v3_video_timestamps_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        index_entry = ctx.datasets_index.get(dataset_key)
        if index_entry is None:
            return jsonify({"error": "dataset is not registered"}), 404

        root = Path(index_entry["root"]).expanduser()
        info = load_json(root / "meta" / "info.json")
        raw_version = str(info.get("codebase_version") or "").lower().removeprefix("v")
        if not (raw_version == "3" or raw_version.startswith("3.")):
            return jsonify({"error": "video timestamp repair requires a v3.0 dataset"}), 400
        dry_run = ctx.bool_option(options, "dry_run", False)
        total = int(info.get("total_episodes") or 1)
        job = _new_job(
            "preprocess_repair_v3_video_timestamps",
            ctx.repo_id_from_key(dataset_key),
            total,
            str(root),
        )

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "message": "Checking and normalizing MP4 frame PTS",
                    },
                )
                result = repair_v3_video_timestamps(
                    root,
                    dry_run=dry_run,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                ctx.finish_job(
                    job,
                    (
                        "v3.0 video timestamp repair preview complete"
                        if result.dry_run
                        else "v3.0 video timestamps remuxed without re-encoding"
                    ),
                    current=result.total_episodes or total,
                    total=result.total_episodes or total,
                    output_root=str(root),
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {result.summary}")
            except Exception as exc:
                logging.exception("Preprocess v3 video timestamp repair failed")
                ctx.fail_job(job, "Preprocess v3 video timestamp repair failed", exc)

        threading.Thread(
            target=_run_job,
            name=f"preprocess-v3-timestamps-{job['id']}",
            daemon=True,
        ).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/drop_field/start", methods=["POST"])
    def api_preprocess_drop_field_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, _ = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        field_name = str(options.get("field_name") or "").strip()
        if not field_name:
            return jsonify({"error": "field_name is required"}), 400
        dry_run = ctx.bool_option(options, "dry_run", False)
        total = int(getattr(dataset_obj, "total_episodes", 1) or 1)
        job = _new_job(
            "preprocess_drop_field", ctx.repo_id_from_key(dataset_key), total, str(_out_root(options) or "")
        )

        def _run_job() -> None:
            try:
                ctx.update_job(job, {"status": "running", "message": f"Starting drop field: {field_name}"})
                result = run_drop_field(
                    dataset_obj.root,
                    out_root=_out_root(options),
                    field_name=field_name,
                    dry_run=dry_run,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                _register_output_dataset(result, job)
            except Exception as exc:
                logging.exception("Preprocess drop_field failed")
                ctx.fail_job(job, "Preprocess drop_field failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-drop-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/value_edit/start", methods=["POST"])
    def api_preprocess_value_edit_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404

        edits = options.get("edits")
        if not isinstance(edits, list) or not edits:
            return jsonify({"error": "edits must contain at least one value edit"}), 400
        scope = str(options.get("episode_scope") or "selected").strip().lower()
        if scope not in {"all", "selected"}:
            return jsonify({"error": "episode_scope must be all or selected"}), 400
        try:
            episode_ids = None if scope == "all" else ctx.parse_int_list(options.get("episodes"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if scope == "selected" and not episode_ids:
            return jsonify({"error": "episodes are required when episode_scope is selected"}), 400

        output_mode = str(options.get("output_mode") or "new_dataset").strip().lower()
        if output_mode not in {"new_dataset", "in_place"}:
            return jsonify({"error": "output_mode must be new_dataset or in_place"}), 400
        in_place = output_mode == "in_place"
        out_root = None if in_place else _out_root(options)
        dry_run = ctx.bool_option(options, "dry_run", False)
        total = (
            int(getattr(dataset_obj, "total_episodes", 1) or 1)
            if episode_ids is None
            else len(set(episode_ids))
        )
        job = _new_job(
            "preprocess_value_edit",
            ctx.repo_id_from_key(dataset_key),
            total,
            str(dataset_obj.root if in_place else out_root or ""),
        )

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "message": (
                            f"Starting {len(edits)} value edit(s) in "
                            f"{'source dataset' if in_place else 'a new dataset'}"
                        ),
                    },
                )
                result = run_value_edits(
                    dataset_obj.root,
                    edits=edits,
                    episode_ids=episode_ids,
                    in_place=in_place,
                    out_root=out_root,
                    dry_run=dry_run,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                if result.dry_run:
                    ctx.finish_job(
                        job,
                        "Value edit dry run complete",
                        current=total,
                        total=total,
                        output_root=str(result.out_root),
                    )
                elif not in_place:
                    _register_output_dataset(result, job)
                else:
                    removed_cache_files = _invalidate_value_edit_cache(ds_static, episode_ids)
                    if ctx.clear_dataset_caches is not None:
                        ctx.clear_dataset_caches(dataset_key)
                    refreshed = ctx.meta_only_dataset_cls(
                        ctx.repo_id_from_key(dataset_key),
                        root=dataset_obj.root,
                    )
                    ctx.register_dataset(refreshed, Path(ds_static).parent)
                    result.summary["invalidated_csv_files"] = removed_cache_files
                    ctx.finish_job(
                        job,
                        "Value edit complete in source dataset",
                        current=total,
                        total=total,
                        output_root=str(dataset_obj.root),
                        viewer_url=f"/{ctx.repo_id_from_key(dataset_key)}/episode_0",
                    )
                    with ctx.jobs_lock:
                        ctx.append_job_log(job, f"Summary: {result.summary}")
            except Exception as exc:
                logging.exception("Preprocess value edit failed")
                ctx.fail_job(job, "Preprocess value edit failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-value-edit-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/smooth_action/start", methods=["POST"])
    def api_preprocess_smooth_action_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, _ = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        window = int(options.get("window") or 5)
        workers = int(options.get("workers") or 0)
        smooth_state = ctx.bool_option(options, "smooth_state", True)
        dry_run = ctx.bool_option(options, "dry_run", False)
        total = int(getattr(dataset_obj, "total_episodes", 1) or 1)
        job = _new_job(
            "preprocess_smooth_action",
            ctx.repo_id_from_key(dataset_key),
            total,
            str(_out_root(options) or ""),
        )

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "message": f"Starting action smoothing, window={window}, workers={workers or 'auto'}",
                    },
                )
                result = run_smooth_action(
                    dataset_obj.root,
                    out_root=_out_root(options),
                    window=window,
                    workers=workers,
                    smooth_state=smooth_state,
                    dry_run=dry_run,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                _register_output_dataset(result, job)
            except Exception as exc:
                logging.exception("Preprocess smooth_action failed")
                ctx.fail_job(job, "Preprocess smooth_action failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-smooth-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/rewrite_prompts/start", methods=["POST"])
    def api_preprocess_rewrite_prompts_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        pattern = str(options.get("pattern") or DEFAULT_PROMPT_PATTERN)
        replacement = str(options.get("replacement") or DEFAULT_PROMPT_REPLACEMENT)
        dry_run = ctx.bool_option(options, "dry_run", False)
        backup = ctx.bool_option(options, "backup", True)
        total = int(getattr(dataset_obj, "total_episodes", 1) or 1)
        job = _new_job(
            "preprocess_rewrite_prompts", ctx.repo_id_from_key(dataset_key), total, str(dataset_obj.root)
        )

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job, {"status": "running", "current": 0, "total": 3, "message": "Starting prompt rewrite"}
                )
                result = run_rewrite_prompts(
                    dataset_obj.root,
                    pattern=pattern,
                    replacement=replacement,
                    dry_run=dry_run,
                    backup=backup,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                if not dry_run:
                    if ctx.clear_dataset_caches:
                        ctx.clear_dataset_caches(dataset_key)
                    try:
                        refreshed = ctx.meta_only_dataset_cls(dataset_obj.repo_id, root=dataset_obj.root)
                        ctx.register_dataset(refreshed, ds_static.parent)
                    except Exception:
                        logging.exception("Failed to refresh dataset metadata after prompt rewrite")
                    if ctx.append_operation_log:
                        ctx.append_operation_log(
                            ds_static,
                            "rewrite_prompts",
                            dataset_key=dataset_key,
                            dataset_root=Path(dataset_obj.root),
                            details={"summary": result.summary, "job_id": job["id"]},
                        )
                ctx.finish_job(
                    job,
                    ("Prompt rewrite dry run complete" if dry_run else "Prompt rewrite complete"),
                    current=3,
                    total=3,
                    output_root=str(dataset_obj.root),
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {result.summary}")
            except Exception as exc:
                logging.exception("Preprocess prompt rewrite failed")
                ctx.fail_job(job, "Preprocess prompt rewrite failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-rewrite-prompts-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/fix_prompt_prepositions/start", methods=["POST"])
    def api_preprocess_fix_prompt_prepositions_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        dry_run = ctx.bool_option(options, "dry_run", False)
        backup = ctx.bool_option(options, "backup", True)
        total = int(getattr(dataset_obj, "total_episodes", 1) or 1)
        job = _new_job(
            "preprocess_fix_prompt_prepositions",
            ctx.repo_id_from_key(dataset_key),
            total,
            str(dataset_obj.root),
        )

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": 3,
                        "message": "Starting prompt preposition fix",
                    },
                )
                result = run_fix_prompt_prepositions(
                    dataset_obj.root,
                    dry_run=dry_run,
                    backup=backup,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                if not dry_run:
                    if ctx.clear_dataset_caches:
                        ctx.clear_dataset_caches(dataset_key)
                    try:
                        refreshed = ctx.meta_only_dataset_cls(dataset_obj.repo_id, root=dataset_obj.root)
                        ctx.register_dataset(refreshed, ds_static.parent)
                    except Exception:
                        logging.exception("Failed to refresh dataset metadata after prompt preposition fix")
                    if ctx.append_operation_log:
                        ctx.append_operation_log(
                            ds_static,
                            "fix_prompt_prepositions",
                            dataset_key=dataset_key,
                            dataset_root=Path(dataset_obj.root),
                            details={"summary": result.summary, "job_id": job["id"]},
                        )
                ctx.finish_job(
                    job,
                    (
                        "Prompt preposition fix dry run complete"
                        if dry_run
                        else "Prompt preposition fix complete"
                    ),
                    current=3,
                    total=3,
                    output_root=str(dataset_obj.root),
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {result.summary}")
            except Exception as exc:
                logging.exception("Preprocess prompt preposition fix failed")
                ctx.fail_job(job, "Preprocess prompt preposition fix failed", exc)

        threading.Thread(
            target=_run_job, name=f"preprocess-fix-prompt-prepositions-{job['id']}", daemon=True
        ).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/apply_prompt_assignments/start", methods=["POST"])
    def api_preprocess_apply_prompt_assignments_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
            requested_episodes = ctx.parse_int_list(options.get("episodes") or options.get("episode_ids"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        except Exception as exc:
            logging.exception("Failed to load dataset for prompt assignment apply")
            return jsonify({"error": str(exc)}), 400

        pending = _load_pending_prompt_assignments(ds_static)
        if requested_episodes is not None:
            requested_set = {int(ep) for ep in requested_episodes}
            pending = {ep: item for ep, item in pending.items() if ep in requested_set}
        total = len(pending)
        job = _new_job(
            "preprocess_apply_prompt_assignments", ctx.repo_id_from_key(dataset_key), max(1, total)
        )

        def _run_job() -> None:
            try:
                if not pending:
                    ctx.finish_job(job, "No pending prompt assignments to apply", current=1, total=1)
                    return
                all_pending = _load_pending_prompt_assignments(ds_static)
                applied: list[int] = []
                failures: list[dict] = []
                results: list[dict] = []
                for idx, (episode_index, item) in enumerate(sorted(pending.items()), start=1):
                    selected_task = str(item.get("selected_task") or "").strip()
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": idx - 1,
                            "total": total,
                            "message": f"Applying prompt assignment ep{episode_index}: {selected_task}",
                        },
                    )
                    try:
                        result = apply_task_assignment_choice(
                            Path(dataset_obj.root),
                            ds_static,
                            int(episode_index),
                            selected_task,
                        )
                    except Exception as exc:
                        failures.append({"episode_index": int(episode_index), "error": str(exc)})
                        ctx.append_job_log(job, f"Failed ep{episode_index}: {exc}")
                        continue
                    applied.append(int(episode_index))
                    results.append(result)
                    all_pending.pop(int(episode_index), None)
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": idx,
                            "total": total,
                            "message": f"Applied prompt assignment ep{episode_index}",
                        },
                    )
                _save_pending_prompt_assignments(ds_static, all_pending)
                if ctx.clear_dataset_caches:
                    ctx.clear_dataset_caches(dataset_key)
                try:
                    refreshed = ctx.meta_only_dataset_cls(dataset_obj.repo_id, root=dataset_obj.root)
                    ctx.register_dataset(refreshed, ds_static.parent)
                except Exception:
                    logging.exception("Failed to refresh dataset metadata after applying prompt assignments")
                summary = {
                    "applied_count": len(applied),
                    "failed_count": len(failures),
                    "remaining_pending_count": len(all_pending),
                    "failures": failures[:20],
                }
                if ctx.append_operation_log:
                    ctx.append_operation_log(
                        ds_static,
                        "apply_prompt_assignments",
                        dataset_key=dataset_key,
                        dataset_root=Path(dataset_obj.root),
                        episode_ids=applied,
                        details={"summary": summary, "job_id": job["id"], "results": results[:20]},
                    )
                ctx.finish_job(
                    job,
                    f"Applied {len(applied)}/{total} pending prompt assignments"
                    + (f" ({len(failures)} failed)" if failures else ""),
                    current=total,
                    total=total,
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {summary}")
            except Exception as exc:
                logging.exception("Apply pending prompt assignments failed")
                ctx.fail_job(job, "Apply pending prompt assignments failed", exc)

        threading.Thread(
            target=_run_job, name=f"preprocess-apply-prompt-assignments-{job['id']}", daemon=True
        ).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/lowercase_prompts/start", methods=["POST"])
    def api_preprocess_lowercase_prompts_start():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key = ctx.dataset_key_from_body(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        index_entry = ctx.datasets_index.get(dataset_key)
        if index_entry is None:
            return jsonify({"error": f"dataset is not registered: {ctx.repo_id_from_key(dataset_key)}"}), 404
        root_path = Path(index_entry["root"]).expanduser()
        output_dir = Path(index_entry["output_dir"]).expanduser()
        job = _new_job("preprocess_lowercase_prompts", ctx.repo_id_from_key(dataset_key), 4)

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": 4,
                        "message": "Normalizing prompt text to lowercase",
                    },
                )
                result = run_lowercase_prompts(
                    root_path,
                    backup=True,
                    static_dir=output_dir / "static",
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                if ctx.clear_dataset_caches:
                    ctx.clear_dataset_caches(dataset_key)
                try:
                    refreshed = ctx.meta_only_dataset_cls(ctx.repo_id_from_key(dataset_key), root=root_path)
                    ctx.register_dataset(refreshed, output_dir)
                except Exception:
                    logging.exception("Failed to refresh dataset metadata after lowercasing prompts")
                if ctx.append_operation_log:
                    ctx.append_operation_log(
                        output_dir / "static",
                        "lowercase_prompts",
                        dataset_key=dataset_key,
                        dataset_root=root_path,
                        details={"summary": result.summary, "job_id": job["id"]},
                    )
                ctx.finish_job(
                    job,
                    "Prompt lowercase complete",
                    current=4,
                    total=4,
                    output_root=str(root_path),
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {result.summary}")
            except Exception as exc:
                logging.exception("Prompt lowercase failed")
                ctx.fail_job(job, "Prompt lowercase failed", exc)

        threading.Thread(
            target=_run_job, name=f"preprocess-lowercase-prompts-{job['id']}", daemon=True
        ).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/quality_flags/start", methods=["POST"])
    def api_preprocess_quality_flags_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        index_entry = ctx.datasets_index.get(dataset_key)
        if index_entry is None:
            return jsonify({"error": f"dataset is not registered: {ctx.repo_id_from_key(dataset_key)}"}), 404
        root_path = Path(index_entry["root"]).expanduser()
        output_dir = Path(index_entry["output_dir"]).expanduser()
        try:
            info = load_json(root_path / "meta" / "info.json")
            default_data_version = infer_data_version_from_features(info.get("features") or {})
            data_version = _data_version_option(options, default_data_version)
            episodes = ctx.parse_int_list(options.get("episodes") or options.get("episode_ids"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except (OSError, KeyError, TypeError) as exc:
            return jsonify({"error": f"failed to read dataset metadata: {exc}"}), 400
        workers = _positive_int_option(options, "workers", 8)
        overwrite = ctx.bool_option(options, "overwrite", False)
        clear_manual_flags = ctx.bool_option(options, "clear_manual_flags", False)
        job = _new_job("preprocess_quality_flags", ctx.repo_id_from_key(dataset_key), 1)

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": 1,
                        "message": f"Starting abnormal flag detection ({data_version})",
                    },
                )
                result = run_quality_flag_detection(
                    root_path,
                    output_dir / "static",
                    episodes=episodes,
                    data_version=data_version,
                    workers=workers,
                    overwrite=overwrite,
                    clear_manual_flags=clear_manual_flags,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                summary = result.summary
                if ctx.append_operation_log:
                    ctx.append_operation_log(
                        output_dir / "static",
                        "quality_flags_detect",
                        dataset_key=dataset_key,
                        dataset_root=root_path,
                        episode_ids=episodes or [],
                        details={
                            "summary": summary,
                            "data_version": data_version,
                            "workers": workers,
                            "overwrite": overwrite,
                            "clear_manual_flags": clear_manual_flags,
                            "job_id": job["id"],
                        },
                    )
                ctx.finish_job(
                    job,
                    (
                        "Quality flag detection complete: "
                        f"{summary.get('quality_episode_count', 0)} episodes flagged, "
                        f"{summary.get('quality_issue_count', 0)} issues"
                    ),
                    current=result.total_episodes or 1,
                    total=result.total_episodes or 1,
                    output_root=str(root_path),
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {summary}")
            except Exception as exc:
                logging.exception("Preprocess quality flag detection failed")
                ctx.fail_job(job, "Preprocess quality flag detection failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-quality-flags-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/clear_flags/start", methods=["POST"])
    def api_preprocess_clear_flags_start():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key = ctx.dataset_key_from_body(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        ds_static = ctx.static_dir_for_key(dataset_key) if ctx.static_dir_for_key else None
        dataset_root = None
        try:
            dataset_obj, loaded_static = ctx.ensure_dataset_loaded(dataset_key)
            dataset_root = Path(dataset_obj.root)
            if ds_static is None:
                ds_static = loaded_static
        except KeyError:
            # Cache-only datasets are registered for static viewer access but do not have
            # a loadable source dataset. Clearing flags only needs the static sidecars.
            pass
        except Exception as exc:
            if ds_static is None:
                logging.exception("Failed to load dataset for clear flags")
                return jsonify({"error": str(exc)}), 400

        if ds_static is None:
            return jsonify({"error": "dataset is not registered"}), 404

        job = _new_job("preprocess_clear_flags", ctx.repo_id_from_key(dataset_key), 2)

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": 2,
                        "message": "Clearing all viewer flags",
                    },
                )
                result = clear_all_flags(
                    dataset_root,
                    ds_static,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                    repo_id=ctx.repo_id_from_key(dataset_key),
                )
                if ctx.append_operation_log:
                    ctx.append_operation_log(
                        ds_static,
                        "clear_flags",
                        dataset_key=dataset_key,
                        dataset_root=dataset_root,
                        details={"summary": result.summary, "job_id": job["id"]},
                    )
                ctx.finish_job(
                    job,
                    "All flags cleared",
                    current=2,
                    total=2,
                    output_root=str(dataset_root or Path(ds_static).parent),
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {result.summary}")
            except Exception as exc:
                logging.exception("Preprocess clear flags failed")
                ctx.fail_job(job, "Preprocess clear flags failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-clear-flags-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/flag_fixes/start", methods=["POST"])
    def api_preprocess_flag_fixes_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            fix_kind = str(options.get("fix_kind") or "").strip()
            if fix_kind not in {
                FLAG_FIX_TRIM_EARLY_GRIPPER,
                FLAG_FIX_STUCK_CLOSED_ACTION,
                FLAG_FIX_STATE_GRIPPER_TRANSITION_ACTION,
                FLAG_FIX_DELETE_ALL_FLAGGED,
            }:
                raise ValueError(f"Unsupported flag fix: {fix_kind}")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
        except KeyError:
            return jsonify({"error": f"dataset is not registered: {ctx.repo_id_from_key(dataset_key)}"}), 404
        except Exception as exc:
            logging.exception("Failed to load dataset for flag fix")
            return jsonify({"error": str(exc)}), 400

        try:
            info = load_json(Path(dataset_obj.root) / "meta" / "info.json")
            default_data_version = infer_data_version_from_features(info.get("features") or {})
            data_version = _data_version_option(options, default_data_version)
            episodes = ctx.parse_int_list(options.get("episodes") or options.get("episode_ids"))
            prepare_workers_key = (
                "prepare_workers" if options.get("prepare_workers") not in (None, "") else "workers"
            )
            prepare_workers = _positive_int_option(options, prepare_workers_key, 8)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except (OSError, KeyError, TypeError) as exc:
            return jsonify({"error": f"failed to read dataset metadata: {exc}"}), 400

        flagged_ids = load_flagged_episode_ids(ds_static)
        total = len(flagged_ids) if fix_kind == FLAG_FIX_DELETE_ALL_FLAGGED else 1
        job = _new_job("preprocess_flag_fixes", ctx.repo_id_from_key(dataset_key), max(1, total))

        def _run_job() -> None:
            try:
                if fix_kind == FLAG_FIX_DELETE_ALL_FLAGGED:
                    if not flagged_ids:
                        ctx.finish_job(job, "No flagged episodes to delete", current=1, total=1)
                        return
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": 0,
                            "total": len(flagged_ids),
                            "message": f"Deleting {len(flagged_ids)} flagged episodes",
                        },
                    )
                    delete_result = delete_episodes_inplace(
                        dataset_obj,
                        flagged_ids,
                        static_folder=ds_static,
                        log=lambda message: ctx.append_job_log(job, message),
                    )
                    remaining_ids = None
                    refresh_dataset = getattr(
                        ctx,
                        "refresh_dataset_after_episode_delete",
                        None,
                    )
                    clear_caches = getattr(ctx, "clear_dataset_caches", None)
                    if refresh_dataset:
                        remaining_ids = refresh_dataset(
                            dataset_key,
                            dataset_obj,
                            ds_static,
                        )
                    elif clear_caches:
                        clear_caches(dataset_key)
                    if ctx.append_operation_log:
                        ctx.append_operation_log(
                            ds_static,
                            "flag_fix_delete_flagged",
                            dataset_key=dataset_key,
                            dataset_root=Path(dataset_obj.root),
                            episode_ids=flagged_ids,
                            details={"result": delete_result, "job_id": job["id"]},
                        )
                    ctx.finish_job(
                        job,
                        "Deleted flagged episodes",
                        current=len(flagged_ids),
                        total=len(flagged_ids),
                        review_url=None,
                        viewer_url=(
                            f"/{ctx.repo_id_from_key(dataset_key)}/episode_{remaining_ids[0]}"
                            if remaining_ids
                            else None
                        ),
                    )
                    with ctx.jobs_lock:
                        ctx.append_job_log(job, f"Summary: {delete_result}")
                    return

                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": 1,
                        "message": f"Starting flag fix: {fix_kind}",
                    },
                )
                result = run_flag_fix(
                    Path(dataset_obj.root),
                    ds_static,
                    fix_kind,
                    episodes=episodes,
                    data_version=data_version,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                fixed_episode_ids = sorted({int(ep) for ep in result.summary.get("episodes", [])})
                if fixed_episode_ids:
                    if ctx.clear_dataset_caches:
                        ctx.clear_dataset_caches(dataset_key)
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": 0,
                            "total": len(fixed_episode_ids),
                            "message": f"Recomputing missing cache for {len(fixed_episode_ids)} fixed episodes",
                        },
                    )
                    cache_result = run_precompute(
                        root=Path(dataset_obj.root),
                        repo_id=getattr(dataset_obj, "repo_id", None),
                        episodes=fixed_episode_ids,
                        output_dir=ds_static.parent,
                        prepare_videos=True,
                        prepare_csv=True,
                        prepare_workers=prepare_workers,
                        overwrite_csv=True,
                        data_version=data_version,
                        progress_callback=lambda payload: ctx.update_job(job, payload),
                        show_progress=False,
                    )
                    result.summary["cache_recomputed_episodes"] = fixed_episode_ids
                    result.summary["cache_total_frames"] = cache_result.total_frames
                    if ctx.clear_dataset_caches:
                        ctx.clear_dataset_caches(dataset_key)
                if ctx.append_operation_log:
                    ctx.append_operation_log(
                        ds_static,
                        "flag_fix_apply",
                        dataset_key=dataset_key,
                        dataset_root=Path(dataset_obj.root),
                        episode_ids=episodes or [],
                        details={
                            "fix_kind": fix_kind,
                            "summary": result.summary,
                            "data_version": data_version,
                            "job_id": job["id"],
                        },
                    )
                ctx.finish_job(
                    job,
                    "Flag fix complete",
                    current=result.total_episodes or 1,
                    total=result.total_episodes or 1,
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Summary: {result.summary}")
            except Exception as exc:
                logging.exception("Preprocess flag fix failed")
                ctx.fail_job(job, "Preprocess flag fix failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-flag-fix-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/standardize/start", methods=["POST"])
    def api_preprocess_standardize_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        index_entry = ctx.datasets_index.get(dataset_key)
        if index_entry is None:
            return jsonify({"error": f"dataset is not registered: {ctx.repo_id_from_key(dataset_key)}"}), 404
        dry_run = ctx.bool_option(options, "dry_run", False)
        overwrite_output = ctx.bool_option(options, "overwrite_output", False)
        prepare_workers = _positive_int_option(options, "prepare_workers", 8)
        try:
            standardize_data_version = _data_version_option(options, DATA_VERSION_DVT2)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            delete_episode_ids = sorted(
                set(
                    ctx.parse_int_list(options.get("delete_episodes") or options.get("delete_episode_ids"))
                    or []
                )
            )
        except ValueError as exc:
            return jsonify({"error": f"invalid delete episodes: {exc}"}), 400
        root_path = Path(index_entry["root"]).expanduser()
        out_root = _out_root(options) or default_standardize_path(root_path)
        job = _new_job("preprocess_standardize", ctx.repo_id_from_key(dataset_key), 100, str(out_root))

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": 100,
                        "message": "Loading dataset metadata",
                    },
                )
                dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
                if delete_episode_ids:
                    existing = set(ctx.dataset_episode_ids(dataset_obj, dataset_key))
                    missing = sorted(set(delete_episode_ids) - existing)
                    if missing:
                        raise ValueError(f"episodes not found: {missing}")
                if out_root.expanduser().resolve() == Path(dataset_obj.root).expanduser().resolve():
                    raise ValueError("output root must be different from source root")
                if out_root.exists() and not overwrite_output and not dry_run:
                    raise FileExistsError(f"output dataset already exists: {out_root}")

                source_data_version = standardize_data_version
                step_count = 4 if delete_episode_ids else 3
                source_label = f"Step 1/{step_count}"
                standardize_label = f"Step 2/{step_count}"
                delete_label = f"Step 3/{step_count}"
                repair_label = f"Step {step_count}/{step_count}"
                repair_start = 65 if delete_episode_ids else 60
                if dry_run:
                    result = run_standardize_dataset(
                        dataset_obj.root,
                        out_root=out_root,
                        data_version=source_data_version,
                        overwrite=overwrite_output,
                        dry_run=True,
                        workers=prepare_workers,
                        progress_callback=_step_progress(job, 0, 100, "Dry run"),
                    )
                    if delete_episode_ids:
                        result.summary["delete_episodes"] = delete_episode_ids
                    _register_output_dataset(result, job)
                    return

                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": 100,
                        "message": f"{source_label}: preparing source video/CSV cache if missing ({source_data_version})",
                    },
                )
                run_precompute(
                    root=dataset_obj.root,
                    repo_id=dataset_obj.repo_id,
                    output_dir=ds_static.parent,
                    prepare_videos=True,
                    prepare_csv=True,
                    prepare_workers=prepare_workers,
                    data_version=source_data_version,
                    progress_callback=_step_progress(job, 0, 25, f"{source_label} source cache"),
                    show_progress=False,
                )
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 25,
                        "total": 100,
                        "message": f"{standardize_label}: writing standardized dataset to {out_root} ({source_data_version})",
                    },
                )
                result = run_standardize_dataset(
                    dataset_obj.root,
                    out_root=out_root,
                    data_version=source_data_version,
                    overwrite=overwrite_output,
                    dry_run=False,
                    workers=prepare_workers,
                    progress_callback=_step_progress(job, 25, 60, f"{standardize_label} standardize"),
                )
                if delete_episode_ids:
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": 60,
                            "total": 100,
                            "message": f"{delete_label}: deleting episodes from standardized output: {delete_episode_ids}",
                        },
                    )
                    standardized_dataset = ctx.meta_only_dataset_cls(result.repo_id, root=result.out_root)

                    def _log_delete(message: str) -> None:
                        with ctx.jobs_lock:
                            ctx.append_job_log(job, message)

                    delete_result = delete_episodes_inplace(
                        standardized_dataset,
                        delete_episode_ids,
                        static_folder=None,
                        log=_log_delete,
                    )
                    if ctx.append_operation_log:
                        ctx.append_operation_log(
                            get_default_output_dir(result.out_root) / "static",
                            "standardize_delete_episodes",
                            dataset_key=dataset_key,
                            dataset_root=result.out_root,
                            episode_ids=delete_episode_ids,
                            details={"result": delete_result, "job_id": job["id"]},
                        )
                    result.total_episodes = int(delete_result["new_total_episodes"])
                    result.total_frames = int(
                        getattr(standardized_dataset, "total_frames", result.total_frames)
                        or result.total_frames
                    )
                    result.summary["delete_episodes"] = delete_result["deleted_episode_ids"]
                    result.summary["episodes_after_delete"] = delete_result["new_total_episodes"]
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": repair_start,
                        "total": 100,
                        "message": f"{repair_label}: repairing indices and writing stage/subtask on standardized dataset",
                    },
                )
                run_precompute(
                    root=result.out_root,
                    repo_id=result.repo_id,
                    output_dir=get_default_output_dir(result.out_root),
                    prepare_videos=True,
                    prepare_csv=True,
                    prepare_workers=prepare_workers,
                    fix_episode_indices_enabled=True,
                    annotate=True,
                    write_parquet=True,
                    force_recompute_stage=True,
                    write_subtask=True,
                    overwrite_csv=True,
                    data_version=source_data_version,
                    progress_callback=_step_progress(job, repair_start, 98, f"{repair_label} output cache"),
                    show_progress=False,
                )
                _register_output_dataset(result, job)
            except Exception as exc:
                logging.exception("Preprocess standardize failed")
                ctx.fail_job(job, "Preprocess standardize failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-standardize-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/split/start", methods=["POST"])
    def api_preprocess_split_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, _ = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        dry_run = ctx.bool_option(options, "dry_run", False)
        total = int(getattr(dataset_obj, "total_episodes", 1) or 1)
        job = _new_job(
            "preprocess_split", ctx.repo_id_from_key(dataset_key), total, str(_out_root(options) or "")
        )

        def _run_job() -> None:
            try:
                ctx.update_job(job, {"status": "running", "message": "Starting dataset split"})
                result = run_split(
                    dataset_obj.root,
                    out_root=_out_root(options),
                    episode_range=options.get("episode_range"),
                    task_filter=options.get("task_filter"),
                    dry_run=dry_run,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                _register_output_dataset(result, job)
            except Exception as exc:
                logging.exception("Preprocess split failed")
                ctx.fail_job(job, "Preprocess split failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-split-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/merge/start", methods=["POST"])
    def api_preprocess_merge_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        raw_keys = options.get("src_keys") or body.get("src_keys") or []
        if isinstance(raw_keys, str):
            raw_keys = [item.strip() for item in raw_keys.replace("\n", ",").split(",") if item.strip()]
        if len(raw_keys) < 2:
            return jsonify({"error": "merge requires at least two source datasets"}), 400
        try:
            keys = [ctx.repo_key(str(key)) for key in raw_keys]
        except KeyError as exc:
            return jsonify({"error": f"dataset is not registered: {exc}"}), 404
        missing = [ctx.repo_id_from_key(key) for key in keys if key not in ctx.datasets_index]
        if missing:
            return jsonify({"error": f"dataset is not registered: {missing[0]}"}), 404
        dry_run = ctx.bool_option(options, "dry_run", False)
        try:
            workers = max(1, int(options.get("workers") or 8))
        except (TypeError, ValueError):
            return jsonify({"error": "workers must be an integer"}), 400
        raw_delete_by_key = options.get("delete_episodes_by_key") or {}
        if raw_delete_by_key and not isinstance(raw_delete_by_key, dict):
            return jsonify({"error": "delete_episodes_by_key must be an object keyed by dataset"}), 400
        try:
            delete_by_key = {}
            for raw_key, key in zip(raw_keys, keys, strict=False):
                repo_id = ctx.repo_id_from_key(key)
                raw_value = (
                    raw_delete_by_key.get(str(raw_key))
                    or raw_delete_by_key.get(repo_id)
                    or raw_delete_by_key.get(key)
                )
                delete_by_key[key] = sorted(set(ctx.parse_int_list(raw_value) or []))
        except ValueError as exc:
            return jsonify({"error": f"invalid merge delete episodes: {exc}"}), 400
        try:
            out_root = _merge_out_root(options)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        job = _new_job(
            "preprocess_merge",
            ",".join(ctx.repo_id_from_key(key) for key in keys),
            len(keys),
            str(out_root or ""),
            related_dataset_keys=[ctx.repo_id_from_key(key) for key in keys],
        )

        def _run_job() -> None:
            try:
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": len(keys),
                        "message": f"Loading {len(keys)} source datasets",
                    },
                )
                entries = []
                for idx, key in enumerate(keys, start=1):
                    repo_id = ctx.repo_id_from_key(key)
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": idx - 1,
                            "total": len(keys),
                            "message": f"Loading source dataset {idx}/{len(keys)}: {repo_id}",
                        },
                    )
                    entries.append(ctx.ensure_dataset_loaded(key))
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": idx,
                            "total": len(keys),
                            "message": f"Loaded source dataset {idx}/{len(keys)}: {repo_id}",
                        },
                    )

                total = (
                    sum(int(getattr(dataset_obj, "total_episodes", 0) or 0) for dataset_obj, _ in entries)
                    or 1
                )
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": total,
                        "message": "Starting dataset merge",
                    },
                )
                result = run_merge(
                    [dataset_obj.root for dataset_obj, _ in entries],
                    out_root=out_root,
                    dry_run=dry_run,
                    src_static_dirs=[ds_static for _dataset_obj, ds_static in entries],
                    out_static_dir=(get_default_output_dir(out_root) / "static")
                    if out_root is not None
                    else None,
                    workers=workers,
                    exclude_episodes=[delete_by_key.get(key) for key in keys],
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                _register_output_dataset(result, job)
            except Exception as exc:
                logging.exception("Preprocess merge failed")
                ctx.fail_job(job, "Preprocess merge failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-merge-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/preprocess/subtract/start", methods=["POST"])
    def api_preprocess_subtract_start():
        body = request.get_json(silent=True) or {}
        options = body.get("options") or body
        raw_base_key = options.get("base_key") or options.get("dataset_key") or body.get("dataset_key")
        if not raw_base_key:
            return jsonify({"error": "subtract requires a base dataset"}), 400
        raw_keys = options.get("src_keys") or body.get("src_keys") or []
        if isinstance(raw_keys, str):
            raw_keys = [item.strip() for item in raw_keys.replace("\n", ",").split(",") if item.strip()]
        if not raw_keys:
            return jsonify({"error": "subtract requires at least one subtract dataset"}), 400
        try:
            base_key = ctx.repo_key(str(raw_base_key))
            keys = [ctx.repo_key(str(key)) for key in raw_keys]
        except KeyError as exc:
            return jsonify({"error": f"dataset is not registered: {exc}"}), 404
        all_keys = [base_key] + keys
        missing = [ctx.repo_id_from_key(key) for key in all_keys if key not in ctx.datasets_index]
        if missing:
            return jsonify({"error": f"dataset is not registered: {missing[0]}"}), 404
        dry_run = ctx.bool_option(options, "dry_run", False)
        try:
            workers = max(1, int(options.get("workers") or 8))
        except (TypeError, ValueError):
            return jsonify({"error": "workers must be an integer"}), 400
        try:
            out_root = _merge_out_root(options)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        base_repo_id = ctx.repo_id_from_key(base_key)
        subtract_repo_ids = [ctx.repo_id_from_key(key) for key in keys]
        job = _new_job(
            "preprocess_subtract",
            base_repo_id,
            1,
            str(out_root or ""),
            related_dataset_keys=[base_repo_id, *subtract_repo_ids],
        )

        def _run_job() -> None:
            try:
                ctx.update_job(job, {"status": "running", "message": f"Loading base dataset: {base_repo_id}"})
                base_dataset, base_static = ctx.ensure_dataset_loaded(base_key)
                subtract_entries = []
                for idx, key in enumerate(keys, start=1):
                    repo_id = ctx.repo_id_from_key(key)
                    ctx.update_job(
                        job,
                        {
                            "status": "running",
                            "current": idx - 1,
                            "total": len(keys),
                            "message": f"Loading subtract dataset {idx}/{len(keys)}: {repo_id}",
                        },
                    )
                    subtract_entries.append(ctx.ensure_dataset_loaded(key))
                ctx.update_job(
                    job,
                    {
                        "status": "running",
                        "current": 0,
                        "total": int(getattr(base_dataset, "total_episodes", 1) or 1),
                        "message": "Starting dataset subtract",
                    },
                )
                result = run_subtract(
                    base_dataset.root,
                    [dataset_obj.root for dataset_obj, _ds_static in subtract_entries],
                    out_root=out_root,
                    dry_run=dry_run,
                    src_static_dir=base_static,
                    out_static_dir=(get_default_output_dir(out_root) / "static")
                    if out_root is not None
                    else None,
                    workers=workers,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                _register_output_dataset(result, job)
            except Exception as exc:
                logging.exception("Preprocess subtract failed")
                ctx.fail_job(job, "Preprocess subtract failed", exc)

        threading.Thread(target=_run_job, name=f"preprocess-subtract-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})
