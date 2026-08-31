import logging
import math
import threading
import time
import uuid
from pathlib import Path

from flask import jsonify, render_template, request

from lerobot.data_platform.cli import get_default_output_dir
from lerobot.data_platform.precompute.construction import (
    default_synthetic_path,
    preview_construction,
    run_construction,
)
from lerobot.data_platform.precompute.construction.review import (
    finalize as finalize_construction,
)
from lerobot.data_platform.precompute.construction.review import (
    load_construction_doc,
    load_construction_records,
)
from lerobot.data_platform.precompute.construction.review import (
    save_decision as save_construction_decision,
)
from lerobot.data_platform.precompute.labeling.review import (
    load_episode_record as load_labeling_episode_record,
)
from lerobot.data_platform.routes.context import RouteContext


def register_construction_routes(app, ctx: RouteContext) -> None:
    @app.route("/api/construction/preview", methods=["POST"])
    def api_construction_preview():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
            threshold = int(body.get("uncertainty_threshold") or 50)
            options = body.get("options") or body
            allow_pick_to_give = ctx.bool_option(options, "allow_pick_to_give", False)
            preview = preview_construction(
                dataset_obj.meta,
                ds_static / "labeling",
                threshold,
                allow_pick_to_give=allow_pick_to_give,
            )
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        except FileNotFoundError as exc:
            if str(exc) == "object_labeling_required":
                return jsonify({"error": "object_labeling_required"}), 400
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logging.exception("Failed to preview data construction")
            return jsonify({"error": str(exc)}), 400
        return jsonify(preview)

    @app.route("/api/construction/start", methods=["POST"])
    def api_start_construction():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        except Exception as exc:
            logging.exception("Failed to load dataset for data construction")
            return jsonify({"error": str(exc)}), 400

        options = body.get("options") or body
        threshold = int(options.get("uncertainty_threshold") or 50)
        oversample_factor = max(1.0, float(options.get("oversample_factor") or 1.0))
        per_scenario_counts = {
            str(key): int(value or 0) for key, value in dict(options.get("per_scenario_counts") or {}).items()
        }
        include_positives = ctx.bool_option(options, "include_positives", False)
        allow_pick_to_give = ctx.bool_option(options, "allow_pick_to_give", False)
        out_root_value = str(options.get("out_root") or "").strip()
        out_root = (
            Path(out_root_value).expanduser() if out_root_value else default_synthetic_path(dataset_obj.root)
        )
        config = {
            "uncertainty_threshold": threshold,
            "per_scenario_counts": per_scenario_counts,
            "include_positives": include_positives,
            "oversample_factor": oversample_factor,
            "allow_pick_to_give": allow_pick_to_give,
        }

        total = max(1, sum(math.ceil(count * oversample_factor) for count in per_scenario_counts.values()))
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        source_repo_id = ctx.repo_id_from_key(dataset_key)
        job = {
            "id": job_id,
            "job_type": "construction",
            "dataset_key": source_repo_id,
            "status": "queued",
            "progress": 0,
            "current": 0,
            "total": total,
            "message": "Queued",
            "error": None,
            "viewer_url": None,
            "review_url": None,
            "output_root": str(out_root),
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
                ctx.update_job(job, {"status": "running", "message": "Starting data construction"})
                result = run_construction(
                    src_root=dataset_obj.root,
                    meta=dataset_obj.meta,
                    labeling_dir=ds_static / "labeling",
                    out_root=out_root,
                    config=config,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                synthetic = ctx.meta_only_dataset_cls(result.repo_id, root=result.out_root)
                synthetic_key = ctx.register_dataset(synthetic, get_default_output_dir(result.out_root))
                review_url = f"/{ctx.repo_id_from_key(synthetic_key)}/construction"
                distribution = {}
                source_distribution = {}
                for plan in result.plans:
                    distribution[plan.missing_object] = distribution.get(plan.missing_object, 0) + 1
                    source_obj = plan.source_visual_object or "unknown"
                    source_distribution[source_obj] = source_distribution.get(source_obj, 0) + 1
                ctx.finish_job(
                    job,
                    "Data construction complete",
                    current=len(result.plans),
                    total=len(result.plans),
                    review_url=review_url,
                    output_dataset_key=ctx.repo_id_from_key(synthetic_key),
                    output_root=str(result.out_root),
                )
                with ctx.jobs_lock:
                    ctx.append_job_log(job, f"Missing-object distribution: {distribution}")
                    ctx.append_job_log(job, f"Source-object distribution: {source_distribution}")
            except Exception as exc:
                logging.exception("Data construction job failed")
                ctx.fail_job(job, "Data construction failed", exc)

        threading.Thread(target=_run_job, name=f"construction-{job_id}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/construction")
    def show_construction(dataset_namespace, dataset_name):
        dataset_key = (dataset_namespace, dataset_name)
        dataset_obj, ds_static = ctx.get_ctx(dataset_namespace, dataset_name)
        repo_id = ctx.repo_id_from_key(dataset_key)
        episode_ids = ctx.dataset_episode_ids(dataset_obj, dataset_key)
        first_episode = episode_ids[0] if episode_ids else 0
        return render_template(
            "visualize_dataset_construction.html",
            dataset_namespace=dataset_namespace,
            dataset_name=dataset_name,
            dataset_key=repo_id,
            viewer_url=f"/{repo_id}/episode_{first_episode}",
            legacy_mutations_enabled=bool(getattr(ctx, "legacy_mutations_enabled", False)),
            **ctx.dataset_nav(repo_id, first_episode, "construction", dataset_obj, ds_static),
        )

    def _construction_record_payload(dataset_key: tuple[str, str], record: dict) -> dict:
        dataset_obj, _ = ctx.ensure_dataset_loaded(dataset_key)
        doc = load_construction_doc(dataset_obj.root)
        source_repo_id = doc.get("source_repo_id")
        image_url = ""
        bboxes = []
        if source_repo_id:
            source_key = ctx.repo_key(source_repo_id)
            try:
                source_obj, source_static = ctx.ensure_dataset_loaded(source_key)
                image_keys = ctx.dataset_image_keys(source_obj)
                if image_keys:
                    image_url = (
                        f"/{source_repo_id}/image?key={image_keys[0]}"
                        f"&episode={int(record['src_episode_index'])}&frame=0"
                    )
                labeling_record = load_labeling_episode_record(
                    source_static / "labeling",
                    int(record["src_episode_index"]),
                )
                if labeling_record is not None:
                    current = labeling_record.get("current") or {}
                    parsed = current.get("parsed") or {}
                    if current.get("selected"):
                        selected = dict(current["selected"])
                        selected["label"] = parsed.get("target") or "target"
                        selected["kind"] = "target"
                        bboxes.append(selected)
                    for ref in current.get("detections_ref", [])[:1]:
                        ref_box = dict(ref)
                        ref_box["label"] = parsed.get("reference") or "reference"
                        ref_box["kind"] = "reference"
                        bboxes.append(ref_box)
            except Exception:
                logging.exception("Failed to build construction source preview for %s", source_repo_id)
        return {**record, "source_repo_id": source_repo_id, "image_url": image_url, "bboxes": bboxes}

    @app.route("/api/construction/<string:dataset_namespace>/<string:dataset_name>/records")
    def api_construction_records(dataset_namespace, dataset_name):
        dataset_key = (dataset_namespace, dataset_name)
        dataset_obj, _ = ctx.get_ctx(dataset_namespace, dataset_name)
        records = load_construction_records(dataset_obj.root)
        return jsonify({"records": [_construction_record_payload(dataset_key, record) for record in records]})

    @app.route("/api/construction/<string:dataset_namespace>/<string:dataset_name>/record/<int:new_idx>")
    def api_construction_record(dataset_namespace, dataset_name, new_idx):
        dataset_key = (dataset_namespace, dataset_name)
        dataset_obj, _ = ctx.get_ctx(dataset_namespace, dataset_name)
        for record in load_construction_records(dataset_obj.root):
            if int(record["new_episode_index"]) == int(new_idx):
                return jsonify(_construction_record_payload(dataset_key, record))
        return jsonify({"error": "not found"}), 404

    @app.route(
        "/api/construction/<string:dataset_namespace>/<string:dataset_name>/decision/<int:new_idx>",
        methods=["POST"],
    )
    def api_construction_decision(dataset_namespace, dataset_name, new_idx):
        dataset_obj, _ = ctx.get_ctx(dataset_namespace, dataset_name)
        body = request.get_json(silent=True) or {}
        try:
            record = save_construction_decision(
                dataset_obj.root,
                new_idx,
                str(body.get("decision") or ""),
                str(body.get("reason") or "") or None,
            )
        except KeyError:
            return jsonify({"error": "not found"}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"record": record})

    @app.route(
        "/api/construction/<string:dataset_namespace>/<string:dataset_name>/finalize", methods=["POST"]
    )
    def api_construction_finalize(dataset_namespace, dataset_name):
        dataset_key = (dataset_namespace, dataset_name)
        dataset_obj, ds_static = ctx.get_ctx(dataset_namespace, dataset_name)
        try:
            result = finalize_construction(dataset_obj.root)
            refreshed = ctx.meta_only_dataset_cls(ctx.repo_id_from_key(dataset_key), root=dataset_obj.root)
            ctx.register_dataset(refreshed, ds_static.parent)
        except Exception as exc:
            logging.exception("Failed to finalize construction review")
            return jsonify({"error": str(exc)}), 500
        return jsonify({"status": "ok", **result})
