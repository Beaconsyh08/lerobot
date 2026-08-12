import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class RouteContext:
    """Shared state and helpers for Data Platform feature route modules."""

    datasets_index: dict
    jobs_registry: dict
    jobs_lock: Any
    tagging_status_cache: dict
    repo_id_from_key: Callable[[tuple[str, str]], str]
    repo_key: Callable[[str], tuple[str, str]]
    get_ctx: Callable[[str, str], tuple[object, Path]]
    ensure_dataset_loaded: Callable[[tuple[str, str]], tuple[object, Path]]
    register_dataset: Callable
    dataset_episode_ids: Callable[[object, tuple[str, str]], list[int]]
    dataset_image_keys: Callable[[object], list[str]]
    dataset_nav: Callable
    serialize_dataset_light: Callable[[tuple[str, str]], dict]
    serialize_job: Callable[[dict], dict]
    append_job_log: Callable[[dict, str], None]
    job_timing_snapshot: Callable[[dict, float | None], tuple[int, int | None]]
    parse_int_list: Callable[[Any], list[int] | None]
    parse_str_list: Callable[[Any], list[str] | None]
    bool_option: Callable[[dict, str, bool], bool]
    dataset_key_from_body: Callable[[dict], tuple[str, str]]
    active_tag_variant: Callable[[Path, str | None], str | None]
    meta_only_dataset_cls: type
    append_operation_log: Callable[..., None] | None = None
    audit_job: Callable[[dict, str, Exception | None], None] | None = None
    clear_dataset_caches: Callable[[tuple[str, str] | None], None] | None = None
    refresh_dataset_after_episode_delete: Callable[[tuple[str, str], object, Path], list[int]] | None = None
    static_dir_for_key: Callable[[tuple[str, str]], Path | None] | None = None

    def update_job(self, job: dict, payload: dict) -> None:
        with self.jobs_lock:
            update_time = time.time()
            status = payload.get("status")
            if status and status != "done":
                job["status"] = status
            if job.get("status") == "running" and job.get("started_at") is None:
                job["started_at"] = update_time
            if "current" in payload:
                job["current"] = payload["current"]
            if "total" in payload:
                job["total"] = payload["total"]
            total_value = job.get("total") or 0
            current = job.get("current") or 0
            job["progress"] = int((current / total_value) * 100) if total_value else 0
            if payload.get("message"):
                job["message"] = payload["message"]
                self.append_job_log(job, payload["message"])
            elapsed_seconds, eta_seconds = self.job_timing_snapshot(job, update_time)
            job["elapsed_seconds"] = elapsed_seconds
            job["eta_seconds"] = eta_seconds
            job["updated_at"] = update_time

    def finish_job(self, job: dict, message: str, **updates) -> None:
        with self.jobs_lock:
            finished_at = time.time()
            job["status"] = "done"
            job["progress"] = 100
            job["message"] = message
            job["finished_at"] = finished_at
            job.update(updates)
            elapsed_seconds, eta_seconds = self.job_timing_snapshot(job, finished_at)
            job["elapsed_seconds"] = elapsed_seconds
            job["eta_seconds"] = eta_seconds
            job["updated_at"] = finished_at
            self.append_job_log(job, message)
        if self.audit_job is not None:
            try:
                self.audit_job(job, "success", None)
            except Exception:
                logging.exception("Failed to record successful job audit")

    def fail_job(self, job: dict, message: str, exc: Exception) -> None:
        with self.jobs_lock:
            finished_at = time.time()
            job["status"] = "error"
            job["error"] = str(exc)
            job["message"] = message
            job["finished_at"] = finished_at
            elapsed_seconds, eta_seconds = self.job_timing_snapshot(job, finished_at)
            job["elapsed_seconds"] = elapsed_seconds
            job["eta_seconds"] = eta_seconds
            job["updated_at"] = finished_at
            self.append_job_log(job, f"Error: {exc}")
        if self.audit_job is not None:
            try:
                self.audit_job(job, "failed", exc)
            except Exception:
                logging.exception("Failed to record failed job audit")

    def invalidate_tagging_status(self, dataset_key: tuple[str, str], ds_static: Path) -> None:
        self.tagging_status_cache.pop(
            (self.repo_id_from_key(dataset_key), str(Path(ds_static).expanduser())),
            None,
        )
