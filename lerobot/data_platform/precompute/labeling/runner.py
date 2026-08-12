from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from lerobot.data_platform.precompute.dataset_io import read_episode_table
from lerobot.data_platform.precompute.labeling.bbox_select import select_bbox_with_context
from lerobot.data_platform.precompute.labeling.detector import (
    DEFAULT_BACKEND,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT_THRESHOLD,
    cuda_devices,
    load_detector,
)
from lerobot.data_platform.precompute.labeling.review import (
    image_keys_from_meta,
    labels_path,
    load_labels_jsonl,
    migrate_latest_labels_to_variant,
    read_first_frame_image,
    read_frame_image,
    resolved_labels_path,
    resolved_reviewed_path,
    reviewed_path,
    source_path,
)
from lerobot.data_platform.precompute.labeling.task_parser import expand_prompts, parse_task
from lerobot.data_platform.precompute.labeling.vis import draw_detections
from lerobot.data_platform.precompute.tagging.review import current_tags

LABELING_FLAGGED_EPISODES = "labeling_flagged_episodes.json"
LABELING_MISSING_TARGET_ISSUE_TYPE = "object_labeling"
LABELING_MISSING_TARGET_REASON = "missing_target_detection"
LABELING_RUN_MODE_MISSING = "missing"
LABELING_RUN_MODE_FULL = "full"
LABELING_RUN_MODES = {LABELING_RUN_MODE_MISSING, LABELING_RUN_MODE_FULL}


@dataclass
class LabelingResult:
    root: Path
    repo_id: str
    static_dir: Path
    labeling_dir: Path
    labels_path: Path
    episodes: list[int]
    image_key: str
    model_id: str
    backend: str


@dataclass
class EpisodeSampleResult:
    episodes: list[int]
    counts: dict[str, int]
    available_counts: dict[str, int]
    seed: int


def _emit_progress(progress_callback: Callable[[dict], None] | None, **payload) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _load_json_any(path: Path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp_path.replace(path)


def _load_flagged_episodes(path: Path) -> set[int]:
    data = _load_json_any(path)
    if not isinstance(data, dict):
        return set()
    out = set()
    for value in data.get("flagged_episodes") or []:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _issue_episode(issue: dict) -> int | None:
    try:
        return int(issue.get("episode"))
    except (TypeError, ValueError):
        return None


def _is_labeling_missing_target_issue(issue: dict) -> bool:
    return (
        isinstance(issue, dict)
        and issue.get("type") == LABELING_MISSING_TARGET_ISSUE_TYPE
        and issue.get("reason") == LABELING_MISSING_TARGET_REASON
    )


def _episode_task_from_meta(meta, episode_index: int) -> str | None:
    episode_info = getattr(meta, "episodes", {}).get(episode_index)
    if episode_info is None:
        episode_info = getattr(meta, "episodes", {}).get(str(episode_index))
    tasks = episode_info.get("tasks", []) if episode_info else []
    return tasks[0] if tasks else None


def _episode_task_from_parquet(root: Path, meta, episode_index: int) -> str | None:
    parquet_path = root / meta.get_data_file_path(episode_index)
    if "task_index" not in pq.read_schema(parquet_path).names:
        return None
    table = read_episode_table(root, meta, episode_index, columns=["task_index"])
    if len(table) == 0:
        return None
    task_index = int(table["task_index"][0].as_py())
    tasks = getattr(meta, "tasks", {}) or {}
    return tasks.get(task_index) or tasks.get(str(task_index))


def _get_episode_task(root: Path, meta, episode_index: int) -> str:
    return (
        _episode_task_from_meta(meta, episode_index)
        or _episode_task_from_parquet(root, meta, episode_index)
        or f"unknown_task_{episode_index}"
    )


def _value_is_zero(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_value_is_zero(item) for item in value)
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _episode_exist_label_zero(root: Path, meta, episode_index: int) -> bool:
    parquet_path = root / meta.get_data_file_path(episode_index)
    column_names = pq.read_schema(parquet_path).names
    column_name = None
    if "exist_label" in column_names:
        column_name = "exist_label"
    elif "exist" in column_names:
        column_name = "exist"
    if column_name is None:
        return False

    table = read_episode_table(root, meta, episode_index, columns=[column_name])
    if len(table) == 0:
        return False
    return _value_is_zero(table[column_name][0].as_py())


def _normalize_devices(devices: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if devices in (None, "", []):
        return []
    if isinstance(devices, str):
        tokens = devices.replace(",", " ").split()
    else:
        tokens = [str(item) for item in devices]
    out = []
    for token in tokens:
        value = token.strip()
        if not value:
            continue
        if value.isdigit():
            value = f"cuda:{value}"
        out.append(value)
    return out


def _object_prompt(noun: str) -> str:
    return ". ".join(expand_prompts(noun))


def _target_prompt_for_backend(parsed: dict, backend: str) -> str:
    return _object_prompt(parsed["target"])


def _episode_length(meta, episode_index: int) -> int | None:
    episode_info = getattr(meta, "episodes", {}).get(episode_index)
    if episode_info is None:
        episode_info = getattr(meta, "episodes", {}).get(str(episode_index))
    if not episode_info:
        return None
    try:
        return int(episode_info.get("length"))
    except (TypeError, ValueError):
        return None


def _load_current_tagging(static_dir: Path) -> dict[int, dict]:
    try:
        return current_tags(Path(static_dir) / "tagging")
    except FileNotFoundError:
        return {}


def _tag_values_for_episode(tags_by_episode: dict[int, dict], episode_index: int) -> dict:
    record = tags_by_episode.get(int(episode_index)) or {}
    tags = record.get("tags") or {}
    return dict(tags) if isinstance(tags, dict) else {}


def _active_arm_from_tags(tags: dict) -> str | None:
    value = str(tags.get("arm") or "").strip().lower()
    if value in {"left", "right"}:
        return value
    return None


def _tag_context_for_labeling(tags: dict) -> dict:
    context = {}
    for key in ("arm", "object_count", "background", "background_color", "prompt_action_match"):
        if key in tags and tags.get(key) not in (None, ""):
            context[key] = tags.get(key)
    return context


def _context_warnings_for_labeling(tag_context: dict) -> list[str]:
    warnings = []
    if tag_context.get("prompt_action_match") == "mismatch":
        warnings.append("prompt_action_mismatch")
    return warnings


def _missing_target_issue_for_record(record: dict, *, backend: str, variant: str) -> dict | None:
    parsed = record.get("parsed")
    if parsed is None:
        return None
    if record.get("skip_reason") == "exist_label_zero":
        return None
    if record.get("error"):
        return None
    if record.get("detections_target"):
        return None
    try:
        episode_index = int(record["episode_index"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "episode": episode_index,
        "type": LABELING_MISSING_TARGET_ISSUE_TYPE,
        "reason": LABELING_MISSING_TARGET_REASON,
        "task": record.get("task"),
        "target": parsed.get("target"),
        "prompt": _target_prompt_for_backend(parsed, backend),
        "backend": backend,
        "variant": variant,
        "metrics": {"target_detection_count": 0},
    }


def _sync_missing_target_flags(
    static_dir: Path,
    records_by_episode: dict[int, dict],
    *,
    backend: str,
    variant: str,
    scanned_episodes: set[int],
) -> dict:
    static_dir = Path(static_dir)
    issues_path = static_dir / "annotation_issues.json"
    auto_path = static_dir / LABELING_FLAGGED_EPISODES
    flagged_path = static_dir / "flagged_episodes.json"

    missing_issues = [
        issue
        for record in records_by_episode.values()
        if (issue := _missing_target_issue_for_record(record, backend=backend, variant=variant)) is not None
    ]
    missing_episodes = {int(issue["episode"]) for issue in missing_issues}

    existing_issues = _load_json_any(issues_path)
    if not isinstance(existing_issues, list):
        existing_issues = []
    kept_issues = [
        issue
        for issue in existing_issues
        if isinstance(issue, dict)
        and not (_is_labeling_missing_target_issue(issue) and (_issue_episode(issue) in scanned_episodes))
    ]
    merged_issues = sorted(
        kept_issues + missing_issues,
        key=lambda issue: (
            int(issue.get("episode", -1)),
            str(issue.get("type", "")),
            str(issue.get("reason", "")),
        ),
    )
    _write_json(issues_path, merged_issues)

    previous_auto = _load_flagged_episodes(auto_path)
    next_auto = (previous_auto - scanned_episodes) | missing_episodes
    current_flagged = _load_flagged_episodes(flagged_path)
    manual_or_other_flags = current_flagged - previous_auto
    combined_flags = manual_or_other_flags | next_auto
    _write_json(auto_path, {"flagged_episodes": sorted(next_auto)})
    _write_json(flagged_path, {"flagged_episodes": sorted(combined_flags)})

    return {
        "missing_target_count": len(missing_episodes),
        "annotation_issue_count": len(merged_issues),
        "flagged_episode_count": len(combined_flags),
    }


def labeling_task_type(parsed: dict | None) -> str:
    if parsed is None:
        return "unsupported"
    if parsed.get("action") == "give":
        return "give"
    if parsed.get("direction") and parsed.get("reference"):
        return "relative"
    if parsed.get("direction"):
        return "absolute"
    return "single"


def sample_episodes_by_task_type(
    root: Path,
    meta,
    per_type: int = 20,
    episodes: list[int] | None = None,
    seed: int | None = None,
) -> EpisodeSampleResult:
    selected_episodes = sorted(getattr(meta, "episodes", {}).keys()) if episodes is None else list(episodes)
    limit = max(1, int(per_type or 20))
    buckets: dict[str, list[int]] = {}
    available_counts: dict[str, int] = {}
    seed_value = int(time.time()) if seed is None else int(seed)
    rng = random.Random(seed_value)

    for episode_index in selected_episodes:
        task_name = _get_episode_task(Path(root), meta, int(episode_index))
        task_type = labeling_task_type(parse_task(task_name))
        if task_type == "unsupported":
            continue
        available_counts[task_type] = available_counts.get(task_type, 0) + 1
        buckets.setdefault(task_type, []).append(int(episode_index))

    sampled = set()
    for bucket in buckets.values():
        if len(bucket) <= limit:
            sampled.update(bucket)
        else:
            sampled.update(rng.sample(bucket, limit))
    ordered = [int(episode_index) for episode_index in selected_episodes if int(episode_index) in sampled]
    return EpisodeSampleResult(
        episodes=ordered,
        counts={task_type: min(limit, len(values)) for task_type, values in sorted(buckets.items())},
        available_counts=dict(sorted(available_counts.items())),
        seed=seed_value,
    )


def _write_source_json(
    path: Path,
    *,
    root: Path,
    repo_id: str,
    model_id: str,
    episodes: list[int],
    image_key: str,
    backend: str,
    box_threshold: float,
    text_threshold: float,
    save_vis: bool,
    workers: int,
    devices: list[str],
    output_variant: str,
    run_mode: str,
    requested_episodes: list[int],
    skipped_existing: int,
    endpoint: str | None = None,
    qwen_model: str | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "repo_id": repo_id,
                "model_id": model_id,
                "backend": backend,
                "endpoint": endpoint,
                "qwen_model": qwen_model,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "command": "run_labeling",
                "episodes": episodes,
                "requested_episodes": requested_episodes,
                "image_key": image_key,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "save_vis": save_vis,
                "workers": workers,
                "devices": devices,
                "output_variant": output_variant,
                "run_mode": run_mode,
                "skipped_existing": skipped_existing,
                "created_at": int(time.time()),
            },
            indent=2,
        )
    )


def run_labeling(
    root: Path,
    meta,
    episodes: list[int] | None,
    static_dir: Path,
    *,
    backend: str = DEFAULT_BACKEND,
    model_id: str = DEFAULT_MODEL_ID,
    endpoint: str | None = None,
    qwen_model: str | None = None,
    qwen_token: str | None = None,
    min_pixels: int = 1024,
    max_pixels: int = 9800,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    save_vis: bool = False,
    workers: int = 1,
    devices: str | list[str] | None = None,
    output_variant: str | None = None,
    run_mode: str = LABELING_RUN_MODE_MISSING,
    progress_callback: Callable[[dict], None] | None = None,
    show_progress: bool = True,
) -> LabelingResult:
    root = Path(root)
    static_dir = Path(static_dir)
    labeling_dir = static_dir / "labeling"
    labeling_dir.mkdir(parents=True, exist_ok=True)
    migrate_latest_labels_to_variant(labeling_dir)
    vis_dir = labeling_dir / "vis"
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    run_mode = str(run_mode or LABELING_RUN_MODE_MISSING).strip().lower()
    if run_mode not in LABELING_RUN_MODES:
        raise ValueError(
            f"Unknown object labeling run_mode: {run_mode}. Expected one of {sorted(LABELING_RUN_MODES)}"
        )

    requested_episodes = sorted(getattr(meta, "episodes", {}).keys()) if episodes is None else list(episodes)
    selected_episodes = []
    skipped_unsupported = 0
    for episode_index in requested_episodes:
        task_name = _get_episode_task(root, meta, int(episode_index))
        if parse_task(task_name) is None:
            skipped_unsupported += 1
            continue
        selected_episodes.append(int(episode_index))
    result_variant = output_variant or backend
    write_latest = output_variant is None
    labels_file = labels_path(labeling_dir)
    variant_labels_file = labels_path(labeling_dir, result_variant)
    image_key = ""
    image_keys = image_keys_from_meta(meta)
    if image_keys:
        image_key = image_keys[0]
    effective_workers = max(1, int(workers or 1))
    device_list = _normalize_devices(devices)
    explicit_devices = bool(device_list)
    if backend == "grounding_dino":
        if effective_workers > 1 and not device_list:
            device_list = cuda_devices()[:effective_workers]
        if effective_workers > 1 and device_list and not explicit_devices:
            effective_workers = min(effective_workers, len(device_list))
        if effective_workers > 1 and not device_list:
            effective_workers = 1
    elif backend not in {"qwen_remote", "qwen_dashscope"}:
        effective_workers = 1

    loaded_model_id = qwen_model if backend in {"qwen_remote", "qwen_dashscope"} and qwen_model else model_id
    label_output_paths = {variant_labels_file}
    source_output_paths = {source_path(labeling_dir, result_variant)}
    if write_latest:
        label_output_paths.add(labels_file)
        source_output_paths.add(source_path(labeling_dir))

    existing_label_records: dict[int, dict] = {}
    existing_reviewed_records: dict[int, dict] = {}
    skipped_existing = 0
    if run_mode == LABELING_RUN_MODE_MISSING:
        existing_label_records = load_labels_jsonl(resolved_labels_path(labeling_dir, result_variant))
        existing_reviewed_records = load_labels_jsonl(resolved_reviewed_path(labeling_dir, result_variant))
        completed_episodes = set(existing_label_records) | set(existing_reviewed_records)
        before_filter = len(selected_episodes)
        selected_episodes = [
            episode_index for episode_index in selected_episodes if episode_index not in completed_episodes
        ]
        skipped_existing = before_filter - len(selected_episodes)

    if not selected_episodes:
        for output_path in label_output_paths:
            if run_mode == LABELING_RUN_MODE_FULL:
                output_path.write_text("")
        for output_path in source_output_paths:
            _write_source_json(
                output_path,
                root=root,
                repo_id=getattr(meta, "repo_id", "local/dataset"),
                model_id=loaded_model_id,
                episodes=selected_episodes,
                image_key=image_key,
                backend=backend,
                endpoint=endpoint,
                qwen_model=qwen_model,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                save_vis=save_vis,
                workers=effective_workers,
                devices=device_list,
                output_variant=result_variant,
                run_mode=run_mode,
                requested_episodes=[int(episode_index) for episode_index in requested_episodes],
                skipped_existing=skipped_existing,
            )
        if run_mode == LABELING_RUN_MODE_MISSING and skipped_existing:
            message = f"No missing object labels to run ({skipped_existing} existing episodes kept)"
        else:
            message = f"No reviewable pick/give episodes found ({skipped_unsupported} unsupported/place episodes skipped)"
        _emit_progress(
            progress_callback,
            status="done",
            step="labeling",
            current=0,
            total=0,
            message=message,
        )
        return LabelingResult(
            root=root,
            repo_id=getattr(meta, "repo_id", "local/dataset"),
            static_dir=static_dir,
            labeling_dir=labeling_dir,
            labels_path=variant_labels_file,
            episodes=[],
            image_key=image_key,
            model_id=loaded_model_id,
            backend=backend,
        )

    _emit_progress(
        progress_callback,
        status="running",
        step="labeling_model",
        current=0,
        total=len(selected_episodes),
        message=(
            f"Loading object detector backend {backend}"
            + (f" ({skipped_unsupported} unsupported/place episodes skipped)" if skipped_unsupported else "")
            + (f" ({skipped_existing} existing episodes skipped)" if skipped_existing else "")
        ),
    )
    detector = None
    if effective_workers == 1:
        detector = load_detector(
            backend,
            model_id=model_id,
            device=device_list[0] if device_list else None,
            endpoint=endpoint,
            model=qwen_model,
            token=qwen_token,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        loaded_model_id = getattr(detector, "model_id", model_id)
        logging.info(
            "Loaded object labeling backend %s model %s on %s",
            backend,
            loaded_model_id,
            detector.device,
        )
    else:
        logging.info(
            "Loading object labeling backend %s model %s with %s workers on %s",
            backend,
            loaded_model_id,
            effective_workers,
            device_list or ["remote"],
        )
    _emit_progress(
        progress_callback,
        status="running",
        step="labeling_model",
        current=0,
        total=len(selected_episodes),
        message=(
            f"Loaded {backend}:{loaded_model_id} on {detector.device}"
            if detector is not None
            else f"Starting {backend}:{loaded_model_id} with {effective_workers} workers"
        ),
    )

    failed_count = 0
    missing_target_count = 0
    progress = (
        tqdm(total=len(selected_episodes), desc="Object labeling", unit="episode", dynamic_ncols=True)
        if show_progress
        else None
    )

    def _write_source_files() -> None:
        for output_path in source_output_paths:
            _write_source_json(
                output_path,
                root=root,
                repo_id=getattr(meta, "repo_id", "local/dataset"),
                model_id=loaded_model_id,
                episodes=selected_episodes,
                image_key=image_key,
                backend=backend,
                endpoint=endpoint,
                qwen_model=qwen_model,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                save_vis=save_vis,
                workers=effective_workers,
                devices=device_list,
                output_variant=result_variant,
                run_mode=run_mode,
                requested_episodes=[int(episode_index) for episode_index in requested_episodes],
                skipped_existing=skipped_existing,
            )

    def _append_record(record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        for output_path in label_output_paths:
            with output_path.open("a") as f:
                f.write(line)

    def _rewrite_records_sorted(records_by_episode: dict[int, dict]) -> None:
        for output_path in label_output_paths:
            with output_path.open("w") as f:
                for episode_index in sorted(records_by_episode):
                    f.write(json.dumps(records_by_episode[episode_index], ensure_ascii=False) + "\n")

    def _remove_reviewed_records(episode_indices: set[int]) -> None:
        if not episode_indices:
            return
        review_output_paths = {reviewed_path(labeling_dir, result_variant)}
        if write_latest:
            review_output_paths.add(reviewed_path(labeling_dir))
        for review_file in review_output_paths:
            if not review_file.is_file():
                continue
            reviewed = load_labels_jsonl(review_file)
            changed = False
            for episode_index in episode_indices:
                changed = reviewed.pop(int(episode_index), None) is not None or changed
            if changed:
                with review_file.open("w") as f:
                    for episode_index in sorted(reviewed):
                        f.write(json.dumps(reviewed[episode_index], ensure_ascii=False) + "\n")

    if run_mode == LABELING_RUN_MODE_FULL:
        for output_path in label_output_paths:
            output_path.write_text("")
    _write_source_files()
    tags_by_episode = _load_current_tagging(static_dir)

    def _close_detector(detector_obj) -> None:
        if detector_obj is None:
            return
        close_detector = getattr(detector_obj, "close", None)
        if close_detector is not None:
            close_detector()

    def _label_episode(detector_obj, episode_index: int) -> tuple[dict, bool]:
        task_name = _get_episode_task(root, meta, episode_index)
        parsed = parse_task(task_name)

        if parsed is None:
            return (
                {
                    "episode_index": episode_index,
                    "task": task_name,
                    "parsed": None,
                    "detections_target": [],
                    "detections_ref": [],
                    "selected": None,
                    "relation_satisfied": None,
                },
                False,
            )

        try:
            tag_values = _tag_values_for_episode(tags_by_episode, episode_index)
            active_arm = _active_arm_from_tags(tag_values)
            tag_context = _tag_context_for_labeling(tag_values)
            context_warnings = _context_warnings_for_labeling(tag_context)
            if _episode_exist_label_zero(root, meta, episode_index):
                return (
                    {
                        "episode_index": episode_index,
                        "task": task_name,
                        "parsed": parsed,
                        "detections_target": [],
                        "detections_ref": [],
                        "selected": None,
                        "relation_satisfied": None,
                        "exist_label": 0,
                        "skip_reason": "exist_label_zero",
                        "tags": tag_values,
                        "tag_context": tag_context,
                        "context_warnings": context_warnings,
                        "active_arm": active_arm,
                    },
                    False,
                )
            image_pil, _ = read_first_frame_image(
                root,
                meta,
                episode_index,
                image_key=image_key or None,
            )
            target_prompt = _target_prompt_for_backend(parsed, backend)
            detections_target = detector_obj.detect_for_prompt(
                image_pil,
                target_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            detections_ref = []
            if parsed["reference"] is not None:
                reference_prompt = _object_prompt(parsed["reference"])
                detections_ref = detector_obj.detect_for_prompt(
                    image_pil,
                    reference_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            detections_target_last = []
            if parsed.get("direction") is None and len(detections_target) > 1:
                episode_length = _episode_length(meta, episode_index)
                last_frame_index = max(0, int(episode_length or 1) - 1)
                if last_frame_index > 0:
                    try:
                        last_image_pil, _ = read_frame_image(
                            root,
                            meta,
                            episode_index,
                            last_frame_index,
                            image_key=image_key or None,
                        )
                        detections_target_last = detector_obj.detect_for_prompt(
                            last_image_pil,
                            target_prompt,
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                        )
                    except Exception as exc:
                        logging.warning(
                            "Could not use last frame for object labeling episode %s: %s", episode_index, exc
                        )
            selected, relation_satisfied, selection_method = select_bbox_with_context(
                detections_target,
                detections_ref if parsed["reference"] is not None else None,
                parsed["direction"],
                arm=active_arm,
                detections_target_last=detections_target_last,
            )
            record = {
                "episode_index": episode_index,
                "task": task_name,
                "parsed": parsed,
                "tags": tag_values,
                "tag_context": tag_context,
                "context_warnings": context_warnings,
                "active_arm": active_arm,
                "target_prompt": target_prompt,
                "detections_all_target": detections_target,
                "detections_target": detections_target,
                "detections_target_last": detections_target_last,
                "detections_ref": detections_ref,
                "selected": selected,
                "selected_target": selected,
                "relation_satisfied": relation_satisfied,
                "target_selection_method": selection_method,
            }

            if save_vis and detections_target:
                vis_img = draw_detections(
                    image_pil,
                    detections_target,
                    detections_ref,
                    selected,
                    parsed,
                )
                vis_img.save(vis_dir / f"episode_{episode_index:06d}.png")
            return record, False
        except Exception as exc:
            if backend not in {"qwen_remote", "qwen_dashscope"}:
                raise
            tag_values = _tag_values_for_episode(tags_by_episode, episode_index)
            tag_context = _tag_context_for_labeling(tag_values)
            return (
                {
                    "episode_index": episode_index,
                    "task": task_name,
                    "parsed": parsed,
                    "tags": tag_values,
                    "tag_context": tag_context,
                    "context_warnings": _context_warnings_for_labeling(tag_context),
                    "active_arm": _active_arm_from_tags(tag_values),
                    "detections_target": [],
                    "detections_ref": [],
                    "selected": None,
                    "relation_satisfied": False,
                    "error": f"{backend}_failed",
                    "error_detail": str(exc),
                },
                True,
            )

    try:
        records_by_episode: dict[int, dict] = (
            dict(existing_label_records) if run_mode == LABELING_RUN_MODE_MISSING else {}
        )
        scanned_records_by_episode: dict[int, dict] = {}
        if effective_workers == 1:
            for idx, episode_index in enumerate(selected_episodes, start=1):
                _emit_progress(
                    progress_callback,
                    status="running",
                    step="labeling",
                    current=idx - 1,
                    total=len(selected_episodes),
                    episode=episode_index,
                    message=f"Labeling episode {episode_index}",
                )
                record, failed = _label_episode(detector, episode_index)
                failed_count += int(failed)
                records_by_episode[episode_index] = record
                scanned_records_by_episode[episode_index] = record
                _append_record(record)
                if failed:
                    _emit_progress(
                        progress_callback,
                        status="running",
                        step="labeling",
                        current=idx - 1,
                        total=len(selected_episodes),
                        episode=episode_index,
                        message=f"{backend} failed for episode {episode_index}: {record['error_detail']}",
                    )
                if progress is not None:
                    progress.update(1)
                _emit_progress(
                    progress_callback,
                    status="running",
                    step="labeling",
                    current=idx,
                    total=len(selected_episodes),
                    episode=episode_index,
                    message=f"Labeled episode {episode_index}",
                )
        else:
            _close_detector(detector)
            worker_state = local()
            worker_detectors = []
            worker_lock = Lock()

            def _worker_detector():
                detector_obj = getattr(worker_state, "detector", None)
                if detector_obj is None:
                    with worker_lock:
                        worker_index = len(worker_detectors)
                        worker_detectors.append(None)
                    device = (
                        device_list[worker_index % len(device_list)]
                        if backend == "grounding_dino" and device_list
                        else None
                    )
                    detector_obj = load_detector(
                        backend,
                        model_id=model_id,
                        device=device,
                        endpoint=endpoint,
                        model=qwen_model,
                        token=qwen_token,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )
                    worker_state.detector = detector_obj
                    with worker_lock:
                        worker_detectors[worker_index] = detector_obj
                return detector_obj

            def _label_episode_in_worker(episode_index: int) -> tuple[dict, bool]:
                try:
                    return _label_episode(_worker_detector(), episode_index)
                except Exception as exc:
                    if backend not in {"qwen_remote", "qwen_dashscope"}:
                        raise
                    task_name = _get_episode_task(root, meta, episode_index)
                    tag_values = _tag_values_for_episode(tags_by_episode, episode_index)
                    tag_context = _tag_context_for_labeling(tag_values)
                    return (
                        {
                            "episode_index": episode_index,
                            "task": task_name,
                            "parsed": parse_task(task_name),
                            "tags": tag_values,
                            "tag_context": tag_context,
                            "context_warnings": _context_warnings_for_labeling(tag_context),
                            "active_arm": _active_arm_from_tags(tag_values),
                            "detections_target": [],
                            "detections_ref": [],
                            "selected": None,
                            "relation_satisfied": False,
                            "error": f"{backend}_failed",
                            "error_detail": str(exc),
                        },
                        True,
                    )

            try:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    future_to_episode = {
                        executor.submit(_label_episode_in_worker, episode_index): episode_index
                        for episode_index in selected_episodes
                    }
                    for completed, future in enumerate(as_completed(future_to_episode), start=1):
                        episode_index = future_to_episode[future]
                        record, failed = future.result()
                        failed_count += int(failed)
                        records_by_episode[episode_index] = record
                        scanned_records_by_episode[episode_index] = record
                        _append_record(record)
                        if failed:
                            _emit_progress(
                                progress_callback,
                                status="running",
                                step="labeling",
                                current=completed - 1,
                                total=len(selected_episodes),
                                episode=episode_index,
                                message=f"{backend} failed for episode {episode_index}: {record['error_detail']}",
                            )
                        if progress is not None:
                            progress.update(1)
                        _emit_progress(
                            progress_callback,
                            status="running",
                            step="labeling",
                            current=completed,
                            total=len(selected_episodes),
                            episode=episode_index,
                            message=f"Labeled episode {episode_index}",
                        )
            finally:
                for detector_obj in worker_detectors:
                    _close_detector(detector_obj)

        _rewrite_records_sorted(records_by_episode)
        if run_mode == LABELING_RUN_MODE_FULL:
            _remove_reviewed_records({int(episode_index) for episode_index in selected_episodes})
        flag_summary = _sync_missing_target_flags(
            static_dir,
            scanned_records_by_episode,
            backend=backend,
            variant=result_variant,
            scanned_episodes={int(episode_index) for episode_index in selected_episodes},
        )
        missing_target_count = int(flag_summary.get("missing_target_count", 0))
        if missing_target_count:
            logging.info(
                "Object labeling auto-flagged %d episodes with no target detections", missing_target_count
            )
    finally:
        if progress is not None:
            progress.close()
        _close_detector(detector)

    done_notes = []
    if failed_count:
        done_notes.append(f"{failed_count} {backend} failures")
    if missing_target_count:
        done_notes.append(f"{missing_target_count} no-detection flags")
    if skipped_existing:
        done_notes.append(f"{skipped_existing} existing skipped")
    _write_source_files()
    _emit_progress(
        progress_callback,
        status="done",
        step="done",
        current=len(selected_episodes),
        total=len(selected_episodes),
        message=f"Object labeling complete ({', '.join(done_notes)})"
        if done_notes
        else "Object labeling complete",
    )
    return LabelingResult(
        root=root,
        repo_id=getattr(meta, "repo_id", "local/dataset"),
        static_dir=static_dir,
        labeling_dir=labeling_dir,
        labels_path=variant_labels_file,
        episodes=selected_episodes,
        image_key=image_key,
        model_id=loaded_model_id,
        backend=backend,
    )
