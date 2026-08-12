from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local

from lerobot.data_platform.precompute.dataset_io import read_episode_table
from lerobot.data_platform.precompute.labeling.review import (
    image_keys_from_meta,
    read_first_frame_image,
    read_frame_image,
)
from lerobot.data_platform.precompute.labeling.task_parser import normalize_object_name, parse_task
from lerobot.data_platform.precompute.tagging.geometric_backend import (
    build_grasp_heatmap,
    grasp_xy_from_trajectory,
    trajectory_xy_details_from_table,
)
from lerobot.data_platform.precompute.tagging.review import (
    load_tags_jsonl,
    merge_tag_record,
    source_path,
    tags_path,
)
from lerobot.data_platform.precompute.tagging.rule_backend import arm_from_action
from lerobot.data_platform.precompute.tagging.schema import DEFAULT_VLM_MODEL, selected_tag_defs
from lerobot.data_platform.precompute.tagging.vlm_backend import VLMTagger
from lerobot.data_platform.precompute.tagging.vlm_backend import (
    get_capabilities as get_vlm_capabilities,
)

PROMPT_ACTION_MISMATCH_ISSUE_TYPE = "tagging_prompt_behavior"
PROMPT_ACTION_MISMATCH_REASON = "prompt_action_mismatch"
PROMPT_ACTION_MISMATCH_FLAGS = "tagging_prompt_mismatch_flagged_episodes.json"


@dataclass
class TaggingResult:
    root: Path
    repo_id: str
    static_dir: Path
    tagging_dir: Path
    tags_path: Path
    episodes: list[int]
    selected_tags: list[str]
    output_variant: str | None = None


def _emit(progress_callback: Callable[[dict], None] | None, **payload) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _episode_task(meta, episode_index: int) -> str:
    episode = getattr(meta, "episodes", {}).get(episode_index) or getattr(meta, "episodes", {}).get(
        str(episode_index)
    )
    tasks = episode.get("tasks", []) if episode else []
    return tasks[0] if tasks else ""


def _prompt_action_allowed_objects(meta) -> list[str]:
    """Return canonical object names that actually appear in this dataset's tasks."""
    out = []
    seen = set()

    def add_task(task: str | None) -> None:
        parsed = parse_task(str(task or ""))
        if not parsed:
            return
        for key in ("target", "reference"):
            value = normalize_object_name(parsed.get(key))
            if value and value not in seen:
                seen.add(value)
                out.append(value)

    for task in (getattr(meta, "tasks", {}) or {}).values():
        add_task(task)
    for episode in (getattr(meta, "episodes", {}) or {}).values():
        for task in (episode or {}).get("tasks", []) or []:
            add_task(task)
    return out


def _load_json_any(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _flag_set(path: Path) -> set[int]:
    data = _load_json_any(path)
    values = (
        data.get("flagged_episodes") if isinstance(data, dict) else data if isinstance(data, list) else []
    )
    out = set()
    for value in values or []:
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


def _is_prompt_action_mismatch_issue(issue: dict) -> bool:
    return (
        isinstance(issue, dict)
        and issue.get("type") == PROMPT_ACTION_MISMATCH_ISSUE_TYPE
        and issue.get("reason") == PROMPT_ACTION_MISMATCH_REASON
    )


def _prompt_action_mismatch_issue(record: dict, *, output_variant: str | None) -> dict | None:
    tags = record.get("tags") or {}
    if tags.get("prompt_action_match") != "mismatch":
        return None
    detail = (record.get("tag_details") or {}).get("prompt_action_match") or {}
    observed_object = normalize_object_name(detail.get("observed_object"))
    return {
        "episode": int(record["episode_index"]),
        "type": PROMPT_ACTION_MISMATCH_ISSUE_TYPE,
        "reason": PROMPT_ACTION_MISMATCH_REASON,
        "task": record.get("task"),
        "observed_object": observed_object,
        "vlm_reason": detail.get("reason"),
        "variant": output_variant or "latest",
        "metrics": {
            "prompt_action_match": "mismatch",
            "observed_object": observed_object,
        },
    }


def _flag_reasons_for_issues(issues: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for issue in issues:
        episode = _issue_episode(issue)
        if episode is None:
            continue
        item = {
            "type": str(issue.get("type") or PROMPT_ACTION_MISMATCH_ISSUE_TYPE),
            "reason": str(issue.get("reason") or "unknown"),
            "metrics": issue.get("metrics") or {},
        }
        if issue.get("task"):
            item["task"] = str(issue.get("task"))
        if issue.get("vlm_reason"):
            item["vlm_reason"] = issue.get("vlm_reason")
        out.setdefault(str(episode), []).append(item)
    return out


def _sync_prompt_action_mismatch_flags(
    static_dir: Path,
    records_by_episode: dict[int, dict],
    scanned_episodes: set[int],
    *,
    output_variant: str | None,
) -> dict:
    static_dir = Path(static_dir)
    issues_path = static_dir / "annotation_issues.json"
    auto_path = static_dir / PROMPT_ACTION_MISMATCH_FLAGS
    flagged_path = static_dir / "flagged_episodes.json"
    mismatch_issues = [
        issue
        for record in records_by_episode.values()
        if (issue := _prompt_action_mismatch_issue(record, output_variant=output_variant)) is not None
    ]
    mismatch_episodes = {_issue_episode(issue) for issue in mismatch_issues}
    mismatch_episodes = {episode for episode in mismatch_episodes if episode is not None}

    existing_issues = _load_json_any(issues_path)
    retained = [
        issue
        for issue in (existing_issues if isinstance(existing_issues, list) else [])
        if not (_is_prompt_action_mismatch_issue(issue) and _issue_episode(issue) in scanned_episodes)
    ]
    merged = retained + sorted(mismatch_issues, key=lambda issue: int(issue.get("episode", -1)))
    _write_json(issues_path, merged)

    previous_auto = _flag_set(auto_path)
    next_auto = (previous_auto - scanned_episodes) | mismatch_episodes
    existing_flagged = _flag_set(flagged_path)
    manual_or_other_auto = existing_flagged - previous_auto
    combined = manual_or_other_auto | next_auto
    _write_json(
        auto_path,
        {
            "flagged_episodes": sorted(next_auto),
            "flag_reasons": _flag_reasons_for_issues(mismatch_issues),
            "summary": {
                "reason": PROMPT_ACTION_MISMATCH_REASON,
                "mismatch_episode_count": len(mismatch_episodes),
                "episodes_scanned": len(scanned_episodes),
                "variant": output_variant or "latest",
            },
        },
    )
    _write_json(flagged_path, {"flagged_episodes": sorted(combined)})
    return {
        "mismatch_episode_count": len(mismatch_episodes),
        "flagged_episode_count": len(combined),
    }


def run_tagging(
    root: Path,
    meta,
    episodes: list[int] | None,
    static_dir: Path,
    selected_tags: list[str] | None = None,
    vlm_backend: str = "qwen_dashscope",
    vlm_model: str = DEFAULT_VLM_MODEL,
    vlm_endpoint: str | None = None,
    vlm_token: str | None = None,
    output_variant: str | None = None,
    workers: int = 1,
    overwrite: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> TaggingResult:
    root = Path(root)
    static_dir = Path(static_dir)
    tagging_dir = static_dir / "tagging"
    tagging_dir.mkdir(parents=True, exist_ok=True)
    if not selected_tags:
        tag_defs = selected_tag_defs(None)
        if not get_vlm_capabilities(vlm_model, backend=vlm_backend)["available"]:
            tag_defs = [tag for tag in tag_defs if tag["backend"] != "vlm"]
    else:
        tag_defs = selected_tag_defs(selected_tags)
    tag_names = [tag["name"] for tag in tag_defs]
    vlm_capabilities = get_vlm_capabilities(vlm_model, backend=vlm_backend)
    vlm_tags = [tag for tag in tag_defs if tag["backend"] == "vlm"]
    prompt_action_tag = next((tag for tag in vlm_tags if tag["name"] == "prompt_action_match"), None)
    first_frame_vlm_tags = [tag for tag in vlm_tags if tag["name"] != "prompt_action_match"]
    prompt_action_objects = _prompt_action_allowed_objects(meta) if prompt_action_tag else []
    image_key = (image_keys_from_meta(meta) or [""])[0]
    effective_workers = max(1, int(workers or 1))

    requested_episodes = sorted(getattr(meta, "episodes", {}).keys()) if episodes is None else list(episodes)
    heatmap_points: list[list[float]] = []
    grasp_xy_source = ""
    grasp_xy_projection = ""
    out_path = tags_path(tagging_dir, output_variant)
    source_file = source_path(tagging_dir, output_variant)
    heatmap_path = tagging_dir / "grasp_heatmap.png"
    if overwrite and heatmap_path.exists():
        heatmap_path.unlink()

    records_by_episode: dict[int, dict] = dict(load_tags_jsonl(out_path))

    def _has_selected_tags(record: dict | None) -> bool:
        tags = (record or {}).get("tags") or {}
        if not isinstance(tags, dict):
            return False
        return all(name in tags for name in tag_names)

    skipped_existing = 0
    if overwrite:
        selected_episodes = list(requested_episodes)
    else:
        selected_episodes = [
            int(episode_index)
            for episode_index in requested_episodes
            if not _has_selected_tags(records_by_episode.get(int(episode_index)))
        ]
        skipped_existing = len(requested_episodes) - len(selected_episodes)

    def _close_tagger(tagger) -> None:
        if tagger is None:
            return
        close_tagger = getattr(tagger, "close", None)
        if close_tagger is not None:
            close_tagger()

    def _tag_episode(episode_index: int, vlm_tagger) -> tuple[dict, list[float] | None, str, str]:
        table = read_episode_table(root, meta, episode_index)
        tags = {}
        vlm_values = {}
        vlm_error = None
        task = _episode_task(meta, episode_index)
        first_image_pil = None
        if first_frame_vlm_tags and vlm_tagger is not None:
            try:
                first_image_pil, _ = read_first_frame_image(
                    root, meta, episode_index, image_key=image_key or None
                )
                vlm_values = vlm_tagger.predict_many(first_image_pil, first_frame_vlm_tags)
            except Exception as exc:
                vlm_error = str(exc)
        elif prompt_action_tag and vlm_tagger is not None:
            try:
                first_image_pil, _ = read_first_frame_image(
                    root, meta, episode_index, image_key=image_key or None
                )
            except Exception:
                first_image_pil = None

        tag_details = {}
        if prompt_action_tag and vlm_tagger is not None:
            try:
                final_frame_index = max(0, int(table.num_rows) - 1)
                final_image_pil, _ = read_frame_image(
                    root, meta, episode_index, final_frame_index, image_key=image_key or None
                )
                match_detail = vlm_tagger.predict_prompt_action_match(
                    first_image_pil,
                    final_image_pil,
                    task,
                    allowed_objects=prompt_action_objects,
                )
                observed_object = normalize_object_name(match_detail.get("observed_object"))
                vlm_values["prompt_action_match"] = match_detail.get("value")
                tag_details["prompt_action_match"] = {
                    "observed_object": observed_object,
                    "reason": match_detail.get("reason"),
                    "final_frame_index": final_frame_index,
                }
            except Exception as exc:
                vlm_error = str(exc)

        grasp_point = None
        grasp_source = ""
        grasp_projection = ""
        for tag in tag_defs:
            name = tag["name"]
            backend = tag["backend"]
            if backend == "rule" and name == "arm":
                tags[name] = arm_from_action(table)
            elif backend == "geometric" and name == "grasp_xy":
                features = getattr(meta, "features", {})
                robot_type = getattr(meta, "robot_type", "") or getattr(meta, "info", {}).get(
                    "robot_type", ""
                )
                details = trajectory_xy_details_from_table(table, features, robot_type=robot_type)
                tags[name] = grasp_xy_from_trajectory(table, features, robot_type=robot_type)
                if tags[name] is not None:
                    grasp_point = tags[name]
                    grasp_source = details.get("source") or ""
                    grasp_projection = details.get("projection") or ""
            elif backend == "vlm":
                tags[name] = vlm_values.get(name)
        record = {"episode_index": int(episode_index), "task": task, "tags": tags}
        if tag_details:
            record["tag_details"] = tag_details
        if vlm_error:
            record["error"] = "vlm_tagging_failed"
            record["error_detail"] = vlm_error
        return record, grasp_point, grasp_source, grasp_projection

    def _write_records_sorted(records_by_episode: dict[int, dict]) -> None:
        with out_path.open("w") as f:
            for episode_index in sorted(records_by_episode):
                f.write(json.dumps(records_by_episode[episode_index], ensure_ascii=False) + "\n")

    def _store_record(record: dict) -> dict:
        episode_index = int(record["episode_index"])
        merged = merge_tag_record(records_by_episode.get(episode_index), record, tag_names)
        if record.get("error"):
            merged["error"] = record.get("error")
            merged["error_detail"] = record.get("error_detail")
        records_by_episode[episode_index] = merged
        updated_records_by_episode[episode_index] = merged
        with out_path.open("a") as f:
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
        return merged

    if not selected_episodes:
        _write_records_sorted(records_by_episode)
        source_file.write_text(
            json.dumps(
                {
                    "dataset_root": str(root),
                    "repo_id": getattr(meta, "repo_id", f"local/{root.name}"),
                    "selected_tags": tag_names,
                    "vlm_backend": vlm_backend,
                    "vlm_model": vlm_model,
                    "vlm_endpoint": vlm_endpoint,
                    "vlm_capabilities": vlm_capabilities,
                    "image_key": image_key,
                    "workers": effective_workers,
                    "output_variant": output_variant,
                    "overwrite": bool(overwrite),
                    "requested_episodes": [int(episode) for episode in requested_episodes],
                    "episodes": [],
                    "skipped_existing": skipped_existing,
                    "prompt_action_allowed_objects": prompt_action_objects,
                    "prompt_action_mismatch": {},
                    "grasp_xy_coordinate_source": "",
                    "grasp_xy_projection": "",
                    "grasp_xy_points": 0,
                },
                indent=2,
            )
        )
        _emit(
            progress_callback,
            status="done",
            step="tagging_done",
            current=0,
            total=0,
            message=f"No missing tags to run ({skipped_existing} existing episodes kept)",
        )
        return TaggingResult(
            root=root,
            repo_id=getattr(meta, "repo_id", f"local/{root.name}"),
            static_dir=static_dir,
            tagging_dir=tagging_dir,
            tags_path=out_path,
            episodes=[],
            selected_tags=tag_names,
            output_variant=output_variant,
        )

    _emit(
        progress_callback,
        status="running",
        step="tagging",
        current=0,
        total=len(selected_episodes),
        message=(
            f"Starting auto-tagging with {effective_workers} worker(s)"
            + (f" ({skipped_existing} existing episodes skipped)" if skipped_existing else "")
        ),
    )
    updated_records_by_episode: dict[int, dict] = {}
    out_path.touch(exist_ok=True)
    if effective_workers == 1:
        vlm_tagger = (
            VLMTagger.load(vlm_model, endpoint=vlm_endpoint, backend=vlm_backend, token=vlm_token)
            if vlm_tags and vlm_capabilities["available"]
            else None
        )
        try:
            for idx, episode_index in enumerate(selected_episodes, start=1):
                record, grasp_point, grasp_source, grasp_projection = _tag_episode(episode_index, vlm_tagger)
                record = _store_record(record)
                if grasp_point is not None:
                    heatmap_points.append(grasp_point)
                    if not grasp_xy_source:
                        grasp_xy_source = grasp_source
                        grasp_xy_projection = grasp_projection
                if record.get("error"):
                    _emit(
                        progress_callback,
                        status="running",
                        step="tagging",
                        current=idx - 1,
                        total=len(selected_episodes),
                        message=f"VLM tagging failed for episode {episode_index}: {record['error_detail']}",
                    )
                _emit(
                    progress_callback,
                    status="running",
                    step="tagging",
                    current=idx,
                    total=len(selected_episodes),
                    message=f"Tagged episode {episode_index}",
                )
        finally:
            _close_tagger(vlm_tagger)
    else:
        worker_state = local()
        worker_taggers = []
        worker_lock = Lock()

        def _worker_tagger():
            if not vlm_tags or not vlm_capabilities["available"]:
                return None
            tagger = getattr(worker_state, "tagger", None)
            if tagger is None:
                tagger = VLMTagger.load(
                    vlm_model, endpoint=vlm_endpoint, backend=vlm_backend, token=vlm_token
                )
                worker_state.tagger = tagger
                with worker_lock:
                    worker_taggers.append(tagger)
            return tagger

        def _tag_episode_in_worker(episode_index: int):
            return _tag_episode(episode_index, _worker_tagger())

        try:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_episode = {
                    executor.submit(_tag_episode_in_worker, episode_index): episode_index
                    for episode_index in selected_episodes
                }
                for completed, future in enumerate(as_completed(future_to_episode), start=1):
                    episode_index = future_to_episode[future]
                    record, grasp_point, grasp_source, grasp_projection = future.result()
                    record = _store_record(record)
                    if grasp_point is not None:
                        heatmap_points.append(grasp_point)
                        if not grasp_xy_source:
                            grasp_xy_source = grasp_source
                            grasp_xy_projection = grasp_projection
                    if record.get("error"):
                        _emit(
                            progress_callback,
                            status="running",
                            step="tagging",
                            current=completed - 1,
                            total=len(selected_episodes),
                            message=f"VLM tagging failed for episode {episode_index}: {record['error_detail']}",
                        )
                    _emit(
                        progress_callback,
                        status="running",
                        step="tagging",
                        current=completed,
                        total=len(selected_episodes),
                        message=f"Tagged episode {episode_index}",
                    )
        finally:
            for tagger in worker_taggers:
                _close_tagger(tagger)

    _write_records_sorted(records_by_episode)
    prompt_action_summary = {}
    if "prompt_action_match" in tag_names:
        prompt_action_summary = _sync_prompt_action_mismatch_flags(
            static_dir,
            updated_records_by_episode,
            {int(episode) for episode in selected_episodes},
            output_variant=output_variant,
        )

    if "grasp_xy" in tag_names:
        heatmap_points = [
            value
            for record in records_by_episode.values()
            if isinstance((record.get("tags") or {}).get("grasp_xy"), list)
            for value in [(record.get("tags") or {}).get("grasp_xy")]
        ]
    if heatmap_points:
        heatmap_path.write_bytes(build_grasp_heatmap(heatmap_points))
    source_file.write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "repo_id": getattr(meta, "repo_id", f"local/{root.name}"),
                "selected_tags": tag_names,
                "vlm_backend": vlm_backend,
                "vlm_model": vlm_model,
                "vlm_endpoint": vlm_endpoint,
                "vlm_capabilities": vlm_capabilities,
                "image_key": image_key,
                "workers": effective_workers,
                "output_variant": output_variant,
                "overwrite": bool(overwrite),
                "requested_episodes": [int(episode) for episode in requested_episodes],
                "episodes": [int(episode) for episode in selected_episodes],
                "skipped_existing": skipped_existing,
                "prompt_action_allowed_objects": prompt_action_objects,
                "prompt_action_mismatch": prompt_action_summary,
                "grasp_xy_coordinate_source": grasp_xy_source,
                "grasp_xy_projection": grasp_xy_projection,
                "grasp_xy_points": len(heatmap_points),
            },
            indent=2,
        )
    )
    done_message = "Auto-tagging complete"
    if skipped_existing and not selected_episodes:
        done_message = f"No missing tags to run ({skipped_existing} existing episodes kept)"
    elif skipped_existing:
        done_message = f"Auto-tagging complete ({skipped_existing} existing episodes kept)"
    _emit(
        progress_callback,
        status="done",
        step="tagging_done",
        current=len(selected_episodes),
        total=len(selected_episodes),
        message=done_message,
    )
    return TaggingResult(
        root=root,
        repo_id=getattr(meta, "repo_id", f"local/{root.name}"),
        static_dir=static_dir,
        tagging_dir=tagging_dir,
        tags_path=out_path,
        episodes=selected_episodes,
        selected_tags=tag_names,
        output_variant=output_variant,
    )
