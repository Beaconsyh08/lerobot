from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.annotation import QUALITY_FLAG_TYPE, compute_quality_flags, series_to_2d
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    load_episode_records,
    load_task_records,
    read_episode_table,
    replace_episode_column,
    write_episode_records,
    write_task_records,
)
from lerobot.data_platform.precompute.labeling.task_parser import normalize_object_name, parse_task
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    emit,
    format_data_path,
    load_json,
    validate_dataset_root,
)
from lerobot.data_platform.precompute.tagging.review import current_tags, latest_tag_variant
from lerobot.data_platform.precompute.timeseries import (
    DATA_VERSION_DVT1,
    DATA_VERSION_DVT2,
    feature_vector_dim,
    infer_data_version_from_features,
)

QUALITY_FLAGGED_EPISODES = "quality_flagged_episodes.json"
MULTIPLE_TASK_ASSIGNMENT_REASON = "multiple_task_assignments"
PROMPT_ACTION_MISMATCH_REASON = "prompt_action_mismatch"
MANUAL_PROMPT_ERROR_REASON = "wrong_prompt"
PROMPT_ACTION_MISMATCH_ISSUE_TYPE = "tagging_prompt_behavior"
TAGGING_PROMPT_MISMATCH_FLAGS = "tagging_prompt_mismatch_flagged_episodes.json"
TASK_ASSIGNMENT_REASONS = {
    MULTIPLE_TASK_ASSIGNMENT_REASON,
    PROMPT_ACTION_MISMATCH_REASON,
    MANUAL_PROMPT_ERROR_REASON,
}
FREEFORM_TASK_ASSIGNMENT_REASONS = {PROMPT_ACTION_MISMATCH_REASON, MANUAL_PROMPT_ERROR_REASON}

# The platform's prompt convention permits one instruction without commas or periods.
# Any occurrence of either character is reported as ``multi_sentence_prompt``.
PROMPT_SENTENCE_PUNCTUATION = (",", ".")


class _LegacyMetadataPath:
    def __init__(self, info: dict):
        self.info = info

    def get_data_file_path(self, episode_index: int) -> Path:
        return format_data_path(self.info, episode_index)


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
    if isinstance(data, dict):
        values = data.get("flagged_episodes") or []
    elif isinstance(data, list):
        values = data
    else:
        values = []
    result = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _records_by_episode(episodes_rows: list[dict]) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for row in episodes_rows:
        try:
            episode_index = int(row["episode_index"])
        except (KeyError, TypeError, ValueError):
            continue
        records[episode_index] = row
    return records


def _issue_episode(issue: dict) -> int | None:
    try:
        return int(issue.get("episode"))
    except (TypeError, ValueError):
        return None


def _array_from_column(table, column_name: str, fallback_dim: int) -> np.ndarray | None:
    if column_name not in table.column_names:
        return None
    values = table[column_name].to_pylist()
    actual_dim = max(
        (len(item) if hasattr(item, "__len__") and not isinstance(item, str) else 1 for item in values),
        default=0,
    )
    dim = actual_dim if actual_dim > 0 else int(fallback_dim or 0)
    if dim <= 0:
        return None
    return series_to_2d(values, dim)


def _tasks_by_index(root: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for row in load_task_records(root):
        try:
            task_index = int(row["task_index"])
        except (KeyError, TypeError, ValueError):
            continue
        task = row.get("task")
        if task is not None:
            tasks[task_index] = str(task)
    return tasks


def _task_rows(root: Path) -> list[dict]:
    return load_task_records(root)


def _task_index_for_prompt(root: Path, prompt: str) -> int:
    rows = _task_rows(root)
    prompt = str(prompt)
    used_indices = []
    for row in rows:
        try:
            task_index = int(row["task_index"])
        except (KeyError, TypeError, ValueError):
            continue
        used_indices.append(task_index)
        if str(row.get("task") or "") == prompt:
            return task_index

    task_index = (max(used_indices) + 1) if used_indices else 0
    rows.append({"task_index": task_index, "task": prompt})
    write_task_records(root, rows)
    return task_index


def _episode_task(row: dict | None) -> str:
    tasks = (row or {}).get("tasks") or []
    return str(tasks[0]) if tasks else ""


_PROMPT_TARGET_PATTERNS = (
    re.compile(
        r"^(?P<prefix>Give\s+(?:the\s+)?)(?P<target>.+?)"
        r"(?P<suffix>\s+to\s+me)(?P<punct>\.?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>Give\s+me\s+(?:the\s+)?)(?P<target>.+?)"
        r"(?P<suffix>)(?P<punct>\.?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>Pick\s+up\s+(?:the\s+)?)(?P<target>.+?)"
        r"(?P<suffix>\s+to\s+the\s+(?:left|right)\s+of\s+(?:the\s+)?.+?)(?P<punct>\.?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>Pick\s+up\s+(?:the\s+)?)(?P<target>.+?)"
        r"(?P<suffix>\s+on\s+the\s+(?:left|right)\s+of\s+(?:the\s+)?.+?)(?P<punct>\.?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>Pick\s+up\s+(?:the\s+)?)(?P<target>.+?)"
        r"(?P<suffix>\s+on\s+the\s+(?:left|right))(?P<punct>\.?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>Pick\s+up\s+(?:the\s+)?)(?P<target>.+?)"
        r"(?P<suffix>\s+to\s+the\s+(?:left|right))(?P<punct>\.?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>Pick\s+up\s+(?:the\s+)?)(?P<target>.+?)"
        r"(?P<suffix>)(?P<punct>\.?)$",
        re.IGNORECASE,
    ),
)


def _match_target_case(value: str, sample: str) -> str:
    value = str(value or "").strip().lower()
    sample = str(sample or "").strip()
    if not value:
        return ""
    if sample.isupper():
        return value.upper()
    if sample.istitle():
        return value.title()
    if sample[:1].isupper() and sample[1:] == sample[1:].lower():
        return value[:1].upper() + value[1:]
    return value


def _prompt_with_target(template_prompt: str, target: str) -> str:
    parsed = parse_task(template_prompt)
    target = str(target or "").strip().lower()
    if not parsed or not target:
        return ""
    if parsed.get("reference"):
        reference = str(parsed.get("reference") or "").strip().lower()
        if reference and target == reference:
            return ""
    prompt = str(template_prompt or "").strip()
    for pattern in _PROMPT_TARGET_PATTERNS:
        match = pattern.match(prompt)
        if match:
            replacement = _match_target_case(target, match.group("target"))
            return f"{match.group('prefix')}{replacement}{match.group('suffix')}{match.group('punct')}"
    if parsed.get("action") == "give":
        return f"Give me the {target}"
    if parsed.get("reference"):
        reference = str(parsed.get("reference") or "").strip().lower()
        return (
            f"Pick up the {target} to the {parsed.get('direction') or 'left'} of the {reference or 'object'}"
        )
    if parsed.get("direction"):
        return f"Pick up the {target} on the {parsed.get('direction') or 'left'}"
    return f"Pick up the {target}"


def _episode_rows_by_index(root: Path) -> dict[int, dict]:
    return _records_by_episode(load_episode_records(root))


def _task_object_vocab(root: Path) -> set[str]:
    objects: set[str] = set()

    def add_task(task: str | None) -> None:
        parsed = parse_task(str(task or ""))
        if not parsed:
            return
        for key in ("target", "reference"):
            value = normalize_object_name(parsed.get(key))
            if value:
                objects.add(value)

    for task in _tasks_by_index(root).values():
        add_task(task)
    for row in _episode_rows_by_index(root).values():
        for task in (row or {}).get("tasks") or []:
            add_task(task)
    return objects


def _task_target_vocab(root: Path) -> list[str]:
    objects: list[str] = []

    def add_task(task: str | None) -> None:
        parsed = parse_task(str(task or ""))
        if not parsed:
            return
        target = normalize_object_name(parsed.get("target"))
        if target and target not in objects:
            objects.append(target)

    for _, task in sorted(_tasks_by_index(root).items()):
        add_task(task)
    for _, row in sorted(_episode_rows_by_index(root).items()):
        for task in (row or {}).get("tasks") or []:
            add_task(task)
    return objects


def _prompt_target_in_vocab(prompt: str, object_vocab: set[str]) -> bool:
    if not object_vocab:
        return True
    parsed = parse_task(str(prompt or ""))
    if not parsed:
        return False
    target = normalize_object_name(parsed.get("target"))
    return bool(target and target in object_vocab)


def _task_assignment_candidates_for_issue(root: Path, issue: dict) -> list[str]:
    metrics = issue.get("metrics") or {}
    raw_candidates = metrics.get("tasks") or metrics.get("prompts") or metrics.get("candidates") or []
    candidates: list[str] = []
    seen: set[str] = set()
    is_freeform_repair = issue.get("reason") in FREEFORM_TASK_ASSIGNMENT_REASONS
    target_vocab = _task_target_vocab(root) if is_freeform_repair else []
    object_vocab = (
        set(target_vocab) if target_vocab else (_task_object_vocab(root) if is_freeform_repair else set())
    )
    if is_freeform_repair:
        episode = _issue_episode(issue)
        episode_row = _episode_rows_by_index(root).get(int(episode)) if episode is not None else None
        template_prompt = str(issue.get("task") or _episode_task(episode_row) or "").strip()
        parsed_template = parse_task(template_prompt)
        if not parsed_template:
            for task in raw_candidates:
                prompt = str(task or "").strip()
                if not _prompt_target_in_vocab(prompt, object_vocab):
                    continue
                if prompt and prompt not in seen:
                    candidates.append(prompt)
                    seen.add(prompt)
            return candidates

        target_objects: list[str] = []

        def add_target(value) -> None:
            value = normalize_object_name(value)
            if not value:
                return
            if object_vocab and value not in object_vocab:
                return
            if value not in target_objects:
                target_objects.append(value)

        add_target(parsed_template.get("target"))
        add_target(metrics.get("observed_object") or issue.get("observed_object"))
        for task in raw_candidates:
            parsed = parse_task(str(task or ""))
            if parsed:
                add_target(parsed.get("target"))
        for target in target_vocab:
            add_target(target)

        for target in target_objects:
            prompt = _prompt_with_target(template_prompt, target)
            if prompt and prompt not in seen:
                candidates.append(prompt)
                seen.add(prompt)
        return candidates

    for task in raw_candidates:
        prompt = str(task or "").strip()
        if prompt and prompt not in seen:
            candidates.append(prompt)
            seen.add(prompt)
    return candidates


def _episode_prompts(row: dict | None, table, tasks_by_index: dict[int, str]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()

    for task in (row or {}).get("tasks") or []:
        prompt = str(task or "").strip()
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)

    if "task_index" in table.column_names:
        for value in table["task_index"].to_pylist():
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            try:
                task_index = int(value)
            except (TypeError, ValueError):
                continue
            prompt = str(tasks_by_index.get(task_index) or "").strip()
            if prompt and prompt not in seen:
                prompts.append(prompt)
                seen.add(prompt)

    return prompts


def _subtask_state_issues(table, episode_index: int, data_version: str) -> list[dict]:
    if "subtask_state" not in table.column_names:
        return [
            {
                "episode": episode_index,
                "type": QUALITY_FLAG_TYPE,
                "reason": "missing_subtask_state",
                "frames": [],
                "data_version": data_version,
                "metrics": {"column": "subtask_state"},
            }
        ]

    raw_values = table["subtask_state"].to_pylist()
    states: list[int] = []
    for value in raw_values:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            states.append(int(numeric))

    if not states:
        return [
            {
                "episode": episode_index,
                "type": QUALITY_FLAG_TYPE,
                "reason": "missing_subtask_state",
                "frames": [],
                "data_version": data_version,
                "metrics": {"column": "subtask_state", "valid_count": 0},
            }
        ]

    unique_states = sorted(set(states))
    if len(unique_states) < 2:
        return [
            {
                "episode": episode_index,
                "type": QUALITY_FLAG_TYPE,
                "reason": "unsegmented_subtask_state",
                "frames": [],
                "data_version": data_version,
                "metrics": {"unique_states": unique_states, "valid_count": len(states)},
            }
        ]
    return []


def _prompt_issues(prompts: list[str], episode_index: int, data_version: str) -> list[dict]:
    bad_prompts = []
    punctuation = set()
    for prompt in prompts:
        found = [char for char in PROMPT_SENTENCE_PUNCTUATION if char in prompt]
        if found:
            bad_prompts.append(prompt)
            punctuation.update(found)
    if not bad_prompts:
        return []
    return [
        {
            "episode": episode_index,
            "type": QUALITY_FLAG_TYPE,
            "reason": "multi_sentence_prompt",
            "frames": [],
            "data_version": data_version,
            "metrics": {
                "prompts": bad_prompts,
                "punctuation": sorted(punctuation),
            },
        }
    ]


def _task_assignment_issues(prompts: list[str], episode_index: int, data_version: str) -> list[dict]:
    if len(prompts) <= 1:
        return []
    return [
        {
            "episode": episode_index,
            "type": QUALITY_FLAG_TYPE,
            "reason": MULTIPLE_TASK_ASSIGNMENT_REASON,
            "frames": [],
            "data_version": data_version,
            "metrics": {
                "task_count": len(prompts),
                "tasks": prompts,
            },
        }
    ]


def _scan_episode(
    root: Path,
    info: dict,
    episode_row: dict,
    data_version: str,
    tasks_by_index: dict[int, str] | None = None,
    meta: V3DatasetMetadata | None = None,
) -> tuple[int, int, list[dict]]:
    episode_index = int(episode_row["episode_index"])
    parquet_path = (
        root / meta.get_data_file_path(episode_index)
        if meta is not None
        else root / format_data_path(info, episode_index)
    )
    if not parquet_path.is_file():
        return (
            episode_index,
            0,
            [
                {
                    "episode": episode_index,
                    "type": QUALITY_FLAG_TYPE,
                    "reason": "missing_parquet",
                    "frames": [],
                    "data_version": data_version,
                    "metrics": {"path": str(parquet_path)},
                }
            ],
        )

    schema = pq.read_schema(parquet_path)
    columns = ["timestamp"]
    if "action" in schema.names:
        columns.append("action")
    if "state" in schema.names:
        columns.append("state")
    if "subtask_state" in schema.names:
        columns.append("subtask_state")
    if "task_index" in schema.names:
        columns.append("task_index")
    table = (
        read_episode_table(root, meta, episode_index, columns=columns)
        if meta is not None
        else pq.read_table(parquet_path, columns=columns)
    )
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)

    features = info.get("features") or {}
    action = _array_from_column(table, "action", feature_vector_dim(features.get("action")))
    state = _array_from_column(table, "state", feature_vector_dim(features.get("state")))
    fps = float(info.get("fps") or 0) or None
    task = _episode_task(episode_row)
    prompts = _episode_prompts(episode_row, table, tasks_by_index or {})
    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=fps,
        episode_id=episode_index,
        task=task,
        data_version=data_version,
    )
    issues.extend(_subtask_state_issues(table, episode_index, data_version))
    issues.extend(_prompt_issues(prompts, episode_index, data_version))
    issues.extend(_task_assignment_issues(prompts, episode_index, data_version))
    return episode_index, int(len(timestamps)), issues


def _annotation_issues(static_dir: Path) -> list[dict]:
    issues = _load_json_any(Path(static_dir) / "annotation_issues.json")
    return issues if isinstance(issues, list) else []


def _refresh_quality_flag_files(static_dir: Path, quality_issues: list[dict]) -> set[int]:
    static_dir = Path(static_dir)
    quality_episodes = {
        episode for episode in (_issue_episode(issue) for issue in quality_issues) if episode is not None
    }
    previous_auto = _flag_set(static_dir / QUALITY_FLAGGED_EPISODES)
    existing_flagged = _flag_set(static_dir / "flagged_episodes.json")
    manual_flagged = existing_flagged - previous_auto
    combined = manual_flagged | quality_episodes
    reason_counts = Counter(str(issue.get("reason", "unknown")) for issue in quality_issues)
    _write_json(
        static_dir / QUALITY_FLAGGED_EPISODES,
        {
            "flagged_episodes": sorted(quality_episodes),
            "flag_reasons": _quality_reason_map(quality_issues),
            "summary": {
                "quality_episode_count": len(quality_episodes),
                "quality_issue_count": len(quality_issues),
                "reason_counts": dict(sorted(reason_counts.items())),
            },
        },
    )
    _write_json(static_dir / "flagged_episodes.json", {"flagged_episodes": sorted(combined)})
    return combined


def list_task_assignment_choices(root: Path, static_dir: Path) -> list[dict]:
    root = validate_dataset_root(Path(root))
    static_dir = Path(static_dir).expanduser()
    issues = list(_annotation_issues(static_dir))
    manual_payload = _load_json_any(static_dir / "manual_flagged_episodes.json")
    manual_reasons = manual_payload.get("flag_reasons") if isinstance(manual_payload, dict) else {}
    if isinstance(manual_reasons, dict):
        for episode_key, items in manual_reasons.items():
            try:
                episode = int(episode_key)
            except (TypeError, ValueError):
                continue
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict) or item.get("reason") not in TASK_ASSIGNMENT_REASONS:
                    continue
                issue = dict(item)
                issue["episode"] = episode
                issues.append(issue)
    auto_payload = _load_json_any(static_dir / TAGGING_PROMPT_MISMATCH_FLAGS)
    auto_reasons = auto_payload.get("flag_reasons") if isinstance(auto_payload, dict) else {}
    if isinstance(auto_reasons, dict):
        for episode_key, items in auto_reasons.items():
            try:
                episode = int(episode_key)
            except (TypeError, ValueError):
                continue
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict) or item.get("reason") != PROMPT_ACTION_MISMATCH_REASON:
                    continue
                issue = dict(item)
                issue["episode"] = episode
                issues.append(issue)
    records = []
    seen_records: set[tuple[int, str]] = set()
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        if issue.get("reason") not in TASK_ASSIGNMENT_REASONS:
            continue
        episode = _issue_episode(issue)
        if episode is None:
            continue
        key = (int(episode), str(issue.get("reason")))
        if key in seen_records:
            continue
        seen_records.add(key)
        metrics = issue.get("metrics") or {}
        candidates = _task_assignment_candidates_for_issue(root, issue)
        records.append(
            {
                "episode_index": int(episode),
                "candidates": candidates,
                "selected_task": candidates[0] if candidates else "",
                "reason": issue.get("reason"),
                "metrics": metrics,
            }
        )
    return sorted(records, key=lambda item: item["episode_index"])


def list_multiple_task_assignments(root: Path, static_dir: Path) -> list[dict]:
    return [
        record
        for record in list_task_assignment_choices(root, static_dir)
        if record.get("reason") == MULTIPLE_TASK_ASSIGNMENT_REASON
    ]


def _auto_flag_set(static_dir: Path, filename: str) -> set[int]:
    return _flag_set(Path(static_dir) / filename)


def _all_auto_flags(static_dir: Path) -> set[int]:
    result: set[int] = set()
    for path in Path(static_dir).glob("*_flagged_episodes.json"):
        if path.name in {"flagged_episodes.json", "manual_flagged_episodes.json"}:
            continue
        result.update(_flag_set(path))
    return result


def _remove_episode_from_auto_flag_file(static_dir: Path, filename: str, episode_index: int) -> None:
    path = Path(static_dir) / filename
    payload = _load_json_any(path)
    if not isinstance(payload, dict):
        return
    flagged = sorted(ep for ep in _flag_set(path) if ep != int(episode_index))
    payload["flagged_episodes"] = flagged
    reasons = payload.get("flag_reasons")
    if isinstance(reasons, dict):
        reasons.pop(str(int(episode_index)), None)
        payload["flag_reasons"] = reasons
    summary = payload.get("summary")
    if isinstance(summary, dict):
        if "quality_episode_count" in summary:
            summary["quality_episode_count"] = len(flagged)
        if "flagged_episode_count" in summary:
            summary["flagged_episode_count"] = len(flagged)
        if "mismatch_episode_count" in summary:
            summary["mismatch_episode_count"] = len(flagged)
        payload["summary"] = summary
    _write_json(path, payload)


def _remove_episode_from_manual_task_assignment_flag(static_dir: Path, episode_index: int) -> None:
    path = Path(static_dir) / "manual_flagged_episodes.json"
    payload = _load_json_any(path)
    if not isinstance(payload, dict):
        return
    reasons = payload.get("flag_reasons")
    if not isinstance(reasons, dict):
        return
    key = str(int(episode_index))
    items = [item for item in reasons.get(key, []) if isinstance(item, dict)]
    retained = [item for item in items if item.get("reason") not in TASK_ASSIGNMENT_REASONS]
    if retained:
        reasons[key] = retained
    else:
        reasons.pop(key, None)
    flagged = sorted(ep for ep in _flag_set(path) if ep != int(episode_index) or retained)
    payload["flagged_episodes"] = flagged
    payload["flag_reasons"] = reasons
    _write_json(path, payload)


def _clear_manual_flags(static_dir: Path) -> int:
    static_dir = Path(static_dir)
    manual_path = static_dir / "manual_flagged_episodes.json"
    previous_manual = _flag_set(manual_path)
    _write_json(manual_path, {"flagged_episodes": [], "flag_reasons": {}})
    flagged_path = static_dir / "flagged_episodes.json"
    retained = _flag_set(flagged_path) - previous_manual
    _write_json(flagged_path, {"flagged_episodes": sorted(retained)})
    return len(previous_manual)


def apply_task_assignment_choice(
    root: Path,
    static_dir: Path,
    episode_index: int,
    selected_task: str,
    reason: str | None = None,
) -> dict:
    root = validate_dataset_root(Path(root))
    static_dir = Path(static_dir).expanduser()
    episode_index = int(episode_index)
    selected_task = str(selected_task or "").strip()
    if not selected_task:
        raise ValueError("selected_task is required")

    existing_flagged = _flag_set(static_dir / "flagged_episodes.json")
    previous_auto = _all_auto_flags(static_dir)
    manual_flagged = existing_flagged - previous_auto

    expected_reason = str(reason or "").strip()
    records = [
        record
        for record in list_task_assignment_choices(root, static_dir)
        if int(record["episode_index"]) == episode_index
    ]
    if expected_reason:
        record = next((record for record in records if record.get("reason") == expected_reason), None)
    else:
        record = records[0] if records else None
    if record is None:
        suffix = f" for reason {expected_reason}" if expected_reason else ""
        raise ValueError(f"episode {episode_index} has no prompt assignment issue{suffix}")
    candidates = set(record.get("candidates") or [])
    if record.get("reason") not in FREEFORM_TASK_ASSIGNMENT_REASONS and selected_task not in candidates:
        raise ValueError(f"selected task is not one of the flagged candidates for episode {episode_index}")

    episodes = load_episode_records(root)
    found_episode = False
    for row in episodes:
        try:
            row_episode = int(row["episode_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if row_episode == episode_index:
            row["tasks"] = [selected_task]
            found_episode = True
            break
    if not found_episode:
        raise ValueError(f"episode {episode_index} not found in dataset metadata")
    write_episode_records(root, episodes)

    info = load_json(root / "meta" / "info.json")
    meta = V3DatasetMetadata(f"local/{root.name}", root) if is_v3_dataset(root) else None
    parquet_path = (
        root / meta.get_data_file_path(episode_index)
        if meta is not None
        else root / format_data_path(info, episode_index)
    )
    task_index_written = None
    if parquet_path.is_file() and "task_index" in pq.read_schema(parquet_path).names:
        task_index_written = _task_index_for_prompt(root, selected_task)
        if meta is not None:
            episode_table = read_episode_table(root, meta, episode_index, columns=["task_index"])
            row_count = episode_table.num_rows
        else:
            row_count = pq.read_metadata(parquet_path).num_rows
        replace_episode_column(
            root,
            meta or _LegacyMetadataPath(info),
            episode_index,
            "task_index",
            [int(task_index_written)] * row_count,
        )

    issues_path = static_dir / "annotation_issues.json"
    issues = _annotation_issues(static_dir)
    retained = []
    removed = 0
    for issue in issues:
        if (
            isinstance(issue, dict)
            and issue.get("reason") in TASK_ASSIGNMENT_REASONS
            and _issue_episode(issue) == episode_index
        ):
            removed += 1
            continue
        retained.append(issue)
    _write_json(issues_path, retained)
    remaining_quality = [
        issue for issue in retained if isinstance(issue, dict) and issue.get("type") == QUALITY_FLAG_TYPE
    ]
    _refresh_quality_flag_files(static_dir, remaining_quality)
    _remove_episode_from_auto_flag_file(static_dir, QUALITY_FLAGGED_EPISODES, episode_index)
    _remove_episode_from_auto_flag_file(static_dir, TAGGING_PROMPT_MISMATCH_FLAGS, episode_index)
    _remove_episode_from_manual_task_assignment_flag(static_dir, episode_index)
    manual_flagged = _flag_set(static_dir / "manual_flagged_episodes.json")
    combined_flags = manual_flagged | _all_auto_flags(static_dir)
    _write_json(static_dir / "flagged_episodes.json", {"flagged_episodes": sorted(combined_flags)})
    return {
        "episode_index": episode_index,
        "selected_task": selected_task,
        "task_index": task_index_written,
        "removed_issues": removed,
        "flagged_episode_count": len(combined_flags),
    }


def _replace_quality_issues(
    static_dir: Path,
    quality_issues: list[dict],
    scanned_episodes: set[int],
    *,
    overwrite: bool = False,
) -> list[dict]:
    issues_path = Path(static_dir) / "annotation_issues.json"
    existing = _load_json_any(issues_path)
    retained = [
        issue
        for issue in (existing if isinstance(existing, list) else [])
        if isinstance(issue, dict)
        and (
            issue.get("type") != QUALITY_FLAG_TYPE
            or (not overwrite and _issue_episode(issue) not in scanned_episodes)
        )
    ]
    merged = retained + sorted(
        quality_issues,
        key=lambda issue: (int(issue.get("episode", -1)), str(issue.get("reason", ""))),
    )
    _write_json(issues_path, merged)
    return merged


def _quality_reason_map(quality_issues: list[dict]) -> dict[str, list[dict]]:
    reasons: dict[str, list[dict]] = {}
    for issue in quality_issues:
        episode = _issue_episode(issue)
        if episode is None:
            continue
        item = {
            "type": str(issue.get("type") or QUALITY_FLAG_TYPE),
            "reason": str(issue.get("reason") or "unknown"),
        }
        if "frames" in issue:
            item["frames"] = issue.get("frames") or []
        if "metrics" in issue:
            item["metrics"] = issue.get("metrics") or {}
        reasons.setdefault(str(episode), []).append(item)
    return reasons


def _is_prompt_action_mismatch_issue(issue: dict) -> bool:
    return (
        isinstance(issue, dict)
        and issue.get("type") == PROMPT_ACTION_MISMATCH_ISSUE_TYPE
        and issue.get("reason") == PROMPT_ACTION_MISMATCH_REASON
    )


def _prompt_action_mismatch_issue_from_tag(record: dict, *, variant: str | None) -> dict | None:
    tags = record.get("tags") or {}
    if tags.get("prompt_action_match") != "mismatch":
        return None
    try:
        episode_index = int(record["episode_index"])
    except (KeyError, TypeError, ValueError):
        return None
    detail = (record.get("tag_details") or {}).get("prompt_action_match") or {}
    observed_object = normalize_object_name(detail.get("observed_object"))
    return {
        "episode": episode_index,
        "type": PROMPT_ACTION_MISMATCH_ISSUE_TYPE,
        "reason": PROMPT_ACTION_MISMATCH_REASON,
        "task": record.get("task"),
        "observed_object": observed_object,
        "vlm_reason": detail.get("reason"),
        "variant": variant or "latest",
        "metrics": {
            "prompt_action_match": "mismatch",
            "observed_object": observed_object,
        },
    }


def _prompt_action_mismatch_reason_map(issues: list[dict]) -> dict[str, list[dict]]:
    reasons: dict[str, list[dict]] = {}
    for issue in issues:
        episode = _issue_episode(issue)
        if episode is None:
            continue
        item = {
            "type": str(issue.get("type") or PROMPT_ACTION_MISMATCH_ISSUE_TYPE),
            "reason": PROMPT_ACTION_MISMATCH_REASON,
            "metrics": issue.get("metrics") or {},
        }
        if issue.get("task"):
            item["task"] = str(issue.get("task"))
        if issue.get("vlm_reason"):
            item["vlm_reason"] = issue.get("vlm_reason")
        reasons.setdefault(str(episode), []).append(item)
    return reasons


def _sync_prompt_action_mismatch_from_tagging(
    static_dir: Path,
    scanned_episodes: set[int],
    *,
    overwrite: bool = False,
) -> tuple[list[dict], set[int]]:
    static_dir = Path(static_dir)
    tagging_dir = static_dir / "tagging"
    variant = latest_tag_variant(tagging_dir)
    records = current_tags(tagging_dir, variant)
    mismatch_issues = [
        issue
        for episode, record in records.items()
        if int(episode) in scanned_episodes
        if (issue := _prompt_action_mismatch_issue_from_tag(record, variant=variant)) is not None
    ]
    mismatch_episodes = {
        episode for episode in (_issue_episode(issue) for issue in mismatch_issues) if episode is not None
    }

    issues_path = static_dir / "annotation_issues.json"
    existing = _load_json_any(issues_path)
    retained = [
        issue
        for issue in (existing if isinstance(existing, list) else [])
        if not (
            _is_prompt_action_mismatch_issue(issue)
            and (overwrite or _issue_episode(issue) in scanned_episodes)
        )
    ]
    merged = retained + sorted(mismatch_issues, key=lambda issue: int(issue.get("episode", -1)))
    _write_json(issues_path, merged)

    auto_path = static_dir / TAGGING_PROMPT_MISMATCH_FLAGS
    flagged_path = static_dir / "flagged_episodes.json"
    previous_auto = _flag_set(auto_path)
    previous_payload = _load_json_any(auto_path)
    previous_reasons = {}
    if isinstance(previous_payload, dict) and isinstance(previous_payload.get("flag_reasons"), dict):
        previous_reasons = {
            str(key): value
            for key, value in previous_payload["flag_reasons"].items()
            if isinstance(value, list)
        }
    retained_previous_auto = set() if overwrite else (previous_auto - scanned_episodes)
    next_auto = retained_previous_auto | mismatch_episodes
    existing_flagged = _flag_set(flagged_path)
    manual_or_other_auto = existing_flagged - previous_auto
    combined = manual_or_other_auto | next_auto
    next_reasons = {
        str(episode): previous_reasons[str(episode)]
        for episode in retained_previous_auto
        if str(episode) in previous_reasons
    }
    next_reasons.update(_prompt_action_mismatch_reason_map(mismatch_issues))
    _write_json(
        auto_path,
        {
            "flagged_episodes": sorted(next_auto),
            "flag_reasons": next_reasons,
            "summary": {
                "reason": PROMPT_ACTION_MISMATCH_REASON,
                "mismatch_episode_count": len(mismatch_episodes),
                "episodes_scanned": len(scanned_episodes),
                "variant": variant or "latest",
            },
        },
    )
    _write_json(flagged_path, {"flagged_episodes": sorted(combined)})
    return merged, combined


def _sync_flagged_episodes(
    static_dir: Path,
    quality_episodes: set[int],
    scanned_episodes: set[int],
    summary: dict,
    quality_issues: list[dict],
    *,
    overwrite: bool = False,
) -> set[int]:
    static_dir = Path(static_dir)
    flagged_path = static_dir / "flagged_episodes.json"
    quality_path = static_dir / QUALITY_FLAGGED_EPISODES
    previous_auto = _flag_set(quality_path)
    previous_payload = _load_json_any(quality_path)
    previous_reasons = {}
    if isinstance(previous_payload, dict) and isinstance(previous_payload.get("flag_reasons"), dict):
        previous_reasons = {
            str(key): value
            for key, value in previous_payload["flag_reasons"].items()
            if isinstance(value, list)
        }
    retained_previous_auto = set() if overwrite else (previous_auto - scanned_episodes)
    next_auto = retained_previous_auto | set(quality_episodes)
    existing_flagged = _flag_set(flagged_path)
    manual_flagged = existing_flagged - previous_auto
    combined = manual_flagged | next_auto
    next_reasons = {
        str(episode): previous_reasons[str(episode)]
        for episode in retained_previous_auto
        if str(episode) in previous_reasons
    }
    next_reasons.update(_quality_reason_map(quality_issues))
    _write_json(
        quality_path,
        {
            "flagged_episodes": sorted(next_auto),
            "flag_reasons": next_reasons,
            "summary": summary,
        },
    )
    _write_json(flagged_path, {"flagged_episodes": sorted(combined)})
    return combined


def run_quality_flag_detection(
    root: Path,
    static_dir: Path,
    episodes: list[int] | None = None,
    data_version: str | None = None,
    workers: int = 8,
    overwrite: bool = False,
    clear_manual_flags: bool = False,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    root = validate_dataset_root(Path(root))
    static_dir = Path(static_dir).expanduser()
    info = load_json(root / "meta" / "info.json")
    selected_data_version = str(
        data_version or infer_data_version_from_features(info.get("features") or {})
    ).upper()
    if selected_data_version not in {DATA_VERSION_DVT1, DATA_VERSION_DVT2}:
        raise ValueError(f"Unsupported data_version: {selected_data_version}")

    episode_rows = load_episode_records(root)
    by_episode = _records_by_episode(episode_rows)
    task_lookup = _tasks_by_index(root)
    meta = V3DatasetMetadata(f"local/{root.name}", root) if is_v3_dataset(root) else None
    selected_ids = sorted({int(episode) for episode in episodes}) if episodes else sorted(by_episode)
    missing = [episode for episode in selected_ids if episode not in by_episode]
    if missing:
        raise ValueError(f"episodes not found: {missing}")
    selected_rows = [by_episode[episode] for episode in selected_ids]

    total = len(selected_rows)
    emit(
        progress_callback,
        status="running",
        current=0,
        total=total,
        message=f"Scanning {total} episodes for abnormal flags ({selected_data_version})",
    )

    quality_issues: list[dict] = []
    total_frames = 0
    worker_count = max(1, int(workers or 1))
    if worker_count == 1 or total <= 1:
        for idx, row in enumerate(selected_rows, start=1):
            episode_index, frame_count, issues = _scan_episode(
                root,
                info,
                row,
                selected_data_version,
                task_lookup,
                meta,
            )
            total_frames += frame_count
            quality_issues.extend(issues)
            emit(
                progress_callback,
                status="running",
                current=idx,
                total=total,
                episode=episode_index,
                message=f"Scanned episode {episode_index}",
            )
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, total)) as executor:
            futures = {
                executor.submit(
                    _scan_episode,
                    root,
                    info,
                    row,
                    selected_data_version,
                    task_lookup,
                    meta,
                ): int(row["episode_index"])
                for row in selected_rows
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                episode_index, frame_count, issues = future.result()
                total_frames += frame_count
                quality_issues.extend(issues)
                emit(
                    progress_callback,
                    status="running",
                    current=idx,
                    total=total,
                    episode=episode_index,
                    message=f"Scanned episode {episode_index}",
                )

    reason_counts = Counter(str(issue.get("reason", "unknown")) for issue in quality_issues)
    scanned_episodes = set(selected_ids)
    quality_episodes = {int(issue["episode"]) for issue in quality_issues if "episode" in issue}
    manual_flags_cleared = _clear_manual_flags(static_dir) if clear_manual_flags else 0
    summary = {
        "data_version": selected_data_version,
        "episodes_scanned": total,
        "quality_episode_count": len(quality_episodes),
        "quality_issue_count": len(quality_issues),
        "reason_counts": dict(sorted(reason_counts.items())),
        "overwrite": bool(overwrite),
        "clear_manual_flags": bool(clear_manual_flags),
        "manual_flags_cleared": manual_flags_cleared,
    }
    all_issues = _replace_quality_issues(static_dir, quality_issues, scanned_episodes, overwrite=overwrite)
    combined_flags = _sync_flagged_episodes(
        static_dir,
        quality_episodes,
        scanned_episodes,
        summary,
        quality_issues,
        overwrite=overwrite,
    )
    all_issues, combined_flags = _sync_prompt_action_mismatch_from_tagging(
        static_dir,
        scanned_episodes,
        overwrite=overwrite,
    )
    summary["annotation_issue_count"] = len(all_issues)
    summary["flagged_episode_count"] = len(combined_flags)
    summary["prompt_action_mismatch_count"] = len(
        [
            issue
            for issue in all_issues
            if _is_prompt_action_mismatch_issue(issue) and _issue_episode(issue) in scanned_episodes
        ]
    )

    emit(
        progress_callback,
        status="done",
        current=total,
        total=total,
        message="Quality flag detection complete",
    )
    return PreprocessResult(
        op="quality_flags",
        src_roots=[root],
        out_root=root,
        repo_id=root.name,
        total_episodes=total,
        total_frames=total_frames,
        dry_run=False,
        summary=summary,
    )
