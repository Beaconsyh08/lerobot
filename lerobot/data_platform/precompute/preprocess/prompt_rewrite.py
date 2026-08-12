import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    load_episode_records,
    load_task_records,
    write_episode_records,
    write_task_records,
)
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    emit,
    format_data_path,
    load_json,
    validate_dataset_root,
    write_json,
)

DEFAULT_PROMPT_PATTERN = r"\bpick up the (.+?) to the (left|right)\b"
DEFAULT_PROMPT_REPLACEMENT = r"pick up the \1 on the \2"

_RELATIVE_WRONG_PATTERN = re.compile(
    r"\bpick up the (.+?) on the (left|right) of the (.+?)\b",
    re.IGNORECASE,
)
_ABSOLUTE_WRONG_PATTERN = re.compile(
    r"\bpick up the (.+?) to the (left|right)\b(?!\s+of\b)",
    re.IGNORECASE,
)


def fix_prompt_prepositions_text(text: str) -> tuple[str, int]:
    """Canonicalize spatial prompt prepositions.

    Absolute position: "pick up the X on the left/right".
    Relative position: "pick up the X to the left/right of the Y".
    """
    rewritten, relative_count = _RELATIVE_WRONG_PATTERN.subn(r"pick up the \1 to the \2 of the \3", str(text))
    rewritten, absolute_count = _ABSOLUTE_WRONG_PATTERN.subn(r"pick up the \1 on the \2", rewritten)
    return rewritten, relative_count + absolute_count


def _rewrite_text(text: str, pattern: re.Pattern, replacement: str) -> tuple[str, int]:
    return pattern.subn(replacement, str(text))


def lowercase_prompt_text(text: str) -> tuple[str, int]:
    original = str(text)
    lowered = original.lower()
    return lowered, int(lowered != original)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _backup_metadata(root: Path) -> dict[str, str]:
    backup_dir = root / "meta" / "prompt_rewrite_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = {}
    names = ("tasks.parquet", "episodes") if is_v3_dataset(root) else ("tasks.jsonl", "episodes.jsonl")
    for name in names:
        src = root / "meta" / name
        if src.exists():
            dst = backup_dir / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            backups[name] = str(dst)
    return backups


def _lowercase_pending_prompt_assignments(
    static_dir: Path | None,
    *,
    dry_run: bool = False,
) -> tuple[int, list[dict]]:
    if static_dir is None:
        return 0, []
    path = Path(static_dir) / "prompt_assignments_pending.json"
    if not path.is_file():
        return 0, []
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0, []
    raw_items = payload.get("assignments") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        return 0, []

    changed = 0
    preview = []
    new_items = []
    for item in raw_items:
        if not isinstance(item, dict):
            new_items.append(item)
            continue
        item = dict(item)
        old_task = str(item.get("selected_task") or "")
        new_task, count = lowercase_prompt_text(old_task)
        if count:
            changed += 1
            item["selected_task"] = new_task
            if len(preview) < 20:
                preview.append(
                    {
                        "kind": "pending_prompt_assignments",
                        "episode_index": item.get("episode_index"),
                        "from": old_task,
                        "to": new_task,
                    }
                )
        new_items.append(item)

    if changed and not dry_run:
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["assignments"] = new_items
        else:
            payload = {"version": 1, "assignments": new_items}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return changed, preview


def _remap_parquet_task_indices(
    root: Path,
    info: dict,
    episode_rows: list[dict],
    old_to_new_task_index: dict[int, int],
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    if not any(old != new for old, new in old_to_new_task_index.items()):
        return 0, 0

    changed_files = 0
    changed_values = 0
    if is_v3_dataset(root):
        meta = V3DatasetMetadata(f"local/{root.name}", root)
        parquet_paths = sorted(
            {root / meta.get_data_file_path(int(row["episode_index"])) for row in episode_rows}
        )
    else:
        parquet_paths = [root / format_data_path(info, int(row["episode_index"])) for row in episode_rows]
    for parquet_path in parquet_paths:
        if not parquet_path.is_file():
            continue
        schema = pq.read_schema(parquet_path)
        if "task_index" not in schema.names:
            continue
        table = pq.read_table(parquet_path)
        field = table.schema.field("task_index")
        values = table["task_index"].to_pylist()
        mapped_values = []
        changed_this_file = 0
        for value in values:
            if value is None:
                mapped_values.append(None)
                continue
            try:
                old_value = int(value)
            except (TypeError, ValueError):
                mapped_values.append(value)
                continue
            new_value = old_to_new_task_index.get(old_value, old_value)
            mapped_values.append(new_value)
            if new_value != old_value:
                changed_this_file += 1
        if not changed_this_file:
            continue
        changed_files += 1
        changed_values += changed_this_file
        if dry_run:
            continue
        idx = table.column_names.index("task_index")
        table = table.set_column(idx, field, pa.array(mapped_values, type=field.type))
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)
    return changed_files, changed_values


def run_rewrite_prompts(
    root: Path,
    pattern: str = DEFAULT_PROMPT_PATTERN,
    replacement: str = DEFAULT_PROMPT_REPLACEMENT,
    dry_run: bool = False,
    backup: bool = True,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    root = validate_dataset_root(root)
    compiled = re.compile(pattern)
    info = load_json(root / "meta" / "info.json")
    task_rows = load_task_records(root)
    episode_rows = load_episode_records(root)

    emit(progress_callback, status="running", current=0, total=3, message="Scanning prompt metadata")
    changed_task_rows = 0
    changed_episode_rows = 0
    changed_episode_task_values = 0
    total_replacements = 0

    new_task_rows = []
    preview = []
    for row in task_rows:
        row = dict(row)
        old_task = str(row.get("task") or "")
        new_task, count = _rewrite_text(old_task, compiled, replacement)
        if count:
            changed_task_rows += 1
            total_replacements += count
            row["task"] = new_task
            if len(preview) < 20:
                preview.append({"kind": "tasks", "from": old_task, "to": new_task})
        new_task_rows.append(row)

    emit(progress_callback, status="running", current=1, total=3, message="Scanning episode task arrays")
    new_episode_rows = []
    for row in episode_rows:
        row = dict(row)
        tasks = row.get("tasks")
        if not isinstance(tasks, list):
            new_episode_rows.append(row)
            continue
        changed_this_episode = False
        new_tasks = []
        for task in tasks:
            new_task, count = _rewrite_text(str(task), compiled, replacement)
            if count:
                changed_this_episode = True
                changed_episode_task_values += 1
                total_replacements += count
                if len(preview) < 20:
                    preview.append(
                        {
                            "kind": "episodes",
                            "episode_index": row.get("episode_index"),
                            "from": task,
                            "to": new_task,
                        }
                    )
            new_tasks.append(new_task)
        if changed_this_episode:
            changed_episode_rows += 1
            row["tasks"] = new_tasks
        new_episode_rows.append(row)

    summary = {
        "pattern": pattern,
        "replacement": replacement,
        "changed_task_rows": changed_task_rows,
        "changed_episode_rows": changed_episode_rows,
        "changed_episode_task_values": changed_episode_task_values,
        "total_replacements": total_replacements,
        "preview": preview,
    }
    result = PreprocessResult(
        op="rewrite_prompts",
        src_roots=[root],
        out_root=root,
        repo_id=f"local/{root.name}",
        total_episodes=int(info.get("total_episodes") or len(episode_rows)),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=dry_run,
        summary=summary,
    )
    if dry_run:
        emit(
            progress_callback,
            status="done",
            current=3,
            total=3,
            message=f"Dry run complete: {total_replacements} replacements",
        )
        return result

    backups = _backup_metadata(root) if backup else {}
    if backups:
        summary["backups"] = backups
    emit(progress_callback, status="running", current=2, total=3, message="Writing prompt metadata")
    if changed_task_rows:
        write_task_records(root, new_task_rows)
    if changed_episode_rows:
        write_episode_records(root, new_episode_rows)
    emit(
        progress_callback,
        status="done",
        current=3,
        total=3,
        message=f"Prompt rewrite complete: {total_replacements} replacements",
    )
    return result


def run_fix_prompt_prepositions(
    root: Path,
    dry_run: bool = False,
    backup: bool = True,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    root = validate_dataset_root(root)
    info = load_json(root / "meta" / "info.json")
    task_rows = load_task_records(root)
    episode_rows = load_episode_records(root)

    emit(progress_callback, status="running", current=0, total=3, message="Scanning prompt prepositions")
    changed_task_rows = 0
    changed_episode_rows = 0
    changed_episode_task_values = 0
    total_replacements = 0
    new_task_rows = []
    preview = []

    for row in task_rows:
        row = dict(row)
        old_task = str(row.get("task") or "")
        new_task, count = fix_prompt_prepositions_text(old_task)
        if count:
            changed_task_rows += 1
            total_replacements += count
            row["task"] = new_task
            if len(preview) < 20:
                preview.append({"kind": "tasks", "from": old_task, "to": new_task})
        new_task_rows.append(row)

    emit(progress_callback, status="running", current=1, total=3, message="Scanning episode task arrays")
    new_episode_rows = []
    for row in episode_rows:
        row = dict(row)
        tasks = row.get("tasks")
        if not isinstance(tasks, list):
            new_episode_rows.append(row)
            continue
        changed_this_episode = False
        new_tasks = []
        for task in tasks:
            new_task, count = fix_prompt_prepositions_text(str(task))
            if count:
                changed_this_episode = True
                changed_episode_task_values += 1
                total_replacements += count
                if len(preview) < 20:
                    preview.append(
                        {
                            "kind": "episodes",
                            "episode_index": row.get("episode_index"),
                            "from": task,
                            "to": new_task,
                        }
                    )
            new_tasks.append(new_task)
        if changed_this_episode:
            changed_episode_rows += 1
            row["tasks"] = new_tasks
        new_episode_rows.append(row)

    summary = {
        "changed_task_rows": changed_task_rows,
        "changed_episode_rows": changed_episode_rows,
        "changed_episode_task_values": changed_episode_task_values,
        "total_replacements": total_replacements,
        "preview": preview,
    }
    result = PreprocessResult(
        op="fix_prompt_prepositions",
        src_roots=[root],
        out_root=root,
        repo_id=f"local/{root.name}",
        total_episodes=int(info.get("total_episodes") or len(episode_rows)),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=dry_run,
        summary=summary,
    )
    if dry_run:
        emit(
            progress_callback,
            status="done",
            current=3,
            total=3,
            message=f"Dry run complete: {total_replacements} prompt fixes",
        )
        return result

    backups = _backup_metadata(root) if backup else {}
    if backups:
        summary["backups"] = backups
    emit(progress_callback, status="running", current=2, total=3, message="Writing prompt metadata")
    if changed_task_rows:
        write_task_records(root, new_task_rows)
    if changed_episode_rows:
        write_episode_records(root, new_episode_rows)
    emit(
        progress_callback,
        status="done",
        current=3,
        total=3,
        message=f"Prompt preposition fix complete: {total_replacements} fixes",
    )
    return result


def run_lowercase_prompts(
    root: Path,
    dry_run: bool = False,
    backup: bool = True,
    static_dir: Path | None = None,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    root = validate_dataset_root(root)
    info = load_json(root / "meta" / "info.json")
    info_path = root / "meta" / "info.json"
    task_rows = load_task_records(root)
    episode_rows = load_episode_records(root)

    emit(progress_callback, status="running", current=0, total=4, message="Scanning prompt case")
    changed_task_rows = 0
    changed_episode_rows = 0
    changed_episode_task_values = 0
    removed_duplicate_task_rows = 0
    total_replacements = 0
    new_task_rows = []
    task_text_to_row = {}
    old_to_new_task_index: dict[int, int] = {}
    preview = []

    for row in task_rows:
        row = dict(row)
        old_task = str(row.get("task") or "")
        new_task, count = lowercase_prompt_text(old_task)
        try:
            old_task_index = int(row["task_index"])
        except (KeyError, TypeError, ValueError):
            old_task_index = None
        if count:
            changed_task_rows += 1
            total_replacements += count
            row["task"] = new_task
            if len(preview) < 20:
                preview.append({"kind": "tasks", "from": old_task, "to": new_task})
        if old_task_index is None:
            new_task_rows.append(row)
            continue
        existing = task_text_to_row.get(new_task)
        if existing is None:
            task_text_to_row[new_task] = row
            new_task_rows.append(row)
            old_to_new_task_index[old_task_index] = old_task_index
        else:
            removed_duplicate_task_rows += 1
            old_to_new_task_index[old_task_index] = int(existing["task_index"])

    emit(progress_callback, status="running", current=1, total=4, message="Scanning episode task arrays")
    new_episode_rows = []
    for row in episode_rows:
        row = dict(row)
        tasks = row.get("tasks")
        if not isinstance(tasks, list):
            new_episode_rows.append(row)
            continue
        changed_this_episode = False
        new_tasks = []
        for task in tasks:
            new_task, count = lowercase_prompt_text(str(task))
            if count:
                changed_this_episode = True
                changed_episode_task_values += 1
                total_replacements += count
                if len(preview) < 20:
                    preview.append(
                        {
                            "kind": "episodes",
                            "episode_index": row.get("episode_index"),
                            "from": task,
                            "to": new_task,
                        }
                    )
            new_tasks.append(new_task)
        if changed_this_episode:
            changed_episode_rows += 1
            row["tasks"] = _dedupe_preserving_order(new_tasks)
        new_episode_rows.append(row)

    pending_prompt_assignments_changed, pending_preview = _lowercase_pending_prompt_assignments(
        static_dir,
        dry_run=True,
    )
    total_replacements += pending_prompt_assignments_changed
    preview.extend(pending_preview[: max(0, 20 - len(preview))])

    emit(
        progress_callback, status="running", current=2, total=4, message="Scanning parquet task_index columns"
    )
    parquet_task_index_files_changed, parquet_task_index_values_changed = _remap_parquet_task_indices(
        root,
        info,
        episode_rows,
        old_to_new_task_index,
        dry_run=True,
    )

    summary = {
        "changed_task_rows": changed_task_rows,
        "changed_episode_rows": changed_episode_rows,
        "changed_episode_task_values": changed_episode_task_values,
        "removed_duplicate_task_rows": removed_duplicate_task_rows,
        "parquet_task_index_files_changed": parquet_task_index_files_changed,
        "parquet_task_index_values_changed": parquet_task_index_values_changed,
        "pending_prompt_assignments_changed": pending_prompt_assignments_changed,
        "total_replacements": total_replacements,
        "preview": preview,
    }
    result = PreprocessResult(
        op="lowercase_prompts",
        src_roots=[root],
        out_root=root,
        repo_id=str(info.get("repo_id") or f"local/{root.name}"),
        total_episodes=int(info.get("total_episodes") or len(episode_rows)),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=dry_run,
        summary=summary,
    )
    if dry_run:
        emit(
            progress_callback,
            status="done",
            current=4,
            total=4,
            message=f"Dry run complete: {total_replacements} prompt case changes",
        )
        return result

    backups = _backup_metadata(root) if backup else {}
    if backups:
        summary["backups"] = backups
    emit(
        progress_callback,
        status="running",
        current=3,
        total=4,
        message="Writing prompt metadata and task_index remap",
    )
    if changed_task_rows or removed_duplicate_task_rows:
        write_task_records(root, new_task_rows)
        info["total_tasks"] = len(new_task_rows)
        write_json(info_path, info)
    if changed_episode_rows:
        write_episode_records(root, new_episode_rows)
    if pending_prompt_assignments_changed:
        _lowercase_pending_prompt_assignments(static_dir, dry_run=False)
    if parquet_task_index_files_changed:
        _remap_parquet_task_indices(root, info, episode_rows, old_to_new_task_index, dry_run=False)
    emit(
        progress_callback,
        status="done",
        current=4,
        total=4,
        message=f"Prompt lowercase complete: {total_replacements} changes",
    )
    return result
