import csv
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.data_profile import resolve_data_profile, write_data_profile
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    copy_episode_videos,
    default_preprocess_path,
    emit,
    ensure_output_root,
    format_data_path,
    load_json,
    load_jsonl,
    update_info_counts,
    validate_dataset_root,
    write_json,
    write_jsonl,
)
from lerobot.data_platform.precompute.preprocess.dataset_version import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    V30,
    detect_dataset_version,
    materialize_v21_from_v3,
    run_convert_v3,
)


def _merge_features(infos: list[dict]) -> dict:
    features: dict[str, dict] = {}
    for info in infos:
        for name, feature in (info.get("features") or {}).items():
            features.setdefault(name, feature)
    return features


def _feature_shape(feature: dict) -> tuple:
    shape = feature.get("shape") or []
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(value) for value in shape)


def _feature_dtype(feature: dict) -> str:
    return str(feature.get("dtype") or "")


def _feature_signature(feature: dict) -> tuple[str, tuple]:
    return _feature_dtype(feature), _feature_shape(feature)


def _validate_compatible(roots: list[Path], infos: list[dict]):
    base = infos[0]
    base_features = base.get("features") or {}
    base_profile = resolve_data_profile(roots[0], base_features)
    for root, info in zip(roots[1:], infos[1:], strict=False):
        if info.get("robot_type") != base.get("robot_type"):
            raise ValueError(
                f"{root} robot_type={info.get('robot_type')} differs from {base.get('robot_type')}"
            )
        if info.get("fps") != base.get("fps"):
            raise ValueError(f"{root} fps={info.get('fps')} differs from {base.get('fps')}")
        features = info.get("features") or {}
        for name in sorted(set(base_features) & set(features)):
            base_feature = base_features[name] or {}
            feature = features[name] or {}
            if _feature_signature(feature) == _feature_signature(base_feature):
                continue
            base_shape = _feature_shape(base_feature)
            shape = _feature_shape(feature)
            if name in {"action", "state"} and base_shape != shape:
                raise ValueError(
                    f"{name} shape mismatch: {roots[0]} has {list(base_shape)}, "
                    f"{root} has {list(shape)}; please standardize datasets first"
                )
            raise ValueError(
                f"feature schema mismatch for {name}: {roots[0]} has "
                f"dtype={_feature_dtype(base_feature)} shape={list(base_shape)}, "
                f"{root} has dtype={_feature_dtype(feature)} shape={list(shape)}; "
                "please standardize datasets first or merge only compatible datasets"
            )
        profile = resolve_data_profile(root, features)
        semantic_fields = ("robot_profile", "signal_schema", "gripper_encoding", "stage_profile")
        mismatches = [
            field
            for field in semantic_fields
            if getattr(profile, field) != getattr(base_profile, field)
        ]
        if mismatches:
            details = ", ".join(
                f"{field}: {getattr(base_profile, field)} != {getattr(profile, field)}"
                for field in mismatches
            )
            raise ValueError(f"dataset data profile mismatch for {root}: {details}")
    return base_profile


def _build_task_map(roots: list[Path]) -> tuple[list[dict], dict[str, dict[int, int]]]:
    text_to_new: dict[str, int] = {}
    merged_tasks = []
    mapping: dict[str, dict[int, int]] = {}
    for root in roots:
        per_root = {}
        for row in load_jsonl(root / "meta" / "tasks.jsonl"):
            text = row["task"]
            if text not in text_to_new:
                new_idx = len(text_to_new)
                text_to_new[text] = new_idx
                merged_tasks.append({"task_index": new_idx, "task": text})
            per_root[int(row["task_index"])] = text_to_new[text]
        mapping[str(root)] = per_root
    return merged_tasks, mapping


def _normalize_exclude_episodes(
    roots: list[Path],
    exclude_episodes: list[list[int] | set[int] | None] | None,
) -> dict[str, set[int]]:
    if exclude_episodes is None:
        return {}
    if len(exclude_episodes) != len(roots):
        raise ValueError("exclude_episodes must align with src_roots")
    out: dict[str, set[int]] = {}
    for root, values in zip(roots, exclude_episodes, strict=False):
        if values:
            out[str(root)] = {int(value) for value in values}
    return out


def _build_episode_map(
    roots: list[Path],
    exclude_by_root: dict[str, set[int]] | None = None,
) -> tuple[list[tuple[Path, int, int]], list[dict], list[dict], int]:
    episode_map = []
    episodes_out = []
    stats_out = []
    total_frames = 0
    next_episode = 0
    exclude_by_root = exclude_by_root or {}
    for root in roots:
        episodes = load_jsonl(root / "meta" / "episodes.jsonl")
        exclude_set = set(exclude_by_root.get(str(root), set()))
        available_indices = {int(row["episode_index"]) for row in episodes}
        missing_excludes = sorted(exclude_set - available_indices)
        if missing_excludes:
            raise ValueError(f"{root} episodes not found for merge delete: {missing_excludes}")
        stats = {int(row["episode_index"]): row for row in load_jsonl(root / "meta" / "episodes_stats.jsonl")}
        for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
            old_idx = int(episode["episode_index"])
            if old_idx in exclude_set:
                continue
            new_idx = next_episode
            length = int(episode["length"])
            episode_map.append((root, old_idx, new_idx))
            episodes_out.append(
                {"episode_index": new_idx, "tasks": episode.get("tasks", []), "length": length}
            )
            if old_idx in stats:
                row = dict(stats[old_idx])
                row["episode_index"] = new_idx
                stats_payload = row.get("stats") or {}
                if "episode_index" in stats_payload:
                    stats_payload["episode_index"].update(
                        {"min": [new_idx], "max": [new_idx], "mean": [float(new_idx)], "std": [0.0]}
                    )
                if "index" in stats_payload:
                    stats_payload["index"].update(
                        {
                            "min": [total_frames],
                            "max": [total_frames + length - 1],
                            "mean": [total_frames + (length - 1) / 2.0],
                        }
                    )
                stats_out.append(row)
            total_frames += length
            next_episode += 1
    return episode_map, episodes_out, stats_out, total_frames


def _replace_or_append_column(table: pa.Table, field: pa.Field, values: list) -> pa.Table:
    array = pa.array(values, type=field.type)
    if field.name in table.column_names:
        column_idx = table.column_names.index(field.name)
        return table.set_column(column_idx, field, array)
    return table.append_column(field, array)


def _collect_arrow_fields(
    episode_map: list[tuple[Path, int, int]],
    roots: list[Path],
    infos: list[dict],
    merged_features: dict,
) -> dict[str, pa.Field]:
    fields: dict[str, pa.Field] = {}
    for src_root, old_idx, _new_idx in episode_map:
        src_info = infos[roots.index(src_root)]
        parquet_path = src_root / format_data_path(src_info, old_idx)
        if not parquet_path.is_file():
            continue
        schema = pq.read_schema(parquet_path)
        for field in schema:
            if field.name in merged_features and field.name not in fields:
                fields[field.name] = field
    return fields


def _remap_episode_table(
    table: pa.Table,
    *,
    new_idx: int,
    frame_offset: int,
    task_map: dict[int, int],
    merged_features: dict,
    arrow_fields: dict[str, pa.Field],
) -> pa.Table:
    if "episode_index" in table.column_names:
        field = table.schema.field("episode_index")
        table = _replace_or_append_column(table, field, [new_idx] * table.num_rows)

    if "task_index" in table.column_names:
        field = table.schema.field("task_index")
        remapped = []
        for value in table["task_index"].to_pylist():
            try:
                remapped.append(task_map[int(value)])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Cannot remap task_index={value!r} for merged episode {new_idx}") from exc
        table = _replace_or_append_column(table, field, remapped)

    if "index" in table.column_names and "frame_index" in table.column_names:
        field = table.schema.field("index")
        indices = []
        for value in table["frame_index"].to_pylist():
            try:
                indices.append(frame_offset + int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Cannot remap frame_index={value!r} for merged episode {new_idx}") from exc
        table = _replace_or_append_column(table, field, indices)

    missing_features = sorted(set(merged_features) - set(table.column_names))
    for name in missing_features:
        field = arrow_fields.get(name)
        if field is None:
            continue
        table = table.append_column(field, pa.nulls(table.num_rows, type=field.type))
    return table


def _write_merged_episode(
    src_root: Path,
    old_idx: int,
    new_idx: int,
    *,
    src_info: dict,
    out_info: dict,
    out_root: Path,
    frame_offset: int,
    task_map: dict[int, int],
    merged_features: dict,
    arrow_fields: dict[str, pa.Field],
) -> tuple[int, int]:
    src = src_root / format_data_path(src_info, old_idx)
    dst = out_root / format_data_path(out_info, new_idx)
    if not src.is_file():
        raise FileNotFoundError(f"Missing source parquet: {src}")
    table = pq.read_table(src)
    table = _remap_episode_table(
        table,
        new_idx=new_idx,
        frame_offset=frame_offset,
        task_map=task_map,
        merged_features=merged_features,
        arrow_fields=arrow_fields,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return old_idx, new_idx


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _remap_episode_filename(name: str, old_idx: int, new_idx: int) -> str:
    return name.replace(f"episode_{old_idx:06d}", f"episode_{new_idx:06d}")


def _copy_cached_videos(src_static: Path, out_static: Path, old_idx: int, new_idx: int) -> int:
    copied = 0
    videos_dir = Path(src_static) / "videos"
    if not videos_dir.is_dir():
        return copied
    for path in videos_dir.glob(f"*/episode_{old_idx:06d}_h264.mp4"):
        rel = path.relative_to(videos_dir)
        dst = out_static / "videos" / rel.parent / _remap_episode_filename(rel.name, old_idx, new_idx)
        if _copy_file(path, dst):
            copied += 1
    return copied


def _copy_labeling_visuals(src_static: Path, out_static: Path, old_idx: int, new_idx: int) -> int:
    copied = 0
    vis_dir = Path(src_static) / "labeling" / "vis"
    if not vis_dir.is_dir():
        return copied
    for path in vis_dir.glob(f"episode_{old_idx:06d}.*"):
        dst = out_static / "labeling" / "vis" / _remap_episode_filename(path.name, old_idx, new_idx)
        if _copy_file(path, dst):
            copied += 1
    return copied


def _copy_cached_csvs(
    src_static: Path, out_static: Path, old_idx: int, new_idx: int, frame_offset: int
) -> int:
    copied = 0
    csv_dir = Path(src_static) / "csv"
    if not csv_dir.is_dir():
        return copied
    for path in csv_dir.glob(f"episode_{old_idx:06d}_ds*.csv"):
        dst = out_static / "csv" / _remap_episode_filename(path.name, old_idx, new_idx)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with path.open(newline="") as src_handle:
            reader = csv.DictReader(src_handle)
            if reader.fieldnames is None:
                _copy_file(path, dst)
                copied += 1
                continue
            rows = list(reader)
        with dst.open("w", newline="") as dst_handle:
            writer = csv.DictWriter(dst_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in rows:
                if "episode_index" in row:
                    row["episode_index"] = str(new_idx)
                if "frame_index" in row and "index" in row:
                    with suppress(TypeError, ValueError):
                        row["index"] = str(frame_offset + int(float(row["frame_index"])))
                writer.writerow(row)
        copied += 1
    return copied


def _load_json_file(path: Path):
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_json_file(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_jsonl_records(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    records = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _merge_record_jsonl(
    src_static_dirs: list[Path | None],
    out_static: Path,
    episode_mapping: dict[tuple[int, int], int],
    feature_dir: str,
    pattern: str,
) -> int:
    records_by_name: dict[str, list[dict]] = {}
    for src_pos, src_static in enumerate(src_static_dirs):
        if src_static is None:
            continue
        src_dir = Path(src_static) / feature_dir
        if not src_dir.is_dir():
            continue
        for path in src_dir.glob(pattern):
            if pattern in {"labels*.jsonl", "tags*.jsonl"} and (
                path.name.startswith("labels_reviewed") or path.name.startswith("tags_reviewed")
            ):
                continue
            out_records = records_by_name.setdefault(path.name, [])
            for record in _read_jsonl_records(path):
                try:
                    old_episode = int(record["episode_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                new_episode = episode_mapping.get((src_pos, old_episode))
                if new_episode is None:
                    continue
                updated = dict(record)
                updated["episode_index"] = int(new_episode)
                out_records.append(updated)
    written = 0
    for name, records in records_by_name.items():
        if not records:
            continue
        records.sort(key=lambda item: int(item.get("episode_index", -1)))
        _write_jsonl_records(out_static / feature_dir / name, records)
        written += 1
    return written


def _merge_source_jsons(
    src_static_dirs: list[Path | None], out_static: Path, feature_dir: str, pattern: str
) -> int:
    written = 0
    grouped: dict[str, list[dict]] = {}
    for src_static in src_static_dirs:
        if src_static is None:
            continue
        src_dir = Path(src_static) / feature_dir
        if not src_dir.is_dir():
            continue
        for path in src_dir.glob(pattern):
            payload = _load_json_file(path)
            if isinstance(payload, dict):
                grouped.setdefault(path.name, []).append(payload)
    for name, sources in grouped.items():
        if not sources:
            continue
        merged = dict(sources[-1])
        merged["merged_from"] = sources
        _write_json_file(out_static / feature_dir / name, merged)
        written += 1
    return written


def _merge_flag_jsons(
    src_static_dirs: list[Path | None],
    out_static: Path,
    episode_mapping: dict[tuple[int, int], int],
) -> int:
    payloads: dict[str, dict] = {}
    for src_pos, src_static in enumerate(src_static_dirs):
        if src_static is None:
            continue
        for path in Path(src_static).glob("*flagged_episodes.json"):
            payload = _load_json_file(path)
            if not isinstance(payload, dict):
                continue
            out = payloads.setdefault(path.name, {"flagged_episodes": [], "flag_reasons": {}, "summary": {}})
            for value in payload.get("flagged_episodes") or []:
                try:
                    new_episode = episode_mapping.get((src_pos, int(value)))
                except (TypeError, ValueError):
                    continue
                if new_episode is not None:
                    out["flagged_episodes"].append(int(new_episode))
            flag_reasons = payload.get("flag_reasons") or {}
            if isinstance(flag_reasons, dict):
                for key, reasons in flag_reasons.items():
                    try:
                        new_episode = episode_mapping.get((src_pos, int(key)))
                    except (TypeError, ValueError):
                        continue
                    if new_episode is not None:
                        out.setdefault("flag_reasons", {})[str(new_episode)] = reasons
    written = 0
    for name, payload in payloads.items():
        payload["flagged_episodes"] = sorted({int(ep) for ep in payload.get("flagged_episodes") or []})
        _write_json_file(out_static / name, payload)
        written += 1
    return written


def _merge_annotation_issues(
    src_static_dirs: list[Path | None],
    out_static: Path,
    episode_mapping: dict[tuple[int, int], int],
) -> int:
    merged = []
    for src_pos, src_static in enumerate(src_static_dirs):
        if src_static is None:
            continue
        issues = _load_json_file(Path(src_static) / "annotation_issues.json")
        if not isinstance(issues, list):
            continue
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            try:
                old_episode = int(issue["episode"])
            except (KeyError, TypeError, ValueError):
                continue
            new_episode = episode_mapping.get((src_pos, old_episode))
            if new_episode is None:
                continue
            updated = dict(issue)
            updated["episode"] = int(new_episode)
            merged.append(updated)
    if not merged:
        return 0
    merged.sort(
        key=lambda issue: (
            int(issue.get("episode", -1)),
            str(issue.get("type", "")),
            str(issue.get("reason", "")),
        )
    )
    _write_json_file(out_static / "annotation_issues.json", merged)
    return len(merged)


def _merge_keyed_episode_jsons(
    src_static_dirs: list[Path | None],
    out_static: Path,
    episode_mapping: dict[tuple[int, int], int],
    filenames: list[str],
) -> int:
    grouped: dict[str, dict] = {}
    for src_pos, src_static in enumerate(src_static_dirs):
        if src_static is None:
            continue
        for filename in filenames:
            payload = _load_json_file(Path(src_static) / filename)
            if not isinstance(payload, dict):
                continue
            out = grouped.setdefault(filename, {})
            for key, value in payload.items():
                try:
                    new_episode = episode_mapping.get((src_pos, int(key)))
                except (TypeError, ValueError):
                    continue
                if new_episode is None:
                    continue
                updated_value = dict(value) if isinstance(value, dict) else value
                if isinstance(updated_value, dict) and "episode_index" in updated_value:
                    updated_value["episode_index"] = int(new_episode)
                out[str(new_episode)] = updated_value

    written = 0
    for filename, payload in grouped.items():
        if not payload:
            continue
        ordered = {key: payload[key] for key in sorted(payload, key=lambda item: int(item))}
        _write_json_file(out_static / filename, ordered)
        written += 1
    return written


def _merge_pending_prompt_assignments(
    src_static_dirs: list[Path | None],
    out_static: Path,
    episode_mapping: dict[tuple[int, int], int],
) -> int:
    assignments = []
    for src_pos, src_static in enumerate(src_static_dirs):
        if src_static is None:
            continue
        payload = _load_json_file(Path(src_static) / "prompt_assignments_pending.json")
        raw_items = payload.get("assignments") if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                new_episode = episode_mapping.get((src_pos, int(item["episode_index"])))
            except (KeyError, TypeError, ValueError):
                continue
            if new_episode is None:
                continue
            updated = dict(item)
            updated["episode_index"] = int(new_episode)
            assignments.append(updated)

    if not assignments:
        return 0
    assignments.sort(key=lambda item: int(item.get("episode_index", -1)))
    _write_json_file(
        out_static / "prompt_assignments_pending.json", {"version": 1, "assignments": assignments}
    )
    return len(assignments)


def _merge_static_artifacts(
    episode_map: list[tuple[int, Path, int, int]],
    frame_offsets: dict[int, int],
    src_static_dirs: list[Path | None] | None,
    out_static_dir: Path | None,
) -> dict:
    if not src_static_dirs or out_static_dir is None:
        return {}
    out_static = Path(out_static_dir)
    src_dirs = [Path(path) if path is not None else None for path in src_static_dirs]
    episode_mapping = {(src_pos, old_idx): new_idx for src_pos, _, old_idx, new_idx in episode_map}
    summary = {
        "csv_files": 0,
        "video_files": 0,
        "labeling_vis_files": 0,
        "labeling_jsonl_files": 0,
        "tagging_jsonl_files": 0,
        "source_json_files": 0,
        "flag_json_files": 0,
        "annotation_issues": 0,
        "keyed_episode_json_files": 0,
        "pending_prompt_assignments": 0,
    }
    for src_pos, _src_root, old_idx, new_idx in episode_map:
        src_static = src_dirs[src_pos] if src_pos < len(src_dirs) else None
        if src_static is None:
            continue
        summary["csv_files"] += _copy_cached_csvs(
            src_static, out_static, old_idx, new_idx, frame_offsets[new_idx]
        )
        summary["video_files"] += _copy_cached_videos(src_static, out_static, old_idx, new_idx)
        summary["labeling_vis_files"] += _copy_labeling_visuals(src_static, out_static, old_idx, new_idx)
    summary["labeling_jsonl_files"] += _merge_record_jsonl(
        src_dirs, out_static, episode_mapping, "labeling", "labels*.jsonl"
    )
    summary["labeling_jsonl_files"] += _merge_record_jsonl(
        src_dirs, out_static, episode_mapping, "labeling", "labels_reviewed*.jsonl"
    )
    summary["tagging_jsonl_files"] += _merge_record_jsonl(
        src_dirs, out_static, episode_mapping, "tagging", "tags*.jsonl"
    )
    summary["tagging_jsonl_files"] += _merge_record_jsonl(
        src_dirs, out_static, episode_mapping, "tagging", "tags_reviewed*.jsonl"
    )
    summary["source_json_files"] += _merge_source_jsons(src_dirs, out_static, "labeling", "source*.json")
    summary["source_json_files"] += _merge_source_jsons(src_dirs, out_static, "tagging", "source*.json")
    summary["flag_json_files"] = _merge_flag_jsons(src_dirs, out_static, episode_mapping)
    summary["annotation_issues"] = _merge_annotation_issues(src_dirs, out_static, episode_mapping)
    summary["keyed_episode_json_files"] = _merge_keyed_episode_jsons(
        src_dirs,
        out_static,
        episode_mapping,
        ["trim_annotations.json", "subtask_annotations.json"],
    )
    summary["pending_prompt_assignments"] = _merge_pending_prompt_assignments(
        src_dirs, out_static, episode_mapping
    )
    return summary


def run_merge(
    src_roots: list[Path],
    out_root: Path | None = None,
    dry_run: bool = False,
    src_static_dirs: list[Path | None] | None = None,
    out_static_dir: Path | None = None,
    workers: int = 8,
    exclude_episodes: list[list[int] | set[int] | None] | None = None,
    progress_callback: ProgressCallback = None,
    _allow_single_source: bool = False,
    _op: str = "merge",
    _default_op: str = "merge",
    _summary_extra: dict | None = None,
) -> PreprocessResult:
    roots = [validate_dataset_root(root) for root in src_roots]
    if not roots:
        raise ValueError("merge requires at least one source dataset")
    if len(roots) < 2 and not _allow_single_source:
        raise ValueError("merge requires at least two source datasets")
    out_root = ensure_output_root(out_root or default_preprocess_path(roots[0], _default_op), dry_run)
    infos = [load_json(root / "meta" / "info.json") for root in roots]
    output_profile = _validate_compatible(roots, infos)
    v3_positions = [position for position, root in enumerate(roots) if detect_dataset_version(root) == V30]
    if v3_positions:
        source_info = infos[v3_positions[0]]
        with tempfile.TemporaryDirectory(
            prefix=f".lerobot-v3-{_op}-",
            dir=out_root.parent,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            legacy_roots = list(roots)
            for position in v3_positions:
                legacy_roots[position] = materialize_v21_from_v3(
                    roots[position],
                    temp_root / f"source_{position:03d}",
                    include_data=True,
                    include_videos=not dry_run,
                    workers=max(1, int(workers or 1)),
                    progress_callback=progress_callback,
                )
            legacy_output = temp_root / "output"
            legacy_result = run_merge(
                legacy_roots,
                out_root=legacy_output,
                dry_run=dry_run,
                src_static_dirs=src_static_dirs,
                out_static_dir=out_static_dir,
                workers=workers,
                exclude_episodes=exclude_episodes,
                progress_callback=progress_callback,
                _allow_single_source=_allow_single_source,
                _op=_op,
                _default_op=_default_op,
                _summary_extra=_summary_extra,
            )
            summary = {
                **legacy_result.summary,
                "source_formats": [detect_dataset_version(root) for root in roots],
                "output_format": V30,
            }
            if not dry_run:
                run_convert_v3(
                    legacy_output,
                    out_root,
                    data_file_size_in_mb=int(
                        source_info.get("data_files_size_in_mb") or DEFAULT_DATA_FILE_SIZE_IN_MB
                    ),
                    video_file_size_in_mb=int(
                        source_info.get("video_files_size_in_mb") or DEFAULT_VIDEO_FILE_SIZE_IN_MB
                    ),
                    workers=max(1, int(workers or 1)),
                    progress_callback=progress_callback,
                )
                converted_info = load_json(out_root / "meta" / "info.json")
                write_data_profile(out_root, output_profile, info=converted_info)
                write_json(out_root / "meta" / "info.json", converted_info)
            return PreprocessResult(
                op=_op,
                src_roots=roots,
                out_root=out_root,
                repo_id=f"local/{out_root.name}",
                total_episodes=legacy_result.total_episodes,
                total_frames=legacy_result.total_frames,
                dry_run=dry_run,
                summary=summary,
                episode_lineage=list(legacy_result.episode_lineage),
            )
    merged_features = _merge_features(infos)
    merged_tasks, task_maps = _build_task_map(roots)
    exclude_by_root = _normalize_exclude_episodes(roots, exclude_episodes)
    raw_episode_map, episodes_out, stats_out, total_frames = _build_episode_map(roots, exclude_by_root)
    if not raw_episode_map:
        raise ValueError(f"{_op} produced no episodes; source datasets may have empty meta/episodes.jsonl")
    root_positions = {str(root): idx for idx, root in enumerate(roots)}
    episode_map = [
        (root_positions[str(root)], root, old_idx, new_idx) for root, old_idx, new_idx in raw_episode_map
    ]
    arrow_fields = _collect_arrow_fields(raw_episode_map, roots, infos, merged_features)
    worker_count = max(1, int(workers or 1))
    result = PreprocessResult(
        op=_op,
        src_roots=roots,
        out_root=out_root,
        repo_id=f"local/{out_root.name}",
        total_episodes=len(episodes_out),
        total_frames=total_frames,
        dry_run=dry_run,
        summary={
            "source_count": len(roots),
            "episodes": len(episodes_out),
            "tasks": len(merged_tasks),
            "workers": worker_count,
            "deleted_source_episodes": {
                str(root): sorted(values) for root, values in exclude_by_root.items()
            },
            **(_summary_extra or {}),
        },
        episode_lineage=[
            {
                "source_position": src_pos,
                "source_root": str(root),
                "source_episode_index": old_idx,
                "output_episode_index": new_idx,
            }
            for src_pos, root, old_idx, new_idx in episode_map
        ],
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(episode_map),
        message=f"Planning {_op}: {result.summary}",
    )
    if dry_run:
        emit(progress_callback, status="done", current=0, total=len(episode_map), message="Dry run complete")
        return result

    try:
        info = update_info_counts(infos[0], len(episodes_out), total_frames, len(merged_tasks))
        info["features"] = merged_features
        write_data_profile(out_root, output_profile, info=info)
        write_json(out_root / "meta" / "info.json", info)
        write_jsonl(out_root / "meta" / "tasks.jsonl", merged_tasks)
        write_jsonl(out_root / "meta" / "episodes.jsonl", episodes_out)
        write_jsonl(out_root / "meta" / "episodes_stats.jsonl", stats_out)

        frame_offsets = {}
        running = 0
        for episode in episodes_out:
            frame_offsets[int(episode["episode_index"])] = running
            running += int(episode["length"])

        info_by_root = {str(root): src_info for root, src_info in zip(roots, infos, strict=False)}
        if worker_count == 1 or len(episode_map) <= 1:
            for idx, (_src_pos, src_root, old_idx, new_idx) in enumerate(episode_map, start=1):
                _write_merged_episode(
                    src_root,
                    old_idx,
                    new_idx,
                    src_info=info_by_root[str(src_root)],
                    out_info=info,
                    out_root=out_root,
                    frame_offset=frame_offsets[new_idx],
                    task_map=task_maps[str(src_root)],
                    merged_features=merged_features,
                    arrow_fields=arrow_fields,
                )
                emit(
                    progress_callback,
                    status="running",
                    current=idx,
                    total=len(episode_map),
                    message=f"Merged episode {old_idx} -> {new_idx}",
                )
        else:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(episode_map))) as executor:
                futures = {
                    executor.submit(
                        _write_merged_episode,
                        src_root,
                        old_idx,
                        new_idx,
                        src_info=info_by_root[str(src_root)],
                        out_info=info,
                        out_root=out_root,
                        frame_offset=frame_offsets[new_idx],
                        task_map=task_maps[str(src_root)],
                        merged_features=merged_features,
                        arrow_fields=arrow_fields,
                    ): (old_idx, new_idx)
                    for _src_pos, src_root, old_idx, new_idx in episode_map
                }
                for idx, future in enumerate(as_completed(futures), start=1):
                    old_idx, new_idx = future.result()
                    emit(
                        progress_callback,
                        status="running",
                        current=idx,
                        total=len(episode_map),
                        message=f"Merged episode {old_idx} -> {new_idx}",
                    )

        written_parquets = list((out_root / "data").rglob("*.parquet"))
        if len(written_parquets) != len(episode_map):
            raise RuntimeError(
                f"{_op} wrote {len(written_parquets)} parquet files, expected {len(episode_map)}"
            )

        copy_episode_videos(
            [(src_root, old_idx, new_idx) for _src_pos, src_root, old_idx, new_idx in episode_map],
            info,
            out_root,
        )
        if out_static_dir is None and src_static_dirs:
            out_static_dir = out_root.parent / "vis" / f"local_vis_{out_root.name}" / "static"
        artifact_summary = _merge_static_artifacts(
            episode_map, frame_offsets, src_static_dirs, out_static_dir
        )
        if artifact_summary:
            result.summary["artifacts"] = artifact_summary
    except Exception:
        if out_root.exists():
            shutil.rmtree(out_root, ignore_errors=True)
        raise
    emit(
        progress_callback,
        status="done",
        current=len(episode_map),
        total=len(episode_map),
        message=f"{_op.title()} complete: {out_root}",
    )
    return result
