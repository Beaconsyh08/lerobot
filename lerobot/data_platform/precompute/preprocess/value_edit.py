from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.dataset_io import is_v3_dataset, update_episode_metadata
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    default_preprocess_path,
    emit,
    load_json,
    load_jsonl,
    parquet_paths,
    validate_dataset_root,
)

EDITABLE_FIELDS = ("action", "state")


def _feature_dimension(info: dict, field_name: str) -> int:
    if field_name not in EDITABLE_FIELDS:
        raise ValueError(f"field_name must be one of: {', '.join(EDITABLE_FIELDS)}")
    feature = (info.get("features") or {}).get(field_name)
    if not isinstance(feature, dict):
        raise ValueError(f"Dataset metadata does not define a {field_name} feature")
    shape = feature.get("shape") or []
    if not shape or int(shape[-1]) <= 0:
        raise ValueError(f"Dataset metadata does not define a vector dimension for {field_name}")
    return int(shape[-1])


def _fallback_episode_index(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return int(stem.removeprefix("episode_"))
    raise ValueError(f"Cannot infer episode index from parquet path: {path}")


def _episode_ids(table: pa.Table, path: Path) -> np.ndarray:
    if "episode_index" in table.column_names:
        return np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    return np.full(table.num_rows, _fallback_episode_index(path), dtype=np.int64)


def _update_stats(accumulator: dict, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected non-empty frame vectors, got shape {values.shape}")
    if not accumulator:
        accumulator.update(
            count=0,
            minimum=np.full(values.shape[1], np.inf, dtype=np.float64),
            maximum=np.full(values.shape[1], -np.inf, dtype=np.float64),
            total=np.zeros(values.shape[1], dtype=np.float64),
            total_sq=np.zeros(values.shape[1], dtype=np.float64),
        )
    accumulator["count"] += int(values.shape[0])
    accumulator["minimum"] = np.minimum(accumulator["minimum"], values.min(axis=0))
    accumulator["maximum"] = np.maximum(accumulator["maximum"], values.max(axis=0))
    accumulator["total"] += values.sum(axis=0)
    accumulator["total_sq"] += np.square(values).sum(axis=0)


def _finish_stats(accumulator: dict) -> dict:
    count = int(accumulator["count"])
    mean = accumulator["total"] / count
    variance = np.maximum(accumulator["total_sq"] / count - np.square(mean), 0.0)
    return {
        "min": accumulator["minimum"].tolist(),
        "max": accumulator["maximum"].tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [count],
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.value-edit-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.value-edit-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_stats(root: Path, stats_by_field: dict[str, dict[int, dict]]) -> None:
    if is_v3_dataset(root):
        updates: dict[int, dict] = {}
        for field_name, stats_by_episode in stats_by_field.items():
            for episode_index, stats in stats_by_episode.items():
                updates.setdefault(episode_index, {}).update(
                    {f"stats/{field_name}/{stat_name}": value for stat_name, value in stats.items()}
                )
        update_episode_metadata(
            root,
            updates,
        )
    else:
        stats_path = root / "meta" / "episodes_stats.jsonl"
        existing = {
            int(record["episode_index"]): record
            for record in load_jsonl(stats_path)
            if "episode_index" in record
        }
        rows = []
        episode_ids = sorted({episode_index for stats in stats_by_field.values() for episode_index in stats})
        for episode_index in episode_ids:
            record = existing.get(episode_index, {"episode_index": episode_index, "stats": {}})
            for field_name, stats_by_episode in stats_by_field.items():
                record.setdefault("stats", {})[field_name] = stats_by_episode[episode_index]
            rows.append(record)
        _atomic_write_jsonl(stats_path, rows)

    stats_path = root / "meta" / "stats.json"
    global_stats = load_json(stats_path) if stats_path.is_file() else {}
    for field_name, stats_by_episode in stats_by_field.items():
        global_accumulator: dict = {}
        for stats in stats_by_episode.values():
            count = int(stats["count"][0])
            mean = np.asarray(stats["mean"], dtype=np.float64)
            std = np.asarray(stats["std"], dtype=np.float64)
            if not global_accumulator:
                global_accumulator.update(
                    count=0,
                    minimum=np.full(mean.shape, np.inf, dtype=np.float64),
                    maximum=np.full(mean.shape, -np.inf, dtype=np.float64),
                    total=np.zeros(mean.shape, dtype=np.float64),
                    total_sq=np.zeros(mean.shape, dtype=np.float64),
                )
            global_accumulator["count"] += count
            global_accumulator["minimum"] = np.minimum(
                global_accumulator["minimum"], np.asarray(stats["min"], dtype=np.float64)
            )
            global_accumulator["maximum"] = np.maximum(
                global_accumulator["maximum"], np.asarray(stats["max"], dtype=np.float64)
            )
            global_accumulator["total"] += mean * count
            global_accumulator["total_sq"] += (np.square(std) + np.square(mean)) * count
        global_stats[field_name] = _finish_stats(global_accumulator)
    _atomic_write_json(stats_path, global_stats)


def _apply_value_edits(
    root: Path,
    *,
    edits: list[dict],
    episode_ids: list[int] | None,
    dry_run: bool,
    progress_callback: ProgressCallback,
) -> dict:
    paths = parquet_paths(root)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    requested = None if episode_ids is None else {int(item) for item in episode_ids}
    if requested is not None and (not requested or min(requested) < 0):
        raise ValueError("episode_ids must contain one or more non-negative integers")

    seen_episodes: set[int] = set()
    matched_episodes: set[int] = set()
    matched_frames = 0
    changed_by_edit = [0 for _ in edits]
    edited_fields = sorted({edit["field"] for edit in edits})
    stats_accumulators: dict[str, dict[int, dict]] = {field: {} for field in edited_fields}
    prepared: list[tuple[Path, Path]] = []
    edit_id = uuid.uuid4().hex
    try:
        for path_index, path in enumerate(paths, start=1):
            table = pq.read_table(path)
            row_episode_ids = _episode_ids(table, path)
            seen_episodes.update(int(item) for item in np.unique(row_episode_ids))
            values_by_field = {}
            for field_name in edited_fields:
                if field_name not in table.column_names:
                    raise ValueError(f"Missing {field_name} column in {path}")
                values = np.asarray(table[field_name].to_pylist(), dtype=np.float64)
                if values.ndim != 2:
                    raise ValueError(f"{field_name} must contain per-frame vectors; got shape {values.shape}")
                values_by_field[field_name] = values

            selected_mask = (
                np.ones(table.num_rows, dtype=bool)
                if requested is None
                else np.isin(row_episode_ids, list(requested))
            )
            if np.any(selected_mask):
                matched_frames += int(np.count_nonzero(selected_mask))
                matched_episodes.update(int(item) for item in np.unique(row_episode_ids[selected_mask]))
                for edit_index, edit in enumerate(edits):
                    values = values_by_field[edit["field"]]
                    dimension = edit["dimension"]
                    if dimension >= values.shape[1]:
                        raise ValueError(
                            f"dimension {dimension} is outside {edit['field']} width "
                            f"{values.shape[1]} in {path}"
                        )
                    matched = values[selected_mask, dimension]
                    changed_by_edit[edit_index] += int(np.count_nonzero(matched != edit["value"]))
                    values[selected_mask, dimension] = edit["value"]
                if not dry_run:
                    rewritten = table
                    for field_name in edited_fields:
                        field = table.schema.field(field_name)
                        column_index = table.column_names.index(field_name)
                        rewritten = rewritten.set_column(
                            column_index,
                            field,
                            pa.array(values_by_field[field_name].tolist(), type=field.type),
                        )
                    temporary = path.with_name(f".{path.name}.value-edit-{edit_id}.tmp")
                    pq.write_table(rewritten, temporary)
                    prepared.append((path, temporary))

            for field_name, values in values_by_field.items():
                for episode_index in np.unique(row_episode_ids):
                    episode_mask = row_episode_ids == episode_index
                    _update_stats(
                        stats_accumulators[field_name].setdefault(int(episode_index), {}),
                        values[episode_mask],
                    )
            emit(
                progress_callback,
                status="running",
                current=path_index,
                total=len(paths),
                message=f"Prepared parquet {path_index}/{len(paths)}",
            )

        missing = sorted((requested or set()) - seen_episodes)
        if missing:
            raise ValueError(f"episodes not found: {missing}")
        if not matched_episodes:
            raise ValueError("No episode rows matched the requested selection")

        stats_by_field = {
            field_name: {
                episode_index: _finish_stats(accumulator) for episode_index, accumulator in by_episode.items()
            }
            for field_name, by_episode in stats_accumulators.items()
        }
        if not dry_run:
            for path, temporary in prepared:
                temporary.replace(path)
            _write_stats(root, stats_by_field)
    finally:
        for _, temporary in prepared:
            temporary.unlink(missing_ok=True)

    return {
        "edits": [{**edit, "changed_values": changed_by_edit[index]} for index, edit in enumerate(edits)],
        "edit_count": len(edits),
        "episode_scope": "all" if episode_ids is None else "selected",
        "episode_count": len(matched_episodes),
        "matched_frames": matched_frames,
        "changed_values": sum(changed_by_edit),
        "parquet_files": len(paths),
        "stats_recomputed": not dry_run,
    }


def _normalize_edits(info: dict, edits: list[dict]) -> list[dict]:
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must contain at least one value edit")
    normalized = []
    seen = set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("each edit must be an object")
        field_name = str(edit.get("field") or edit.get("field_name") or "").strip()
        feature_dimension = _feature_dimension(info, field_name)
        try:
            dimension = int(edit.get("dimension"))
            value = float(edit.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValueError("each edit requires an integer dimension and numeric value") from exc
        if dimension < 0 or dimension >= feature_dimension:
            raise ValueError(f"dimension must satisfy 0 <= dimension < {feature_dimension} for {field_name}")
        if not math.isfinite(value):
            raise ValueError("edit values must be finite")
        key = (field_name, dimension)
        if key in seen:
            raise ValueError(f"duplicate edit for {field_name}[{dimension}]")
        seen.add(key)
        normalized.append({"field": field_name, "dimension": dimension, "value": value})
    return normalized


def run_value_edits(
    src_root: Path,
    *,
    edits: list[dict],
    episode_ids: list[int] | None = None,
    in_place: bool = False,
    out_root: Path | None = None,
    dry_run: bool = False,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    info = load_json(src_root / "meta" / "info.json")
    edits = _normalize_edits(info, edits)
    if in_place and out_root is not None:
        raise ValueError("out_root cannot be used with in_place=True")

    target_root = (
        src_root
        if in_place
        else Path(out_root or default_preprocess_path(src_root, "value_edit")).expanduser()
    )
    if not in_place and target_root.exists() and not dry_run:
        raise FileExistsError(f"Output dataset already exists: {target_root}")

    result = PreprocessResult(
        op="set_value",
        src_roots=[src_root],
        out_root=target_root,
        repo_id=f"local/{target_root.name}",
        total_episodes=(
            int(info.get("total_episodes") or 0) if episode_ids is None else len(set(episode_ids))
        ),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=dry_run,
        summary={
            "mode": "in_place" if in_place else "new_dataset",
            "edits": edits,
            "edit_count": len(edits),
            "episode_scope": "all" if episode_ids is None else "selected",
        },
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=1,
        message=(
            f"Planning {len(edits)} value edit(s) "
            f"for {'all episodes' if episode_ids is None else f'{len(set(episode_ids))} episodes'}"
        ),
    )

    if dry_run:
        result.summary.update(
            _apply_value_edits(
                src_root,
                edits=edits,
                episode_ids=episode_ids,
                dry_run=True,
                progress_callback=progress_callback,
            )
        )
        emit(progress_callback, status="done", current=1, total=1, message="Dry run complete")
        return result

    if in_place:
        result.summary.update(
            _apply_value_edits(
                src_root,
                edits=edits,
                episode_ids=episode_ids,
                dry_run=False,
                progress_callback=progress_callback,
            )
        )
    else:
        building_root = target_root.with_name(f".{target_root.name}.building-{uuid.uuid4().hex}")
        try:
            emit(
                progress_callback,
                status="running",
                current=0,
                total=1,
                message=f"Copying source dataset to {target_root}",
            )
            shutil.copytree(src_root, building_root, symlinks=True)
            result.summary.update(
                _apply_value_edits(
                    building_root,
                    edits=edits,
                    episode_ids=episode_ids,
                    dry_run=False,
                    progress_callback=progress_callback,
                )
            )
            building_root.replace(target_root)
        except Exception:
            if building_root.exists():
                shutil.rmtree(building_root)
            raise

    emit(progress_callback, status="done", current=1, total=1, message="Value edit complete")
    return result


def run_set_value(
    src_root: Path,
    *,
    field_name: str,
    dimension: int,
    value: float,
    **kwargs,
) -> PreprocessResult:
    """Compatibility wrapper for callers that only need one edit."""
    return run_value_edits(
        src_root,
        edits=[{"field": field_name, "dimension": dimension, "value": value}],
        **kwargs,
    )
