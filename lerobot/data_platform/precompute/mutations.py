from __future__ import annotations

import json
import logging
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.data_platform.precompute.annotation import assign_subtask_states
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    is_v3_metadata,
    read_episode_table,
    replace_episode_column,
    update_episode_metadata,
    upsert_episode_column,
)
from lerobot.data_platform.task_text import generate_subtask_text


def update_info_features(dataset_root: Path, new_features: dict[str, dict]) -> bool:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return False

    info = json.loads(info_path.read_text())
    info_features = info.setdefault("features", {})
    changed = False
    for key, value in new_features.items():
        if key not in info_features:
            info_features[key] = value
            changed = True

    if changed:
        info_path.write_text(json.dumps(info, indent=2))
    return changed


def update_episode_stats_for_subtask_state(dataset_root: Path, episode_stats: dict[int, dict]) -> bool:
    if is_v3_dataset(dataset_root):
        updates = {
            int(episode_id): {
                f"stats/subtask_state/{stat_name}": stat_value for stat_name, stat_value in stats.items()
            }
            for episode_id, stats in episode_stats.items()
        }
        changed = bool(update_episode_metadata(dataset_root, updates))
        if not changed:
            return False

        meta = V3DatasetMetadata(f"local/{Path(dataset_root).name}", dataset_root)
        all_feature_stats = [
            stats["subtask_state"]
            for stats in meta.episodes_stats.values()
            if "subtask_state" in stats and stats["subtask_state"].get("count") is not None
        ]
        counts = np.asarray(
            [float(stats["count"][0]) for stats in all_feature_stats],
            dtype=np.float64,
        )
        means = np.asarray(
            [float(stats["mean"][0]) for stats in all_feature_stats],
            dtype=np.float64,
        )
        stds = np.asarray(
            [float(stats["std"][0]) for stats in all_feature_stats],
            dtype=np.float64,
        )
        total = float(counts.sum())
        mean = float(np.sum(counts * means) / total) if total else 0.0
        variance = float(np.sum(counts * (stds**2 + means**2)) / total - mean**2) if total else 0.0
        stats_path = Path(dataset_root) / "meta" / "stats.json"
        global_stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
        global_stats["subtask_state"] = {
            "min": [min(int(stats["min"][0]) for stats in all_feature_stats)],
            "max": [max(int(stats["max"][0]) for stats in all_feature_stats)],
            "mean": [mean],
            "std": [float(np.sqrt(max(0.0, variance)))],
            "count": [int(total)],
        }
        stats_path.write_text(json.dumps(global_stats, indent=2))
        return True

    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.is_file() or not episode_stats:
        return False

    rows = []
    changed = False
    with jsonlines.open(stats_path, mode="r") as reader:
        for row in reader:
            episode_id = row["episode_index"]
            if episode_id in episode_stats:
                row.setdefault("stats", {})
                row["stats"]["subtask_state"] = episode_stats[episode_id]
                changed = True
            rows.append(row)

    if changed:
        with jsonlines.open(stats_path, mode="w") as writer:
            writer.write_all(rows)
    return changed


def write_subtask_state_to_parquet(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    boundaries_by_episode: dict[int | str, dict],
) -> dict[int, dict]:
    all_episode_stats: dict[int, dict] = {}
    with tqdm(
        total=len(boundaries_by_episode),
        desc="Writing subtask_state to parquet",
        unit="episode",
        dynamic_ncols=True,
    ) as progress:
        for episode_key, bounds in boundaries_by_episode.items():
            episode_id = int(episode_key)
            parquet_path = dataset_root / meta.get_data_file_path(episode_id)
            if not parquet_path.is_file():
                progress.update(1)
                continue

            table = read_episode_table(dataset_root, meta, episode_id)
            timestamps = table.column("timestamp").to_numpy()
            states = assign_subtask_states(timestamps, bounds)
            none_indices = [idx for idx, state in enumerate(states) if state is None]
            if none_indices:
                logging.warning(
                    "Episode %d: subtask_state has %d None value(s) at indices: %s",
                    episode_id,
                    len(none_indices),
                    none_indices,
                )

            if is_v3_metadata(meta):
                upsert_episode_column(
                    dataset_root,
                    meta,
                    episode_id,
                    "subtask_state",
                    states,
                    pa.int32(),
                )
            else:
                column = pa.array(states, type=pa.int32())
                if "subtask_state" in table.column_names:
                    column_idx = table.column_names.index("subtask_state")
                    table = table.set_column(column_idx, "subtask_state", column)
                else:
                    table = table.append_column("subtask_state", column)
                tmp_path = parquet_path.with_suffix(".tmp")
                pq.write_table(table, tmp_path)
                tmp_path.rename(parquet_path)

            arr = np.array(states, dtype=np.float64)
            all_episode_stats[episode_id] = {
                "min": [int(arr.min())],
                "max": [int(arr.max())],
                "mean": [float(arr.mean())],
                "std": [float(arr.std())],
                "count": [len(arr)],
            }
            progress.update(1)

    return all_episode_stats


def write_subtask_text_to_parquet(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    episodes: list[int],
) -> int:
    written = 0
    with tqdm(
        total=len(episodes),
        desc="Writing subtask text to parquet",
        unit="episode",
        dynamic_ncols=True,
    ) as progress:
        for episode_id in episodes:
            parquet_path = dataset_root / meta.get_data_file_path(episode_id)
            if not parquet_path.is_file():
                progress.update(1)
                continue

            table = read_episode_table(dataset_root, meta, episode_id)
            if "subtask_state" not in table.column_names:
                logging.warning("Episode %d: no subtask_state in parquet, skipping subtask text", episode_id)
                progress.update(1)
                continue

            states = table.column("subtask_state").to_pylist()
            if not states or all(state is None for state in states):
                logging.warning(
                    "Episode %d: subtask_state is empty, skipping subtask text",
                    episode_id,
                )
                progress.update(1)
                continue
            task = ""
            if hasattr(meta, "episodes") and episode_id in meta.episodes:
                tasks = meta.episodes[episode_id].get("tasks", [])
                task = tasks[0] if tasks else ""

            if "exist" in table.column_names:
                exist_vals = table.column("exist").to_pylist()
                states = [-1 if int(exist_vals[idx]) == 0 else state for idx, state in enumerate(states)]
                if is_v3_metadata(meta):
                    replace_episode_column(
                        dataset_root,
                        meta,
                        episode_id,
                        "subtask_state",
                        states,
                    )
                else:
                    state_col = pa.array(states, type=pa.int32())
                    state_idx = table.column_names.index("subtask_state")
                    table = table.set_column(state_idx, "subtask_state", state_col)

            subtask_texts = [generate_subtask_text(task, state) for state in states]
            if is_v3_metadata(meta):
                upsert_episode_column(
                    dataset_root,
                    meta,
                    episode_id,
                    "subtask",
                    subtask_texts,
                    pa.string(),
                )
            else:
                subtask_col = pa.array(subtask_texts, type=pa.string())
                if "subtask" in table.column_names:
                    subtask_idx = table.column_names.index("subtask")
                    table = table.set_column(subtask_idx, "subtask", subtask_col)
                else:
                    table = table.append_column("subtask", subtask_col)
                tmp_path = parquet_path.with_suffix(".tmp")
                pq.write_table(table, tmp_path)
                tmp_path.rename(parquet_path)
            written += 1
            progress.update(1)

    return written


def _column_to_numpy(column) -> np.ndarray:
    try:
        return column.to_numpy(zero_copy_only=False)
    except Exception:
        return np.asarray(column.to_pylist())


def _is_contiguous_range(column, start: int, length: int) -> bool:
    if length == 0:
        return True
    values = _column_to_numpy(column)
    if len(values) != length:
        return False
    try:
        if int(values[0]) != start or int(values[-1]) != start + length - 1:
            return False
        return bool(np.all(values == np.arange(start, start + length, dtype=values.dtype)))
    except (TypeError, ValueError, OverflowError):
        return False


def _first_timestamp_offset(column) -> float | None:
    if len(column) == 0:
        return None
    try:
        first = column[0].as_py()
        if first is None:
            return None
        offset = float(first)
    except (TypeError, ValueError, OverflowError):
        return None
    return offset if abs(offset) > 1e-6 else None


def _range_array(start: int, length: int, pa_type) -> pa.Array:
    return pa.array(range(start, start + length), type=pa_type)


def fix_episode_indices(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    episodes: list[int],
) -> bool:
    """Fix frame_index/index/timestamp so each episode starts at zero and global indices are contiguous."""
    if is_v3_metadata(meta):
        any_fixed = False
        global_offset = 0
        length_updates = {}
        metadata_updates = {}
        for episode_id in tqdm(
            sorted(episodes),
            desc="Fixing episode indices",
            unit="episode",
            dynamic_ncols=True,
        ):
            table = read_episode_table(
                dataset_root,
                meta,
                episode_id,
                columns=[
                    name
                    for name in ("frame_index", "timestamp", "index")
                    if name in pq.read_schema(Path(dataset_root) / meta.get_data_file_path(episode_id)).names
                ],
            )
            num_rows = int(table.num_rows)
            if "frame_index" in table.column_names and not _is_contiguous_range(
                table["frame_index"], 0, num_rows
            ):
                replace_episode_column(
                    dataset_root,
                    meta,
                    episode_id,
                    "frame_index",
                    list(range(num_rows)),
                )
                any_fixed = True
            if "timestamp" in table.column_names:
                offset = _first_timestamp_offset(table["timestamp"])
                if offset is not None:
                    values = (
                        _column_to_numpy(table["timestamp"]).astype(np.float64, copy=False) - offset
                    ).tolist()
                    replace_episode_column(
                        dataset_root,
                        meta,
                        episode_id,
                        "timestamp",
                        values,
                    )
                    any_fixed = True
            if "index" in table.column_names and not _is_contiguous_range(
                table["index"], global_offset, num_rows
            ):
                replace_episode_column(
                    dataset_root,
                    meta,
                    episode_id,
                    "index",
                    list(range(global_offset, global_offset + num_rows)),
                )
                any_fixed = True

            recorded_length = int(meta.episodes.get(episode_id, {}).get("length", num_rows))
            if recorded_length != num_rows:
                length_updates[episode_id] = num_rows
                any_fixed = True
            metadata_updates[episode_id] = {
                "length": num_rows,
                "dataset_from_index": global_offset,
                "dataset_to_index": global_offset + num_rows,
            }
            global_offset += num_rows

        if metadata_updates:
            update_episode_metadata(dataset_root, metadata_updates)
            for episode_id, fields in metadata_updates.items():
                meta.episodes[episode_id].update(fields)
        info_path = Path(dataset_root) / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        if int(info.get("total_frames") or 0) != global_offset:
            info["total_frames"] = global_offset
            info_path.write_text(json.dumps(info, indent=2))
            meta.info["total_frames"] = global_offset
            any_fixed = True
        return any_fixed

    any_fixed = False
    global_offset = 0
    total_frames = 0
    length_updates = {}

    for episode_id in tqdm(
        sorted(episodes),
        desc="Fixing episode indices",
        unit="episode",
        dynamic_ncols=True,
    ):
        parquet_path = dataset_root / meta.get_data_file_path(episode_id)
        if not parquet_path.is_file():
            logging.warning(
                "Episode %d: parquet not found at %s, skipping index fix", episode_id, parquet_path
            )
            continue

        parquet_file = pq.ParquetFile(parquet_path)
        num_rows = int(parquet_file.metadata.num_rows)
        column_names = set(parquet_file.schema_arrow.names)
        check_columns = [name for name in ["frame_index", "timestamp", "index"] if name in column_names]
        check_table = pq.read_table(parquet_path, columns=check_columns) if check_columns else None
        fixes: dict[str, object] = {}

        if check_table is not None and "frame_index" in check_table.column_names:
            frame_index_col = check_table.column("frame_index")
            if not _is_contiguous_range(frame_index_col, 0, num_rows):
                first_value = frame_index_col[0].as_py() if len(frame_index_col) else "?"
                logging.info(
                    "Episode %d: fixing frame_index (starts at %s, expected 0)",
                    episode_id,
                    first_value,
                )
                fixes["frame_index"] = True

        if check_table is not None and "timestamp" in check_table.column_names:
            offset = _first_timestamp_offset(check_table.column("timestamp"))
            if offset is not None:
                logging.info(
                    "Episode %d: shifting timestamps by %.6f (was starting at %.6f)",
                    episode_id,
                    -offset,
                    offset,
                )
                fixes["timestamp_offset"] = offset

        if check_table is not None and "index" in check_table.column_names:
            index_col = check_table.column("index")
            if not _is_contiguous_range(index_col, global_offset, num_rows):
                first_value = index_col[0].as_py() if len(index_col) else "?"
                logging.info(
                    "Episode %d: fixing global index (%s -> %s)",
                    episode_id,
                    first_value,
                    global_offset,
                )
                fixes["index"] = True

        if fixes:
            table = pq.read_table(parquet_path)
            if fixes.get("frame_index") and "frame_index" in table.column_names:
                field = table.schema.field("frame_index")
                idx = table.column_names.index("frame_index")
                table = table.set_column(idx, field, _range_array(0, num_rows, field.type))
            if "timestamp_offset" in fixes and "timestamp" in table.column_names:
                offset = float(fixes["timestamp_offset"])
                field = table.schema.field("timestamp")
                idx = table.column_names.index("timestamp")
                timestamps = _column_to_numpy(table.column("timestamp")).astype(np.float64, copy=False)
                shifted = timestamps - offset
                table = table.set_column(idx, field, pa.array(shifted, type=field.type))
            if fixes.get("index") and "index" in table.column_names:
                field = table.schema.field("index")
                idx = table.column_names.index("index")
                table = table.set_column(idx, field, _range_array(global_offset, num_rows, field.type))
            tmp_path = parquet_path.with_suffix(".tmp")
            pq.write_table(table, tmp_path)
            tmp_path.rename(parquet_path)
            any_fixed = True

        recorded_length = meta.episodes.get(episode_id, {}).get("length", num_rows)
        if recorded_length != num_rows:
            length_updates[episode_id] = num_rows

        global_offset += num_rows
        total_frames += num_rows

    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        if info.get("total_frames") != total_frames:
            logging.info("Fixing info.json total_frames: %s -> %d", info.get("total_frames"), total_frames)
            info["total_frames"] = total_frames
            info_path.write_text(json.dumps(info, indent=2))

    if length_updates:
        episodes_path = dataset_root / "meta" / "episodes.jsonl"
        if episodes_path.is_file():
            rows = []
            with jsonlines.open(episodes_path, mode="r") as reader:
                for row in reader:
                    episode_id = row["episode_index"]
                    if episode_id in length_updates:
                        logging.info(
                            "Fixing episodes.jsonl length for episode %d: %s -> %d",
                            episode_id,
                            row.get("length"),
                            length_updates[episode_id],
                        )
                        row["length"] = length_updates[episode_id]
                    rows.append(row)
            with jsonlines.open(episodes_path, mode="w") as writer:
                writer.write_all(rows)

    return any_fixed
