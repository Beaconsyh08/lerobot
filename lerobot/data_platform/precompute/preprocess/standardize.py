import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.data_profile import (
    SIGNAL_SCHEMA_TRAIN_16D,
    resolve_data_profile,
    write_data_profile,
)
from lerobot.data_platform.precompute.dataset_io import is_v3_info
from lerobot.data_platform.precompute.preprocess.action_dim import _trim_arrow_type
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    copy_meta_files,
    copy_sidecar_dirs,
    emit,
    load_json,
    load_jsonl,
    parquet_paths,
    validate_dataset_root,
    write_json,
    write_jsonl,
)
from lerobot.data_platform.precompute.preprocess.field_ops import _drop_field_from_v3_stats
from lerobot.data_platform.precompute.preprocess.smooth_action import _rewrite_v3_stats
from lerobot.data_platform.precompute.timeseries import (
    DATA_VERSION_DVT2,
    normalize_gripper_columns,
)

ACTION_COLUMN = "action"
STATE_COLUMN = "state"
EXIST_LABEL_COLUMN = "exist_label"
EXIST_LABEL_FEATURE = {"dtype": "int32", "shape": [1], "names": None}
TARGET_DIM = 16
DEPTH_FIELD_SUFFIXES = {"head_depth", "left_wrist_depth", "right_wrist_depth"}
STANDARDIZE_META = "preprocess_standardize.json"


def default_standardize_path(src_root: Path) -> Path:
    return Path(src_root).expanduser().parent / f"{Path(src_root).name}_preprocessed"


def _is_depth_field(name: str) -> bool:
    suffix = str(name).replace("/", ".").split(".")[-1]
    return suffix in DEPTH_FIELD_SUFFIXES


def _prepare_output_root(src_root: Path, out_root: Path, dry_run: bool, overwrite: bool) -> Path:
    out_root = Path(out_root).expanduser()
    if out_root.resolve() == src_root.resolve():
        raise ValueError("Standardize output root must be different from source root")
    if out_root.exists() and not dry_run:
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {out_root}")
        shutil.rmtree(out_root)
    return out_root


def _trim_vector(value: Any, target_dim: int = TARGET_DIM) -> Any:
    if value is None or not isinstance(value, (list, tuple)):
        return value
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return [_trim_vector(item, target_dim) for item in value]
    if len(value) < target_dim:
        raise ValueError(f"Cannot trim vector of dim {len(value)} to {target_dim}")
    return list(value[:target_dim])


def _normalize_then_trim(values: list, column_name: str, data_version: str) -> list:
    arr = normalize_gripper_columns(np.asarray(values, dtype=np.float64), column_name, data_version)
    if arr.ndim < 2:
        return values
    if arr.shape[1] < TARGET_DIM:
        raise ValueError(f"{column_name} dim {arr.shape[1]} is smaller than required {TARGET_DIM}")
    return arr[:, :TARGET_DIM].tolist()


def _column_stats(values: list) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _fallback_episode_index(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return int(stem.removeprefix("episode_"))
    raise ValueError(f"Cannot infer episode index from parquet path: {path}")


def _rewrite_parquet(
    src: Path,
    dst: Path,
    data_version: str,
) -> tuple[dict[int, dict[str, dict]], set[str], dict[str, int]]:
    table = pq.read_table(src)
    arrays = []
    fields = []
    rewritten_columns: dict[str, list] = {}
    dropped_fields: set[str] = set()
    dims_after: dict[str, int] = {}
    has_exist_label = EXIST_LABEL_COLUMN in table.column_names

    for field in table.schema:
        if _is_depth_field(field.name):
            dropped_fields.add(field.name)
            continue
        if field.name in {ACTION_COLUMN, STATE_COLUMN}:
            values = _normalize_then_trim(table[field.name].to_pylist(), field.name, data_version)
            rewritten_columns[field.name] = values
            target_type = _trim_arrow_type(field.type, TARGET_DIM)
            arrays.append(pa.array(values, type=target_type))
            fields.append(pa.field(field.name, target_type, nullable=field.nullable, metadata=field.metadata))
            dims_after[field.name] = TARGET_DIM
            continue
        arrays.append(table[field.name])
        fields.append(field)

    if not has_exist_label:
        values = [1] * table.num_rows
        rewritten_columns[EXIST_LABEL_COLUMN] = values
        arrays.append(pa.array(values, type=pa.int32()))
        fields.append(pa.field(EXIST_LABEL_COLUMN, pa.int32()))

    if "episode_index" in table.column_names:
        episode_values = [int(value) for value in table["episode_index"].to_pylist()]
        episode_indices = list(dict.fromkeys(episode_values))
        positions_by_episode = {
            episode_index: [
                position for position, value in enumerate(episode_values) if value == episode_index
            ]
            for episode_index in episode_indices
        }
    else:
        episode_index = _fallback_episode_index(src)
        positions_by_episode = {episode_index: list(range(table.num_rows))}

    stats_by_episode = {}
    for episode_index, positions in positions_by_episode.items():
        stats_by_episode[episode_index] = {
            column: _column_stats([values[position] for position in positions])
            for column, values in rewritten_columns.items()
        }

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata)), dst
    )
    return stats_by_episode, dropped_fields, dims_after


def _rewrite_parquet_worker(
    args: tuple[str, str, str],
) -> tuple[dict[int, dict[str, dict]], set[str], dict[str, int]]:
    src, dst, data_version = args
    return _rewrite_parquet(Path(src), Path(dst), data_version)


def _resolve_workers(workers: int | None, total: int) -> int:
    if total <= 1:
        return 1
    if workers is None or int(workers) == 0:
        return min(8, total)
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be 0 for auto or >= 1")
    return min(workers, total)


def _update_info(info: dict, dropped_fields: set[str]) -> dict:
    updated = json.loads(json.dumps(info))
    features = updated.setdefault("features", {})
    for key in list(features.keys()):
        if key in dropped_fields or _is_depth_field(key):
            features.pop(key, None)
            continue
        if key in {ACTION_COLUMN, STATE_COLUMN} and isinstance(features.get(key), dict):
            shape = list(features[key].get("shape") or [])
            if shape:
                shape[-1] = TARGET_DIM
            else:
                shape = [TARGET_DIM]
            features[key]["shape"] = shape
    exist_label_feature = dict(EXIST_LABEL_FEATURE)
    if is_v3_info(updated):
        exist_label_feature["fps"] = int(updated["fps"])
    features.setdefault(EXIST_LABEL_COLUMN, exist_label_feature)
    return updated


def _rewrite_stats(
    src: Path, dst: Path, stats_by_episode: dict[int, dict[str, dict]], dropped_fields: set[str]
) -> None:
    rows = load_jsonl(src)
    if not rows:
        rows = [
            {"episode_index": episode_index, "stats": stats}
            for episode_index, stats in sorted(stats_by_episode.items())
        ]
    else:
        for row in rows:
            episode_index = int(row["episode_index"])
            stats = row.setdefault("stats", {})
            for key in list(stats.keys()):
                if key in dropped_fields or _is_depth_field(key):
                    stats.pop(key, None)
            if episode_index in stats_by_episode:
                stats.update(stats_by_episode[episode_index])
    write_jsonl(dst, rows)


def run_standardize_dataset(
    src_root: Path,
    out_root: Path | None = None,
    data_version: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    workers: int | None = 8,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    info = load_json(src_root / "meta" / "info.json")
    source_profile = resolve_data_profile(
        src_root,
        info.get("features") or {},
        data_version_override=data_version,
        default_data_version=DATA_VERSION_DVT2,
    )
    data_version = source_profile.legacy_data_version
    output_profile = source_profile.for_signal_schema(
        SIGNAL_SCHEMA_TRAIN_16D,
        gripper_encoding=(
            "normalized_0_1" if data_version == DATA_VERSION_DVT2 else source_profile.gripper_encoding
        ),
        resolution_source="standardize",
    )
    out_root = _prepare_output_root(
        src_root, out_root or default_standardize_path(src_root), dry_run, overwrite
    )
    paths = parquet_paths(src_root)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")
    worker_count = _resolve_workers(workers, len(paths))

    result = PreprocessResult(
        op="standardize",
        src_roots=[src_root],
        out_root=out_root,
        repo_id=f"local/{out_root.name}",
        total_episodes=int(info.get("total_episodes") or len(paths)),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=dry_run,
        summary={
            "data_version": data_version,
            "robot_profile": output_profile.robot_profile,
            "signal_schema": output_profile.signal_schema,
            "stage_profile": output_profile.stage_profile,
            "gripper_encoding": output_profile.gripper_encoding,
            "target_dim": TARGET_DIM,
            "drop_depth_suffixes": sorted(DEPTH_FIELD_SUFFIXES),
            "overwrite": overwrite,
            "workers": worker_count,
        },
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(paths),
        message=f"Planning standardization: {result.summary}",
    )
    if dry_run:
        emit(progress_callback, status="done", current=0, total=len(paths), message="Dry run complete")
        return result

    (out_root / "data").mkdir(parents=True, exist_ok=True)
    copy_meta_files(src_root, out_root)
    copy_sidecar_dirs(src_root, out_root)

    stats_by_episode: dict[int, dict[str, dict]] = {}
    dropped_fields: set[str] = set()
    dims_after: dict[str, int] = {}
    tasks = [
        (str(src), str(out_root / "data" / src.relative_to(src_root / "data")), data_version) for src in paths
    ]
    if worker_count == 1:
        for idx, task in enumerate(tasks, start=1):
            parquet_stats, parquet_dropped, parquet_dims = _rewrite_parquet_worker(task)
            stats_by_episode.update(parquet_stats)
            dropped_fields.update(parquet_dropped)
            dims_after.update(parquet_dims)
            emit(
                progress_callback,
                status="running",
                current=idx,
                total=len(paths),
                message=f"Standardized parquet {idx}/{len(paths)}",
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_rewrite_parquet_worker, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                parquet_stats, parquet_dropped, parquet_dims = future.result()
                stats_by_episode.update(parquet_stats)
                dropped_fields.update(parquet_dropped)
                dims_after.update(parquet_dims)
                emit(
                    progress_callback,
                    status="running",
                    current=idx,
                    total=len(paths),
                    message=f"Standardized parquet {idx}/{len(paths)} with {worker_count} workers",
                )

    result.summary["dropped_fields"] = sorted(dropped_fields)
    result.summary["dims_after"] = dims_after
    output_info = _update_info(info, dropped_fields)
    write_data_profile(out_root, output_profile, info=output_info)
    write_json(out_root / "meta" / "info.json", output_info)
    if is_v3_info(info):
        _rewrite_v3_stats(out_root, stats_by_episode)
        for field_name in dropped_fields:
            _drop_field_from_v3_stats(out_root, field_name)
    else:
        _rewrite_stats(
            src_root / "meta" / "episodes_stats.jsonl",
            out_root / "meta" / "episodes_stats.jsonl",
            stats_by_episode,
            dropped_fields,
        )
    write_json(
        out_root / "meta" / STANDARDIZE_META,
        {
            "op": "standardize",
            "source_root": str(src_root),
            "output_root": str(out_root),
            "source_data_version": data_version,
            "robot_profile": output_profile.robot_profile,
            "signal_schema": output_profile.signal_schema,
            "stage_profile": output_profile.stage_profile,
            "gripper_encoding": output_profile.gripper_encoding,
            "target_dim": TARGET_DIM,
            "dropped_fields": sorted(dropped_fields),
            "workers": worker_count,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    emit(
        progress_callback,
        status="done",
        current=len(paths),
        total=len(paths),
        message=f"Standardization complete: {out_root}",
    )
    return result
