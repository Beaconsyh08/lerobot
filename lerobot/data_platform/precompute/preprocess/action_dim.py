import copy
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    copy_meta_files,
    copy_sidecar_dirs,
    default_preprocess_path,
    emit,
    ensure_output_root,
    load_json,
    parquet_paths,
    validate_dataset_root,
    write_json,
)

ACTION_COLUMN = "action"
STATE_COLUMN = "state"


def _leaf_dim(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        return None
    first = value[0]
    if isinstance(first, (list, tuple)):
        dims = {dim for item in value if (dim := _leaf_dim(item)) is not None}
        if not dims:
            return None
        if len(dims) != 1:
            raise ValueError(f"Mixed nested vector dimensions: {sorted(dims)}")
        return next(iter(dims))
    return len(value)


def _trim_value(value: Any, target_dim: int) -> Any:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return value
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return [_trim_value(item, target_dim) for item in value]
    if len(value) < target_dim:
        raise ValueError(f"Cannot trim vector of dim {len(value)} to {target_dim}")
    return list(value[:target_dim])


def _trim_arrow_type(source_type: pa.DataType, target_dim: int) -> pa.DataType:
    if pa.types.is_fixed_size_list(source_type):
        child = source_type.value_type
        if pa.types.is_fixed_size_list(child) or pa.types.is_list(child) or pa.types.is_large_list(child):
            return pa.list_(_trim_arrow_type(child, target_dim), source_type.list_size)
        if source_type.list_size >= target_dim:
            return pa.list_(child, target_dim)
        raise ValueError(f"Cannot trim Arrow list size {source_type.list_size} to {target_dim}")
    if pa.types.is_list(source_type):
        child = source_type.value_type
        if pa.types.is_fixed_size_list(child) or pa.types.is_list(child) or pa.types.is_large_list(child):
            return pa.list_(_trim_arrow_type(child, target_dim))
        return pa.list_(child)
    if pa.types.is_large_list(source_type):
        child = source_type.value_type
        if pa.types.is_fixed_size_list(child) or pa.types.is_list(child) or pa.types.is_large_list(child):
            return pa.large_list(_trim_arrow_type(child, target_dim))
        return pa.large_list(child)
    return source_type


def _inspect_dims(paths: list[Path], columns: list[str]) -> dict[str, int]:
    dims: dict[str, set[int]] = {column: set() for column in columns}
    for path in paths:
        schema = pq.read_schema(path)
        available = [column for column in columns if column in schema.names]
        if not available:
            continue
        table = pq.read_table(path, columns=available)
        for column in available:
            for value in table[column].to_pylist():
                dim = _leaf_dim(value)
                if dim is not None:
                    dims[column].add(dim)
    out = {}
    for column, values in dims.items():
        if values:
            if len(values) != 1:
                raise ValueError(f"Observed mixed dimensions for {column}: {sorted(values)}")
            out[column] = next(iter(values))
    return out


def _rewrite_parquet(src: Path, dst: Path, columns_to_trim: set[str], target_dim: int) -> None:
    table = pq.read_table(src)
    arrays = []
    fields = []
    for field in table.schema:
        if field.name in columns_to_trim:
            values = [_trim_value(value, target_dim) for value in table[field.name].to_pylist()]
            target_type = _trim_arrow_type(field.type, target_dim)
            arrays.append(pa.array(values, type=target_type))
            fields.append(pa.field(field.name, target_type, nullable=field.nullable, metadata=field.metadata))
        else:
            arrays.append(table[field.name])
            fields.append(field)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata)), dst
    )


def _updated_info(info: dict, dims: dict[str, int], target_dim: int) -> dict:
    updated = copy.deepcopy(info)
    for column in (ACTION_COLUMN, STATE_COLUMN):
        feature = (updated.get("features") or {}).get(column)
        if not isinstance(feature, dict):
            continue
        shape = list(feature.get("shape") or [])
        if shape and column in dims and shape[-1] != target_dim:
            shape[-1] = target_dim
            feature["shape"] = shape
    return updated


def _trim_stats_file(src: Path, dst: Path, columns: set[str], target_dim: int) -> int:
    if not src.is_file():
        return 0
    count = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as input_handle, dst.open("w", encoding="utf-8") as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            stats = record.get("stats") or {}
            for column in columns:
                column_stats = stats.get(column)
                if not isinstance(column_stats, dict):
                    continue
                for key in ("min", "max", "mean", "std"):
                    if key in column_stats:
                        column_stats[key] = _trim_value(column_stats[key], target_dim)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _trim_v3_stats(out_root: Path, columns: set[str], target_dim: int) -> None:
    stats_path = out_root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = load_json(stats_path)
        for column in columns:
            column_stats = stats.get(column)
            if not isinstance(column_stats, dict):
                continue
            for key in ("min", "max", "mean", "std"):
                if key in column_stats:
                    column_stats[key] = _trim_value(column_stats[key], target_dim)
        write_json(stats_path, stats)

    for path in sorted((out_root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        arrays = []
        fields = []
        stats_columns = {
            f"stats/{column}/{stat_name}" for column in columns for stat_name in ("min", "max", "mean", "std")
        }
        for field in table.schema:
            if field.name in stats_columns:
                values = [_trim_value(value, target_dim) for value in table[field.name].to_pylist()]
                target_type = _trim_arrow_type(field.type, target_dim)
                arrays.append(pa.array(values, type=target_type))
                fields.append(
                    pa.field(
                        field.name,
                        target_type,
                        nullable=field.nullable,
                        metadata=field.metadata,
                    )
                )
            else:
                arrays.append(table[field.name])
                fields.append(field)
        pq.write_table(
            pa.Table.from_arrays(
                arrays,
                schema=pa.schema(fields, metadata=table.schema.metadata),
            ),
            path,
        )


def run_convert_action(
    src_root: Path,
    out_root: Path | None = None,
    target_dim: int = 16,
    dry_run: bool = False,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    out_root = ensure_output_root(
        out_root or default_preprocess_path(src_root, f"action{target_dim}"), dry_run
    )
    paths = parquet_paths(src_root)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")

    info = load_json(src_root / "meta" / "info.json")
    candidate_columns = [ACTION_COLUMN]
    if STATE_COLUMN in (info.get("features") or {}):
        candidate_columns.append(STATE_COLUMN)
    dims = _inspect_dims(paths, candidate_columns)
    columns_to_trim = {column for column, dim in dims.items() if dim > target_dim}

    result = PreprocessResult(
        op=f"action{target_dim}",
        src_roots=[src_root],
        out_root=out_root,
        repo_id=f"local/{out_root.name}",
        total_episodes=int(info.get("total_episodes") or 0),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=dry_run,
        summary={"observed_dims": dims, "trimmed_columns": sorted(columns_to_trim), "target_dim": target_dim},
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(paths),
        message=f"Planning action/state dim conversion: {result.summary}",
    )
    if dry_run:
        emit(progress_callback, status="done", current=0, total=len(paths), message="Dry run complete")
        return result

    (out_root / "data").mkdir(parents=True, exist_ok=True)
    copy_meta_files(src_root, out_root)
    copy_sidecar_dirs(src_root, out_root)

    for idx, src in enumerate(paths, start=1):
        dst = out_root / "data" / src.relative_to(src_root / "data")
        if columns_to_trim:
            _rewrite_parquet(src, dst, columns_to_trim, target_dim)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        emit(
            progress_callback,
            status="running",
            current=idx,
            total=len(paths),
            message=f"Converted parquet {idx}/{len(paths)}",
        )

    write_json(out_root / "meta" / "info.json", _updated_info(info, dims, target_dim))
    _trim_stats_file(
        src_root / "meta" / "episodes_stats.jsonl",
        out_root / "meta" / "episodes_stats.jsonl",
        columns_to_trim,
        target_dim,
    )
    _trim_v3_stats(out_root, columns_to_trim, target_dim)
    emit(
        progress_callback,
        status="done",
        current=len(paths),
        total=len(paths),
        message=f"Action/state conversion complete: {out_root}",
    )
    return result
