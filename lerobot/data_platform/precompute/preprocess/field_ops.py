import json
from pathlib import Path

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


def _drop_field_from_stats(src: Path, dst: Path, field_name: str) -> int:
    if not src.is_file():
        return 0
    removed = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as input_handle, dst.open("w", encoding="utf-8") as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            stats = record.get("stats")
            if isinstance(stats, dict) and field_name in stats:
                del stats[field_name]
                removed += 1
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return removed


def _drop_field_from_v3_stats(out_root: Path, field_name: str) -> int:
    removed = 0
    stats_path = out_root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = load_json(stats_path)
        if field_name in stats:
            del stats[field_name]
            removed += 1
        write_json(stats_path, stats)

    prefix = f"stats/{field_name}/"
    for path in sorted((out_root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        columns = [name for name in table.column_names if name.startswith(prefix)]
        if columns:
            pq.write_table(table.drop(columns), path)
            removed += len(columns)
    return removed


def run_drop_field(
    src_root: Path,
    out_root: Path | None = None,
    field_name: str = "",
    dry_run: bool = False,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    field_name = str(field_name or "").strip()
    if not field_name:
        raise ValueError("field_name is required")
    out_root = ensure_output_root(
        out_root or default_preprocess_path(src_root, f"drop_{field_name}"), dry_run
    )
    paths = parquet_paths(src_root)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {src_root / 'data'}")

    info = load_json(src_root / "meta" / "info.json")
    present_count = 0
    for path in paths:
        if field_name in pq.read_schema(path).names:
            present_count += 1

    result = PreprocessResult(
        op=f"drop_{field_name}",
        src_roots=[src_root],
        out_root=out_root,
        repo_id=f"local/{out_root.name}",
        total_episodes=int(info.get("total_episodes") or 0),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=dry_run,
        summary={
            "field": field_name,
            "parquet_files_with_field": present_count,
            "parquet_files_total": len(paths),
        },
    )
    if present_count == 0 and field_name not in (info.get("features") or {}):
        raise ValueError(f"Field not found in parquet schema or meta/info features: {field_name}")
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(paths),
        message=f"Planning drop field: {result.summary}",
    )
    if dry_run:
        emit(progress_callback, status="done", current=0, total=len(paths), message="Dry run complete")
        return result

    (out_root / "data").mkdir(parents=True, exist_ok=True)
    copy_meta_files(src_root, out_root)
    copy_sidecar_dirs(src_root, out_root)

    for idx, src in enumerate(paths, start=1):
        table = pq.read_table(src)
        if field_name in table.column_names:
            table = table.drop([field_name])
        dst = out_root / "data" / src.relative_to(src_root / "data")
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst)
        emit(
            progress_callback,
            status="running",
            current=idx,
            total=len(paths),
            message=f"Dropped field from parquet {idx}/{len(paths)}",
        )

    features = dict(info.get("features") or {})
    features.pop(field_name, None)
    info["features"] = features
    write_json(out_root / "meta" / "info.json", info)
    removed_stats = _drop_field_from_stats(
        src_root / "meta" / "episodes_stats.jsonl", out_root / "meta" / "episodes_stats.jsonl", field_name
    )
    removed_stats += _drop_field_from_v3_stats(out_root, field_name)
    result.summary["stats_rows_removed"] = removed_stats
    emit(
        progress_callback,
        status="done",
        current=len(paths),
        total=len(paths),
        message=f"Drop field complete: {out_root}",
    )
    return result
