import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    load_episode_records,
    read_episode_table,
)
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    emit,
    format_data_path,
    load_json,
    validate_dataset_root,
)
from lerobot.data_platform.precompute.preprocess.dataset_merge import run_merge

VOLATILE_FINGERPRINT_COLUMNS = {"episode_index", "index", "task_index"}


def _episode_index(episode: dict) -> int:
    return int(episode["episode_index"])


def _canonical_tasks(episode: dict) -> list[str]:
    return sorted({str(task) for task in episode.get("tasks", [])})


def _stable_table_bytes(table: pa.Table) -> tuple[list[tuple[str, str]], bytes]:
    stable_names = sorted(name for name in table.column_names if name not in VOLATILE_FINGERPRINT_COLUMNS)
    columns = [(name, str(table.schema.field(name).type)) for name in stable_names]
    if not stable_names:
        return columns, b""
    stable_table = pa.Table.from_arrays(
        [table[name].combine_chunks() for name in stable_names], names=stable_names
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, stable_table.schema) as writer:
        writer.write_table(stable_table)
    return columns, sink.getvalue().to_pybytes()


def _episode_fingerprint(
    root: Path,
    info: dict,
    episode: dict,
    meta: V3DatasetMetadata | None = None,
) -> str:
    episode_idx = _episode_index(episode)
    parquet_path = (
        root / meta.get_data_file_path(episode_idx)
        if meta is not None
        else root / format_data_path(info, episode_idx)
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing source parquet: {parquet_path}")

    table = read_episode_table(root, meta, episode_idx) if meta is not None else pq.read_table(parquet_path)
    stable_columns, table_bytes = _stable_table_bytes(table)
    metadata = {
        "length": int(episode.get("length") or table.num_rows),
        "stable_columns": stable_columns,
        "tasks": _canonical_tasks(episode),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(table_bytes)
    return digest.hexdigest()


def _fingerprints_for_root(
    root: Path,
    *,
    progress_callback: ProgressCallback = None,
    progress_start: int = 0,
    progress_total: int = 1,
    message_prefix: str = "Fingerprinting",
) -> tuple[dict[int, str], Counter[str]]:
    info = load_json(root / "meta" / "info.json")
    episodes = sorted(load_episode_records(root), key=_episode_index)
    meta = V3DatasetMetadata(f"local/{root.name}", root) if is_v3_dataset(root) else None
    fingerprints: dict[int, str] = {}
    counter: Counter[str] = Counter()
    for offset, episode in enumerate(episodes, start=1):
        episode_idx = _episode_index(episode)
        fingerprint = _episode_fingerprint(root, info, episode, meta)
        fingerprints[episode_idx] = fingerprint
        counter[fingerprint] += 1
        emit(
            progress_callback,
            status="running",
            current=progress_start + offset,
            total=progress_total,
            message=f"{message_prefix} {root.name} episode {episode_idx}",
        )
    return fingerprints, counter


def _duplicate_summary(counter: Counter[str]) -> dict:
    return {
        "fingerprints": sum(1 for count in counter.values() if count > 1),
        "episodes": sum(count for count in counter.values() if count > 1),
    }


def run_subtract(
    base_root: Path,
    subtract_roots: list[Path],
    out_root: Path | None = None,
    dry_run: bool = False,
    src_static_dir: Path | None = None,
    out_static_dir: Path | None = None,
    workers: int = 8,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    base_root = validate_dataset_root(base_root)
    subtract_roots = [validate_dataset_root(root) for root in subtract_roots]
    if not subtract_roots:
        raise ValueError("subtract requires at least one subtract dataset")

    base_episodes = load_episode_records(base_root)
    subtract_episode_counts = [len(load_episode_records(root)) for root in subtract_roots]
    fingerprint_total = max(1, len(base_episodes) + sum(subtract_episode_counts))

    base_fingerprints, base_counter = _fingerprints_for_root(
        base_root,
        progress_callback=progress_callback,
        progress_total=fingerprint_total,
        message_prefix="Fingerprinting base",
    )

    subtract_counter: Counter[str] = Counter()
    current = len(base_fingerprints)
    for root, episode_count in zip(subtract_roots, subtract_episode_counts, strict=False):
        _fingerprints, counter = _fingerprints_for_root(
            root,
            progress_callback=progress_callback,
            progress_start=current,
            progress_total=fingerprint_total,
            message_prefix="Fingerprinting subtract",
        )
        subtract_counter.update(counter)
        current += episode_count

    subtract_fingerprints = set(subtract_counter)
    removed_base_episodes = sorted(
        episode_idx
        for episode_idx, fingerprint in base_fingerprints.items()
        if fingerprint in subtract_fingerprints
    )
    kept_episodes = len(base_fingerprints) - len(removed_base_episodes)
    if kept_episodes <= 0:
        raise ValueError("subtract would remove all base episodes; refusing to write an empty dataset")

    base_fingerprint_set = set(base_fingerprints.values())
    unmatched_subtract = subtract_fingerprints - base_fingerprint_set
    summary = {
        "base_episodes": len(base_fingerprints),
        "subtract_source_count": len(subtract_roots),
        "subtract_episodes": sum(subtract_episode_counts),
        "removed_base_episodes": removed_base_episodes,
        "removed_base_episode_count": len(removed_base_episodes),
        "kept_base_episode_count": kept_episodes,
        "matched_fingerprint_count": len(subtract_fingerprints & base_fingerprint_set),
        "matched_subtract_episode_count": sum(
            count for fingerprint, count in subtract_counter.items() if fingerprint in base_fingerprint_set
        ),
        "unmatched_subtract_fingerprint_count": len(unmatched_subtract),
        "unmatched_subtract_episode_count": sum(
            count for fingerprint, count in subtract_counter.items() if fingerprint in unmatched_subtract
        ),
        "duplicate_fingerprint_counts": {
            "base": _duplicate_summary(base_counter),
            "subtract": _duplicate_summary(subtract_counter),
        },
    }

    return run_merge(
        [base_root],
        out_root=out_root,
        dry_run=dry_run,
        src_static_dirs=[src_static_dir] if src_static_dir is not None else None,
        out_static_dir=out_static_dir,
        workers=workers,
        exclude_episodes=[removed_base_episodes],
        progress_callback=progress_callback,
        _allow_single_source=True,
        _op="subtract",
        _default_op="subtract",
        _summary_extra=summary,
    )
