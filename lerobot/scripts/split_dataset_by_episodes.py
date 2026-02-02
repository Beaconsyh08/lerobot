#!/usr/bin/env python3
"""Split a LeRobot dataset into multiple datasets with a fixed number of episodes.

This script:
- copies meta/tasks.jsonl and norm_stats.json as-is
- slices meta/episodes.jsonl and meta/episodes_stats.jsonl
- reindexes episodes per split starting at 0
- rewrites parquet files to update episode_index and index columns
- updates meta/info.json totals/splits/chunks_size
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(path: Path) -> list[dict]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def write_jsonlines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(rows)


def make_scalar_stats(value: int, count: int) -> dict:
    return {
        "min": [int(value)],
        "max": [int(value)],
        "mean": [float(value)],
        "std": [0.0],
        "count": [int(count)],
    }


def make_range_stats(start: int, count: int) -> dict:
    if count <= 0:
        return {
            "min": [int(start)],
            "max": [int(start)],
            "mean": [float(start)],
            "std": [0.0],
            "count": [int(count)],
        }
    end = start + count - 1
    mean = (start + end) / 2.0
    var = 0.0 if count <= 1 else (count**2 - 1) / 12.0
    return {
        "min": [int(start)],
        "max": [int(end)],
        "mean": [float(mean)],
        "std": [float(math.sqrt(var))],
        "count": [int(count)],
    }


def update_episode_row(row: dict, new_episode_index: int) -> dict:
    updated = copy.deepcopy(row)
    updated["episode_index"] = int(new_episode_index)
    return updated


def update_episode_stats_row(row: dict, new_episode_index: int, index_start: int, length: int) -> dict:
    updated = copy.deepcopy(row)
    updated["episode_index"] = int(new_episode_index)
    stats = updated.get("stats", {})
    if "episode_index" in stats:
        stats["episode_index"] = make_scalar_stats(new_episode_index, length)
    if "index" in stats:
        stats["index"] = make_range_stats(index_start, length)
    updated["stats"] = stats
    return updated


def rewrite_parquet(
    src_path: Path,
    dst_path: Path,
    new_episode_index: int,
    index_start: int,
) -> int:
    table = pq.read_table(src_path)
    num_rows = table.num_rows
    columns = table.column_names
    if "episode_index" in columns:
        ep_arr = pa.array([new_episode_index] * num_rows, type=pa.int64())
        table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", ep_arr)
    if "index" in columns:
        idx_arr = pa.array(np.arange(index_start, index_start + num_rows, dtype=np.int64))
        table = table.set_column(table.schema.get_field_index("index"), "index", idx_arr)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)
    return num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a LeRobot dataset by episode count.")
    parser.add_argument("--root", type=Path, required=True, help="Input dataset root.")
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Directory to write splits (subfolders will be created).",
    )
    parser.add_argument("--chunk-size", type=int, default=100, help="Episodes per split.")
    parser.add_argument(
        "--name-prefix",
        type=str,
        default=None,
        help="Optional prefix for output dataset names (default: input dir name).",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    meta_dir = root / "meta"
    info = load_json(meta_dir / "info.json")

    episodes = load_jsonlines(meta_dir / "episodes.jsonl")
    episodes = sorted(episodes, key=lambda x: x["episode_index"])
    episodes_by_index = {row["episode_index"]: row for row in episodes}

    episodes_stats = load_jsonlines(meta_dir / "episodes_stats.jsonl")
    stats_by_index = {row["episode_index"]: row for row in episodes_stats}

    episode_indices = [row["episode_index"] for row in episodes]
    chunk_size = args.chunk_size
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    base_name = args.name_prefix or root.name
    total_splits = math.ceil(len(episode_indices) / chunk_size)

    for split_idx in range(total_splits):
        start = split_idx * chunk_size
        group = episode_indices[start : start + chunk_size]
        if not group:
            continue

        out_dir = args.out_root / f"{base_name}_part-{split_idx:03d}"
        out_meta = out_dir / "meta"
        out_meta.mkdir(parents=True, exist_ok=True)

        # Copy tasks and norm stats as-is.
        tasks_src = meta_dir / "tasks.jsonl"
        if tasks_src.exists():
            shutil.copy2(tasks_src, out_meta / "tasks.jsonl")
        norm_stats_src = root / "norm_stats.json"
        if norm_stats_src.exists():
            shutil.copy2(norm_stats_src, out_dir / "norm_stats.json")

        new_episodes: list[dict] = []
        new_stats: list[dict] = []
        index_cursor = 0
        total_frames = 0

        for new_ep_idx, old_ep_idx in enumerate(group):
            ep_row = episodes_by_index[old_ep_idx]
            length = int(ep_row.get("length", 0))
            total_frames += length

            new_episodes.append(update_episode_row(ep_row, new_ep_idx))
            if old_ep_idx in stats_by_index:
                new_stats.append(update_episode_stats_row(stats_by_index[old_ep_idx], new_ep_idx, index_cursor, length))

            src_path = root / info["data_path"].format(
                episode_chunk=old_ep_idx // info["chunks_size"],
                episode_index=old_ep_idx,
            )
            dst_path = out_dir / info["data_path"].format(
                episode_chunk=new_ep_idx // chunk_size,
                episode_index=new_ep_idx,
            )
            rewrite_parquet(src_path, dst_path, new_ep_idx, index_cursor)
            index_cursor += length

        write_jsonlines(out_meta / "episodes.jsonl", new_episodes)
        if new_stats:
            new_stats = sorted(new_stats, key=lambda x: x["episode_index"])
            write_jsonlines(out_meta / "episodes_stats.jsonl", new_stats)

        new_info = copy.deepcopy(info)
        new_info["total_episodes"] = len(group)
        new_info["total_frames"] = total_frames
        new_info["chunks_size"] = chunk_size
        new_info["total_chunks"] = math.ceil(len(group) / chunk_size)
        new_info["splits"] = {"train": f"0:{len(group)}"}
        write_json(out_meta / "info.json", new_info)

        print(
            f"[{split_idx+1}/{total_splits}] Wrote {len(group)} episodes, "
            f"{total_frames} frames -> {out_dir}"
        )


if __name__ == "__main__":
    main()
