from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.dataset_io import read_episode_table


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metadata_stats(root: Path, meta) -> dict:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = getattr(meta, "tasks", {}) or _read_jsonl(root / "meta" / "tasks.jsonl")
    episodes = getattr(meta, "episodes", {}) or _read_jsonl(root / "meta" / "episodes.jsonl")
    image_keys = [
        key for key, ft in info.get("features", {}).items() if ft.get("dtype") in {"image", "video"}
    ]
    action_feature = info.get("features", {}).get("action", {})
    return {
        "root": str(root),
        "total_episodes": int(info.get("total_episodes") or len(episodes)),
        "total_frames": int(info.get("total_frames") or 0),
        "fps": info.get("fps"),
        "total_tasks": int(info.get("total_tasks") or len(tasks)),
        "image_keys": image_keys,
        "action_shape": action_feature.get("shape"),
    }


def action_stats(root: Path, meta, max_episodes: int = 64) -> dict:
    values = []
    for episode_index in sorted(getattr(meta, "episodes", {}).keys())[:max_episodes]:
        path = Path(root) / meta.get_data_file_path(episode_index)
        table = (
            read_episode_table(root, meta, episode_index, columns=["action"])
            if "action" in pq.read_schema(path).names
            else None
        )
        if table is None:
            continue
        for value in table["action"].to_pylist():
            if isinstance(value, (list, tuple)):
                values.append([float(v) for v in value])
            elif value is not None:
                values.append([float(value)])
    if not values:
        return {"available": False}
    width = max(len(row) for row in values)
    arr = np.asarray([row + [0.0] * (width - len(row)) for row in values], dtype=np.float32)
    return {
        "available": True,
        "dims": int(arr.shape[1]),
        "mean": np.mean(arr, axis=0).round(6).tolist(),
        "std": np.std(arr, axis=0).round(6).tolist(),
        "min": np.min(arr, axis=0).round(6).tolist(),
        "max": np.max(arr, axis=0).round(6).tolist(),
    }
