"""Shared local dataset access for LeRobot v2.1 and v3.0 layouts."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import av
import numpy as np
import packaging.version
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from lerobot.common.datasets.video_utils import decode_video_frames
from lerobot.data_platform.precompute.image_io import read_image_bytes


def is_v3_info(info: dict) -> bool:
    value = str(info.get("codebase_version") or "").lower().removeprefix("v")
    return value == "3" or value.startswith("3.")


def is_v3_dataset(root: Path) -> bool:
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.is_file():
        return False
    return is_v3_info(json.loads(info_path.read_text()))


def is_v3_metadata(meta: Any) -> bool:
    return bool(getattr(meta, "is_v3", False)) or is_v3_info(getattr(meta, "info", {}) or {})


def _python_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_python_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_python_value(item) for item in value]
    if isinstance(value, list):
        return [_python_value(item) for item in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _set_nested(target: dict, path: list[str], value: Any) -> None:
    current = target
    for name in path[:-1]:
        current = current.setdefault(name, {})
    current[path[-1]] = _python_value(value)


def _load_v3_tasks(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    frame = pd.read_parquet(Path(root) / "meta" / "tasks.parquet")
    tasks: dict[int, str] = {}
    task_values = frame["task"].tolist() if "task" in frame.columns else frame.index.tolist()
    task_indices = frame["task_index"].tolist() if "task_index" in frame.columns else list(range(len(frame)))
    for task_index, task in zip(task_indices, task_values, strict=True):
        tasks[int(task_index)] = str(task)
    return tasks, {task: task_index for task_index, task in tasks.items()}


def _load_v3_episode_frame(root: Path) -> pd.DataFrame:
    paths = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing v3.0 episode metadata under {Path(root) / 'meta' / 'episodes'}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True).sort_values(
        "episode_index"
    )


class V3DatasetMetadata:
    """Metadata surface compatible with the Data Platform's v2.1 callers."""

    is_v3 = True

    def __init__(self, repo_id: str, root: Path):
        self.repo_id = repo_id
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.revision = str(self.info.get("codebase_version") or "v3.0")
        self.tasks, self.task_to_task_index = _load_v3_tasks(self.root)
        episode_frame = _load_v3_episode_frame(self.root)
        self.episodes: dict[int, dict] = {}
        self.episodes_stats: dict[int, dict] = {}
        for record in episode_frame.to_dict(orient="records"):
            episode_index = int(record["episode_index"])
            episode = {}
            stats = {}
            for key, value in record.items():
                if str(key).startswith("stats/"):
                    _set_nested(stats, str(key).split("/")[1:], value)
                else:
                    episode[str(key)] = _python_value(value)
            episode["episode_index"] = episode_index
            episode["tasks"] = [str(task) for task in episode.get("tasks") or []]
            self.episodes[episode_index] = episode
            self.episodes_stats[episode_index] = stats
        stats_path = self.root / "meta" / "stats.json"
        self.stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}

    @property
    def _version(self) -> packaging.version.Version:
        return packaging.version.parse(str(self.info["codebase_version"]))

    @property
    def data_path(self) -> str:
        return str(self.info["data_path"])

    @property
    def video_path(self) -> str | None:
        value = self.info.get("video_path")
        return str(value) if value else None

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict[str, dict]:
        return self.info.get("features") or {}

    @property
    def image_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") in {"image", "video"}]

    @property
    def names(self) -> dict[str, list | dict | None]:
        return {key: feature.get("names") for key, feature in self.features.items()}

    @property
    def shapes(self) -> dict[str, tuple]:
        return {key: tuple(feature.get("shape") or ()) for key, feature in self.features.items()}

    @property
    def total_episodes(self) -> int:
        return int(self.info.get("total_episodes") or len(self.episodes))

    @property
    def total_frames(self) -> int:
        return int(self.info.get("total_frames") or 0)

    @property
    def total_tasks(self) -> int:
        return int(self.info.get("total_tasks") or len(self.tasks))

    @property
    def chunks_size(self) -> int:
        return int(self.info.get("chunks_size") or 1000)

    @property
    def total_chunks(self) -> int:
        return math.ceil(self.total_episodes / self.chunks_size) if self.total_episodes else 0

    def get_episode_chunk(self, episode_index: int) -> int:
        return int(episode_index) // self.chunks_size

    def get_data_file_path(self, episode_index: int) -> Path:
        episode = self.episodes[int(episode_index)]
        return Path(
            self.data_path.format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
                episode_index=int(episode_index),
                episode_chunk=self.get_episode_chunk(episode_index),
            )
        )

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        if not self.video_path:
            raise FileNotFoundError("Dataset does not define video_path")
        episode = self.episodes[int(episode_index)]
        return Path(
            self.video_path.format(
                video_key=video_key,
                chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
                file_index=int(episode[f"videos/{video_key}/file_index"]),
                episode_index=int(episode_index),
                episode_chunk=self.get_episode_chunk(episode_index),
            )
        )

    def get_video_time_range(self, episode_index: int, video_key: str) -> tuple[float, float]:
        episode = self.episodes[int(episode_index)]
        return (
            float(episode[f"videos/{video_key}/from_timestamp"]),
            float(episode[f"videos/{video_key}/to_timestamp"]),
        )


def read_episode_table(
    root: Path,
    meta: Any,
    episode_index: int,
    columns: list[str] | None = None,
) -> pa.Table:
    path = Path(root) / meta.get_data_file_path(int(episode_index))
    if is_v3_metadata(meta):
        return pq.read_table(
            path,
            columns=columns,
            filters=[("episode_index", "=", int(episode_index))],
        )
    return pq.read_table(path, columns=columns)


def load_episode_records(root: Path) -> list[dict]:
    root = Path(root)
    if is_v3_dataset(root):
        meta = V3DatasetMetadata(f"local/{root.name}", root)
        return [dict(meta.episodes[index]) for index in sorted(meta.episodes)]
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_task_records(root: Path) -> list[dict]:
    root = Path(root)
    if is_v3_dataset(root):
        tasks, _ = _load_v3_tasks(root)
        return [{"task_index": int(task_index), "task": task} for task_index, task in sorted(tasks.items())]
    path = root / "meta" / "tasks.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_task_records(root: Path, records: list[dict]) -> None:
    root = Path(root)
    records = sorted(
        ({"task_index": int(record["task_index"]), "task": str(record["task"])} for record in records),
        key=lambda record: record["task_index"],
    )
    if not is_v3_dataset(root):
        path = root / "meta" / "tasks.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    path = root / "meta" / "tasks.parquet"
    original = pd.read_parquet(path)
    if "task" in original.columns:
        frame = pd.DataFrame.from_records(records)
    else:
        frame = pd.DataFrame(
            {"task_index": [record["task_index"] for record in records]},
            index=pd.Index(
                [record["task"] for record in records],
                name=original.index.name,
            ),
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def write_episode_records(root: Path, records: list[dict]) -> None:
    root = Path(root)
    if not is_v3_dataset(root):
        path = root / "meta" / "episodes.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda item: int(item["episode_index"])):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    update_episode_metadata(
        root,
        {
            int(record["episode_index"]): {
                str(key): value for key, value in record.items() if key != "episode_index"
            }
            for record in records
        },
    )


def update_episode_metadata(root: Path, updates: dict[int, dict[str, Any]]) -> list[int]:
    """Update episode-level metadata in either JSONL or chunked v3 Parquet."""
    root = Path(root)
    normalized = {
        int(episode_index): {str(key): value for key, value in fields.items()}
        for episode_index, fields in updates.items()
    }
    if not normalized:
        return []

    if not is_v3_dataset(root):
        path = root / "meta" / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        merged = []
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                episode_index = int(row["episode_index"])
                if episode_index in normalized:
                    row.update(normalized[episode_index])
                    merged.append(episode_index)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return merged

    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    all_columns = sorted({key for fields in normalized.values() for key in fields})
    merged = []
    for path in paths:
        frame = pd.read_parquet(path)
        for column in all_columns:
            if column not in frame.columns:
                frame[column] = pd.Series([None] * len(frame), dtype=object)
        for row_index, episode_index in frame["episode_index"].items():
            episode_index = int(episode_index)
            fields = normalized.get(episode_index)
            if fields is None:
                continue
            for column, value in fields.items():
                frame.at[row_index, column] = value
            merged.append(episode_index)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    return sorted(set(merged))


def replace_episode_column(
    root: Path,
    meta: Any,
    episode_index: int,
    column_name: str,
    values: list[Any],
) -> int:
    """Replace one episode's column values without touching sibling episodes in a v3 shard."""
    root = Path(root)
    episode_index = int(episode_index)
    path = root / meta.get_data_file_path(episode_index)
    table = pq.read_table(path)
    if column_name not in table.column_names:
        raise ValueError(f"Missing {column_name} column in {path}")

    if is_v3_metadata(meta):
        episode_values = table["episode_index"].to_pylist()
        positions = [position for position, value in enumerate(episode_values) if int(value) == episode_index]
    else:
        positions = list(range(table.num_rows))
    if len(values) != len(positions):
        raise ValueError(
            f"Episode {episode_index} has {len(positions)} rows in {path}, "
            f"but {len(values)} replacement values were provided"
        )

    field = table.schema.field(column_name)
    rewritten = table[column_name].to_pylist()
    for position, value in zip(positions, values, strict=True):
        rewritten[position] = value
    column_index = table.column_names.index(column_name)
    table = table.set_column(
        column_index,
        field,
        pa.array(rewritten, type=field.type),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    return len(positions)


def upsert_episode_column(
    root: Path,
    meta: Any,
    episode_index: int,
    column_name: str,
    values: list[Any],
    value_type: pa.DataType,
) -> int:
    """Insert or replace one episode's column while preserving sibling shard rows."""
    root = Path(root)
    episode_index = int(episode_index)
    path = root / meta.get_data_file_path(episode_index)
    table = pq.read_table(path)
    if is_v3_metadata(meta):
        positions = [
            position
            for position, value in enumerate(table["episode_index"].to_pylist())
            if int(value) == episode_index
        ]
    else:
        positions = list(range(table.num_rows))
    if len(values) != len(positions):
        raise ValueError(
            f"Episode {episode_index} has {len(positions)} rows in {path}, "
            f"but {len(values)} replacement values were provided"
        )

    if column_name in table.column_names:
        rewritten = table[column_name].to_pylist()
        column_index = table.column_names.index(column_name)
        field = table.schema.field(column_name)
        output_type = field.type
    else:
        rewritten = [None] * table.num_rows
        column_index = -1
        output_type = value_type
    for position, value in zip(positions, values, strict=True):
        rewritten[position] = value
    column = pa.array(rewritten, type=output_type)
    if column_index >= 0:
        table = table.set_column(column_index, table.schema.field(column_name), column)
    else:
        table = table.append_column(pa.field(column_name, output_type), column)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    return len(positions)


def _decode_image_value(value: Any, root: Path) -> Image.Image:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            return Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
        if image_path:
            path = Path(str(image_path))
            return Image.open(path if path.is_absolute() else Path(root) / path).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value)
        return Image.open(path if path.is_absolute() else Path(root) / path).convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value.astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value).__name__}")


def read_episode_frame_image(
    root: Path,
    meta: Any,
    episode_index: int,
    frame_index: int,
    image_key: str | None = None,
) -> tuple[Image.Image, str]:
    camera_keys = list(getattr(meta, "camera_keys", []) or [])
    if not camera_keys:
        features = getattr(meta, "features", {}) or {}
        camera_keys = [key for key, feature in features.items() if feature.get("dtype") in {"image", "video"}]
    if image_key is None:
        if not camera_keys:
            raise ValueError("Dataset metadata does not define an image or video feature.")
        image_key = camera_keys[0]
    if image_key not in camera_keys:
        raise ValueError(f"Unknown camera feature: {image_key}")

    feature = (getattr(meta, "features", {}) or {}).get(image_key) or {}
    if feature.get("dtype") == "video":
        video_path = Path(root) / meta.get_video_file_path(int(episode_index), image_key)
        start, end = meta.get_video_time_range(int(episode_index), image_key)
        fps = float(getattr(meta, "fps", 0) or feature.get("video.fps") or 0)
        if fps <= 0:
            raise ValueError("Dataset FPS must be positive to decode video frames")
        timestamp = start + int(frame_index) / fps
        if timestamp >= end:
            raise IndexError(
                f"Frame {frame_index} is outside episode {episode_index} video range for {image_key}"
            )
        try:
            frame = decode_video_frames(
                video_path,
                [timestamp],
                tolerance_s=max(1e-4, 0.51 / fps),
            )[0]
            array = frame.mul(255).round().clamp(0, 255).to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
        except (AssertionError, IndexError, RuntimeError) as decode_error:
            target_index = int(round(start * fps)) + int(frame_index)
            array = None
            with av.open(str(video_path)) as container:
                for decoded_index, decoded_frame in enumerate(container.decode(video=0)):
                    if decoded_index == target_index:
                        array = decoded_frame.to_ndarray(format="rgb24")
                        break
            if array is None:
                raise IndexError(
                    f"Could not decode frame {frame_index} for episode {episode_index} from {video_path}"
                ) from decode_error
        return Image.fromarray(array, mode="RGB"), image_key

    if is_v3_metadata(meta):
        table = read_episode_table(root, meta, episode_index, columns=[image_key])
        if frame_index < 0 or frame_index >= table.num_rows:
            raise IndexError(f"Frame {frame_index} is outside episode {episode_index}")
        return _decode_image_value(table[image_key][int(frame_index)].as_py(), Path(root)), image_key

    parquet_path = Path(root) / meta.get_data_file_path(int(episode_index))
    image_bytes = read_image_bytes(parquet_path, Path(root), image_key, int(frame_index))
    if image_bytes is None:
        raise FileNotFoundError(
            f"Could not read frame {frame_index} image for episode {episode_index} from {parquet_path}"
        )
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), image_key


__all__ = [
    "V3DatasetMetadata",
    "is_v3_dataset",
    "is_v3_info",
    "is_v3_metadata",
    "load_episode_records",
    "load_task_records",
    "read_episode_frame_image",
    "read_episode_table",
    "replace_episode_column",
    "upsert_episode_column",
    "update_episode_metadata",
    "write_episode_records",
    "write_task_records",
]
