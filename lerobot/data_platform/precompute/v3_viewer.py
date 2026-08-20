"""Prepare the existing HTML viewer cache from a LeRobot v3.0 dataset."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from lerobot.data_platform.precompute.preprocess.dataset_version import (
    V30,
    detect_dataset_version,
    validate_v3_dataset,
)
from lerobot.data_platform.precompute.timeseries import normalize_gripper_columns
from lerobot.data_platform.precompute.video import temporary_output_path

ProgressCallback = Callable[[dict], None] | None


@dataclass
class V3ViewerResult:
    root: Path
    repo_id: str
    output_dir: Path
    episodes: list[int]


def _emit(progress_callback: ProgressCallback, **payload) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _has_nonempty_cache(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _load_episode_metadata(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing v3.0 episode metadata under {root / 'meta' / 'episodes'}")
    frames = []
    for path in paths:
        columns = [name for name in pq.read_schema(path).names if not name.startswith("stats/")]
        frames.append(pd.read_parquet(path, columns=columns))
    episodes = pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
    required = {
        "episode_index",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"v3.0 episode metadata is missing columns: {missing}")
    return episodes


def _format_data_path(info: dict, episode: pd.Series) -> Path:
    episode_index = int(episode["episode_index"])
    return Path(
        info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
            episode_index=episode_index,
            episode_chunk=episode_index // int(info.get("chunks_size") or 1000),
        )
    )


def _format_video_path(info: dict, episode: pd.Series, video_key: str) -> Path:
    episode_index = int(episode["episode_index"])
    return Path(
        info["video_path"].format(
            video_key=video_key,
            chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
            file_index=int(episode[f"videos/{video_key}/file_index"]),
            episode_index=episode_index,
            episode_chunk=episode_index // int(info.get("chunks_size") or 1000),
        )
    )


def _feature_dim(feature: dict) -> int:
    shape = feature.get("shape") or []
    if isinstance(shape, int):
        return int(shape)
    if len(shape) == 1:
        return int(shape[0])
    return 0


def _feature_names(key: str, feature: dict, dim: int) -> list[str]:
    names = feature.get("names")
    while isinstance(names, dict) and names:
        names = next(iter(names.values()))
    if isinstance(names, (list, tuple)) and len(names) == dim:
        return [str(name) for name in names]
    if key == "exist_label" and dim == 1:
        return ["exist_label"]
    return [f"{key}_{index}" for index in range(dim)]


def _cell_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _write_episode_csv(
    root: Path,
    info: dict,
    episode: pd.Series,
    out_path: Path,
    downsample: int | None,
    data_version: str,
) -> None:
    features = info.get("features") or {}
    plot_keys = []
    for key, feature in features.items():
        dtype = str(feature.get("dtype") or "")
        numeric = dtype in {"float32", "int32"} or (
            key == "exist_label" and dtype.startswith(("float", "int", "uint"))
        )
        if numeric and key not in {"timestamp", "subtask_state"} and _feature_dim(feature) > 0:
            plot_keys.append(key)

    data_path = root / _format_data_path(info, episode)
    available = set(pq.read_schema(data_path).names)
    plot_keys = [key for key in plot_keys if key in available]
    columns = ["timestamp", *plot_keys]
    table = pq.read_table(
        data_path,
        columns=columns,
        filters=[("episode_index", "=", int(episode["episode_index"]))],
    )
    data = table.to_pandas()
    if downsample is not None and downsample > 1:
        data = data.iloc[::downsample].reset_index(drop=True)

    header = ["timestamp"]
    matrices = [np.asarray(data["timestamp"], dtype=np.float64).reshape(-1, 1)]
    for key in plot_keys:
        feature = features[key]
        fallback_dim = _feature_dim(feature)
        values = [_cell_values(value) for value in data[key]]
        actual_dim = max((len(value) for value in values), default=0)
        dim = actual_dim or fallback_dim
        rows = []
        for value in values:
            row = [np.nan] * dim
            row[: min(dim, len(value))] = value[:dim]
            rows.append(row)
        matrix = normalize_gripper_columns(np.asarray(rows, dtype=np.float64), key, data_version)
        matrices.append(matrix)
        header.extend(_feature_names(key, feature, dim))

    temporary = temporary_output_path(out_path)
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(np.hstack(matrices).tolist())
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)


def _clip_video(source: Path, target: Path, start: float, end: float) -> None:
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid v3.0 video interval {start}..{end}: {source}")
    temporary = temporary_output_path(target)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.9f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.9f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg did not create a valid viewer video: {temporary}")
        temporary.replace(target)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to prepare v3.0 viewer videos") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _decode_image(value: Any, root: Path) -> np.ndarray:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
    else:
        image_bytes = None
        image_path = value
    if image_bytes is not None:
        image = Image.open(io.BytesIO(bytes(image_bytes)))
    elif image_path:
        path = Path(str(image_path))
        image = Image.open(path if path.is_absolute() else root / path)
    else:
        raise ValueError("Image feature has neither bytes nor path")
    return np.asarray(image.convert("RGB"))


def _encode_image_feature(
    root: Path,
    info: dict,
    episode: pd.Series,
    image_key: str,
    target: Path,
) -> None:
    data_path = root / _format_data_path(info, episode)
    table = pq.read_table(
        data_path,
        columns=[image_key],
        filters=[("episode_index", "=", int(episode["episode_index"]))],
    )
    values = table.column(image_key).to_pylist()
    if not values:
        raise ValueError(f"No frames found for {image_key}, episode {int(episode['episode_index'])}")
    temporary = temporary_output_path(target)
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=int(info["fps"]))
            first = _decode_image(values[0], root)
            stream.width = int(first.shape[1])
            stream.height = int(first.shape[0])
            stream.pix_fmt = "yuv420p"
            for value in values:
                frame = av.VideoFrame.from_ndarray(_decode_image(value, root), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"PyAV did not create a valid viewer video: {temporary}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _task_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value]
    return [str(value)]


def _write_manifest(
    root: Path,
    repo_id: str,
    info: dict,
    episodes: pd.DataFrame,
    visual_keys: list[str],
    static_dir: Path,
    data_version: str,
    downsample: int | None,
) -> None:
    episode_rows = []
    total_frames = 0
    for _, episode in episodes.iterrows():
        length = int(
            episode.get("length") or (int(episode["dataset_to_index"]) - int(episode["dataset_from_index"]))
        )
        total_frames += length
        episode_rows.append(
            {
                "episode_index": int(episode["episode_index"]),
                "length": length,
                "tasks": _task_list(episode.get("tasks")),
            }
        )
    manifest = {
        "version": 1,
        "codebase_version": V30,
        "repo_id": repo_id,
        "root": str(root),
        "data_version": data_version,
        "fps": int(info["fps"]),
        "total_episodes": len(episode_rows),
        "total_frames": total_frames,
        "episodes": episode_rows,
        "features": info.get("features") or {},
        "image_keys": visual_keys,
        "video_keys": visual_keys,
        "downsample": int(downsample) if downsample and downsample > 1 else 1,
    }
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "viewer_manifest.json").write_text(json.dumps(manifest, indent=2))


def run_v3_viewer_precompute(
    root: Path,
    repo_id: str,
    output_dir: Path,
    episodes: list[int] | None = None,
    prepare_videos: bool = True,
    prepare_csv: bool = True,
    workers: int = 8,
    downsample: int | None = None,
    overwrite_videos: bool = False,
    overwrite_csv: bool = False,
    data_version: str = "DVT1",
    progress_callback: ProgressCallback = None,
) -> V3ViewerResult:
    """Build per-episode viewer artifacts without changing the v3.0 dataset."""
    root = Path(root).expanduser()
    output_dir = Path(output_dir).expanduser()
    if detect_dataset_version(root) != V30:
        raise ValueError(f"Expected a v3.0 dataset: {root}")
    validate_v3_dataset(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    episode_table = _load_episode_metadata(root)
    available_ids = {int(value) for value in episode_table["episode_index"]}
    selected_ids = sorted(available_ids if episodes is None else {int(value) for value in episodes})
    missing = sorted(set(selected_ids) - available_ids)
    if missing:
        raise ValueError(f"Unknown v3.0 episode indices: {missing}")
    selected = episode_table[episode_table["episode_index"].isin(selected_ids)].copy()

    features = info.get("features") or {}
    video_keys = sorted(key for key, feature in features.items() if feature.get("dtype") == "video")
    image_keys = sorted(key for key, feature in features.items() if feature.get("dtype") == "image")
    visual_keys = [*video_keys, *image_keys]
    static_dir = output_dir / "static"
    csv_dir = static_dir / "csv"
    videos_dir = static_dir / "videos"
    workers = max(1, int(workers or 1))
    completed = 0
    completed_lock = threading.Lock()

    def _prepare_episode(episode: pd.Series) -> None:
        nonlocal completed
        episode_index = int(episode["episode_index"])
        if prepare_csv:
            csv_path = csv_dir / f"episode_{episode_index:06d}_ds{downsample or 1}.csv"
            if overwrite_csv or not _has_nonempty_cache(csv_path):
                _write_episode_csv(root, info, episode, csv_path, downsample, data_version)
        if prepare_videos:
            for video_key in video_keys:
                target = videos_dir / video_key / f"episode_{episode_index:06d}_h264.mp4"
                if overwrite_videos or not _has_nonempty_cache(target):
                    source = root / _format_video_path(info, episode, video_key)
                    _clip_video(
                        source,
                        target,
                        float(episode[f"videos/{video_key}/from_timestamp"]),
                        float(episode[f"videos/{video_key}/to_timestamp"]),
                    )
            for image_key in image_keys:
                target = videos_dir / image_key / f"episode_{episode_index:06d}_h264.mp4"
                if overwrite_videos or not _has_nonempty_cache(target):
                    _encode_image_feature(root, info, episode, image_key, target)
        with completed_lock:
            completed += 1
            _emit(
                progress_callback,
                status="running",
                current=completed,
                total=len(selected_ids),
                message=f"Prepared v3.0 viewer cache for episode {episode_index}",
            )

    _emit(
        progress_callback,
        status="running",
        current=0,
        total=len(selected_ids),
        message=f"Preparing read-only v3.0 viewer cache for {len(selected_ids)} episodes",
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_prepare_episode, episode) for _, episode in selected.iterrows()]
        for future in as_completed(futures):
            future.result()

    _write_manifest(
        root,
        repo_id,
        info,
        selected,
        visual_keys,
        static_dir,
        data_version,
        downsample,
    )
    _emit(
        progress_callback,
        status="done",
        current=len(selected_ids),
        total=len(selected_ids),
        message="v3.0 viewer cache ready",
    )
    return V3ViewerResult(root=root, repo_id=repo_id, output_dir=output_dir, episodes=selected_ids)


__all__ = ["V3ViewerResult", "run_v3_viewer_precompute"]
