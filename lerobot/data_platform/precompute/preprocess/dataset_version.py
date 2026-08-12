"""Convert local LeRobot v2.1 datasets to the LeRobot v3.0 layout.

The target layout intentionally follows LeRobot 0.4.4. Conversion is
non-destructive: v2.1 inputs are written to a new sibling directory, while an
already-v3.0 input is reported unchanged because no conversion is needed.
"""

from __future__ import annotations

import copy
import io
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import packaging.version
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image as PILImage

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import cast_stats_to_numpy, serialize_dict
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    read_episode_frame_image,
    read_episode_table,
    update_episode_metadata,
)
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    emit,
    ensure_output_root,
    format_data_path,
    format_video_path,
    load_json,
    load_jsonl,
    validate_dataset_root,
    video_feature_keys,
    write_json,
    write_jsonl,
)

# SVT-AV1 logs directly to stderr and reads this once before its first encoder starts.
# Keep fatal/error messages while suppressing its verbose warning/info configuration dump.
os.environ["SVT_LOG"] = "1"

V21 = "v2.1"
V30 = "v3.0"
V3_BASELINE = "LeRobot 0.4.4"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
DEFAULT_V3_CONVERT_WORKERS = 8
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL = "lerobot_official"
IMAGE_VIDEO_MODE_RGB_LOSSLESS = "rgb_lossless"
IMAGE_VIDEO_MODES = frozenset(
    {
        IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL,
        IMAGE_VIDEO_MODE_RGB_LOSSLESS,
    }
)
LEROBOT_OFFICIAL_IMAGE_VIDEO_CODEC = "libsvtav1"
LEROBOT_OFFICIAL_IMAGE_VIDEO_PIX_FMT = "yuv420p"
LEROBOT_OFFICIAL_IMAGE_VIDEO_CRF = 30
LEROBOT_OFFICIAL_IMAGE_VIDEO_GOP = 2
LEROBOT_OFFICIAL_IMAGE_VIDEO_PRESET = 12
LOSSLESS_IMAGE_VIDEO_CODEC = "libx264rgb"
LOSSLESS_IMAGE_VIDEO_PIX_FMT = "gbrp"
LOSSLESS_IMAGE_VIDEO_PRESET = "slow"

_LEGACY_META_FILES = {
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "info.json",
    "stats.json",
    "tasks.jsonl",
}
_FORMAT_DIRS = {"data", "meta", "videos"}
_IGNORED_SIDECAR_DIRS = {".git", ".cache", "__pycache__"}
_V3_META_ENTRIES = {"episodes", "info.json", "stats.json", "tasks.parquet"}


def lossless_rgb_h264_options(
    fps: int,
    *,
    encoder_threads: int | None = None,
) -> dict[str, str]:
    """Return size-optimized x264 RGB options without changing decoded pixels."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    options = {
        "g": str(fps),
        "crf": "0",
        "preset": LOSSLESS_IMAGE_VIDEO_PRESET,
    }
    if encoder_threads is not None:
        options["threads"] = str(encoder_threads)
    return options


def lerobot_official_av1_options(
    *,
    encoder_threads: int | None = None,
) -> dict[str, str]:
    """Return the LeRobot 0.4.4 default RGB-camera encoder options."""
    options = {
        "g": str(LEROBOT_OFFICIAL_IMAGE_VIDEO_GOP),
        "crf": str(LEROBOT_OFFICIAL_IMAGE_VIDEO_CRF),
        "preset": str(LEROBOT_OFFICIAL_IMAGE_VIDEO_PRESET),
    }
    if encoder_threads is not None:
        options["svtav1-params"] = f"lp={encoder_threads}"
    return options


def normalize_image_video_mode(value: str | None) -> str:
    mode = str(value or IMAGE_VIDEO_MODE_RGB_LOSSLESS).strip().lower()
    if mode not in IMAGE_VIDEO_MODES:
        choices = ", ".join(sorted(IMAGE_VIDEO_MODES))
        raise ValueError(f"image_video_mode must be one of: {choices}")
    return mode


def default_v3_path(src_root: Path) -> Path:
    src_root = Path(src_root).expanduser()
    return src_root.parent / f"{src_root.name}_v3"


def detect_dataset_version(root: Path) -> str:
    """Return the normalized LeRobot dataset format version."""
    root = validate_dataset_root(root)
    info = load_json(root / "meta" / "info.json")
    raw_version = str(info.get("codebase_version") or "").strip()
    if not raw_version:
        raise ValueError(f"Missing codebase_version in {root / 'meta' / 'info.json'}")
    try:
        version = packaging.version.parse(raw_version)
    except packaging.version.InvalidVersion as exc:
        raise ValueError(f"Invalid LeRobot codebase_version: {raw_version!r}") from exc
    return f"v{version.major}.{version.minor}"


def _next_file_index(chunk_index: int, file_index: int) -> tuple[int, int]:
    if file_index == DEFAULT_CHUNK_SIZE - 1:
        return chunk_index + 1, 0
    return chunk_index, file_index + 1


def _parquet_size_in_mb(path: Path, excluded_columns: list[str] | None = None) -> float:
    if excluded_columns:
        columns = [name for name in pq.read_schema(path).names if name not in excluded_columns]
        return pq.read_table(path, columns=columns).nbytes / (1024**2)
    metadata = pq.read_metadata(path)
    total = 0
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            total += row_group.column(column_index).total_uncompressed_size
    return total / (1024**2)


def _video_size_in_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def _video_frame_timestamps_in_s(path: Path) -> list[float]:
    """Read display timestamps from MP4 packets without decoding image pixels."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        packet_timestamps = [
            float(packet.pts * packet.time_base)
            for packet in container.demux(stream)
            if packet.pts is not None
        ]
    if not packet_timestamps:
        raise ValueError(f"Video has no timestamped frames: {path}")
    return sorted(packet_timestamps)


def _timestamps_follow_fps_grid(
    timestamps: Iterable[float],
    fps: int,
    *,
    tolerance_s: float = 1e-4,
) -> bool:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return all(
        abs(timestamp - frame_index / fps) < tolerance_s for frame_index, timestamp in enumerate(timestamps)
    )


def _remux_video_to_fps_grid(
    path: Path,
    fps: int,
    *,
    expected_frames: int,
) -> bool:
    """Normalize packet PTS to frame_index/fps without re-encoding video frames."""
    timestamps = _video_frame_timestamps_in_s(path)
    if len(timestamps) != expected_frames:
        raise ValueError(f"{path} has {len(timestamps)} timestamped frames, expected {expected_frames}")
    if _timestamps_follow_fps_grid(timestamps, fps):
        return False

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to repair v3.0 video timestamps")

    temp_output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            prefix=f".{path.stem}-timestamp-repair-",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_output_path = Path(temp_file.name)

        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-bsf:v",
                f"setts=pts=N/({fps}*TB):dts=DTS-PTS+N/({fps}*TB)",
                "-movflags",
                "+faststart",
                str(temp_output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to repair video timestamps for {path}: {completed.stderr.strip()}"
            )

        repaired_timestamps = _video_frame_timestamps_in_s(temp_output_path)
        if len(repaired_timestamps) != expected_frames:
            raise ValueError(
                f"Repaired {path} has {len(repaired_timestamps)} timestamped frames, "
                f"expected {expected_frames}"
            )
        if not _timestamps_follow_fps_grid(repaired_timestamps, fps):
            raise ValueError(f"Repaired video timestamps do not follow the {fps} FPS grid: {path}")

        shutil.copymode(path, temp_output_path)
        temp_output_path.replace(path)
        temp_output_path = None
        return True
    finally:
        if temp_output_path is not None:
            temp_output_path.unlink(missing_ok=True)


def _episode_timestamp_ranges(
    video_path: Path,
    episode_lengths: Iterable[int],
    fps: int,
) -> list[tuple[float, float]]:
    lengths = [int(length) for length in episode_lengths]
    if fps <= 0 or any(length <= 0 for length in lengths):
        raise ValueError("fps and episode lengths must be positive")
    timestamps = _video_frame_timestamps_in_s(video_path)
    expected_frames = sum(lengths)
    if len(timestamps) != expected_frames:
        raise ValueError(f"{video_path} has {len(timestamps)} timestamped frames, expected {expected_frames}")

    ranges = []
    frame_offset = 0
    frame_duration = 1.0 / fps
    for length in lengths:
        episode_timestamps = timestamps[frame_offset : frame_offset + length]
        ranges.append(
            (
                episode_timestamps[0],
                episode_timestamps[-1] + frame_duration,
            )
        )
        frame_offset += length
    return ranges


def _image_feature_keys(info: dict) -> list[str]:
    return sorted(
        key for key, feature in (info.get("features") or {}).items() if feature.get("dtype") == "image"
    )


def _concatenate_video_files(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise ValueError("No input videos to concatenate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        shutil.copy2(input_paths[0], output_path)
        return

    concat_path: Path | None = None
    temp_output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ffconcat",
            dir=output_path.parent,
            delete=False,
        ) as concat_file:
            concat_file.write("ffconcat version 1.0\n")
            for input_path in input_paths:
                resolved = str(input_path.resolve())
                if "'" in resolved:
                    raise ValueError(f"Video path containing a quote is not supported: {input_path}")
                concat_file.write(f"file '{resolved}'\n")
            concat_path = Path(concat_file.name)

        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=output_path.parent, delete=False) as temp_file:
            temp_output_path = Path(temp_file.name)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to concatenate v3.0 video files")
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(temp_output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to concatenate {len(input_paths)} videos: {completed.stderr.strip()}"
            )
        temp_output_path.replace(output_path)
        temp_output_path = None
    finally:
        if concat_path is not None:
            concat_path.unlink(missing_ok=True)
        if temp_output_path is not None:
            temp_output_path.unlink(missing_ok=True)


def _validate_v21(root: Path, info: dict) -> tuple[list[dict], list[dict], list[dict]]:
    required = {
        "episodes": root / "meta" / "episodes.jsonl",
        "episode stats": root / "meta" / "episodes_stats.jsonl",
        "tasks": root / "meta" / "tasks.jsonl",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing v2.1 {label}: {path}")

    episodes = sorted(load_jsonl(required["episodes"]), key=lambda row: int(row["episode_index"]))
    tasks = sorted(load_jsonl(required["tasks"]), key=lambda row: int(row["task_index"]))
    episode_stats = sorted(
        load_jsonl(required["episode stats"]),
        key=lambda row: int(row["episode_index"]),
    )
    expected_indices = list(range(len(episodes)))
    episode_indices = [int(row["episode_index"]) for row in episodes]
    stats_indices = [int(row["episode_index"]) for row in episode_stats]
    if episode_indices != expected_indices:
        raise ValueError(f"v2.1 episode indices must be contiguous from 0: {episode_indices[:10]}")
    if stats_indices != expected_indices:
        raise ValueError("v2.1 episodes_stats.jsonl must contain one row for every episode")
    if len(episodes) != int(info.get("total_episodes") or 0):
        raise ValueError(
            f"info.json total_episodes={info.get('total_episodes')} does not match "
            f"episodes.jsonl rows={len(episodes)}"
        )

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        parquet_path = root / format_data_path(info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing v2.1 episode parquet: {parquet_path}")
        num_frames = pq.read_metadata(parquet_path).num_rows
        if num_frames != int(episode["length"]):
            raise ValueError(
                f"Episode {episode_index} length={episode['length']} but parquet has {num_frames} rows"
            )
        for video_key in video_feature_keys(info):
            video_path = root / format_video_path(info, episode_index, video_key)
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing v2.1 episode video: {video_path}")
    return episodes, tasks, episode_stats


def _read_v3_episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing v3.0 episode metadata under {root / 'meta' / 'episodes'}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def validate_v3_dataset(root: Path) -> dict[str, int]:
    """Validate the v3.0 files needed by the LeRobot 0.4.4 reader."""
    root = validate_dataset_root(root)
    info = load_json(root / "meta" / "info.json")
    if detect_dataset_version(root) != V30:
        raise ValueError(f"Expected a v3.0 dataset: {root}")
    tasks_path = root / DEFAULT_TASKS_PATH
    stats_path = root / "meta" / "stats.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing v3.0 tasks metadata: {tasks_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing v3.0 dataset stats: {stats_path}")

    episodes = _read_v3_episodes(root)
    required_columns = {
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "length",
        "tasks",
    }
    missing = sorted(required_columns - set(episodes.columns))
    if missing:
        raise ValueError(f"v3.0 episode metadata is missing columns: {missing}")

    episodes = episodes.sort_values("episode_index")
    episode_indices = [int(value) for value in episodes["episode_index"].tolist()]
    if episode_indices != list(range(len(episodes))):
        raise ValueError(f"v3.0 episode indices must be contiguous from 0: {episode_indices[:10]}")
    if len(episodes) != int(info.get("total_episodes") or 0):
        raise ValueError(
            f"info.json total_episodes={info.get('total_episodes')} does not match "
            f"v3.0 episode rows={len(episodes)}"
        )

    data_template = info.get("data_path") or DEFAULT_DATA_PATH
    for chunk_index, file_index in {
        (int(row["data/chunk_index"]), int(row["data/file_index"])) for _, row in episodes.iterrows()
    }:
        path = root / data_template.format(chunk_index=chunk_index, file_index=file_index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing v3.0 data file: {path}")

    video_template = info.get("video_path") or DEFAULT_VIDEO_PATH
    for video_key in video_feature_keys(info):
        chunk_column = f"videos/{video_key}/chunk_index"
        file_column = f"videos/{video_key}/file_index"
        if chunk_column not in episodes or file_column not in episodes:
            raise ValueError(f"v3.0 episode metadata is missing video references for {video_key}")
        for chunk_index, file_index in {
            (int(row[chunk_column]), int(row[file_column])) for _, row in episodes.iterrows()
        }:
            path = root / video_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing v3.0 video file: {path}")

    tasks = pd.read_parquet(tasks_path)
    if len(tasks) != int(info.get("total_tasks") or 0):
        raise ValueError(
            f"info.json total_tasks={info.get('total_tasks')} does not match tasks.parquet rows={len(tasks)}"
        )
    total_frames = sum(
        int(row["dataset_to_index"]) - int(row["dataset_from_index"]) for _, row in episodes.iterrows()
    )
    if total_frames != int(info.get("total_frames") or 0):
        raise ValueError(
            f"info.json total_frames={info.get('total_frames')} does not match "
            f"episode frame ranges={total_frames}"
        )
    return {
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
    }


def _write_tasks(tasks: list[dict], out_root: Path) -> None:
    task_indices = [int(row["task_index"]) for row in tasks]
    task_strings = [str(row["task"]) for row in tasks]
    if task_indices != list(range(len(tasks))):
        raise ValueError(f"v2.1 task indices must be contiguous from 0: {task_indices[:10]}")
    dataframe = pd.DataFrame({"task_index": task_indices}, index=task_strings)
    path = out_root / DEFAULT_TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path)


def _write_data_file(
    paths: list[Path],
    out_root: Path,
    chunk_index: int,
    file_index: int,
    image_keys: list[str],
) -> None:
    dataframes = []
    for source_path in paths:
        data_columns = [name for name in pq.read_schema(source_path).names if name not in image_keys]
        dataframes.append(pd.read_parquet(source_path, columns=data_columns))
    dataframe = pd.concat(dataframes, ignore_index=True)
    path = out_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path, index=False)


def _convert_data(
    root: Path,
    out_root: Path,
    info: dict,
    episodes: list[dict],
    data_file_size_in_mb: int,
    progress_callback: ProgressCallback,
) -> list[dict]:
    image_keys = _image_feature_keys(info)
    chunk_index = 0
    file_index = 0
    file_size = 0.0
    frame_offset = 0
    pending_paths: list[Path] = []
    converted: list[dict] = []

    for position, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        path = root / format_data_path(info, episode_index)
        episode_size = _parquet_size_in_mb(path, image_keys)
        if pending_paths and file_size + episode_size >= data_file_size_in_mb:
            _write_data_file(pending_paths, out_root, chunk_index, file_index, image_keys)
            chunk_index, file_index = _next_file_index(chunk_index, file_index)
            pending_paths = []
            file_size = 0.0

        length = int(episode["length"])
        converted.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": frame_offset,
                "dataset_to_index": frame_offset + length,
            }
        )
        pending_paths.append(path)
        file_size += episode_size
        frame_offset += length
        emit(
            progress_callback,
            status="running",
            step="data",
            current=position,
            total=len(episodes),
            message=f"Packing data episode {episode_index}",
        )

    if pending_paths:
        _write_data_file(pending_paths, out_root, chunk_index, file_index, image_keys)
    return converted


def _decode_parquet_image(value: Any, root: Path) -> PILImage.Image:
    if isinstance(value, PILImage.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            return PILImage.open(io.BytesIO(bytes(image_bytes))).convert("RGB")
        if image_path:
            path = Path(str(image_path))
            return PILImage.open(path if path.is_absolute() else root / path).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return PILImage.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value)
        return PILImage.open(path if path.is_absolute() else root / path).convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(array.max()) if array.size else 0.0
            array = np.clip(array * 255 if max_value <= 1 else array, 0, 255)
        return PILImage.fromarray(array.astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported image value stored in Parquet: {type(value).__name__}")


def _encode_image_episode_video(
    root: Path,
    parquet_path: Path,
    image_key: str,
    target: Path,
    fps: int,
    expected_frames: int,
    encoder_threads: int | None = None,
    image_video_mode: str = IMAGE_VIDEO_MODE_RGB_LOSSLESS,
) -> None:
    image_video_mode = normalize_image_video_mode(image_video_mode)
    values = pq.read_table(parquet_path, columns=[image_key]).column(image_key).to_pylist()
    if len(values) != expected_frames:
        raise ValueError(
            f"{image_key} has {len(values)} frames in {parquet_path}, expected {expected_frames}"
        )
    if not values:
        raise ValueError(f"{image_key} has no frames in {parquet_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if image_video_mode == IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL:
        codec = LEROBOT_OFFICIAL_IMAGE_VIDEO_CODEC
        pixel_format = LEROBOT_OFFICIAL_IMAGE_VIDEO_PIX_FMT
        options = lerobot_official_av1_options(encoder_threads=encoder_threads)
    else:
        codec = LOSSLESS_IMAGE_VIDEO_CODEC
        pixel_format = "rgb24"
        options = lossless_rgb_h264_options(
            fps,
            encoder_threads=encoder_threads,
        )
    try:
        with av.open(str(target), mode="w") as container:
            stream = container.add_stream(
                codec,
                rate=fps,
                options=options,
            )
            first_image = _decode_parquet_image(values[0], root)
            stream.width, stream.height = first_image.size
            stream.pix_fmt = pixel_format
            for value in values:
                image = _decode_parquet_image(value, root)
                if image.size != first_image.size:
                    raise ValueError(
                        f"{image_key} frame size changed from {first_image.size} to {image.size}"
                    )
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _encode_image_episodes(
    root: Path,
    info: dict,
    episodes: list[dict],
    image_key: str,
    temp_dir: Path,
    workers: int,
    completed: int,
    total_work: int,
    progress_callback: ProgressCallback,
    image_video_mode: str = IMAGE_VIDEO_MODE_RGB_LOSSLESS,
) -> list[Path]:
    jobs = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        jobs.append(
            (
                episode_index,
                root / format_data_path(info, episode_index),
                temp_dir / f"episode_{episode_index:06d}.mp4",
                int(episode["length"]),
            )
        )

    effective_workers = min(workers, len(jobs))
    encoder_threads = 1 if effective_workers > 1 else None

    def _encode(job: tuple[int, Path, Path, int]) -> int:
        episode_index, source_path, target_path, expected_frames = job
        _encode_image_episode_video(
            root,
            source_path,
            image_key,
            target_path,
            int(info["fps"]),
            expected_frames,
            encoder_threads=encoder_threads,
            image_video_mode=image_video_mode,
        )
        return episode_index

    if effective_workers <= 1:
        completed_episode_indices = map(_encode, jobs)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=effective_workers)
        futures = [executor.submit(_encode, job) for job in jobs]
        completed_episode_indices = (future.result() for future in as_completed(futures))

    try:
        for episode_index in completed_episode_indices:
            emit(
                progress_callback,
                status="running",
                step="videos",
                current=completed,
                total=total_work,
                message=(
                    f"Encoded {image_key} episode {episode_index} "
                    f"using {image_video_mode} "
                    f"with {effective_workers} worker(s)"
                ),
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    return [job[2] for job in jobs]


def _pack_episode_videos(
    episode_paths: Iterable[Path],
    episode_lengths: Iterable[int],
    out_root: Path,
    video_key: str,
    video_file_size_in_mb: int,
    fps: int,
    cleanup_inputs: bool = False,
) -> list[dict]:
    episode_paths = list(episode_paths)
    episode_lengths = [int(length) for length in episode_lengths]
    if len(episode_paths) != len(episode_lengths):
        raise ValueError("episode video paths and lengths must have the same size")

    chunk_index = 0
    file_index = 0
    file_size = 0.0
    pending_paths: list[Path] = []
    pending_lengths: list[int] = []
    pending_metadata_indices: list[int] = []
    metadata: list[dict] = []

    def _flush_pending() -> None:
        if not pending_paths:
            return
        output = out_root / DEFAULT_VIDEO_PATH.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        _concatenate_video_files(pending_paths, output)
        _remux_video_to_fps_grid(
            output,
            fps,
            expected_frames=sum(pending_lengths),
        )
        timestamp_ranges = _episode_timestamp_ranges(output, pending_lengths, fps)
        for metadata_index, (from_timestamp, to_timestamp) in zip(
            pending_metadata_indices,
            timestamp_ranges,
            strict=True,
        ):
            metadata[metadata_index].update(
                {
                    f"videos/{video_key}/from_timestamp": from_timestamp,
                    f"videos/{video_key}/to_timestamp": to_timestamp,
                }
            )
        if cleanup_inputs:
            for pending_path in pending_paths:
                pending_path.unlink(missing_ok=True)

    for path, episode_length in zip(episode_paths, episode_lengths, strict=True):
        episode_size = _video_size_in_mb(path)
        if pending_paths and file_size + episode_size >= video_file_size_in_mb:
            _flush_pending()
            chunk_index, file_index = _next_file_index(chunk_index, file_index)
            pending_paths = []
            pending_lengths = []
            pending_metadata_indices = []
            file_size = 0.0

        metadata.append(
            {
                f"videos/{video_key}/chunk_index": chunk_index,
                f"videos/{video_key}/file_index": file_index,
            }
        )
        pending_paths.append(path)
        pending_lengths.append(episode_length)
        pending_metadata_indices.append(len(metadata) - 1)
        file_size += episode_size

    _flush_pending()
    return metadata


def _convert_videos(
    root: Path,
    out_root: Path,
    info: dict,
    episodes: list[dict],
    video_file_size_in_mb: int,
    workers: int,
    progress_callback: ProgressCallback,
    image_video_mode: str = IMAGE_VIDEO_MODE_RGB_LOSSLESS,
) -> list[dict]:
    source_video_keys = sorted(video_feature_keys(info))
    image_keys = _image_feature_keys(info)
    visual_keys = [*source_video_keys, *image_keys]
    if not visual_keys:
        return [{} for _ in episodes]

    all_metadata = [{} for _ in episodes]
    episode_lengths = [int(episode["length"]) for episode in episodes]
    total_work = len(visual_keys) * len(episodes)
    completed = 0
    for video_key in visual_keys:
        if video_key in source_video_keys:
            episode_paths = [
                root / format_video_path(info, int(episode["episode_index"]), video_key)
                for episode in episodes
            ]
            key_metadata = _pack_episode_videos(
                episode_paths,
                episode_lengths,
                out_root,
                video_key,
                video_file_size_in_mb,
                int(info["fps"]),
            )
        else:
            with tempfile.TemporaryDirectory(prefix="lerobot-v3-images-") as temp_dir:
                episode_paths = _encode_image_episodes(
                    root,
                    info,
                    episodes,
                    video_key,
                    Path(temp_dir),
                    workers,
                    completed,
                    total_work,
                    progress_callback,
                    image_video_mode,
                )
                key_metadata = _pack_episode_videos(
                    episode_paths,
                    episode_lengths,
                    out_root,
                    video_key,
                    video_file_size_in_mb,
                    int(info["fps"]),
                    cleanup_inputs=True,
                )

        for position, episode_metadata in enumerate(key_metadata):
            all_metadata[position].update(episode_metadata)
            completed += 1
            emit(
                progress_callback,
                status="running",
                step="videos",
                current=completed,
                total=total_work,
                message=f"Packed {video_key} episode {int(episodes[position]['episode_index'])}",
            )
    return all_metadata


def _write_episode_metadata(
    out_root: Path,
    episodes: list[dict],
    episode_stats: list[dict],
    data_metadata: list[dict],
    video_metadata: list[dict],
) -> None:
    stats_by_index = {int(row["episode_index"]): row.get("stats") or {} for row in episode_stats}
    records = []
    for episode, data_row, video_row in zip(
        episodes,
        data_metadata,
        video_metadata,
        strict=True,
    ):
        episode_index = int(episode["episode_index"])
        stats = stats_by_index[episode_index]
        record = {
            **data_row,
            **video_row,
            **episode,
            **_flatten_dict({"stats": stats}),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        records.append(record)
    path = out_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(path, index=False)

    stats_values = [cast_stats_to_numpy(row.get("stats") or {}) for row in episode_stats]
    write_json(out_root / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_values)))


def _flatten_dict(value: dict[str, Any], parent: str = "") -> dict[str, Any]:
    flattened = {}
    for key, item in value.items():
        name = f"{parent}/{key}" if parent else key
        if isinstance(item, dict):
            flattened.update(_flatten_dict(item, name))
        else:
            flattened[name] = item
    return flattened


def _write_info(
    out_root: Path,
    info: dict,
    data_file_size_in_mb: int,
    video_file_size_in_mb: int,
    image_video_mode: str = IMAGE_VIDEO_MODE_RGB_LOSSLESS,
) -> None:
    image_video_mode = normalize_image_video_mode(image_video_mode)
    converted = copy.deepcopy(info)
    image_keys = set(_image_feature_keys(info))
    converted["codebase_version"] = V30
    converted.pop("total_chunks", None)
    converted.pop("total_videos", None)
    converted["data_files_size_in_mb"] = data_file_size_in_mb
    converted["video_files_size_in_mb"] = video_file_size_in_mb
    converted["data_path"] = DEFAULT_DATA_PATH
    converted["video_path"] = DEFAULT_VIDEO_PATH if video_feature_keys(info) or image_keys else None
    converted["fps"] = int(converted["fps"])
    converted["chunks_size"] = int(converted.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    for key, feature in (converted.get("features") or {}).items():
        if key in image_keys:
            feature["dtype"] = "video"
            feature.pop("fps", None)
            if image_video_mode == IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL:
                codec = "av1"
                pixel_format = LEROBOT_OFFICIAL_IMAGE_VIDEO_PIX_FMT
                encoder_metadata = {
                    "video.g": LEROBOT_OFFICIAL_IMAGE_VIDEO_GOP,
                    "video.crf": LEROBOT_OFFICIAL_IMAGE_VIDEO_CRF,
                    "video.preset": LEROBOT_OFFICIAL_IMAGE_VIDEO_PRESET,
                }
            else:
                codec = "h264"
                pixel_format = LOSSLESS_IMAGE_VIDEO_PIX_FMT
                encoder_metadata = {
                    "video.g": converted["fps"],
                    "video.crf": 0,
                    "video.preset": LOSSLESS_IMAGE_VIDEO_PRESET,
                }
            feature.update(
                {
                    "video.fps": converted["fps"],
                    "video.codec": codec,
                    "video.pix_fmt": pixel_format,
                    "video.is_depth_map": False,
                    "has_audio": False,
                    **encoder_metadata,
                }
            )
            continue
        if feature.get("dtype") != "video":
            feature["fps"] = converted["fps"]
    write_json(out_root / "meta" / "info.json", converted)


def _copy_preserved_sidecars(root: Path, out_root: Path) -> None:
    for child in root.iterdir():
        if child.name in _FORMAT_DIRS or child.name in _IGNORED_SIDECAR_DIRS:
            continue
        destination = out_root / child.name
        if child.is_dir():
            shutil.copytree(child, destination, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)

    source_meta = root / "meta"
    target_meta = out_root / "meta"
    for child in source_meta.iterdir():
        if child.name in _LEGACY_META_FILES:
            continue
        destination = target_meta / child.name
        if destination.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, destination, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)


def _v21_info_from_v3(info: dict) -> dict:
    converted = copy.deepcopy(info)
    total_episodes = int(converted.get("total_episodes") or 0)
    chunks_size = int(converted.get("chunks_size") or DEFAULT_CHUNK_SIZE)
    converted["codebase_version"] = V21
    converted["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    converted["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        if video_feature_keys(converted)
        else None
    )
    converted["chunks_size"] = chunks_size
    converted["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size if total_episodes else 0
    converted["total_videos"] = len(video_feature_keys(converted)) * total_episodes
    converted.pop("data_files_size_in_mb", None)
    converted.pop("video_files_size_in_mb", None)
    return converted


def _v21_episode_record(record: dict) -> dict:
    return {
        str(key): value
        for key, value in record.items()
        if key == "episode_index"
        or (
            not str(key).startswith(("data/", "videos/", "meta/episodes/"))
            and key not in {"dataset_from_index", "dataset_to_index"}
        )
    }


def _encode_v3_episode_video(
    root: Path,
    meta: V3DatasetMetadata,
    episode_index: int,
    video_key: str,
    target: Path,
    encoder_threads: int | None,
) -> None:
    length = int(meta.episodes[episode_index]["length"])
    if length <= 0:
        raise ValueError(f"Episode {episode_index} has no frames")
    first_image, _ = read_episode_frame_image(
        root,
        meta,
        episode_index,
        0,
        image_key=video_key,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with av.open(str(target), mode="w") as container:
            stream = container.add_stream(
                LOSSLESS_IMAGE_VIDEO_CODEC,
                rate=meta.fps,
                options=lossless_rgb_h264_options(
                    meta.fps,
                    encoder_threads=encoder_threads,
                ),
            )
            stream.width, stream.height = first_image.size
            stream.pix_fmt = "rgb24"
            for frame_index in range(length):
                image = (
                    first_image
                    if frame_index == 0
                    else read_episode_frame_image(
                        root,
                        meta,
                        episode_index,
                        frame_index,
                        image_key=video_key,
                    )[0]
                )
                if image.size != first_image.size:
                    raise ValueError(
                        f"{video_key} frame size changed from {first_image.size} to {image.size}"
                    )
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def materialize_v21_from_v3(
    src_root: Path,
    out_root: Path,
    *,
    include_data: bool = True,
    include_videos: bool = True,
    workers: int = DEFAULT_V3_CONVERT_WORKERS,
    progress_callback: ProgressCallback = None,
) -> Path:
    """Materialize a temporary v2.1 view for legacy dataset operations.

    Video frames are decoded at episode timestamps and re-encoded with the same
    lossless RGB H.264 settings as the v3 converter, so legacy operations do not
    introduce an additional lossy generation.
    """
    src_root = validate_dataset_root(src_root)
    if detect_dataset_version(src_root) != V30:
        raise ValueError(f"Expected a v3.0 dataset: {src_root}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    out_root = ensure_output_root(out_root)
    meta = V3DatasetMetadata(f"local/{src_root.name}", src_root)
    info = _v21_info_from_v3(meta.info)
    episodes = [_v21_episode_record(meta.episodes[index]) for index in sorted(meta.episodes)]
    tasks = [{"task_index": int(index), "task": task} for index, task in sorted(meta.tasks.items())]
    episode_stats = [
        {"episode_index": int(index), "stats": meta.episodes_stats.get(index, {})}
        for index in sorted(meta.episodes)
    ]

    try:
        write_json(out_root / "meta" / "info.json", info)
        write_jsonl(out_root / "meta" / "tasks.jsonl", tasks)
        write_jsonl(out_root / "meta" / "episodes.jsonl", episodes)
        write_jsonl(out_root / "meta" / "episodes_stats.jsonl", episode_stats)
        (out_root / "data").mkdir(parents=True, exist_ok=True)

        if include_data:
            for position, episode in enumerate(episodes, start=1):
                episode_index = int(episode["episode_index"])
                table = read_episode_table(src_root, meta, episode_index)
                target = out_root / format_data_path(info, episode_index)
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, target)
                emit(
                    progress_callback,
                    status="running",
                    step="materialize_v21_data",
                    current=position,
                    total=len(episodes),
                    message=f"Materialized v3 episode {episode_index} data",
                )

        if include_videos and include_data and meta.video_keys:
            jobs = [
                (
                    episode_index,
                    video_key,
                    out_root / format_video_path(info, episode_index, video_key),
                )
                for episode_index in sorted(meta.episodes)
                for video_key in meta.video_keys
            ]
            effective_workers = min(workers, len(jobs))
            encoder_threads = 1 if effective_workers > 1 else None

            def _encode(job: tuple[int, str, Path]) -> tuple[int, str]:
                episode_index, video_key, target = job
                _encode_v3_episode_video(
                    src_root,
                    meta,
                    episode_index,
                    video_key,
                    target,
                    encoder_threads,
                )
                return episode_index, video_key

            if effective_workers <= 1:
                completed_jobs = map(_encode, jobs)
                executor = None
            else:
                executor = ThreadPoolExecutor(max_workers=effective_workers)
                futures = [executor.submit(_encode, job) for job in jobs]
                completed_jobs = (future.result() for future in as_completed(futures))
            try:
                for position, (episode_index, video_key) in enumerate(
                    completed_jobs,
                    start=1,
                ):
                    emit(
                        progress_callback,
                        status="running",
                        step="materialize_v21_videos",
                        current=position,
                        total=len(jobs),
                        message=f"Materialized {video_key} episode {episode_index}",
                    )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

        for child in src_root.iterdir():
            if child.name in _FORMAT_DIRS or child.name in _IGNORED_SIDECAR_DIRS:
                continue
            destination = out_root / child.name
            if child.is_dir():
                shutil.copytree(child, destination, symlinks=True)
            else:
                shutil.copy2(child, destination, follow_symlinks=False)
        for child in (src_root / "meta").iterdir():
            if child.name in _V3_META_ENTRIES:
                continue
            destination = out_root / "meta" / child.name
            if destination.exists():
                continue
            if child.is_dir():
                shutil.copytree(child, destination, symlinks=True)
            else:
                shutil.copy2(child, destination, follow_symlinks=False)
    except Exception:
        if out_root.exists():
            shutil.rmtree(out_root, ignore_errors=True)
        raise
    return out_root


def repair_v3_video_timestamps(
    root: Path,
    *,
    dry_run: bool = False,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    """Normalize v3 MP4 PTS and align episode video ranges without re-encoding."""
    root = validate_dataset_root(root)
    if detect_dataset_version(root) != V30:
        raise ValueError(f"Expected a {V30} dataset: {root}")
    counts = validate_v3_dataset(root)
    meta = V3DatasetMetadata(f"local/{root.name}", root)
    updates: dict[int, dict[str, float]] = {}
    shard_count = 0
    shards_needing_remux = 0
    shards_remuxed = 0
    max_abs_shift_s = 0.0
    grouped_episodes_by_key: dict[str, dict[tuple[int, int], list[int]]] = {}
    for video_key in meta.video_keys:
        grouped_episodes: dict[tuple[int, int], list[int]] = {}
        for episode_index, episode in meta.episodes.items():
            shard = (
                int(episode[f"videos/{video_key}/chunk_index"]),
                int(episode[f"videos/{video_key}/file_index"]),
            )
            grouped_episodes.setdefault(shard, []).append(int(episode_index))
        grouped_episodes_by_key[video_key] = grouped_episodes

    total_shards = sum(len(groups) for groups in grouped_episodes_by_key.values())
    for video_key, grouped_episodes in grouped_episodes_by_key.items():
        for shard, episode_indices in sorted(grouped_episodes.items()):
            episode_indices.sort(
                key=lambda episode_index: (
                    float(meta.episodes[episode_index][f"videos/{video_key}/from_timestamp"]),
                    episode_index,
                )
            )
            video_path = root / meta.get_video_file_path(episode_indices[0], video_key)
            episode_lengths = [int(meta.episodes[index]["length"]) for index in episode_indices]
            expected_frames = sum(episode_lengths)
            current_timestamps = _video_frame_timestamps_in_s(video_path)
            if len(current_timestamps) != expected_frames:
                raise ValueError(
                    f"{video_path} has {len(current_timestamps)} timestamped frames, "
                    f"expected {expected_frames}"
                )
            needs_remux = not _timestamps_follow_fps_grid(current_timestamps, meta.fps)
            if needs_remux:
                shards_needing_remux += 1
                if not dry_run and _remux_video_to_fps_grid(
                    video_path,
                    meta.fps,
                    expected_frames=expected_frames,
                ):
                    shards_remuxed += 1

            if dry_run and needs_remux:
                frame_offset = 0
                timestamp_ranges = []
                for length in episode_lengths:
                    timestamp_ranges.append(
                        (
                            frame_offset / meta.fps,
                            (frame_offset + length) / meta.fps,
                        )
                    )
                    frame_offset += length
            else:
                timestamp_ranges = _episode_timestamp_ranges(
                    video_path,
                    episode_lengths,
                    meta.fps,
                )
            for episode_index, (from_timestamp, to_timestamp) in zip(
                episode_indices,
                timestamp_ranges,
                strict=True,
            ):
                episode = meta.episodes[episode_index]
                from_key = f"videos/{video_key}/from_timestamp"
                to_key = f"videos/{video_key}/to_timestamp"
                max_abs_shift_s = max(
                    max_abs_shift_s,
                    abs(float(episode[from_key]) - from_timestamp),
                    abs(float(episode[to_key]) - to_timestamp),
                )
                updates.setdefault(episode_index, {}).update(
                    {
                        from_key: from_timestamp,
                        to_key: to_timestamp,
                    }
                )
            shard_count += 1
            emit(
                progress_callback,
                status="running",
                step="repair_v3_video_timestamps",
                current=shard_count,
                total=total_shards,
                message=(
                    f"{'Checked' if dry_run else 'Normalized'} PTS for {video_key} "
                    f"chunk-{shard[0]:03d}/file-{shard[1]:03d}"
                ),
            )

    if not dry_run and updates:
        updated_episodes = update_episode_metadata(root, updates)
        if len(updated_episodes) != len(meta.episodes):
            raise ValueError(
                f"Updated timestamp metadata for {len(updated_episodes)} episodes, "
                f"expected {len(meta.episodes)}"
            )

    result = PreprocessResult(
        op="repair_v3_video_timestamps",
        src_roots=[root],
        out_root=root,
        repo_id=f"local/{root.name}",
        total_episodes=counts["total_episodes"],
        total_frames=counts["total_frames"],
        dry_run=dry_run,
        summary={
            "action": "repair_v3_video_timestamps",
            "video_keys": len(meta.video_keys),
            "video_shards": shard_count,
            "video_shards_needing_remux": shards_needing_remux,
            "video_shards_remuxed": shards_remuxed,
            "episode_camera_ranges": sum(len(fields) // 2 for fields in updates.values()),
            "max_abs_shift_s": max_abs_shift_s,
            "videos_remuxed": shards_remuxed > 0,
            "videos_reencoded": False,
            "norm_stats_recomputed": False,
        },
    )
    emit(
        progress_callback,
        status="done",
        current=shard_count,
        total=shard_count,
        message=(f"{'Previewed' if dry_run else 'Repaired'} v3 video timestamps without re-encoding: {root}"),
    )
    return result


def run_convert_v3(
    src_root: Path,
    out_root: Path | None = None,
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    workers: int = DEFAULT_V3_CONVERT_WORKERS,
    image_video_mode: str = IMAGE_VIDEO_MODE_RGB_LOSSLESS,
    overwrite: bool = False,
    dry_run: bool = False,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    """Convert v2.1 to v3.0 or report that an input is already v3.0.

    The encoding mode only changes Parquet-backed image cameras. Existing
    video features and v3.0 inputs remain unchanged.
    """
    src_root = validate_dataset_root(src_root)
    source_version = detect_dataset_version(src_root)
    if source_version not in {V21, V30}:
        raise ValueError(
            f"Unsupported LeRobot dataset version {source_version!r}. "
            f"Convert it to {V21} first; this low-risk converter follows {V3_BASELINE}."
        )

    if source_version == V30:
        counts = validate_v3_dataset(src_root)
        result = PreprocessResult(
            op="convert_v3",
            src_roots=[src_root],
            out_root=src_root,
            repo_id=f"local/{src_root.name}",
            total_episodes=counts["total_episodes"],
            total_frames=counts["total_frames"],
            dry_run=dry_run,
            summary={
                "source_version": V30,
                "target_version": V30,
                "baseline": V3_BASELINE,
                "already_v3": True,
                "action": "already_v3",
            },
        )
        emit(
            progress_callback,
            status="done",
            current=1,
            total=1,
            message=f"Dataset is already {V30}; no conversion is needed: {src_root}",
        )
        return result

    image_video_mode = normalize_image_video_mode(image_video_mode)
    if data_file_size_in_mb <= 0 or video_file_size_in_mb <= 0 or workers <= 0:
        raise ValueError("data/video file size limits and workers must be positive")
    info = load_json(src_root / "meta" / "info.json")
    episodes, tasks, episode_stats = _validate_v21(src_root, info)
    target_root = Path(out_root or default_v3_path(src_root)).expanduser()
    source_resolved = src_root.resolve()
    target_resolved = target_root.resolve()
    if (
        target_resolved == source_resolved
        or source_resolved in target_resolved.parents
        or target_resolved in source_resolved.parents
    ):
        raise ValueError(
            "v2.1 to v3.0 conversion requires a sibling output directory, "
            "not the source, its parent, or its child"
        )

    backup_root: Path | None = None
    if target_root.exists() and not dry_run:
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {target_root}")
        if target_root.is_symlink() or not target_root.is_dir():
            raise ValueError(f"Refusing to overwrite a non-directory output: {target_root}")
    else:
        ensure_output_root(target_root, dry_run)
    result = PreprocessResult(
        op="convert_v3",
        src_roots=[src_root],
        out_root=target_root,
        repo_id=f"local/{target_root.name}",
        total_episodes=len(episodes),
        total_frames=int(info.get("total_frames") or sum(int(row["length"]) for row in episodes)),
        dry_run=dry_run,
        summary={
            "source_version": V21,
            "target_version": V30,
            "baseline": V3_BASELINE,
            "already_v3": False,
            "action": "convert",
            "data_file_size_in_mb": data_file_size_in_mb,
            "video_file_size_in_mb": video_file_size_in_mb,
            "workers": workers,
            "tasks": len(tasks),
            "image_video_mode": image_video_mode,
        },
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(episodes),
        message=f"Validated {V21} source; planning {V30} conversion",
    )
    if dry_run:
        emit(
            progress_callback,
            status="done",
            current=0,
            total=len(episodes),
            message="Dry run complete",
        )
        return result

    if target_root.exists():
        with tempfile.NamedTemporaryFile(
            prefix=f".{target_root.name}.overwrite-backup-",
            dir=target_root.parent,
            delete=False,
        ) as backup_marker:
            backup_root = Path(backup_marker.name)
        backup_root.unlink()
        target_root.rename(backup_root)

    try:
        _write_info(
            target_root,
            info,
            data_file_size_in_mb,
            video_file_size_in_mb,
            image_video_mode,
        )
        _write_tasks(tasks, target_root)
        data_metadata = _convert_data(
            src_root,
            target_root,
            info,
            episodes,
            data_file_size_in_mb,
            progress_callback,
        )
        video_metadata = _convert_videos(
            src_root,
            target_root,
            info,
            episodes,
            video_file_size_in_mb,
            workers,
            progress_callback,
            image_video_mode,
        )
        _write_episode_metadata(
            target_root,
            episodes,
            episode_stats,
            data_metadata,
            video_metadata,
        )
        _copy_preserved_sidecars(src_root, target_root)
        validate_v3_dataset(target_root)
    except Exception:
        if target_root.exists():
            shutil.rmtree(target_root, ignore_errors=True)
        if backup_root is not None and backup_root.exists():
            backup_root.rename(target_root)
        raise
    if backup_root is not None and backup_root.exists():
        shutil.rmtree(backup_root)

    emit(
        progress_callback,
        status="done",
        current=len(episodes),
        total=len(episodes),
        message=f"Converted {V21} to {V30}: {target_root}",
    )
    logging.info("Converted LeRobot dataset %s -> %s using %s", src_root, target_root, V3_BASELINE)
    return result


__all__ = [
    "DEFAULT_DATA_FILE_SIZE_IN_MB",
    "DEFAULT_V3_CONVERT_WORKERS",
    "DEFAULT_VIDEO_FILE_SIZE_IN_MB",
    "IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL",
    "IMAGE_VIDEO_MODE_RGB_LOSSLESS",
    "IMAGE_VIDEO_MODES",
    "V21",
    "V30",
    "V3_BASELINE",
    "default_v3_path",
    "detect_dataset_version",
    "lerobot_official_av1_options",
    "materialize_v21_from_v3",
    "normalize_image_video_mode",
    "repair_v3_video_timestamps",
    "run_convert_v3",
    "validate_v3_dataset",
]
