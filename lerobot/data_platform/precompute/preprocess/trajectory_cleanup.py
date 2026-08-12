"""Stage-aware trajectory cleanup for LeRobot v2.1 and v3.0 datasets.

The source dataset is never modified. Dry-run is the default; ``--apply`` writes
an explicit sibling output and keeps video frame indices aligned with Parquet.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.utils import cast_stats_to_numpy, serialize_dict
from lerobot.data_platform.precompute.dataset_io import V3DatasetMetadata, is_v3_dataset
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    default_preprocess_path,
    emit,
    ensure_output_root,
    format_data_path,
    load_json,
    load_jsonl,
    parquet_paths,
    validate_dataset_root,
    write_json,
    write_jsonl,
)
from lerobot.data_platform.precompute.preprocess.dataset_version import validate_v3_dataset

ACTION_COLUMN = "action"
STATE_COLUMN = "state"
STAGE_COLUMN = "subtask_state"
TRAJECTORY_CLEANUP_META = "preprocess_trajectory_cleanup.json"
DEFAULT_ARM_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
DEFAULT_GRIPPER_INDICES = (7, 15)


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required executable is unavailable: {name}")
    return executable


def _matrix(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float64)


def _replace_column(table: pa.Table, name: str, values) -> pa.Table:
    field = table.schema.field(name)
    return table.set_column(
        table.column_names.index(name),
        field,
        pa.array(values, type=field.type),
    )


def _butterworth_zero_phase(values: np.ndarray, cutoff_hz: float, fps: float) -> np.ndarray:
    """Apply a dependency-free second-order Butterworth filter forward/backward."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("joint values must be a 2D [frames, joints] array")
    if len(values) < 5:
        return values.copy()
    if not 0 < cutoff_hz < fps / 2:
        raise ValueError(f"cutoff_hz must be between 0 and Nyquist ({fps / 2:g})")

    k = math.tan(math.pi * cutoff_hz / fps)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    b0 = k * k * norm
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * k + k * k) * norm

    def _forward(array: np.ndarray) -> np.ndarray:
        output = np.empty_like(array)
        x1 = array[0].copy()
        x2 = array[0].copy()
        y1 = array[0].copy()
        y2 = array[0].copy()
        for index, current in enumerate(array):
            value = b0 * current + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            output[index] = value
            x2, x1 = x1, current
            y2, y1 = y1, value
        return output

    pad = min(12, len(values) - 1)
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="reflect")
    filtered = _forward(padded)
    filtered = _forward(filtered[::-1])[::-1]
    filtered = filtered[pad:-pad]
    filtered[0] = values[0]
    filtered[-1] = values[-1]
    return filtered


def _stable_tail_start(
    state: np.ndarray,
    action: np.ndarray,
    stages: np.ndarray,
    arm_indices: tuple[int, ...],
    gripper_indices: tuple[int, ...],
    window_frames: int,
    max_range_rad: float,
) -> int | None:
    valid = stages[np.isfinite(stages)]
    if len(valid) == 0 or len(state) < window_frames:
        return None
    final_stage = float(valid.max())
    final_positions = np.flatnonzero(stages == final_stage)
    if len(final_positions) < window_frames:
        return None
    first = int(final_positions[0])
    for start in range(first, len(state) - window_frames + 1):
        if not np.all(stages[start:] == final_stage):
            continue
        if float(np.ptp(state[start:, arm_indices], axis=0).max()) > max_range_rad:
            continue
        if gripper_indices and np.any(np.ptp(action[start:, gripper_indices], axis=0) > 0):
            continue
        return start
    return None


def _smooth_episode_action(
    table: pa.Table,
    *,
    fps: float,
    cutoff_hz: float,
    arm_indices: tuple[int, ...],
    gripper_indices: tuple[int, ...],
    max_smooth_step_rad: float,
    stabilize_final_stage: bool,
    stable_window_frames: int,
    stable_max_range_rad: float,
) -> tuple[pa.Table, dict]:
    action = _matrix(table, ACTION_COLUMN)
    state = _matrix(table, STATE_COLUMN)
    if action.shape != state.shape:
        raise ValueError(f"action/state shape mismatch: {action.shape} != {state.shape}")
    if not arm_indices or max(arm_indices) >= action.shape[1]:
        raise ValueError(f"arm indices {arm_indices} do not fit action dimension {action.shape[1]}")

    smoothed = action.copy()
    max_step = float(np.abs(np.diff(action[:, arm_indices], axis=0)).max()) if len(action) > 1 else 0.0
    skipped = max_step > max_smooth_step_rad
    if not skipped:
        smoothed[:, arm_indices] = _butterworth_zero_phase(action[:, arm_indices], cutoff_hz, fps)

    stable_start = None
    if stabilize_final_stage and STAGE_COLUMN in table.column_names:
        stages = np.asarray(table[STAGE_COLUMN].to_pylist(), dtype=np.float64)
        stable_start = _stable_tail_start(
            state,
            action,
            stages,
            arm_indices,
            tuple(index for index in gripper_indices if index < action.shape[1]),
            stable_window_frames,
            stable_max_range_rad,
        )
        if stable_start is not None:
            target = np.median(smoothed[stable_start:, arm_indices], axis=0)
            blend = min(3, len(smoothed) - stable_start)
            for offset in range(blend):
                alpha = offset / max(1, blend - 1)
                smoothed[stable_start + offset, arm_indices] = (1.0 - alpha) * smoothed[
                    stable_start + offset, arm_indices
                ] + alpha * target
            smoothed[stable_start + blend :, arm_indices] = target

    for index in gripper_indices:
        if index < action.shape[1] and not np.array_equal(smoothed[:, index], action[:, index]):
            raise AssertionError(f"gripper dimension {index} was modified")
    changed = np.abs(smoothed[:, arm_indices] - action[:, arm_indices])
    table = _replace_column(table, ACTION_COLUMN, smoothed.astype(np.float32).tolist())
    return table, {
        "skipped_smoothing": skipped,
        "max_raw_step_rad": max_step,
        "stable_tail_start": stable_start,
        "max_abs_change_rad": float(changed.max()) if changed.size else 0.0,
        "rms_change_rad": float(np.sqrt(np.mean(np.square(changed)))) if changed.size else 0.0,
    }


def _numeric_episode_stats(table: pa.Table, features: dict, old_stats: dict) -> dict:
    stats = {
        key: copy.deepcopy(value)
        for key, value in old_stats.items()
        if (features.get(key) or {}).get("dtype") in {"image", "video"}
    }
    for key, feature in features.items():
        if key not in table.column_names or feature.get("dtype") in {"string", "image", "video"}:
            continue
        values = table[key].to_pylist()
        if any(value is None for value in values):
            values = [value for value in values if value is not None]
        if not values:
            continue
        array = np.asarray(values)
        stats[key] = compute_episode_stats({key: array}, {key: feature})[key]
    return cast_stats_to_numpy(stats)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([int(start), int(end)])
        else:
            merged[-1][1] = max(merged[-1][1], int(end))
    return [(start, end) for start, end in merged]


def _removed_before(ranges: list[tuple[int, int]], frame: int) -> int:
    return sum(max(0, min(end + 1, frame) - start) for start, end in ranges if start < frame)


def _video_plan(
    meta: V3DatasetMetadata,
    kept_old_ids: list[int],
    deleted_ids: set[int],
    drop_first_ids: set[int],
    new_lengths: dict[int, int],
) -> tuple[dict[tuple[str, int, int], list[tuple[int, int]]], dict[int, dict]]:
    fps = float(meta.fps)
    drop_ranges: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    updates: dict[int, dict] = {episode_id: {} for episode_id in kept_old_ids}
    all_ids = sorted(meta.episodes)
    for video_key in meta.video_keys:
        for episode_id in all_ids:
            episode = meta.episodes[episode_id]
            chunk = int(episode[f"videos/{video_key}/chunk_index"])
            file_index = int(episode[f"videos/{video_key}/file_index"])
            start = int(round(float(episode[f"videos/{video_key}/from_timestamp"]) * fps))
            end = int(round(float(episode[f"videos/{video_key}/to_timestamp"]) * fps))
            if end - start != int(episode["length"]):
                raise ValueError(
                    f"episode {episode_id} {video_key} video range has {end - start} frames, "
                    f"expected {episode['length']}"
                )
            key = (video_key, chunk, file_index)
            if episode_id in deleted_ids:
                drop_ranges.setdefault(key, []).append((start, end - 1))
            elif episode_id in drop_first_ids:
                drop_ranges.setdefault(key, []).append((start, start))

        for key in [value for value in drop_ranges if value[0] == video_key]:
            drop_ranges[key] = _merge_ranges(drop_ranges[key])

        for episode_id in kept_old_ids:
            episode = meta.episodes[episode_id]
            chunk = int(episode[f"videos/{video_key}/chunk_index"])
            file_index = int(episode[f"videos/{video_key}/file_index"])
            key = (video_key, chunk, file_index)
            start = int(round(float(episode[f"videos/{video_key}/from_timestamp"]) * fps))
            shifted_start = start - _removed_before(drop_ranges.get(key, []), start)
            updates[episode_id].update(
                {
                    f"videos/{video_key}/chunk_index": chunk,
                    f"videos/{video_key}/file_index": file_index,
                    f"videos/{video_key}/from_timestamp": shifted_start / fps,
                    f"videos/{video_key}/to_timestamp": (shifted_start + new_lengths[episode_id]) / fps,
                }
            )
    return drop_ranges, updates


@functools.lru_cache(maxsize=1)
def _ffmpeg_encoders() -> set[str]:
    completed = subprocess.run(
        [_require_executable("ffmpeg"), "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.split()[1]
        for line in completed.stdout.splitlines()
        if len(line.split()) >= 2 and line.lstrip().startswith("V")
    }


def _video_encoder_args(feature: dict) -> list[str]:
    codec = str(feature.get("video.codec") or "").lower()
    pixel_format = str(feature.get("video.pix_fmt") or "yuv420p")
    crf = str(feature.get("video.crf", 30))
    preset = str(feature.get("video.preset", 12))
    gop = str(feature.get("video.g", 2))
    if codec in {"av1", "libsvtav1"}:
        encoders = _ffmpeg_encoders()
        if "libsvtav1" in encoders:
            return [
                "-c:v",
                "libsvtav1",
                "-crf",
                crf,
                "-preset",
                preset,
                "-g",
                gop,
                "-pix_fmt",
                pixel_format,
            ]
        if "libaom-av1" in encoders:
            return [
                "-c:v",
                "libaom-av1",
                "-crf",
                crf,
                "-b:v",
                "0",
                "-cpu-used",
                "8",
                "-row-mt",
                "1",
                "-g",
                gop,
                "-pix_fmt",
                pixel_format,
            ]
        raise RuntimeError("ffmpeg provides neither libsvtav1 nor libaom-av1")
    if codec in {"h264", "avc", "avc1", "libx264"}:
        encoder = "libx264rgb" if pixel_format == "rgb24" else "libx264"
        return ["-c:v", encoder, "-crf", crf, "-preset", "medium", "-g", gop, "-pix_fmt", pixel_format]
    raise ValueError(f"Unsupported video codec for frame removal: {codec!r}")


def _probe_video_frames(path: Path) -> int:
    completed = subprocess.run(
        [
            _require_executable("ffprobe"),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def _rewrite_av1_with_pyav(
    src: Path,
    dst: Path,
    ranges: list[tuple[int, int]],
    feature: dict,
    fps: int,
) -> None:
    options = {
        "crf": str(feature.get("video.crf", 30)),
        "preset": str(feature.get("video.preset", 12)),
        "g": str(feature.get("video.g", 2)),
        "svtav1-params": "lp=4",
    }
    range_index = 0
    try:
        with av.open(str(src)) as input_container:
            input_stream = input_container.streams.video[0]
            with av.open(str(dst), mode="w", options={"movflags": "+faststart"}) as output_container:
                output_stream = output_container.add_stream("libsvtav1", rate=fps, options=options)
                output_stream.width = input_stream.codec_context.width
                output_stream.height = input_stream.codec_context.height
                output_stream.pix_fmt = str(feature.get("video.pix_fmt") or "yuv420p")
                output_index = 0
                for frame_index, frame in enumerate(input_container.decode(input_stream)):
                    while range_index < len(ranges) and frame_index > ranges[range_index][1]:
                        range_index += 1
                    if (
                        range_index < len(ranges)
                        and ranges[range_index][0] <= frame_index <= ranges[range_index][1]
                    ):
                        continue
                    frame.pts = output_index
                    frame.time_base = Fraction(1, fps)
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
                    output_index += 1
                for packet in output_stream.encode():
                    output_container.mux(packet)
    except Exception:
        dst.unlink(missing_ok=True)
        raise


def _rewrite_video_without_ranges(
    src: Path,
    dst: Path,
    ranges: list[tuple[int, int]],
    feature: dict,
    fps: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    codec = str(feature.get("video.codec") or "").lower()
    expected = _probe_video_frames(src) - sum(end - start + 1 for start, end in ranges)
    if codec in {"av1", "libsvtav1"} and "libsvtav1" in av.codec.codecs_available:
        _rewrite_av1_with_pyav(src, dst, ranges, feature, fps)
        actual = _probe_video_frames(dst)
        if actual != expected:
            dst.unlink(missing_ok=True)
            raise ValueError(f"rewritten video {dst} has {actual} frames, expected {expected}")
        return

    clauses = [f"between(n\\,{start}\\,{end})" for start, end in ranges]
    filter_graph = f"select=not({'+'.join(clauses)}),setpts=N/{fps}/TB"
    completed = subprocess.run(
        [
            _require_executable("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            filter_graph,
            "-an",
            *_video_encoder_args(feature),
            "-r",
            str(fps),
            "-vsync",
            "cfr",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {src}: {completed.stderr.strip()}")
    actual = _probe_video_frames(dst)
    if actual != expected:
        dst.unlink(missing_ok=True)
        raise ValueError(f"rewritten video {dst} has {actual} frames, expected {expected}")


def _copy_or_rewrite_videos(
    src_root: Path,
    out_root: Path,
    info: dict,
    drop_ranges: dict[tuple[str, int, int], list[tuple[int, int]]],
    workers: int,
    progress_callback: ProgressCallback,
) -> int:
    paths = sorted((src_root / "videos").rglob("*.mp4"))
    affected = []
    for path in paths:
        rel = path.relative_to(src_root / "videos")
        video_key = rel.parts[0]
        chunk = int(path.parent.name.removeprefix("chunk-"))
        file_index = int(path.stem.removeprefix("file-"))
        dst = out_root / "videos" / rel
        ranges = drop_ranges.get((video_key, chunk, file_index), [])
        if ranges:
            affected.append((path, dst, ranges, info["features"][video_key]))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)

    if not affected:
        return 0
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required to remove aligned video frames")

    worker_count = min(max(1, workers), len(affected))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _rewrite_video_without_ranges,
                src,
                dst,
                ranges,
                feature,
                int(info["fps"]),
            ): src
            for src, dst, ranges, feature in affected
        }
        for position, future in enumerate(as_completed(futures), start=1):
            future.result()
            emit(
                progress_callback,
                status="running",
                step="videos",
                current=position,
                total=len(affected),
                message=f"Rewrote video shard {position}/{len(affected)}",
            )
    return len(affected)


def _copy_preserved_files(src_root: Path, out_root: Path) -> None:
    for child in src_root.iterdir():
        if child.name in {"data", "meta", "videos"}:
            continue
        destination = out_root / child.name
        if child.is_dir():
            shutil.copytree(child, destination, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    for child in (src_root / "meta").iterdir():
        if child.name in {"episodes", "info.json", "stats.json"}:
            continue
        destination = out_root / "meta" / child.name
        if child.is_dir():
            shutil.copytree(child, destination, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)


def _rewrite_v21_episode(
    *,
    src: Path,
    dst: Path,
    old_id: int,
    new_id: int,
    drop_first: bool,
    global_offset: int,
    old_stats: dict,
    features: dict,
    fps: float,
    cutoff_hz: float,
    arm_indices: tuple[int, ...],
    gripper_indices: tuple[int, ...],
    max_smooth_step_rad: float,
    stabilize_final_stage: bool,
    stable_window_frames: int,
    stable_max_range_rad: float,
) -> dict:
    table = pq.read_table(src)
    if drop_first:
        table = table.slice(1)
    length = table.num_rows
    table = _replace_column(table, "episode_index", [new_id] * length)
    if "frame_index" in table.column_names:
        table = _replace_column(table, "frame_index", list(range(length)))
    if "timestamp" in table.column_names:
        table = _replace_column(table, "timestamp", (np.arange(length) / fps).tolist())
    if "index" in table.column_names:
        table = _replace_column(table, "index", list(range(global_offset, global_offset + length)))
    table, smooth_info = _smooth_episode_action(
        table,
        fps=fps,
        cutoff_hz=cutoff_hz,
        arm_indices=arm_indices,
        gripper_indices=gripper_indices,
        max_smooth_step_rad=max_smooth_step_rad,
        stabilize_final_stage=stabilize_final_stage,
        stable_window_frames=stable_window_frames,
        stable_max_range_rad=stable_max_range_rad,
    )
    stats = _numeric_episode_stats(table, features, old_stats)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return {
        "old_id": old_id,
        "new_id": new_id,
        "length": length,
        "stats": stats,
        "smooth_info": smooth_info,
    }


def _validate_v21_cleanup(root: Path) -> dict[str, int]:
    info = load_json(root / "meta" / "info.json")
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    episode_stats = load_jsonl(root / "meta" / "episodes_stats.jsonl")
    expected_ids = list(range(len(episodes)))
    if [int(row["episode_index"]) for row in episodes] != expected_ids:
        raise ValueError("v2.1 cleanup output episode indices are not contiguous")
    if [int(row["episode_index"]) for row in episode_stats] != expected_ids:
        raise ValueError("v2.1 cleanup output stats do not match episode indices")
    if int(info.get("total_episodes") or -1) != len(episodes):
        raise ValueError("v2.1 cleanup output total_episodes is inconsistent")

    total_frames = 0
    global_offset = 0
    fps = float(info["fps"])
    for episode_id, record in enumerate(episodes):
        path = root / format_data_path(info, episode_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cleaned v2.1 episode parquet: {path}")
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "timestamp", "index"],
        )
        length = int(record["length"])
        if table.num_rows != length:
            raise ValueError(f"Cleaned episode {episode_id} has {table.num_rows} rows, expected {length}")
        if table["episode_index"].to_pylist() != [episode_id] * length:
            raise ValueError(f"Cleaned episode {episode_id} contains wrong episode_index values")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"Cleaned episode {episode_id} contains wrong frame_index values")
        if table["index"].to_pylist() != list(range(global_offset, global_offset + length)):
            raise ValueError(f"Cleaned episode {episode_id} contains wrong global index values")
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        timestamp_dtype = (
            np.float32 if pa.types.is_float32(table.schema.field("timestamp").type) else np.float64
        )
        expected_timestamps = np.asarray(np.arange(length) / fps, dtype=timestamp_dtype).astype(np.float64)
        if not np.array_equal(timestamps, expected_timestamps):
            raise ValueError(f"Cleaned episode {episode_id} contains wrong timestamps")
        total_frames += length
        global_offset += length
    if int(info.get("total_frames") or -1) != total_frames:
        raise ValueError("v2.1 cleanup output total_frames is inconsistent")
    return {"episodes": len(episodes), "frames": total_frames}


def _detect_initial_jumps(
    root: Path,
    threshold_rad: float,
    arm_indices: tuple[int, ...],
) -> dict[int, float]:
    jumps = {}
    for path in parquet_paths(root):
        table = pq.read_table(path, columns=["episode_index", STATE_COLUMN])
        episode_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        states = _matrix(table, STATE_COLUMN)
        starts = np.r_[0, np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(episode_ids)]
        for start, end in zip(starts, ends, strict=True):
            if end - start < 2:
                continue
            jump = float(np.abs(states[start + 1, arm_indices] - states[start, arm_indices]).max())
            if jump > threshold_rad:
                jumps[int(episode_ids[start])] = jump
    return jumps


def run_trajectory_cleanup(
    src_root: Path,
    out_root: Path | None = None,
    *,
    delete_episode_ids: tuple[int, ...] = (),
    initial_jump_threshold_rad: float = 0.5,
    cutoff_hz: float = 4.0,
    arm_indices: tuple[int, ...] = DEFAULT_ARM_INDICES,
    gripper_indices: tuple[int, ...] = DEFAULT_GRIPPER_INDICES,
    max_smooth_step_rad: float = 0.1,
    stabilize_final_stage: bool = True,
    stable_window_frames: int = 10,
    stable_max_range_rad: float = 0.02,
    workers: int = 2,
    dry_run: bool = True,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    if initial_jump_threshold_rad <= 0 or cutoff_hz <= 0 or max_smooth_step_rad <= 0:
        raise ValueError("jump threshold, cutoff, and max smoothing step must be positive")
    if stable_window_frames < 2 or stable_max_range_rad <= 0 or workers < 1:
        raise ValueError("stable window/range and workers must be positive")

    info = load_json(src_root / "meta" / "info.json")
    source_is_v3 = is_v3_dataset(src_root)
    if source_is_v3:
        meta = V3DatasetMetadata(f"local/{src_root.name}", src_root)
        episodes = meta.episodes
        episode_stats = meta.episodes_stats
        features = meta.features
        fps = float(meta.fps)
        source_frames = meta.total_frames
    else:
        version = str(info.get("codebase_version") or "").lower().removeprefix("v")
        if version != "2.1":
            raise ValueError(f"trajectory cleanup supports only LeRobot v2.1 and v3.0, got {version!r}")
        if any(feature.get("dtype") == "video" for feature in (info.get("features") or {}).values()):
            raise ValueError("v2.1 cleanup currently supports image-backed datasets, not per-episode videos")
        episode_rows = load_jsonl(src_root / "meta" / "episodes.jsonl")
        stats_rows = load_jsonl(src_root / "meta" / "episodes_stats.jsonl")
        episodes = {int(row["episode_index"]): row for row in episode_rows}
        episode_stats = {int(row["episode_index"]): row.get("stats") or {} for row in stats_rows}
        features = info.get("features") or {}
        fps = float(info["fps"])
        source_frames = int(info.get("total_frames") or sum(int(row["length"]) for row in episode_rows))
        expected_ids = set(range(len(episodes)))
        if set(episodes) != expected_ids or set(episode_stats) != expected_ids:
            raise ValueError("v2.1 episodes and episode stats must be contiguous and complete")

    all_ids = set(episodes)
    deleted_ids = {int(value) for value in delete_episode_ids}
    missing = sorted(deleted_ids - all_ids)
    if missing:
        raise ValueError(f"episodes not found: {missing}")
    if deleted_ids == all_ids:
        raise ValueError("cleanup would delete every episode")

    jumps = _detect_initial_jumps(src_root, initial_jump_threshold_rad, arm_indices)
    drop_first_ids = set(jumps) - deleted_ids
    kept_old_ids = sorted(all_ids - deleted_ids)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}
    new_lengths = {
        old_id: int(episodes[old_id]["length"]) - (1 if old_id in drop_first_ids else 0)
        for old_id in kept_old_ids
    }
    if any(length <= 0 for length in new_lengths.values()):
        raise ValueError("dropping frame 0 would leave an empty episode")
    total_frames = sum(new_lengths.values())
    target_root = ensure_output_root(
        out_root or default_preprocess_path(src_root, "trajectory_clean"), dry_run
    )
    summary = {
        "delete_episodes": sorted(deleted_ids),
        "drop_first_frame_episodes": sorted(drop_first_ids),
        "initial_jump_rad": {str(key): value for key, value in sorted(jumps.items())},
        "cutoff_hz": cutoff_hz,
        "arm_indices": list(arm_indices),
        "gripper_indices": list(gripper_indices),
        "max_smooth_step_rad": max_smooth_step_rad,
        "stabilize_final_stage": stabilize_final_stage,
        "stable_window_frames": stable_window_frames,
        "stable_max_range_rad": stable_max_range_rad,
        "source_format": "v3.0" if source_is_v3 else "v2.1",
        "source_frames": source_frames,
        "output_frames": total_frames,
        "stats_policy": "numeric_recomputed_image_video_preserved",
    }
    result = PreprocessResult(
        op="trajectory_cleanup",
        src_roots=[src_root],
        out_root=target_root,
        repo_id=f"local/{target_root.name}",
        total_episodes=len(kept_old_ids),
        total_frames=total_frames,
        dry_run=dry_run,
        summary=summary,
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(kept_old_ids),
        message=f"Cleanup plan: {summary}",
    )
    if dry_run:
        emit(progress_callback, status="done", current=0, total=len(kept_old_ids), message="Dry run complete")
        return result

    if not source_is_v3:
        try:
            _copy_preserved_files(src_root, target_root)
            offsets = {}
            global_offset = 0
            for old_id in kept_old_ids:
                offsets[old_id] = global_offset
                global_offset += new_lengths[old_id]

            worker_count = min(workers, len(kept_old_ids), os.cpu_count() or 1)
            cleaned: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {}
                for old_id in kept_old_ids:
                    new_id = old_to_new[old_id]
                    future = executor.submit(
                        _rewrite_v21_episode,
                        src=src_root / format_data_path(info, old_id),
                        dst=target_root / format_data_path(info, new_id),
                        old_id=old_id,
                        new_id=new_id,
                        drop_first=old_id in drop_first_ids,
                        global_offset=offsets[old_id],
                        old_stats=episode_stats.get(old_id, {}),
                        features=features,
                        fps=fps,
                        cutoff_hz=cutoff_hz,
                        arm_indices=arm_indices,
                        gripper_indices=gripper_indices,
                        max_smooth_step_rad=max_smooth_step_rad,
                        stabilize_final_stage=stabilize_final_stage,
                        stable_window_frames=stable_window_frames,
                        stable_max_range_rad=stable_max_range_rad,
                    )
                    futures[future] = old_id
                for position, future in enumerate(as_completed(futures), start=1):
                    item = future.result()
                    cleaned[int(item["old_id"])] = item
                    emit(
                        progress_callback,
                        status="running",
                        step="data",
                        current=position,
                        total=len(kept_old_ids),
                        message=f"Cleaned v2.1 episode {item['old_id']} -> {item['new_id']}",
                    )

            output_episode_rows = []
            output_stats_rows = []
            smoothing_skipped = []
            stabilized = []
            stats_by_new = {}
            for old_id in kept_old_ids:
                item = cleaned[old_id]
                new_id = int(item["new_id"])
                record = dict(episodes[old_id])
                record["episode_index"] = new_id
                record["length"] = int(item["length"])
                output_episode_rows.append(record)
                stats_by_new[new_id] = item["stats"]
                output_stats_rows.append({"episode_index": new_id, "stats": serialize_dict(item["stats"])})
                smooth_info = item["smooth_info"]
                if smooth_info["skipped_smoothing"]:
                    smoothing_skipped.append(old_id)
                if smooth_info["stable_tail_start"] is not None:
                    stabilized.append(old_id)

            write_jsonl(target_root / "meta" / "episodes.jsonl", output_episode_rows)
            write_jsonl(target_root / "meta" / "episodes_stats.jsonl", output_stats_rows)
            if (src_root / "meta" / "stats.json").is_file():
                global_stats = aggregate_stats(
                    [cast_stats_to_numpy(stats_by_new[index]) for index in sorted(stats_by_new)],
                    sample_ids=sorted(stats_by_new),
                    sample_label="episode_index",
                )
                write_json(target_root / "meta" / "stats.json", serialize_dict(global_stats))

            output_info = dict(info)
            output_info["total_episodes"] = len(kept_old_ids)
            output_info["total_frames"] = total_frames
            output_info["total_chunks"] = math.ceil(
                len(kept_old_ids) / int(output_info.get("chunks_size") or 1000)
            )
            output_info["splits"] = {"train": f"0:{len(kept_old_ids)}"}
            write_json(target_root / "meta" / "info.json", output_info)
            summary.update(
                {
                    "smoothing_skipped_episodes": smoothing_skipped,
                    "stable_tail_episodes": stabilized,
                    "rewritten_video_shards": 0,
                }
            )
            write_json(
                target_root / "meta" / TRAJECTORY_CLEANUP_META,
                {
                    "op": "trajectory_cleanup",
                    "source_root": str(src_root),
                    "output_root": str(target_root),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    **summary,
                },
            )
            _validate_v21_cleanup(target_root)
        except Exception:
            if target_root.exists():
                shutil.rmtree(target_root, ignore_errors=True)
            raise

        emit(
            progress_callback,
            status="done",
            current=len(kept_old_ids),
            total=len(kept_old_ids),
            message=f"Trajectory cleanup complete: {target_root}",
        )
        return result

    try:
        _copy_preserved_files(src_root, target_root)
        (target_root / "data").mkdir(parents=True, exist_ok=True)
        stats_by_new: dict[int, dict] = {}
        episode_records: dict[int, dict] = {}
        smoothing_skipped = []
        stabilized = []
        global_offset = 0
        source_episode_paths = {
            old_id: (
                int(meta.episodes[old_id]["data/chunk_index"]),
                int(meta.episodes[old_id]["data/file_index"]),
            )
            for old_id in kept_old_ids
        }
        groups: dict[tuple[int, int], list[int]] = {}
        for old_id in kept_old_ids:
            groups.setdefault(source_episode_paths[old_id], []).append(old_id)

        original_episode_paths = sorted((src_root / "meta" / "episodes").rglob("*.parquet"))
        original_frame = pd.concat(
            (pd.read_parquet(path) for path in original_episode_paths), ignore_index=True
        ).sort_values("episode_index")
        original_records = {
            int(record["episode_index"]): record for record in original_frame.to_dict(orient="records")
        }

        processed = 0
        for (chunk, file_index), episode_ids in sorted(groups.items()):
            src = src_root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
            shard = pq.read_table(src)
            output_tables = []
            for old_id in episode_ids:
                table = shard.filter(pc.equal(shard["episode_index"], old_id))
                if old_id in drop_first_ids:
                    table = table.slice(1)
                new_id = old_to_new[old_id]
                length = table.num_rows
                table = _replace_column(table, "episode_index", [new_id] * length)
                if "frame_index" in table.column_names:
                    table = _replace_column(table, "frame_index", list(range(length)))
                if "timestamp" in table.column_names:
                    table = _replace_column(table, "timestamp", (np.arange(length) / meta.fps).tolist())
                if "index" in table.column_names:
                    table = _replace_column(
                        table, "index", list(range(global_offset, global_offset + length))
                    )
                table, smooth_info = _smooth_episode_action(
                    table,
                    fps=float(meta.fps),
                    cutoff_hz=cutoff_hz,
                    arm_indices=arm_indices,
                    gripper_indices=gripper_indices,
                    max_smooth_step_rad=max_smooth_step_rad,
                    stabilize_final_stage=stabilize_final_stage,
                    stable_window_frames=stable_window_frames,
                    stable_max_range_rad=stable_max_range_rad,
                )
                if smooth_info["skipped_smoothing"]:
                    smoothing_skipped.append(old_id)
                if smooth_info["stable_tail_start"] is not None:
                    stabilized.append(old_id)
                stats = _numeric_episode_stats(
                    table,
                    meta.features,
                    meta.episodes_stats.get(old_id, {}),
                )
                stats_by_new[new_id] = stats
                record = dict(original_records[old_id])
                record.update(
                    {
                        "episode_index": new_id,
                        "length": length,
                        "data/chunk_index": chunk,
                        "data/file_index": file_index,
                        "dataset_from_index": global_offset,
                        "dataset_to_index": global_offset + length,
                    }
                )
                serialized = serialize_dict(stats)
                for feature, feature_stats in serialized.items():
                    for stat_name, value in feature_stats.items():
                        column = f"stats/{feature}/{stat_name}"
                        if column in original_frame.columns:
                            record[column] = value
                episode_records[old_id] = record
                output_tables.append(table)
                global_offset += length
                processed += 1
                emit(
                    progress_callback,
                    status="running",
                    step="data",
                    current=processed,
                    total=len(kept_old_ids),
                    message=f"Cleaned episode {old_id} -> {new_id}",
                )
            dst = target_root / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.concat_tables(output_tables, promote_options="default"), dst)

        drop_ranges, video_updates = _video_plan(
            meta,
            kept_old_ids,
            deleted_ids,
            drop_first_ids,
            new_lengths,
        )
        for old_id, updates in video_updates.items():
            episode_records[old_id].update(updates)

        rewritten_video_shards = _copy_or_rewrite_videos(
            src_root,
            target_root,
            meta.info,
            drop_ranges,
            workers,
            progress_callback,
        )

        output_records = [episode_records[old_id] for old_id in kept_old_ids]
        episode_path = target_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(output_records, columns=original_frame.columns).to_parquet(
            episode_path, index=False
        )
        global_stats = aggregate_stats(
            [cast_stats_to_numpy(stats_by_new[index]) for index in sorted(stats_by_new)],
            sample_ids=sorted(stats_by_new),
            sample_label="episode_index",
        )
        write_json(target_root / "meta" / "stats.json", serialize_dict(global_stats))

        info = load_json(src_root / "meta" / "info.json")
        info["total_episodes"] = len(kept_old_ids)
        info["total_frames"] = total_frames
        info["splits"] = {"train": f"0:{len(kept_old_ids)}"}
        write_json(target_root / "meta" / "info.json", info)
        summary.update(
            {
                "smoothing_skipped_episodes": smoothing_skipped,
                "stable_tail_episodes": stabilized,
                "rewritten_video_shards": rewritten_video_shards,
            }
        )
        write_json(
            target_root / "meta" / TRAJECTORY_CLEANUP_META,
            {
                "op": "trajectory_cleanup",
                "source_root": str(src_root),
                "output_root": str(target_root),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                **summary,
            },
        )
        validate_v3_dataset(target_root)
    except Exception:
        if target_root.exists():
            shutil.rmtree(target_root, ignore_errors=True)
        raise

    emit(
        progress_callback,
        status="done",
        current=len(kept_old_ids),
        total=len(kept_old_ids),
        message=f"Trajectory cleanup complete: {target_root}",
    )
    return result


def _parse_indices(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--delete-episodes", type=_parse_indices, default=())
    parser.add_argument("--initial-jump-threshold-rad", type=float, default=0.5)
    parser.add_argument("--cutoff-hz", type=float, default=4.0)
    parser.add_argument("--max-smooth-step-rad", type=float, default=0.1)
    parser.add_argument("--stable-window-frames", type=int, default=10)
    parser.add_argument("--stable-max-range-rad", type=float, default=0.02)
    parser.add_argument("--no-stable-tail", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--apply", action="store_true", help="Write the sibling output; default is dry-run")
    args = parser.parse_args()
    result = run_trajectory_cleanup(
        args.src_root,
        args.out_root,
        delete_episode_ids=args.delete_episodes,
        initial_jump_threshold_rad=args.initial_jump_threshold_rad,
        cutoff_hz=args.cutoff_hz,
        max_smooth_step_rad=args.max_smooth_step_rad,
        stabilize_final_stage=not args.no_stable_tail,
        stable_window_frames=args.stable_window_frames,
        stable_max_range_rad=args.stable_max_range_rad,
        workers=args.workers,
        dry_run=not args.apply,
    )
    print(json.dumps({"out_root": str(result.out_root), **result.summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
