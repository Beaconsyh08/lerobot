#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Precompute mp4 videos from image streams for faster visualization."""

import argparse
import csv
import json
import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.utils.utils import init_logging


@lru_cache(maxsize=8)
def _get_parquet_file(path: str) -> pq.ParquetFile:
    return pq.ParquetFile(path)


def _iter_image_bytes(
    parquet_path: Path,
    dataset_root: Path,
    image_key: str,
    max_frames: int | None = None,
):
    pf = _get_parquet_file(str(parquet_path))
    total = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[image_key])
        col = table[image_key]
        for i in range(len(col)):
            if max_frames is not None and total >= max_frames:
                return
            value = col[i].as_py()
            img_bytes = None
            if isinstance(value, dict):
                img_bytes = value.get("bytes")
                if img_bytes is None and value.get("path"):
                    candidates = [
                        dataset_root / value["path"],
                        dataset_root / "images" / value["path"],
                        dataset_root / "images" / image_key / value["path"],
                    ]
                    for img_path in candidates:
                        if img_path.is_file():
                            img_bytes = img_path.read_bytes()
                            break
            if img_bytes is not None:
                yield img_bytes
                total += 1


def _encode_with_ffmpeg(
    out_path: Path,
    frame_iter,
    width: int,
    height: int,
    fps: int,
) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logging.error("ffmpeg not found in PATH.")
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frame_iter:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.stdin = None
    _, err = proc.communicate()
    if proc.returncode != 0:
        logging.warning("ffmpeg failed to encode video: %s", err.decode(errors="ignore"))
        return False
    return True


def _encode_episode_key(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    episode_id: int,
    image_key: str,
    static_dir: Path,
    max_frames: int | None,
    overwrite: bool,
) -> Path | None:
    parquet_path = dataset_root / meta.get_data_file_path(episode_id)
    if not parquet_path.is_file():
        return None

    rel_path = Path("videos") / image_key / f"episode_{episode_id:06d}_h264.mp4"
    out_path = static_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    frame_iter = _iter_image_bytes(parquet_path, dataset_root, image_key, max_frames=max_frames)
    first_bytes = next(frame_iter, None)
    if first_bytes is None:
        return None
    first_arr = np.frombuffer(first_bytes, dtype=np.uint8)
    first_img = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
    if first_img is None:
        return None
    height, width = first_img.shape[:2]
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        return None
    if (first_img.shape[1], first_img.shape[0]) != (width, height):
        first_img = cv2.resize(first_img, (width, height))

    def frames_rgb():
        yield cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
        for img_bytes in frame_iter:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
            yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not _encode_with_ffmpeg(out_path, frames_rgb(), width, height, meta.info["fps"]):
        return None

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    return None


def _get_columns_info(meta: LeRobotDatasetMetadata):
    columns = []
    selected_columns = [col for col, ft in meta.features.items() if ft["dtype"] in ["float32", "int32"]]
    if "timestamp" in selected_columns:
        selected_columns.remove("timestamp")
    # subtask_state is computed fresh during precompute — never read from parquet
    if "subtask_state" in selected_columns:
        selected_columns.remove("subtask_state")

    ignored_columns = []
    filtered_columns = []
    for column_name in selected_columns:
        shape = meta.features[column_name]["shape"]
        if len(shape) > 1:
            ignored_columns.append(column_name)
        else:
            filtered_columns.append(column_name)
    selected_columns = filtered_columns

    for column_name in selected_columns:
        dim_state = meta.shapes[column_name][0]
        names = meta.features[column_name].get("names")
        if names:
            column_names = names
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
            if not isinstance(column_names, list) or len(column_names) != dim_state:
                column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

    selected_columns.insert(0, "timestamp")
    return columns, ignored_columns, selected_columns


def _series_to_2d(series, dim: int) -> np.ndarray:
    """Convert pandas Series of array-like values to (N, dim) float64 array."""
    rows = []
    for item in series:
        if item is None:
            rows.append([np.nan] * dim)
            continue
        vals = list(item)
        if len(vals) >= dim:
            rows.append(vals[:dim])
        else:
            rows.append(vals + [np.nan] * (dim - len(vals)))
    return np.asarray(rows, dtype=np.float64)


def _compute_subtask_boundaries(
    timestamps: np.ndarray,
    action_data: np.ndarray,
    state_data: np.ndarray | None,
    fps: float,
    episode_id: int = -1,
    task: str = "",
    gripper_margin: float = 0.6,
) -> dict | None:
    """Auto-detect subtask stage boundaries.

    Priority: gripper detection (stage 2) is absolute — other stages adjust around it.
    Each stage is guaranteed at least MIN_FRAMES (10) frames.

    Stages (frame ranges, all half-open except stage 2 end inclusive):
        0  [0, b1)        initial static — arm not moving
        1  [b1, b2)       arm moving toward target
        2  [b2, b3)       gripper transition  (last action_change − 0.6 s … state_change + 0.6 s)
        3  [b3, b4)       arm moving after gripper event
        4  [b4, N)        final static — arm not moving

    Returns (boundaries_dict_or_None, issues_list).
    """
    issues = []
    N = len(timestamps)
    MIN_FRAMES = 10
    if N < MIN_FRAMES * 5:
        msg = f"too few frames ({N}) for 5 stages"
        logging.warning("Episode %d: %s, skipping annotation.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "error", "reason": msg})
        return None, issues

    action_dim = action_data.shape[1]
    state_dim = state_data.shape[1] if state_data is not None else 0
    is_pick = "pick" in task.lower() if task else False

    # ── 1. Gripper detection → stage 2 frame window (HIGHEST PRIORITY) ──
    gripper_indices = [i for i in [7, 15] if i < action_dim]
    # Collect ALL gripper transitions across all gripper columns
    all_transitions = []  # list of (action_frame, state_frame, gripper_index)

    for gi in gripper_indices:
        ga = action_data[:, gi]
        binary_action = (ga > 0.5).astype(int)
        action_diffs = np.diff(binary_action)
        action_transition_frames = np.where(action_diffs != 0)[0]

        for af in action_transition_frames:
            action_frame = af + 1
            # corresponding state transition
            state_frame = action_frame  # fallback
            if state_data is not None and gi < state_dim:
                gs = state_data[:, gi]
                binary_state = (gs > 0.5).astype(int)
                state_diffs = np.diff(binary_state)
                state_transitions = np.where(state_diffs != 0)[0]
                after = state_transitions[state_transitions >= af]
                if len(after) > 0:
                    state_frame = after[0] + 1
            all_transitions.append((action_frame, state_frame, gi))

    if not all_transitions:
        msg = f"no gripper transition found (task: '{task}')" if is_pick else "no gripper transition found"
        logging.warning("Episode %d: %s, skipping.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "error", "reason": msg})
        return None, issues

    if len(all_transitions) > 1:
        frames_list = [int(t[0]) for t in all_transitions]
        msg = f"{len(all_transitions)} gripper transitions at frames {frames_list}, using LAST"
        logging.info("Episode %d: %s for stage 2.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "multi_gripper", "reason": msg, "frames": frames_list})

    # Use the LAST transition for stage 2
    last_action_frame, last_state_frame, _ = all_transitions[-1]

    # Convert 0.6 s margin to frame indices via timestamp search (handles variable fps)
    s2_start_time = float(timestamps[last_action_frame]) - gripper_margin
    s2_end_time = float(timestamps[min(last_state_frame, N - 1)]) + gripper_margin
    b2 = int(np.searchsorted(timestamps, s2_start_time, side="left"))
    b3 = int(np.searchsorted(timestamps, s2_end_time, side="right"))
    b2 = max(0, b2)
    b3 = min(N, b3)
    if b3 - b2 < MIN_FRAMES:
        mid = (b2 + b3) // 2
        b2 = max(0, mid - MIN_FRAMES // 2)
        b3 = b2 + MIN_FRAMES
        if b3 > N:
            b3 = N
            b2 = max(0, b3 - MIN_FRAMES)

    # ── 2. Arm-motion detection → stages 0 & 4 ──
    # Use only action columns (exclude grippers 7,15 and flag 16).
    # Sliding window: 1s centered window, slides per-frame.
    arm_idx = [i for i in range(action_dim) if i not in (7, 15, 16)]
    if not arm_idx:
        msg = "no arm action columns found"
        logging.warning("Episode %d: %s, skipping annotation.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "error", "reason": msg})
        return None, issues

    arm_actions = action_data[:, arm_idx]
    win = max(round(fps), 2)  # 1-second window in frames
    CHANGE_THR = 0.1

    # Per-column rolling range (max − min), sliding per-frame
    max_range = np.zeros(N)
    for j in range(arm_actions.shape[1]):
        s = pd.Series(arm_actions[:, j])
        roll_max = s.rolling(win, center=True, min_periods=1).max().values
        roll_min = s.rolling(win, center=True, min_periods=1).min().values
        max_range = np.maximum(max_range, roll_max - roll_min)

    # b1 = first frame where 1s-range >= 0.1 AND state[1] or state[9] > -0.4
    detected_b1 = b2  # fallback
    for i in range(N):
        if max_range[i] >= CHANGE_THR:
            # Additional condition: state[1] or state[9] must be > -0.4
            if state_data is not None:
                s1 = state_data[i, 1] if state_dim > 1 else np.nan
                s9 = state_data[i, 9] if state_dim > 9 else np.nan
                if (not np.isnan(s1) and s1 > -0.4) or (not np.isnan(s9) and s9 > -0.4):
                    detected_b1 = i
                    break
            else:
                detected_b1 = i
                break

    # b4 = frame after last frame where 1s-range >= threshold (stage 3 → 4)
    CHANGE_THR_END = 0.06
    detected_b4 = b3  # fallback
    for i in range(N - 1, -1, -1):
        if max_range[i] >= CHANGE_THR_END:
            detected_b4 = i + 1
            break

    # ── 3. Enforce MIN_FRAMES per stage ──
    # Stage 2 (b2, b3) is fixed.  Adjust b1 and b4 around it.
    b1 = detected_b1
    b4 = detected_b4

    # b1 ∈ [MIN_FRAMES, b2 − MIN_FRAMES]  (room for stages 0 and 1)
    b1_lo = MIN_FRAMES
    b1_hi = b2 - MIN_FRAMES
    if b1_lo <= b1_hi:
        b1 = max(b1_lo, min(b1_hi, b1))
    else:
        b1 = max(1, b2 // 2)

    # Ensure stage 1 [b1, b2) is at least 2 seconds — push b1 earlier into stage 0
    min_stage1_frames = max(round(fps * 2), MIN_FRAMES)
    if b2 - b1 < min_stage1_frames:
        b1 = max(MIN_FRAMES, b2 - min_stage1_frames)

    # b4 ∈ [b3 + MIN_FRAMES, N − MIN_FRAMES]  (room for stages 3 and 4)
    b4_lo = b3 + MIN_FRAMES
    b4_hi = N - MIN_FRAMES
    if b4_lo <= b4_hi:
        b4 = max(b4_lo, min(b4_hi, b4))
    else:
        b4 = (b3 + N) // 2

    # ── 4. Convert frame indices → timestamps ──
    return {
        "stage0_end": float(timestamps[min(b1, N - 1)]),
        "stage2_start": float(timestamps[min(b2, N - 1)]),
        "stage2_end": float(timestamps[min(b3 - 1, N - 1)]),
        "stage4_start": float(timestamps[min(b4, N - 1)]),
    }, issues


def _assign_subtask_states(timestamps, boundaries) -> list[int]:
    """Map each timestamp to a subtask state (0-4) based on boundaries."""
    if boundaries is None:
        return [0] * len(timestamps)
    result = []
    for t in timestamps:
        tf = float(t)
        if tf < boundaries["stage0_end"]:
            result.append(0)
        elif tf < boundaries["stage2_start"]:
            result.append(1)
        elif tf <= boundaries["stage2_end"]:
            result.append(2)
        elif tf < boundaries["stage4_start"]:
            result.append(3)
        else:
            result.append(4)
    return result


def _write_episode_csv(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    episode_id: int,
    out_path: Path,
    max_frames: int | None,
    downsample: int | None,
    overwrite: bool,
) -> tuple[bool, dict | None, list]:
    """Write precomputed CSV for one episode.

    Returns (success, subtask_boundaries_or_None, issues_list).
    """
    if out_path.exists() and not overwrite:
        return True, None, []
    parquet_path = dataset_root / meta.get_data_file_path(episode_id)
    if not parquet_path.is_file():
        return False, None, []

    columns, _, selected_columns = _get_columns_info(meta)
    data = pd.read_parquet(parquet_path, columns=selected_columns)
    if max_frames is not None:
        data = data.head(max_frames)

    # Compute subtask boundaries from full-rate data (before downsample)
    boundaries = None
    ep_issues = []
    action_dim = meta.shapes.get("action", [0])[0] if "action" in meta.features else 0
    state_dim = meta.shapes.get("state", [0])[0] if "state" in meta.features else 0
    if action_dim > 0 and "action" in data.columns and len(data) > 1:
        fps = 1.0 / np.median(np.diff(data["timestamp"].values))
        act_arr = _series_to_2d(data["action"], action_dim)
        st_arr = _series_to_2d(data["state"], state_dim) if state_dim > 0 and "state" in data.columns else None
        task = ""
        if hasattr(meta, "episodes") and episode_id in meta.episodes:
            tasks = meta.episodes[episode_id].get("tasks", [])
            task = tasks[0] if tasks else ""
        boundaries, ep_issues = _compute_subtask_boundaries(
            data["timestamp"].values, act_arr, st_arr, fps,
            episode_id=episode_id, task=task,
        )

    if downsample is not None and downsample > 1:
        data = data.iloc[::downsample].reset_index(drop=True)

    def _get_len(item) -> int:
        if item is None:
            return 0
        try:
            return len(item)
        except TypeError:
            return 0

    def _normalize_series(series, dim: int) -> np.ndarray:
        rows = []
        for item in series:
            row = [np.nan] * dim
            if item is not None:
                values = list(item)
                if len(values) > dim:
                    values = values[:dim]
                row[: len(values)] = values
            rows.append(row)
        return np.asarray(rows)

    data_arrays = []
    for col in selected_columns[1:]:
        fallback_dim = meta.shapes[col][0]
        series = data[col]
        actual_dim = max((_get_len(item) for item in series), default=0)
        dim = actual_dim if actual_dim > 0 else fallback_dim

        col_entry = next((c for c in columns if c["key"] == col), None)
        if col_entry is not None and len(col_entry["value"]) != dim:
            col_entry["value"] = [f"{col}_{i}" for i in range(dim)]

        data_arrays.append(_normalize_series(series, dim))

    header = ["timestamp"]
    for col_entry in columns:
        header += col_entry["value"]

    rows = np.hstack((np.expand_dims(data["timestamp"], axis=1), *data_arrays)).tolist()

    # Add stage column from auto-annotation (as decimal: current_stage / 4)
    if boundaries is not None:
        states = _assign_subtask_states(data["timestamp"].values, boundaries)
        header.append("stage")
        for i, s in enumerate(states):
            rows[i].append(s / 4.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return True, boundaries, ep_issues


def _get_default_output_dir(root: Path) -> Path:
    dataset_name = root.name or "dataset"
    return root.parent / f"local_vis_{dataset_name}"


def _all_precomputed_files_exist(
    static_dir: Path,
    episodes: list[int],
    image_keys: list[str],
    prepare_videos: bool,
    prepare_csv: bool,
    downsample: int | None,
) -> bool:
    if prepare_videos:
        for ep_id in episodes:
            for image_key in image_keys:
                video_path = static_dir / "videos" / image_key / f"episode_{ep_id:06d}_h264.mp4"
                if not video_path.is_file() or video_path.stat().st_size <= 0:
                    return False

    if prepare_csv:
        ds = downsample if downsample and downsample > 1 else 1
        csv_dir = static_dir / "csv"
        for ep_id in episodes:
            csv_path = csv_dir / f"episode_{ep_id:06d}_ds{ds}.csv"
            if not csv_path.is_file():
                return False

    return True


def _run_visualize_dataset_html(
    root: Path,
    repo_id: str,
    output_dir: Path,
    episodes: list[int] | None,
    max_frames: int | None,
    downsample: int | None,
    precomputed_only: bool,
    serve: bool,
    host: str,
    port: int,
    annotate: bool = False,
) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.visualize_dataset_html import MetaOnlyDataset, visualize_dataset_html

    dataset = (
        MetaOnlyDataset(repo_id, root=root)
        if precomputed_only
        else LeRobotDataset(repo_id, root=root)
    )
    visualize_dataset_html(
        dataset=dataset,
        episodes=episodes,
        output_dir=output_dir,
        serve=serve,
        host=host,
        port=port,
        max_frames=max_frames,
        prepare_videos=False,
        downsample=downsample,
        precompute_csv=False,
        precomputed_only=precomputed_only,
        annotate=annotate,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory for a dataset stored locally.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repo id for naming outputs when using --output-dir (e.g. lerobot/pusht).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to prepare. Default: all episodes.",
    )
    parser.add_argument(
        "--image-keys",
        type=str,
        nargs="*",
        default=None,
        help="Image keys to encode (default: all image features).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output dir used by visualize_dataset_html (will write to <output-dir>/static/videos). "
            "Default: <root parent>/local_vis_<root name>."
        ),
    )
    parser.add_argument(
        "--prepare-videos",
        type=int,
        default=1,
        help="Prepare mp4 videos from image streams.",
    )
    parser.add_argument(
        "--prepare-csv",
        type=int,
        default=1,
        help="Prepare downsampled CSV for time-series plots.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames per episode (for quick tests).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=5,
        help="Downsample time series by keeping one every N frames (e.g. 5).",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="Overwrite all existing precomputed files (videos + csv).",
    )
    parser.add_argument(
        "--overwrite-csv",
        type=int,
        default=0,
        help="Overwrite only existing precomputed CSV files (not videos).",
    )
    parser.add_argument(
        "--annotate",
        type=int,
        default=0,
        help="Enable interactive subtask annotation editing (press A to advance stage in frontend).",
    )
    parser.add_argument(
        "--write-parquet",
        type=int,
        default=0,
        help="Write computed subtask_state back into original parquet files (requires --annotate 1).",
    )
    parser.add_argument(
        "--run-visualize",
        type=int,
        default=1,
        help="Automatically run visualize_dataset_html after preparing files.",
    )
    parser.add_argument(
        "--precomputed-only",
        type=int,
        default=1,
        help="When running visualize_dataset_html, only load precomputed CSV/videos.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server when visualize_dataset_html runs.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by visualize_dataset_html.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9091,
        help="Web port used by visualize_dataset_html.",
    )

    args = parser.parse_args()
    init_logging()

    if not args.root.exists():
        raise FileNotFoundError(f"Local dataset root does not exist: {args.root}")
    info_path = args.root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata at: {info_path}")

    repo_id = args.repo_id or f"local/{args.root.name or 'dataset'}"
    meta = LeRobotDatasetMetadata(repo_id, root=args.root)

    image_keys = args.image_keys
    if image_keys is None:
        image_keys = [key for key, ft in meta.features.items() if ft["dtype"] == "image"]

    if args.episodes is None:
        episodes = sorted(meta.episodes.keys())
    else:
        episodes = args.episodes

    if args.output_dir is None:
        output_dir = _get_default_output_dir(args.root)
    else:
        output_dir = args.output_dir

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = static_dir / "csv"
    if args.prepare_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    overwrite_video = bool(args.overwrite)
    overwrite_csv = bool(args.overwrite) or bool(args.overwrite_csv)
    if args.write_parquet and not args.annotate:
        logging.warning("--write-parquet requires --annotate 1. Enabling --annotate automatically.")
        args.annotate = 1
    needs_prepare = overwrite_video or overwrite_csv or not _all_precomputed_files_exist(
        static_dir=static_dir,
        episodes=episodes,
        image_keys=image_keys,
        prepare_videos=bool(args.prepare_videos),
        prepare_csv=bool(args.prepare_csv),
        downsample=args.downsample,
    )

    all_boundaries = {}
    all_issues = []

    if needs_prepare:
        if args.prepare_videos or args.prepare_csv:
            with tqdm(total=len(episodes), desc="Precomputing", unit="episode", dynamic_ncols=True) as pbar:
                for ep_id in episodes:
                    if args.prepare_videos:
                        for image_key in image_keys:
                            _encode_episode_key(
                                args.root,
                                meta,
                                ep_id,
                                image_key,
                                static_dir,
                                args.max_frames,
                                overwrite_video,
                            )

                    if args.prepare_csv:
                        ds = args.downsample if args.downsample and args.downsample > 1 else 1
                        csv_path = csv_dir / f"episode_{ep_id:06d}_ds{ds}.csv"
                        success, boundaries, ep_issues = _write_episode_csv(
                            args.root,
                            meta,
                            ep_id,
                            csv_path,
                            args.max_frames,
                            args.downsample,
                            overwrite_csv,
                        )
                        if boundaries is not None:
                            all_boundaries[str(ep_id)] = boundaries
                        all_issues.extend(ep_issues)
                    pbar.update(1)
        else:
            logging.info("No precompute tasks selected. Skip prepare stage.")
    else:
        logging.info("Detected all precomputed files in '%s'. Skip prepare stage.", static_dir)

    # Save annotation issues JSON for review
    if all_issues:
        issues_path = static_dir / "annotation_issues.json"
        issues_path.write_text(json.dumps(all_issues, indent=2))
        logging.info(
            "Saved %d annotation issues to %s (errors: %d, multi_gripper: %d)",
            len(all_issues), issues_path,
            sum(1 for i in all_issues if i["type"] == "error"),
            sum(1 for i in all_issues if i["type"] == "multi_gripper"),
        )

    # Write subtask_state column into original parquet files + update meta
    if all_boundaries and args.write_parquet:
        all_episode_stats = {}  # ep_id -> stats dict for subtask_state
        with tqdm(total=len(all_boundaries), desc="Writing subtask_state to parquet", unit="episode", dynamic_ncols=True) as pbar:
            for ep_str, bounds in all_boundaries.items():
                ep_id = int(ep_str)
                parquet_path = args.root / meta.get_data_file_path(ep_id)
                if not parquet_path.is_file():
                    pbar.update(1)
                    continue
                table = pq.read_table(parquet_path)
                ts = table.column("timestamp").to_numpy()
                states = _assign_subtask_states(ts, bounds)
                # Verify no None values
                none_indices = [i for i, s in enumerate(states) if s is None]
                if none_indices:
                    logging.warning(
                        "Episode %d: subtask_state has %d None value(s) at indices: %s",
                        ep_id, len(none_indices), none_indices,
                    )
                col = pa.array(states, type=pa.int32())
                if "subtask_state" in table.column_names:
                    idx = table.column_names.index("subtask_state")
                    table = table.set_column(idx, "subtask_state", col)
                else:
                    table = table.append_column("subtask_state", col)
                tmp_path = parquet_path.with_suffix(".tmp")
                pq.write_table(table, tmp_path)
                tmp_path.rename(parquet_path)
                # Compute stats for this episode
                arr = np.array(states, dtype=np.float64)
                all_episode_stats[ep_id] = {
                    "min": [int(arr.min())],
                    "max": [int(arr.max())],
                    "mean": [float(arr.mean())],
                    "std": [float(arr.std())],
                    "count": [len(arr)],
                }
                pbar.update(1)

        # Update meta/info.json — add subtask_state to features
        info_path = args.root / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            if "subtask_state" not in info.get("features", {}):
                info["features"]["subtask_state"] = {
                    "dtype": "int32",
                    "shape": [1],
                    "names": None,
                }
                info_path.write_text(json.dumps(info, indent=2))
                logging.info("Added subtask_state to %s", info_path)

        # Update meta/episodes_stats.jsonl — add subtask_state stats per episode
        stats_path = args.root / "meta" / "episodes_stats.jsonl"
        if stats_path.is_file() and all_episode_stats:
            import jsonlines

            rows = []
            with jsonlines.open(stats_path, mode="r") as reader:
                for row in reader:
                    ep_id = row["episode_index"]
                    if ep_id in all_episode_stats:
                        row["stats"]["subtask_state"] = all_episode_stats[ep_id]
                    rows.append(row)
            with jsonlines.open(stats_path, mode="w") as writer:
                writer.write_all(rows)
            logging.info("Updated subtask_state stats in %s", stats_path)

    # Save auto-annotated subtask boundaries to JSON for the visualization frontend
    if all_boundaries:
        ann_path = static_dir / "subtask_annotations.json"
        existing = {}
        if ann_path.is_file():
            try:
                existing = json.loads(ann_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        # Convert boundaries to transitions format expected by frontend
        for ep_str, bounds in all_boundaries.items():
            if overwrite_csv or ep_str not in existing:
                existing[ep_str] = [
                    {"time": bounds["stage0_end"], "state": 1},
                    {"time": bounds["stage2_start"], "state": 2},
                    {"time": bounds["stage2_end"], "state": 3},
                    {"time": bounds["stage4_start"], "state": 4},
                ]
        ann_path.write_text(json.dumps(existing, indent=2))
        logging.info(
            "Saved auto subtask annotations for %d episodes to %s",
            len(all_boundaries),
            ann_path,
        )

    if args.run_visualize:
        logging.info("Launching visualize_dataset_html with output dir: %s", output_dir)
        _run_visualize_dataset_html(
            root=args.root,
            repo_id=repo_id,
            output_dir=output_dir,
            episodes=args.episodes,
            max_frames=args.max_frames,
            downsample=args.downsample,
            precomputed_only=bool(args.precomputed_only),
            serve=bool(args.serve),
            host=args.host,
            port=args.port,
            annotate=bool(args.annotate),
        )


if __name__ == "__main__":
    main()
