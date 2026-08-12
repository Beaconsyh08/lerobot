from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.data_platform.precompute.dataset_io import read_episode_table
from lerobot.data_platform.precompute.timeseries import (
    BODY_JOINT_INDICES,
    DATA_VERSION_DVT1,
    DATA_VERSION_DVT2,
    normalize_gripper_columns,
)

DVT1_GRIPPER_STAGE_MARGIN_SECONDS = 0.6
DVT2_GRIPPER_STAGE_MARGIN_SECONDS = 0.2
DVT2_LEFT_ARM_JOINT_INDICES = tuple(range(0, 7))
DVT2_RIGHT_ARM_JOINT_INDICES = tuple(range(8, 15))
DVT2_LEFT_ARM_START_MOVE_RAD = np.deg2rad(
    np.array(
        [-20.7723, 50.0851, -16.0979, -46.6055, -62.3497, 49.7502, -3.2868],
        dtype=np.float64,
    )
)
DVT2_RIGHT_ARM_START_MOVE_RAD = np.deg2rad(
    np.array(
        [-20.7723, -50.0851, 16.0979, -46.6055, 62.3497, 49.7502, 3.2868],
        dtype=np.float64,
    )
)
DVT2_START_POSE_RMS_TOLERANCE_RAD = np.deg2rad(1.0)
DVT2_START_POSE_MAX_TOLERANCE_RAD = np.deg2rad(3.0)
DVT2_START_POSE_STAGE0_DELAY_SECONDS = 0.2
INITIAL_CLOSED_GRIPPER_OPEN_STAGE0_DELAY_SECONDS = 0.1
INITIAL_CLOSED_GRIPPER_OPEN_THRESHOLD = 0.1
DVT2_STAGE4_STABLE_SECONDS = 0.4
DVT2_PLACE_STAGE4_STABLE_SECONDS = 0.2
DVT2_STAGE4_STABLE_THRESHOLD = 0.08
DEFAULT_FALLBACK_STAGE_COUNT = 5
QUALITY_FLAG_TYPE = "quality_flag"
QUALITY_EARLY_WINDOW_SECONDS = 0.3
QUALITY_GRIPPER_THRESHOLD = 0.5
QUALITY_STUCK_CLOSED_RATIO = 0.6
QUALITY_STUCK_ACTION_MAX_ABS = 0.1
QUALITY_STUCK_ACTION_MAX_DELTA = 0.1
QUALITY_GRIPPER_ACTION_MATCH_PRE_SECONDS = 0.3
QUALITY_GRIPPER_ACTION_MATCH_POST_SECONDS = 0.1
QUALITY_ZERO_RESET_ABS = 0.03
QUALITY_ZERO_RESET_NEIGHBOR_ABS = 0.25
QUALITY_ZERO_RESET_DELTA = 0.45
QUALITY_ZERO_RESET_MAX_RUN_FRAMES = 2
QUALITY_ZERO_RESET_MIN_EVENTS = 6
QUALITY_ZERO_RESET_MIN_SYNC_DIMS = 4
GRIPPER_SIDE_BY_INDEX = {7: "left", 15: "right"}


def _gripper_stage_margin_seconds(data_version: str) -> float:
    return (
        DVT2_GRIPPER_STAGE_MARGIN_SECONDS
        if str(data_version).upper() == DATA_VERSION_DVT2
        else DVT1_GRIPPER_STAGE_MARGIN_SECONDS
    )


def _is_pose_near(target: np.ndarray, pose: np.ndarray) -> bool:
    if pose.shape[0] != target.shape[0] or not np.all(np.isfinite(pose)):
        return False
    diff = np.abs(pose - target)
    return bool(
        np.sqrt(np.mean(np.square(diff))) <= DVT2_START_POSE_RMS_TOLERANCE_RAD
        and np.max(diff) <= DVT2_START_POSE_MAX_TOLERANCE_RAD
    )


def _first_dvt2_start_pose_frame(state_data: np.ndarray | None, upper_bound: int) -> int | None:
    if state_data is None or state_data.ndim != 2 or upper_bound <= 0:
        return None

    state_dim = state_data.shape[1]
    check_left = state_dim > max(DVT2_LEFT_ARM_JOINT_INDICES)
    check_right = state_dim > max(DVT2_RIGHT_ARM_JOINT_INDICES)
    if not check_left and not check_right:
        return None

    search_end = min(max(upper_bound, 0), state_data.shape[0])
    for frame_idx in range(search_end):
        if check_left:
            left_pose = state_data[frame_idx, list(DVT2_LEFT_ARM_JOINT_INDICES)]
            if _is_pose_near(DVT2_LEFT_ARM_START_MOVE_RAD, left_pose):
                return frame_idx
        if check_right:
            right_pose = state_data[frame_idx, list(DVT2_RIGHT_ARM_JOINT_INDICES)]
            if _is_pose_near(DVT2_RIGHT_ARM_START_MOVE_RAD, right_pose):
                return frame_idx
    return None


def _first_dvt2_stage4_stable_frame(
    max_range: np.ndarray,
    start_frame: int,
    fps: float,
    stable_seconds: float = DVT2_STAGE4_STABLE_SECONDS,
) -> int | None:
    if max_range.size == 0 or start_frame >= max_range.shape[0]:
        return None
    if not np.isfinite(fps) or fps <= 0:
        return None
    stable_frames = max(1, round(float(fps) * float(stable_seconds)))
    search_start = max(0, int(start_frame))
    search_end = max_range.shape[0] - stable_frames + 1
    for frame_idx in range(search_start, max(search_start, search_end)):
        window = max_range[frame_idx : frame_idx + stable_frames]
        if window.shape[0] < stable_frames:
            break
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            continue
        if float(np.max(finite)) <= DVT2_STAGE4_STABLE_THRESHOLD:
            return frame_idx
    return None


def _is_closed_to_open_gripper_transition(
    action_data: np.ndarray,
    state_data: np.ndarray | None,
    gripper_idx: int,
) -> bool:
    source = state_data if state_data is not None and gripper_idx < state_data.shape[1] else action_data
    if gripper_idx >= source.shape[1] or source.shape[0] < 2:
        return False
    series = np.asarray(source[:, gripper_idx], dtype=np.float64)
    finite = series[np.isfinite(series)]
    if finite.size < 2:
        return False
    binary = (finite > 0.5).astype(np.int8)
    return bool(binary[0] == 1 and binary[-1] == 0)


def _initial_closed_gripper_open_frame(
    action_data: np.ndarray,
    state_data: np.ndarray | None,
    upper_bound: int,
) -> int | None:
    """Return the frame where observed initially closed grippers have opened.

    Stage 0 can require both "robot reached the pre-grasp start pose" and "the
    gripper is open". This must use observed state, not action, so the boundary
    waits until the physical gripper state has changed. If state is unavailable,
    leave stage timing unchanged.
    """
    if upper_bound <= 0 or state_data is None:
        return None
    required_open_frames: list[int] = []
    for gripper_idx in (7, 15):
        if gripper_idx >= state_data.shape[1] or state_data.shape[0] < 2:
            continue
        series = np.asarray(
            state_data[: min(int(upper_bound), state_data.shape[0]), gripper_idx], dtype=np.float64
        )
        finite_indices = np.flatnonzero(np.isfinite(series))
        if finite_indices.size == 0:
            continue
        first_idx = int(finite_indices[0])
        if series[first_idx] <= 0.5:
            continue
        open_candidates = finite_indices[series[finite_indices] <= INITIAL_CLOSED_GRIPPER_OPEN_THRESHOLD]
        open_candidates = open_candidates[open_candidates > first_idx]
        if open_candidates.size == 0:
            continue
        required_open_frames.append(int(open_candidates[0]))
    if not required_open_frames:
        return None
    return max(required_open_frames)


def _initial_closed_to_open_transition_count(
    transitions: list[tuple[int, int, int]],
    action_data: np.ndarray,
    state_data: np.ndarray | None,
) -> int:
    count = 0
    seen_grippers: set[int] = set()
    for action_frame, state_frame, gripper_idx in sorted(transitions, key=lambda item: item[0]):
        if gripper_idx in seen_grippers:
            continue
        seen_grippers.add(gripper_idx)
        source = state_data if state_data is not None and gripper_idx < state_data.shape[1] else action_data
        if source is None or gripper_idx >= source.shape[1] or source.shape[0] < 2:
            continue
        series = np.asarray(source[:, gripper_idx], dtype=np.float64)
        finite_indices = np.flatnonzero(np.isfinite(series))
        if finite_indices.size == 0 or series[int(finite_indices[0])] <= 0.5:
            continue
        check_frame = min(max(int(state_frame), int(action_frame), 0), source.shape[0] - 1)
        if np.isfinite(series[check_frame]) and series[check_frame] <= 0.5:
            count += 1
    return count


def _safe_2d_float_array(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    if array.ndim != 2:
        return None
    return array


def _infer_fps_from_timestamps(timestamps: np.ndarray, fallback: float | None = None) -> float | None:
    if fallback is not None and np.isfinite(fallback) and fallback > 0:
        return float(fallback)
    if len(timestamps) < 2:
        return None
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    median = float(np.median(diffs))
    return 1.0 / median if median > 0 else None


def _early_frame_count(timestamps: np.ndarray, fps: float | None, window_seconds: float) -> int:
    frame_count = len(timestamps)
    if frame_count <= 0:
        return 0
    timestamps = np.asarray(timestamps, dtype=np.float64)
    finite_timestamps = np.isfinite(timestamps)
    if finite_timestamps.all() and frame_count > 1:
        start_time = float(timestamps[0])
        by_time = int(np.count_nonzero(timestamps <= start_time + window_seconds))
        if by_time > 0:
            return min(frame_count, max(3, by_time))
    if fps is not None and np.isfinite(fps) and fps > 0:
        return min(frame_count, max(3, int(round(float(fps) * window_seconds))))
    return min(frame_count, 3)


def _gripper_transition_frames(values: np.ndarray | None, source: str) -> list[dict]:
    if values is None or values.shape[0] < 2:
        return []
    events: list[dict] = []
    for gripper_idx in (7, 15):
        if gripper_idx >= values.shape[1]:
            continue
        series = values[:, gripper_idx]
        finite = np.isfinite(series)
        if not finite.any():
            continue
        binary = (np.where(finite, series, 0.0) > QUALITY_GRIPPER_THRESHOLD).astype(np.int8)
        frames = np.where(np.diff(binary) != 0)[0] + 1
        for frame in frames:
            events.append({"source": source, "gripper_index": int(gripper_idx), "frame": int(frame)})
    return events


def _transition_metric(events: list[dict]) -> dict:
    return {
        "count": len(events),
        "frames": [int(event["frame"]) for event in events],
        "sources": sorted({str(event["source"]) for event in events}),
        "gripper_indices": sorted({int(event["gripper_index"]) for event in events}),
    }


def _stuck_closed_gripper_no_action_issues(
    action_norm: np.ndarray | None,
    state_norm: np.ndarray | None,
    episode_id: int,
    data_version: str,
) -> list[dict]:
    if action_norm is None or state_norm is None:
        return []
    if action_norm.shape[0] == 0 or state_norm.shape[0] == 0:
        return []

    issues = []
    frame_count = min(action_norm.shape[0], state_norm.shape[0])
    for gripper_idx, side in GRIPPER_SIDE_BY_INDEX.items():
        if gripper_idx >= action_norm.shape[1] or gripper_idx >= state_norm.shape[1]:
            continue

        action_series = np.asarray(action_norm[:frame_count, gripper_idx], dtype=np.float64)
        state_series = np.asarray(state_norm[:frame_count, gripper_idx], dtype=np.float64)
        action_finite = action_series[np.isfinite(action_series)]
        state_finite = state_series[np.isfinite(state_series)]
        if action_finite.size == 0 or state_finite.size == 0:
            continue

        state_closed_mask = state_finite > QUALITY_GRIPPER_THRESHOLD
        state_closed_ratio = float(np.mean(state_closed_mask))
        action_abs_max = float(np.max(np.abs(action_finite)))
        action_delta_max = float(np.max(np.abs(np.diff(action_finite)))) if action_finite.size >= 2 else 0.0

        if (
            state_closed_ratio >= QUALITY_STUCK_CLOSED_RATIO
            and action_abs_max <= QUALITY_STUCK_ACTION_MAX_ABS
            and action_delta_max <= QUALITY_STUCK_ACTION_MAX_DELTA
        ):
            closed_frames = np.where(np.isfinite(state_series) & (state_series > QUALITY_GRIPPER_THRESHOLD))[
                0
            ]
            issues.append(
                {
                    "episode": int(episode_id),
                    "type": QUALITY_FLAG_TYPE,
                    "reason": "stuck_closed_gripper_no_action",
                    "frames": [int(closed_frames[0]), int(closed_frames[-1])] if closed_frames.size else [],
                    "data_version": data_version,
                    "metrics": {
                        "side": side,
                        "gripper_index": int(gripper_idx),
                        "state_closed_ratio": state_closed_ratio,
                        "state_median": float(np.median(state_finite)),
                        "action_abs_max": action_abs_max,
                        "action_delta_max": action_delta_max,
                        "closed_threshold": QUALITY_GRIPPER_THRESHOLD,
                        "closed_ratio_threshold": QUALITY_STUCK_CLOSED_RATIO,
                        "action_abs_threshold": QUALITY_STUCK_ACTION_MAX_ABS,
                        "action_delta_threshold": QUALITY_STUCK_ACTION_MAX_DELTA,
                    },
                }
            )
    return issues


def _state_gripper_transition_without_action_issues(
    action_norm: np.ndarray | None,
    state_norm: np.ndarray | None,
    timestamps: np.ndarray,
    fps: float | None,
    episode_id: int,
    data_version: str,
) -> list[dict]:
    if action_norm is None or state_norm is None:
        return []
    if action_norm.ndim != 2 or state_norm.ndim != 2:
        return []
    frame_count = min(action_norm.shape[0], state_norm.shape[0], len(timestamps))
    if frame_count < 2:
        return []

    inferred_fps = _infer_fps_from_timestamps(timestamps[:frame_count], fps)
    if inferred_fps is not None and np.isfinite(inferred_fps) and inferred_fps > 0:
        pre_frames = max(1, int(round(float(inferred_fps) * QUALITY_GRIPPER_ACTION_MATCH_PRE_SECONDS)))
        post_frames = max(1, int(round(float(inferred_fps) * QUALITY_GRIPPER_ACTION_MATCH_POST_SECONDS)))
    else:
        pre_frames = 3
        post_frames = 1

    missing_events = []
    for gripper_idx, side in GRIPPER_SIDE_BY_INDEX.items():
        if gripper_idx >= action_norm.shape[1] or gripper_idx >= state_norm.shape[1]:
            continue
        state_series = np.asarray(state_norm[:frame_count, gripper_idx], dtype=np.float64)
        action_series = np.asarray(action_norm[:frame_count, gripper_idx], dtype=np.float64)
        if not np.all(np.isfinite(state_series)) or not np.all(np.isfinite(action_series)):
            continue
        state_binary = (state_series > QUALITY_GRIPPER_THRESHOLD).astype(np.int8)
        action_binary = (action_series > QUALITY_GRIPPER_THRESHOLD).astype(np.int8)
        state_transition_frames = np.where(np.diff(state_binary) != 0)[0] + 1
        if state_transition_frames.size == 0:
            continue
        action_transition_frames = np.where(np.diff(action_binary) != 0)[0] + 1
        for frame in state_transition_frames:
            frame = int(frame)
            window_start = max(0, frame - pre_frames)
            window_end = min(frame_count - 1, frame + post_frames)
            matched = bool(
                action_transition_frames.size
                and np.any(
                    (action_transition_frames >= window_start) & (action_transition_frames <= window_end)
                )
            )
            if matched:
                continue
            missing_events.append(
                {
                    "side": side,
                    "gripper_index": int(gripper_idx),
                    "frame": frame,
                    "from_state": int(state_binary[frame - 1]),
                    "to_state": int(state_binary[frame]),
                    "action_before": int(action_binary[max(0, frame - 1)]),
                    "action_at": int(action_binary[frame]),
                    "match_window": [int(window_start), int(window_end)],
                }
            )

    if not missing_events:
        return []
    return [
        {
            "episode": int(episode_id),
            "type": QUALITY_FLAG_TYPE,
            "reason": "state_gripper_transition_without_action",
            "frames": sorted({int(event["frame"]) for event in missing_events}),
            "data_version": data_version,
            "metrics": {
                "events": missing_events,
                "event_count": len(missing_events),
                "pre_seconds": QUALITY_GRIPPER_ACTION_MATCH_PRE_SECONDS,
                "post_seconds": QUALITY_GRIPPER_ACTION_MATCH_POST_SECONDS,
            },
        }
    ]


def _quality_joint_indices(values: np.ndarray | None) -> list[int]:
    if values is None or values.ndim != 2:
        return []
    return [idx for idx in range(values.shape[1]) if idx not in GRIPPER_SIDE_BY_INDEX]


def _zero_reset_events(values: np.ndarray | None, source: str) -> list[dict]:
    """Detect short reset-to-zero artifacts in non-gripper joints.

    This intentionally requires a near-zero run to be surrounded by clearly
    non-zero neighbors, so normal smooth trajectories crossing zero are ignored.
    """
    if values is None or values.ndim != 2 or values.shape[0] < 3:
        return []
    events: list[dict] = []
    for joint_idx in _quality_joint_indices(values):
        series = np.asarray(values[:, joint_idx], dtype=np.float64)
        finite = np.isfinite(series)
        if np.count_nonzero(finite) < 3:
            continue
        zero_mask = finite & (np.abs(series) <= QUALITY_ZERO_RESET_ABS)
        start = None
        for frame_idx, is_zero in enumerate(zero_mask.tolist() + [False]):
            if is_zero and start is None:
                start = frame_idx
                continue
            if is_zero or start is None:
                continue
            end = frame_idx - 1
            run_len = end - start + 1
            if start > 0 and end < len(series) - 1 and run_len <= QUALITY_ZERO_RESET_MAX_RUN_FRAMES:
                prev_value = float(series[start - 1])
                next_value = float(series[end + 1])
                if (
                    np.isfinite(prev_value)
                    and np.isfinite(next_value)
                    and abs(prev_value) >= QUALITY_ZERO_RESET_NEIGHBOR_ABS
                    and abs(next_value) >= QUALITY_ZERO_RESET_NEIGHBOR_ABS
                    and abs(prev_value - float(series[start])) >= QUALITY_ZERO_RESET_DELTA
                    and abs(next_value - float(series[end])) >= QUALITY_ZERO_RESET_DELTA
                ):
                    events.append(
                        {
                            "source": source,
                            "joint_index": int(joint_idx),
                            "frame": int(start),
                            "run_length": int(run_len),
                            "prev": prev_value,
                            "next": next_value,
                        }
                    )
            start = None
    return events


def _joint_zero_reset_issues(
    action_values: np.ndarray | None,
    state_values: np.ndarray | None,
    episode_id: int,
    data_version: str,
) -> list[dict]:
    events = _zero_reset_events(action_values, "action") + _zero_reset_events(state_values, "state")
    if not events:
        return []
    frame_counts: dict[int, int] = {}
    for event in events:
        frame_counts[int(event["frame"])] = frame_counts.get(int(event["frame"]), 0) + 1
    sync_frames = sorted(
        frame for frame, count in frame_counts.items() if count >= QUALITY_ZERO_RESET_MIN_SYNC_DIMS
    )
    affected_joints = sorted({(str(event["source"]), int(event["joint_index"])) for event in events})
    if not (
        len(events) >= QUALITY_ZERO_RESET_MIN_EVENTS
        or sync_frames
        or (len(affected_joints) >= 3 and len(events) >= 4)
    ):
        return []
    return [
        {
            "episode": int(episode_id),
            "type": QUALITY_FLAG_TYPE,
            "reason": "joint_zero_reset_spike",
            "frames": sorted({int(event["frame"]) for event in events})[:50],
            "data_version": data_version,
            "metrics": {
                "event_count": len(events),
                "affected_joint_count": len(affected_joints),
                "sync_frames": sync_frames[:50],
                "zero_abs_threshold": QUALITY_ZERO_RESET_ABS,
                "neighbor_abs_threshold": QUALITY_ZERO_RESET_NEIGHBOR_ABS,
                "delta_threshold": QUALITY_ZERO_RESET_DELTA,
                "max_run_frames": QUALITY_ZERO_RESET_MAX_RUN_FRAMES,
                "sample_events": events[:20],
            },
        }
    ]


def compute_quality_flags(
    timestamps: np.ndarray,
    action_data: np.ndarray | None,
    state_data: np.ndarray | None = None,
    fps: float | None = None,
    episode_id: int = -1,
    task: str = "",
    data_version: str = DATA_VERSION_DVT1,
    early_window_seconds: float = QUALITY_EARLY_WINDOW_SECONDS,
) -> list[dict]:
    """Detect suspicious early gripper transitions without mutating cache, stage, or parquet data."""
    issues: list[dict] = []
    timestamps = np.asarray(timestamps, dtype=np.float64)
    frame_count = len(timestamps)
    if frame_count == 0:
        return issues

    data_version = str(data_version or DATA_VERSION_DVT1).upper()
    inferred_fps = _infer_fps_from_timestamps(timestamps, fps)
    early_count = _early_frame_count(timestamps, inferred_fps, early_window_seconds)

    action_array = _safe_2d_float_array(action_data)
    state_array = _safe_2d_float_array(state_data)

    action_norm = (
        normalize_gripper_columns(action_array, "action", data_version) if action_array is not None else None
    )
    state_norm = (
        normalize_gripper_columns(state_array, "state", data_version) if state_array is not None else None
    )
    all_gripper_events = _gripper_transition_frames(action_norm, "action") + _gripper_transition_frames(
        state_norm, "state"
    )
    early_gripper_events = [event for event in all_gripper_events if int(event["frame"]) < early_count]
    if early_gripper_events:
        issues.append(
            {
                "episode": int(episode_id),
                "type": QUALITY_FLAG_TYPE,
                "reason": "early_gripper_transition",
                "frames": sorted({int(event["frame"]) for event in early_gripper_events}),
                "data_version": data_version,
                "metrics": {
                    **_transition_metric(early_gripper_events),
                    "early_window_seconds": float(early_window_seconds),
                    "early_frame_count": int(early_count),
                },
            }
        )

    issues.extend(
        _stuck_closed_gripper_no_action_issues(
            action_norm,
            state_norm,
            int(episode_id),
            data_version,
        )
    )
    issues.extend(
        _state_gripper_transition_without_action_issues(
            action_norm,
            state_norm,
            timestamps,
            inferred_fps,
            int(episode_id),
            data_version,
        )
    )
    issues.extend(
        _joint_zero_reset_issues(
            action_norm,
            state_norm,
            int(episode_id),
            data_version,
        )
    )

    return issues


def _is_exist_label_feature(column_name: str, feature: dict) -> bool:
    if column_name != "exist_label":
        return False
    dtype = str(feature.get("dtype", ""))
    return dtype.startswith(("float", "int", "uint")) or dtype in {"bool", "boolean"}


def _is_plot_feature(column_name: str, feature: dict) -> bool:
    return feature.get("dtype") in ["float32", "int32"] or _is_exist_label_feature(column_name, feature)


def _feature_shape(feature: dict) -> tuple:
    shape = feature.get("shape") or []
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def get_columns_info(meta: LeRobotDatasetMetadata) -> tuple[list[dict], list[str], list[str]]:
    columns = []
    selected_columns = [col for col, feature in meta.features.items() if _is_plot_feature(col, feature)]
    if "timestamp" in selected_columns:
        selected_columns.remove("timestamp")
    if "subtask_state" in selected_columns:
        selected_columns.remove("subtask_state")

    ignored_columns = []
    filtered_columns = []
    for column_name in selected_columns:
        shape = _feature_shape(meta.features[column_name])
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
        elif column_name == "exist_label" and dim_state == 1:
            column_names = ["exist_label"]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

    selected_columns.insert(0, "timestamp")
    return columns, ignored_columns, selected_columns


def series_to_2d(series, dim: int) -> np.ndarray:
    """Convert pandas Series of array-like values to an (N, dim) float64 array."""
    rows = []
    for item in series:
        if item is None:
            rows.append([np.nan] * dim)
            continue
        values = list(item)
        if len(values) >= dim:
            rows.append(values[:dim])
        else:
            rows.append(values + [np.nan] * (dim - len(values)))
    return np.asarray(rows, dtype=np.float64)


def compute_subtask_boundaries(
    timestamps: np.ndarray,
    action_data: np.ndarray,
    state_data: np.ndarray | None,
    fps: float,
    episode_id: int = -1,
    task: str = "",
    gripper_margin: float | None = None,
    data_version: str = DATA_VERSION_DVT1,
    fallback_stage_count: int = DEFAULT_FALLBACK_STAGE_COUNT,
) -> tuple[dict | None, list[dict]]:
    """Auto-detect subtask stage boundaries."""
    issues: list[dict] = []
    data_version = str(data_version or DATA_VERSION_DVT1).upper()
    if gripper_margin is None:
        gripper_margin = _gripper_stage_margin_seconds(data_version)
    num_frames = len(timestamps)
    is_pick = "pick" in task.lower() if task else False
    is_place = any(token in task.lower() for token in ("place", "put")) if task else False
    is_give = ("give" in task.lower() or "hand" in task.lower()) if task else False

    if not (is_pick or is_place or is_give):
        fallback_stage_count = int(fallback_stage_count)
        if fallback_stage_count < 2:
            raise ValueError("fallback_stage_count must be at least 2")
        if num_frames < 2:
            msg = f"too few frames ({num_frames}) for time-equal stages"
            logging.warning("Episode %d: %s, skipping annotation.", episode_id, msg)
            issues.append({"episode": episode_id, "type": "error", "reason": msg})
            return None, issues
        start_time = float(timestamps[0])
        end_time = float(timestamps[-1])
        if not np.isfinite(start_time) or not np.isfinite(end_time) or end_time <= start_time:
            msg = "timestamps must have a positive finite duration for time-equal stages"
            logging.warning("Episode %d: %s, skipping annotation.", episode_id, msg)
            issues.append({"episode": episode_id, "type": "error", "reason": msg})
            return None, issues
        stage_boundaries = np.linspace(start_time, end_time, fallback_stage_count + 1)[1:-1]
        return {
            "equal_time": True,
            "num_stages": fallback_stage_count,
            "stage_boundaries": [float(value) for value in stage_boundaries],
        }, issues

    min_frames = 10
    if num_frames < min_frames * 5:
        msg = f"too few frames ({num_frames}) for 5 stages"
        logging.warning("Episode %d: %s, skipping annotation.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "error", "reason": msg})
        return None, issues

    action_data = normalize_gripper_columns(action_data, "action", data_version)
    state_data = (
        normalize_gripper_columns(state_data, "state", data_version) if state_data is not None else None
    )
    action_dim = action_data.shape[1]
    state_dim = state_data.shape[1] if state_data is not None else 0

    gripper_indices = [idx for idx in [7, 15] if idx < action_dim]
    all_transitions = []

    for gripper_idx in gripper_indices:
        gripper_action = action_data[:, gripper_idx]
        binary_action = (gripper_action > 0.5).astype(int)
        action_diffs = np.diff(binary_action)
        action_transition_frames = np.where(action_diffs != 0)[0]

        for action_frame_idx in action_transition_frames:
            action_frame = action_frame_idx + 1
            state_frame = action_frame
            if state_data is not None and gripper_idx < state_dim:
                gripper_state = state_data[:, gripper_idx]
                binary_state = (gripper_state > 0.5).astype(int)
                state_diffs = np.diff(binary_state)
                state_transitions = np.where(state_diffs != 0)[0]
                after = state_transitions[state_transitions >= action_frame_idx]
                if len(after) > 0:
                    state_frame = after[0] + 1
            all_transitions.append((action_frame, state_frame, gripper_idx))

    if not all_transitions:
        msg = f"no gripper transition found (task: '{task}')" if is_pick else "no gripper transition found"
        logging.warning("Episode %d: %s, skipping.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "error", "reason": msg})
        return None, issues

    direct_give = False
    if is_give and len(all_transitions) < 2:
        if len(all_transitions) == 1 and _is_closed_to_open_gripper_transition(
            action_data,
            state_data,
            all_transitions[0][2],
        ):
            direct_give = True
            logging.info(
                "Episode %d: direct give detected from single closed-to-open gripper transition at frame %d.",
                episode_id,
                all_transitions[0][0],
            )
        else:
            msg = f"give task needs >=2 gripper transitions or one closed-to-open direct-give transition, found {len(all_transitions)}"
            logging.warning("Episode %d: %s, skipping.", episode_id, msg)
            issues.append({"episode": episode_id, "type": "error", "reason": msg})
            return None, issues

    initial_closed_to_open_count = _initial_closed_to_open_transition_count(
        all_transitions, action_data, state_data
    )
    expected_transitions = (2 if is_give else 1) + initial_closed_to_open_count
    should_flag_multi_gripper = len(all_transitions) > expected_transitions
    if should_flag_multi_gripper:
        frames_list = [int(transition[0]) for transition in all_transitions]
        if is_give:
            msg = (
                f"{len(all_transitions)} gripper transitions at frames {frames_list}, "
                f"expected {expected_transitions} for give; using second-to-last for grasp, last for release"
            )
        else:
            msg = f"{len(all_transitions)} gripper transitions at frames {frames_list}, using LAST"
        logging.info("Episode %d: %s for stage 2.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "multi_gripper", "reason": msg, "frames": frames_list})

    if direct_give:
        release_action_frame, release_state_frame, _ = all_transitions[-1]
        boundary_2 = None
        boundary_3 = None
    elif is_give:
        grasp_action_frame, grasp_state_frame, _ = all_transitions[-2]
        release_action_frame, release_state_frame, _ = all_transitions[-1]
    else:
        grasp_action_frame, grasp_state_frame, _ = all_transitions[-1]

    if not direct_give:
        stage2_start_time = float(timestamps[grasp_action_frame]) - gripper_margin
        stage2_end_time = float(timestamps[min(grasp_state_frame, num_frames - 1)]) + gripper_margin
        boundary_2 = int(np.searchsorted(timestamps, stage2_start_time, side="left"))
        boundary_3 = int(np.searchsorted(timestamps, stage2_end_time, side="right"))
        boundary_2 = max(0, boundary_2)
        boundary_3 = min(num_frames, boundary_3)
        if boundary_3 - boundary_2 < min_frames:
            mid = (boundary_2 + boundary_3) // 2
            boundary_2 = max(0, mid - min_frames // 2)
            boundary_3 = boundary_2 + min_frames
            if boundary_3 > num_frames:
                boundary_3 = num_frames
                boundary_2 = max(0, boundary_3 - min_frames)

    if is_give:
        stage5_start_time = float(timestamps[release_action_frame]) - gripper_margin
        stage5_end_time = float(timestamps[min(release_state_frame, num_frames - 1)]) + gripper_margin
        boundary_5 = int(np.searchsorted(timestamps, stage5_start_time, side="left"))
        boundary_6 = int(np.searchsorted(timestamps, stage5_end_time, side="right"))
        boundary_5_min = min_frames if direct_give else boundary_3 + min_frames
        boundary_5 = max(boundary_5_min, boundary_5)
        boundary_6 = min(num_frames, boundary_6)
        if boundary_6 - boundary_5 < min_frames:
            mid = (boundary_5 + boundary_6) // 2
            boundary_5 = max(boundary_5_min, mid - min_frames // 2)
            boundary_6 = boundary_5 + min_frames
            if boundary_6 > num_frames:
                boundary_6 = num_frames
                boundary_5 = max(boundary_5_min, boundary_6 - min_frames)

    excluded_motion_indices = {7, 15, *BODY_JOINT_INDICES}
    arm_indices = [idx for idx in range(action_dim) if idx not in excluded_motion_indices]
    if not arm_indices:
        msg = "no arm action columns found"
        logging.warning("Episode %d: %s, skipping annotation.", episode_id, msg)
        issues.append({"episode": episode_id, "type": "error", "reason": msg})
        return None, issues

    arm_actions = action_data[:, arm_indices]
    window = max(round(fps), 2)
    change_threshold = 0.1

    max_range = np.zeros(num_frames)
    for column_idx in range(arm_actions.shape[1]):
        series = pd.Series(arm_actions[:, column_idx])
        roll_max = series.rolling(window, center=True, min_periods=1).max().values
        roll_min = series.rolling(window, center=True, min_periods=1).min().values
        max_range = np.maximum(max_range, roll_max - roll_min)

    stage0_upper_bound = boundary_5 if direct_give else boundary_2
    detected_boundary_1 = stage0_upper_bound
    dvt2_start_pose_frame = (
        _first_dvt2_start_pose_frame(state_data, stage0_upper_bound)
        if data_version == DATA_VERSION_DVT2
        else None
    )
    used_dvt2_start_pose = dvt2_start_pose_frame is not None
    if dvt2_start_pose_frame is not None:
        stage0_delay_frames = max(1, round(fps * DVT2_START_POSE_STAGE0_DELAY_SECONDS))
        detected_boundary_1 = min(dvt2_start_pose_frame + stage0_delay_frames, stage0_upper_bound - 1)
    else:
        for frame_idx in range(num_frames):
            if max_range[frame_idx] >= change_threshold:
                if state_data is not None:
                    state_1 = state_data[frame_idx, 1] if state_dim > 1 else np.nan
                    state_9 = state_data[frame_idx, 9] if state_dim > 9 else np.nan
                    if (not np.isnan(state_1) and state_1 > -0.4) or (
                        not np.isnan(state_9) and state_9 > -0.4
                    ):
                        detected_boundary_1 = frame_idx
                        break
                else:
                    detected_boundary_1 = frame_idx
                    break

    initial_gripper_open_frame = (
        _initial_closed_gripper_open_frame(action_data, state_data, stage0_upper_bound)
        if (is_pick or is_give)
        else None
    )
    if initial_gripper_open_frame is not None:
        open_delay_frames = max(1, round(float(fps) * INITIAL_CLOSED_GRIPPER_OPEN_STAGE0_DELAY_SECONDS))
        detected_boundary_1 = max(detected_boundary_1, initial_gripper_open_frame + open_delay_frames)

    end_threshold = 0.06
    detected_boundary_4 = boundary_3 if boundary_3 is not None else boundary_5
    dvt2_stage4_stable_frame = (
        _first_dvt2_stage4_stable_frame(
            max_range,
            boundary_3,
            fps,
            stable_seconds=DVT2_PLACE_STAGE4_STABLE_SECONDS if is_place else DVT2_STAGE4_STABLE_SECONDS,
        )
        if data_version == DATA_VERSION_DVT2 and not is_give
        else None
    )
    if dvt2_stage4_stable_frame is not None:
        detected_boundary_4 = dvt2_stage4_stable_frame
    else:
        for frame_idx in range(num_frames - 1, -1, -1):
            if max_range[frame_idx] >= end_threshold:
                detected_boundary_4 = frame_idx + 1
                break

    boundary_1 = detected_boundary_1
    boundary_4 = detected_boundary_4

    boundary_1_low = min_frames
    boundary_1_high = stage0_upper_bound - (1 if used_dvt2_start_pose else min_frames)
    if boundary_1_low <= boundary_1_high:
        boundary_1 = max(boundary_1_low, min(boundary_1_high, boundary_1))
    else:
        boundary_1 = max(1, stage0_upper_bound // 2)

    min_stage1_frames = max(round(fps * 2), min_frames)
    if not direct_give and not used_dvt2_start_pose and boundary_2 - boundary_1 < min_stage1_frames:
        boundary_1 = max(min_frames, boundary_2 - min_stage1_frames)

    if is_give:
        result = {
            "stage0_end": float(timestamps[min(boundary_1, num_frames - 1)]),
            "stage4_start": float(timestamps[min(boundary_5, num_frames - 1)]),
            "stage4_end": float(timestamps[min(boundary_6 - 1, num_frames - 1)]),
            "is_give": True,
        }
        if direct_give:
            result["direct_give"] = True
        else:
            result["stage2_start"] = float(timestamps[min(boundary_2, num_frames - 1)])
            result["stage2_end"] = float(timestamps[min(boundary_3 - 1, num_frames - 1)])
        return result, issues

    boundary_4_low = boundary_3 if data_version == DATA_VERSION_DVT2 else boundary_3 + min_frames
    boundary_4_high = num_frames - min_frames
    if boundary_4_low <= boundary_4_high:
        boundary_4 = max(boundary_4_low, min(boundary_4_high, boundary_4))
    else:
        boundary_4 = (boundary_3 + num_frames) // 2

    return {
        "stage0_end": float(timestamps[min(boundary_1, num_frames - 1)]),
        "stage2_start": float(timestamps[min(boundary_2, num_frames - 1)]),
        "stage2_end": float(timestamps[min(boundary_3 - 1, num_frames - 1)]),
        "stage4_start": float(timestamps[min(boundary_4, num_frames - 1)]),
    }, issues


def assign_subtask_states(timestamps, boundaries) -> list[int]:
    """Map timestamps to subtask states using computed boundaries."""
    if boundaries is None:
        return [0] * len(timestamps)

    if boundaries.get("equal_time"):
        num_stages = int(boundaries["num_stages"])
        stage_boundaries = np.asarray(boundaries["stage_boundaries"], dtype=np.float64)
        return [
            min(
                int(np.searchsorted(stage_boundaries, float(timestamp), side="right")),
                num_stages - 1,
            )
            for timestamp in timestamps
        ]

    is_give = boundaries.get("is_give", False)
    direct_give = bool(boundaries.get("direct_give", False))
    result = []
    for timestamp in timestamps:
        time_value = float(timestamp)
        if time_value < boundaries["stage0_end"]:
            result.append(0)
        elif direct_give:
            if time_value < boundaries["stage4_start"]:
                result.append(3)
            elif time_value <= boundaries["stage4_end"]:
                result.append(4)
            else:
                result.append(5)
        elif time_value < boundaries["stage2_start"]:
            result.append(1)
        elif time_value <= boundaries["stage2_end"]:
            result.append(2)
        elif is_give:
            if time_value < boundaries["stage4_start"]:
                result.append(3)
            elif time_value <= boundaries["stage4_end"]:
                result.append(4)
            else:
                result.append(5)
        elif time_value < boundaries["stage4_start"]:
            result.append(3)
        else:
            result.append(4)
    return result


def write_episode_csv(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    episode_id: int,
    out_path: Path,
    max_frames: int | None,
    downsample: int | None,
    overwrite: bool,
    force_recompute_stage: bool = False,
    data_version: str = DATA_VERSION_DVT1,
    fallback_stage_count: int = DEFAULT_FALLBACK_STAGE_COUNT,
) -> tuple[bool, dict | None, list[dict]]:
    """Write a precomputed CSV for one episode."""
    if out_path.exists() and not overwrite:
        return True, None, []

    parquet_path = dataset_root / meta.get_data_file_path(episode_id)
    if not parquet_path.is_file():
        return False, None, []

    columns, _, selected_columns = get_columns_info(meta)
    parquet_schema = pq.read_schema(parquet_path)
    has_parquet_stage = "subtask_state" in parquet_schema.names

    read_columns = list(selected_columns)
    use_existing_stage = has_parquet_stage and not force_recompute_stage
    if use_existing_stage and "subtask_state" not in read_columns:
        read_columns.append("subtask_state")

    data = read_episode_table(
        dataset_root,
        meta,
        episode_id,
        columns=read_columns,
    ).to_pandas()
    if max_frames is not None:
        data = data.head(max_frames)

    boundaries = None
    episode_issues: list[dict] = []
    existing_states = None

    if use_existing_stage and "subtask_state" in data.columns:
        existing_states = data["subtask_state"].values
        logging.debug("Episode %d: using existing subtask_state from parquet", episode_id)
    else:
        action_dim = meta.shapes.get("action", [0])[0] if "action" in meta.features else 0
        state_dim = meta.shapes.get("state", [0])[0] if "state" in meta.features else 0
        if action_dim > 0 and "action" in data.columns and len(data) > 1:
            fps = 1.0 / np.median(np.diff(data["timestamp"].values))
            action_array = series_to_2d(data["action"], action_dim)
            state_array = (
                series_to_2d(data["state"], state_dim) if state_dim > 0 and "state" in data.columns else None
            )
            task = ""
            if hasattr(meta, "episodes") and episode_id in meta.episodes:
                tasks = meta.episodes[episode_id].get("tasks", [])
                task = tasks[0] if tasks else ""
            boundaries, episode_issues = compute_subtask_boundaries(
                data["timestamp"].values,
                action_array,
                state_array,
                fps,
                episode_id=episode_id,
                task=task,
                data_version=data_version,
                fallback_stage_count=fallback_stage_count,
            )

    if downsample is not None and downsample > 1:
        if existing_states is not None:
            existing_states = existing_states[::downsample]
        data = data.iloc[::downsample].reset_index(drop=True)

    def _get_len(item) -> int:
        if item is None:
            return 0
        try:
            return len(item)
        except TypeError:
            return 0

    def _normalize_series(series, dim: int, column_name: str) -> np.ndarray:
        rows = []
        for item in series:
            row = [np.nan] * dim
            if item is not None:
                try:
                    values = list(item)
                except TypeError:
                    values = [item]
                if len(values) > dim:
                    values = values[:dim]
                row[: len(values)] = values
            rows.append(row)
        return normalize_gripper_columns(np.asarray(rows), column_name, data_version)

    data_arrays = []
    for column_name in selected_columns[1:]:
        fallback_dim = meta.shapes[column_name][0]
        series = data[column_name]
        actual_dim = max((_get_len(item) for item in series), default=0)
        dim = actual_dim if actual_dim > 0 else fallback_dim

        column_entry = next((entry for entry in columns if entry["key"] == column_name), None)
        if column_entry is not None and len(column_entry["value"]) != dim:
            column_entry["value"] = [f"{column_name}_{i}" for i in range(dim)]

        data_arrays.append(_normalize_series(series, dim, column_name))

    header = ["timestamp"]
    for column_entry in columns:
        header += column_entry["value"]

    rows = np.hstack((np.expand_dims(data["timestamp"], axis=1), *data_arrays)).tolist()

    if existing_states is not None:
        header.append("stage")
        max_stage = max(int(state) for state in existing_states) if len(existing_states) > 0 else 4
        max_stage = max(max_stage, 1)
        for row_idx, state in enumerate(existing_states):
            rows[row_idx].append(int(state) / float(max_stage))
    elif boundaries is not None:
        states = assign_subtask_states(data["timestamp"].values, boundaries)
        header.append("stage")
        num_stages = int(boundaries.get("num_stages", 6 if boundaries.get("is_give") else 5))
        max_stage = num_stages - 1
        for row_idx, state in enumerate(states):
            rows[row_idx].append(state / float(max_stage))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)

    return True, boundaries, episode_issues
