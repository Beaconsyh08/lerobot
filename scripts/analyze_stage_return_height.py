#!/usr/bin/env python3
"""Analyze H10W DVT2 TCP heights at final-stage return boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.preprocess.common import parquet_paths
from lerobot.data_platform.precompute.preprocess.h10w_kinematics import (
    DEFAULT_H10W_DVT2_URDF,
    H10WDVT2ArmKinematics,
)
from lerobot.data_platform.precompute.preprocess.stage_return_alignment import (
    ACTION_COLUMN,
    STAGE_COLUMN,
    STATE_COLUMN,
    _active_side,
    _load_tasks,
    _tail_interval,
    _task_category,
)

ARM_INDICES = {
    "left": tuple(range(7)),
    "right": tuple(range(8, 15)),
}


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _backtrack_m(heights: np.ndarray, target: float) -> float:
    distance = np.abs(np.asarray(heights, dtype=np.float64) - target)
    return float(np.maximum(np.diff(distance), 0.0).sum()) if len(distance) > 1 else 0.0


def analyze(dataset_root: Path, urdf_path: Path) -> dict:
    dataset_root = dataset_root.resolve()
    urdf_path = urdf_path.resolve()
    tasks = _load_tasks(dataset_root)
    kinematics = {side: H10WDVT2ArmKinematics(urdf_path, side=side) for side in ARM_INDICES}
    episodes: list[dict] = []
    skipped: list[dict] = []
    columns = ["episode_index", "task_index", STAGE_COLUMN, STATE_COLUMN, ACTION_COLUMN]

    for path in parquet_paths(dataset_root):
        table = pq.read_table(path, columns=columns)
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        task_values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        stage_values = np.asarray(table[STAGE_COLUMN].to_pylist(), dtype=np.int64)
        state_values = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float64)
        action_values = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float64)
        for episode_id in dict.fromkeys(episode_values.tolist()):
            positions = np.flatnonzero(episode_values == episode_id)
            episode_tasks = np.unique(task_values[positions])
            if len(episode_tasks) != 1:
                skipped.append({"episode_index": int(episode_id), "reason": "multiple task indices"})
                continue
            task = tasks.get(int(episode_tasks[0]), "")
            category = _task_category(task)
            if category is None:
                skipped.append({"episode_index": int(episode_id), "reason": f"unsupported task {task!r}"})
                continue
            stages = stage_values[positions]
            state = state_values[positions]
            action = action_values[positions]
            try:
                start, boundary = _tail_interval(stages, category)
                side = _active_side(action)
            except ValueError as exc:
                skipped.append({"episode_index": int(episode_id), "reason": str(exc)})
                continue
            arm = list(ARM_INDICES[side])
            fk = kinematics[side]
            state_boundary_position = fk.pose(state[boundary, arm])[:3, 3]
            action_positions = fk.positions(action[start:, arm])
            episodes.append(
                {
                    "episode_index": int(episode_id),
                    "task": task,
                    "category": category,
                    "active_side": side,
                    "group": f"{category}:{side}",
                    "frames": int(len(positions)),
                    "return_start_frame": int(start),
                    "final_stage_start_frame": int(boundary),
                    "state_boundary_tcp_xyz_m": state_boundary_position.tolist(),
                    "action_start_height_m": float(action_positions[0, 2]),
                    "action_boundary_height_m": float(action_positions[boundary - start, 2]),
                    "action_final_height_m": float(action_positions[-1, 2]),
                    "action_return_heights_m": action_positions[: boundary - start + 1, 2].tolist(),
                }
            )

    groups = sorted({row["group"] for row in episodes})
    targets = {
        group: float(
            np.median([row["state_boundary_tcp_xyz_m"][2] for row in episodes if row["group"] == group])
        )
        for group in groups
    }
    for row in episodes:
        target = targets[row["group"]]
        return_heights = np.asarray(row.pop("action_return_heights_m"), dtype=np.float64)
        row["target_height_m"] = target
        row["state_boundary_height_error_m"] = float(row["state_boundary_tcp_xyz_m"][2] - target)
        row["action_boundary_height_error_m"] = float(row["action_boundary_height_m"] - target)
        row["raw_return_backtrack_m"] = _backtrack_m(return_heights, target)
        row["raw_return_height_min_m"] = float(return_heights.min())
        row["raw_return_height_max_m"] = float(return_heights.max())

    group_summary = []
    representatives = []
    for group in groups:
        rows = [row for row in episodes if row["group"] == group]
        errors = [abs(row["state_boundary_height_error_m"]) for row in rows]
        backtracks = [row["raw_return_backtrack_m"] for row in rows]
        median_error = float(np.median(errors))
        representative = min(
            rows,
            key=lambda row: (
                abs(abs(row["state_boundary_height_error_m"]) - median_error),
                row["episode_index"],
            ),
        )
        worst = max(rows, key=lambda row: (row["raw_return_backtrack_m"], -row["episode_index"]))
        representatives.append(
            {
                "group": group,
                "median_change_episode": representative["episode_index"],
                "worst_backtrack_episode": worst["episode_index"],
            }
        )
        group_summary.append(
            {
                "group": group,
                "episodes": len(rows),
                "target_height_m": targets[group],
                "state_boundary_height_m": _quantiles([row["state_boundary_tcp_xyz_m"][2] for row in rows]),
                "abs_state_boundary_error_m": _quantiles(errors),
                "action_boundary_height_m": _quantiles([row["action_boundary_height_m"] for row in rows]),
                "raw_return_backtrack_m": _quantiles(backtracks),
            }
        )

    return {
        "dataset_root": str(dataset_root),
        "urdf_path": str(urdf_path),
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "height_frame": "Torso",
        "tcp_links": {side: fk.tcp_link for side, fk in kinematics.items()},
        "target_statistic": "median state TCP height at final-stage start, by task category and active arm",
        "eligible_episodes": len(episodes),
        "skipped_episodes": skipped,
        "target_heights_m": targets,
        "group_summary": group_summary,
        "representatives": representatives,
        "episodes": sorted(episodes, key=lambda row: row["episode_index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_H10W_DVT2_URDF)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.dataset_root, args.urdf)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
