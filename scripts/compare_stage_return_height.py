#!/usr/bin/env python3
"""Create static original-vs-height-aligned H10W stage-return comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.preprocess.h10w_kinematics import (
    DEFAULT_H10W_DVT2_URDF,
    H10WDVT2ArmKinematics,
)
from lerobot.data_platform.precompute.preprocess.stage_return_height_alignment import (
    ARM_INDICES,
    STAGE_RETURN_HEIGHT_EPISODES,
    STAGE_RETURN_HEIGHT_META,
)

GROUP_LABELS = {
    "pick:left": "Pick · left arm",
    "pick:right": "Pick · right arm",
    "place:left": "Place · left arm",
    "place:right": "Place · right arm",
}
COLORS = {
    "original": "#2563EB",
    "aligned": "#EA580C",
    "target": "#4B5563",
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_path(root: Path, episode_index: int) -> Path:
    matches = list((root / "data").rglob(f"episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"Expected one parquet for episode {episode_index}, found {matches}")
    return matches[0]


def _episode_arrays(root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(_episode_path(root, episode_index), columns=["action", "subtask_state"])
    return (
        np.asarray(table["action"].to_pylist(), dtype=np.float64),
        np.asarray(table["subtask_state"].to_pylist(), dtype=np.int64),
    )


def _representatives(rows: list[dict], mode: str) -> list[dict]:
    result = []
    for group in GROUP_LABELS:
        group_rows = [row for row in rows if row["target_group"] == group]
        if mode == "worst_backtrack":
            selected = max(
                group_rows,
                key=lambda row: (row["raw_return_height_backtrack_m"], -row["episode_index"]),
            )
        else:
            median = float(np.median([row["rms_change_rad"] for row in group_rows]))
            selected = min(
                group_rows,
                key=lambda row: (abs(row["rms_change_rad"] - median), row["episode_index"]),
            )
        result.append(selected)
    return result


def _plot_trajectories(
    src_root: Path,
    aligned_root: Path,
    rows: list[dict],
    kinematics: dict[str, H10WDVT2ArmKinematics],
    fps: float,
    mode: str,
    output: Path,
) -> None:
    selected_rows = _representatives(rows, mode)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for axis, row in zip(axes.flat, selected_rows, strict=True):
        episode_index = int(row["episode_index"])
        side = row["active_side"]
        arm = list(ARM_INDICES[side])
        raw_action, raw_stages = _episode_arrays(src_root, episode_index)
        aligned_action, aligned_stages = _episode_arrays(aligned_root, episode_index)
        if not np.array_equal(raw_stages, aligned_stages):
            raise ValueError(f"Stage mismatch for episode {episode_index}")
        start = int(row["return_start_frame"])
        boundary = int(row["final_stage_start_frame"])
        time = (np.arange(start, len(raw_action)) - boundary) / fps
        raw_height = kinematics[side].positions(raw_action[start:, arm])[:, 2]
        aligned_height = kinematics[side].positions(aligned_action[start:, arm])[:, 2]
        target = float(row["target_height_m"])

        axis.plot(time, raw_height, color=COLORS["original"], linewidth=2.0, label="Original")
        axis.plot(
            time,
            aligned_height,
            color=COLORS["aligned"],
            linewidth=2.0,
            linestyle="--",
            label="Height-aligned",
        )
        axis.axhline(
            target,
            color=COLORS["target"],
            linewidth=1.5,
            linestyle=":",
            label="Target height",
        )
        axis.axvline(0.0, color="#9CA3AF", linewidth=1.0, linestyle=":")
        axis.set_title(f"{GROUP_LABELS[row['target_group']]} · Episode {episode_index}")
        axis.set_xlabel("Time relative to final-stage start (s)")
        axis.set_ylabel("TCP height in Torso frame (m)")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.02,
            0.04,
            f"raw backtrack: {row['raw_return_height_backtrack_m'] * 100:.1f} cm\n"
            f"aligned: {row['aligned_return_height_backtrack_m'] * 100:.3f} cm",
            transform=axis.transAxes,
            fontsize=9,
            color="#374151",
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    title = "Typical stage-return TCP height trajectories"
    if mode == "worst_backtrack":
        title = "Worst original height-backtrack episodes by task and arm"
    figure.suptitle(title, fontsize=16, y=0.985)
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_distributions(rows: list[dict], output: Path) -> None:
    groups = list(GROUP_LABELS)
    labels = [GROUP_LABELS[group].replace(" · ", "\n") for group in groups]
    metrics = [
        (
            "Boundary height error",
            "Absolute error from group target (cm)",
            [
                [
                    abs(row["state_boundary_height_error_m"]) * 100
                    for row in rows
                    if row["target_group"] == group
                ]
                for group in groups
            ],
        ),
        (
            "Action modification",
            "Episode RMS joint change (rad)",
            [[row["rms_change_rad"] for row in rows if row["target_group"] == group] for group in groups],
        ),
        (
            "Accompanying XY shift",
            "Maximum TCP XY shift per episode (cm)",
            [
                [row["max_tcp_xy_change_m"] * 100 for row in rows if row["target_group"] == group]
                for group in groups
            ],
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), constrained_layout=True)
    for axis, (title, ylabel, values) in zip(axes, metrics, strict=True):
        boxplot = axis.boxplot(
            values,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.5},
            whiskerprops={"color": "#64748B"},
            capprops={"color": "#64748B"},
        )
        for box in boxplot["boxes"]:
            box.set(facecolor="#DBEAFE", edgecolor="#2563EB", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle(
        "Full-dataset height alignment: error and modification distributions", fontsize=16, y=1.04
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_H10W_DVT2_URDF)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    provenance = json.loads((args.aligned_root / "meta" / STAGE_RETURN_HEIGHT_META).read_text())
    rows = _load_jsonl(args.aligned_root / "meta" / STAGE_RETURN_HEIGHT_EPISODES)
    kinematics = {side: H10WDVT2ArmKinematics(args.urdf, side=side) for side in ARM_INDICES}
    fps = float(json.loads((args.src_root / "meta" / "info.json").read_text())["fps"])
    _plot_trajectories(
        args.src_root,
        args.aligned_root,
        rows,
        kinematics,
        fps,
        "typical",
        args.output_dir / "height_trajectories_typical.png",
    )
    _plot_trajectories(
        args.src_root,
        args.aligned_root,
        rows,
        kinematics,
        fps,
        "worst_backtrack",
        args.output_dir / "height_trajectories_worst_backtrack.png",
    )
    _plot_distributions(rows, args.output_dir / "height_alignment_distributions.png")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "source_root": str(args.src_root),
                "aligned_root": str(args.aligned_root),
                "aligned_episodes": provenance["aligned_episodes"],
                "changed_frame_fraction": provenance["changed_frame_fraction"],
                "target_groups": provenance["target_groups"],
                "representatives": {
                    mode: [row["episode_index"] for row in _representatives(rows, mode)]
                    for mode in ("typical", "worst_backtrack")
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
