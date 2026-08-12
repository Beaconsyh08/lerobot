#!/usr/bin/env python3
"""Build reviewed comparison data for a stage-return-aligned LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_path(root: Path, episode_index: int) -> Path:
    info = _load_json(root / "meta" / "info.json")
    chunks_size = int(info.get("chunks_size") or 1000)
    template = info.get("data_path") or ("data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return root / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def _episode_arrays(root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        _episode_path(root, episode_index),
        columns=["action", "subtask_state"],
    )
    return (
        np.asarray(table["action"].to_pylist(), dtype=np.float64),
        np.asarray(table["subtask_state"].to_pylist(), dtype=np.int64),
    )


def _sample_indices(length: int, required: set[int], maximum: int = 150) -> list[int]:
    if length <= maximum:
        return list(range(length))
    stride = math.ceil(length / maximum)
    sampled = set(range(0, length, stride)) | required | {length - 1}
    return sorted(index for index in sampled if 0 <= index < length)


def _quantile(values: list[float], value: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), value))


def build_comparison(
    src_root: Path,
    aligned_root: Path,
    *,
    previous_aligned_root: Path | None = None,
    reference_representatives: list[dict] | None = None,
) -> dict:
    provenance = _load_json(aligned_root / "meta" / "preprocess_stage_return_alignment.json")
    episode_reports = _load_jsonl(aligned_root / "meta" / "preprocess_stage_return_alignment_episodes.jsonl")
    arm_indices = [int(value) for value in provenance["arm_indices"]]
    fps = float(_load_json(src_root / "meta" / "info.json")["fps"])

    episode_metrics = []
    for row in episode_reports:
        episode_metrics.append(
            {
                "episode_index": int(row["episode_index"]),
                "group": row["target_group"],
                "category": row["category"],
                "active_side": row["active_side"],
                "rms_change_rad": float(row["rms_change_rad"]),
                "max_abs_change_rad": float(row["max_abs_change_rad"]),
                "target_l2_change_rad": float(row["target_l2_change_rad"]),
                "raw_tail_max_step_rad": float(row["raw_tail_max_step_rad"]),
                "aligned_tail_max_step_rad": float(row["aligned_tail_max_step_rad"]),
                "raw_tail_second_difference_rms": float(row["raw_tail_second_difference_rms"]),
                "aligned_tail_second_difference_rms": float(row["aligned_tail_second_difference_rms"]),
                "changed_frames": int(row["changed_frames"]),
                "return_frames": int(row["return_frames"]),
                "final_stage_frames": int(row["final_stage_frames"]),
                "cubic_mix": float(row["cubic_mix"]),
            }
        )

    groups = sorted({row["group"] for row in episode_metrics})
    reference_by_group = {row["group"]: row for row in (reference_representatives or [])}
    group_summary = []
    representatives = []
    representative_curves = []
    for group in groups:
        rows = [row for row in episode_reports if row["target_group"] == group]
        rms_values = [float(row["rms_change_rad"]) for row in rows]
        median_rms = _quantile(rms_values, 0.5)
        reference = reference_by_group.get(group)
        if reference is None:
            representative = min(
                rows,
                key=lambda row: (
                    abs(float(row["rms_change_rad"]) - median_rms),
                    row["episode_index"],
                ),
            )
        else:
            representative = next(
                row for row in rows if int(row["episode_index"]) == int(reference["episode_index"])
            )
        episode_index = int(representative["episode_index"])
        raw_action, stages = _episode_arrays(src_root, episode_index)
        aligned_action, aligned_stages = _episode_arrays(aligned_root, episode_index)
        if not np.array_equal(stages, aligned_stages):
            raise ValueError(f"Stage mismatch for representative episode {episode_index}")

        if reference is None:
            side_indices = list(range(7)) if representative["active_side"] == "left" else list(range(8, 15))
            joint_index = max(
                side_indices,
                key=lambda index: float(np.max(np.abs(aligned_action[:, index] - raw_action[:, index]))),
            )
        else:
            joint_index = int(reference["joint"].removeprefix("action[").removesuffix("]"))
        previous_action = None
        if previous_aligned_root is not None:
            previous_action, previous_stages = _episode_arrays(previous_aligned_root, episode_index)
            if not np.array_equal(stages, previous_stages):
                raise ValueError(f"Previous-stage mismatch for representative episode {episode_index}")
        start = int(representative["return_start_frame"])
        boundary = int(representative["final_stage_start_frame"])
        indices = _sample_indices(len(raw_action), {start, boundary, boundary - 1})
        target_position = arm_indices.index(joint_index)
        target = float(provenance["target_groups"][group]["arm_target"][target_position])
        for frame_index in indices:
            common = {
                "group": group,
                "episode_index": episode_index,
                "joint": f"action[{joint_index}]",
                "frame": frame_index,
                "relative_frame": frame_index - boundary,
                "relative_time_s": (frame_index - boundary) / fps,
                "return_progress": (frame_index - start) / (boundary - start),
                "stage": int(stages[frame_index]),
            }
            representative_curves.append(
                {
                    **common,
                    "series": "Original",
                    "line_style": "solid",
                    "action_rad": float(raw_action[frame_index, joint_index]),
                }
            )
            if previous_action is not None:
                representative_curves.append(
                    {
                        **common,
                        "series": "Previous aligned",
                        "line_style": "dotted",
                        "action_rad": float(previous_action[frame_index, joint_index]),
                    }
                )
            representative_curves.append(
                {
                    **common,
                    "series": "Direct aligned",
                    "line_style": "dashed",
                    "action_rad": float(aligned_action[frame_index, joint_index]),
                }
            )
            if frame_index >= start:
                representative_curves.append(
                    {
                        **common,
                        "series": "Fixed target",
                        "line_style": "dotted",
                        "action_rad": target,
                    }
                )
        representatives.append(
            {
                "group": group,
                "episode_index": episode_index,
                "joint": f"action[{joint_index}]",
                "return_start_frame": start,
                "final_stage_start_frame": boundary,
                "target_rad": target,
                "rms_change_rad": float(representative["rms_change_rad"]),
                "max_abs_change_rad": float(representative["max_abs_change_rad"]),
            }
        )
        group_summary.append(
            {
                "group": group,
                "episodes": len(rows),
                "rms_change_median_rad": median_rms,
                "rms_change_p90_rad": _quantile(rms_values, 0.9),
                "max_change_p90_rad": _quantile([float(row["max_abs_change_rad"]) for row in rows], 0.9),
                "target_l2_median_rad": _quantile([float(row["target_l2_change_rad"]) for row in rows], 0.5),
                "aligned_step_p90_rad": _quantile(
                    [float(row["aligned_tail_max_step_rad"]) for row in rows], 0.9
                ),
            }
        )

    worst = max(episode_reports, key=lambda row: float(row["max_abs_change_rad"]))
    worst_episode = int(worst["episode_index"])
    raw_action, stages = _episode_arrays(src_root, worst_episode)
    aligned_action, _ = _episode_arrays(aligned_root, worst_episode)
    start = int(worst["return_start_frame"])
    boundary = int(worst["final_stage_start_frame"])
    worst_heatmap = []
    for frame_index in range(start, len(raw_action)):
        for joint_index in arm_indices:
            delta = float(aligned_action[frame_index, joint_index] - raw_action[frame_index, joint_index])
            worst_heatmap.append(
                {
                    "episode_index": worst_episode,
                    "group": worst["target_group"],
                    "relative_frame": frame_index - boundary,
                    "relative_time_s": (frame_index - boundary) / fps,
                    "joint": f"action[{joint_index}]",
                    "stage": int(stages[frame_index]),
                    "delta_rad": delta,
                    "abs_delta_rad": abs(delta),
                }
            )

    top_episodes = sorted(
        episode_metrics,
        key=lambda row: (row["max_abs_change_rad"], row["rms_change_rad"]),
        reverse=True,
    )[:15]
    return {
        "summary": {
            "source_episodes": int(provenance["source_episodes"]),
            "source_frames": int(provenance["source_frames"]),
            "aligned_episodes": int(provenance["aligned_episodes"]),
            "changed_frames": int(provenance["changed_frames"]),
            "changed_frame_fraction": float(provenance["changed_frame_fraction"]),
            "episode_rms_change_rad_median": float(provenance["episode_rms_change_rad_median"]),
            "episode_rms_change_rad_p90": float(provenance["episode_rms_change_rad_p90"]),
            "max_abs_change_rad": float(provenance["max_abs_change_rad"]),
            "aligned_tail_max_step_rad_max": float(provenance["aligned_tail_max_step_rad_max"]),
            "step_limited_episodes": int(provenance["step_limited_episodes"]),
            "target_statistic": provenance["target_statistic"],
            "trajectory_method": provenance.get("trajectory_method", "unknown"),
            "comparison_basis": (
                "previous_report_representatives" if reference_representatives else "current_group_median_rms"
            ),
            "includes_previous_aligned": previous_aligned_root is not None,
            "worst_episode": worst_episode,
            "worst_group": worst["target_group"],
        },
        "group_summary": group_summary,
        "episode_metrics": episode_metrics,
        "representatives": representatives,
        "representative_curves": representative_curves,
        "worst_heatmap": worst_heatmap,
        "top_episodes": top_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--previous-aligned-root", type=Path)
    parser.add_argument("--reference-analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference_representatives = None
    if args.reference_analysis is not None:
        reference_representatives = _load_json(args.reference_analysis)["representatives"]
    comparison = build_comparison(
        args.src_root,
        args.aligned_root,
        previous_aligned_root=args.previous_aligned_root,
        reference_representatives=reference_representatives,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
    )
    pd.DataFrame(comparison["episode_metrics"]).to_csv(args.output_dir / "episode_metrics.csv", index=False)
    pd.DataFrame(comparison["representative_curves"]).to_csv(
        args.output_dir / "representative_curves.csv", index=False
    )
    pd.DataFrame(comparison["worst_heatmap"]).to_csv(
        args.output_dir / "worst_episode_heatmap.csv", index=False
    )
    print(json.dumps(comparison["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
