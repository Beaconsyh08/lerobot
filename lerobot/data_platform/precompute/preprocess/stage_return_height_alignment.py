"""Align final-stage return actions to robust H10W DVT2 TCP heights.

Only the active arm is changed.  Pick/place use the stage 3 -> 4 interval;
give uses stage 4 -> 5.  State, images, grippers, inactive-arm actions, and
stage labels are preserved in a new sibling dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    copy_meta_files,
    copy_sidecar_dirs,
    default_preprocess_path,
    emit,
    ensure_output_root,
    load_json,
    parquet_paths,
    validate_dataset_root,
    write_json,
    write_jsonl,
)
from lerobot.data_platform.precompute.preprocess.h10w_kinematics import (
    DEFAULT_H10W_DVT2_URDF,
    H10WDVT2ArmKinematics,
)
from lerobot.data_platform.precompute.preprocess.stage_return_alignment import (
    ACTION_COLUMN,
    CATEGORY_STAGES,
    DEFAULT_GRIPPER_INDICES,
    STAGE_COLUMN,
    STATE_COLUMN,
    _active_side,
    _column_stats,
    _load_tasks,
    _max_step,
    _replace_column,
    _resolve_workers,
    _rewrite_stats,
    _second_difference_rms,
    _tail_interval,
    _task_category,
)

ARM_INDICES = {
    "left": tuple(range(7)),
    "right": tuple(range(8, 15)),
}
STAGE_RETURN_HEIGHT_META = "preprocess_stage_return_height_alignment.json"
STAGE_RETURN_HEIGHT_EPISODES = "preprocess_stage_return_height_alignment_episodes.jsonl"


def _height_backtrack(heights: np.ndarray, target: float) -> float:
    distance = np.abs(np.asarray(heights, dtype=np.float64) - target)
    return float(np.maximum(np.diff(distance), 0.0).sum()) if len(distance) > 1 else 0.0


def _align_episode_height(
    action: np.ndarray,
    *,
    start: int,
    boundary: int,
    target_height: float,
    side: str,
    kinematics: H10WDVT2ArmKinematics,
    height_tolerance_m: float,
    max_step_rad: float,
) -> tuple[np.ndarray, dict]:
    action = np.asarray(action, dtype=np.float64)
    if not 0 <= start < boundary < len(action):
        raise ValueError(f"Invalid tail interval [{start}, {boundary}) for {len(action)} frames")
    arm = np.asarray(ARM_INDICES[side], dtype=np.int64)
    original_arm = action[:, arm].copy()
    aligned = action.copy()
    start_height = float(kinematics.pose(original_arm[start])[2, 3])

    progress = np.arange(boundary - start + 1, dtype=np.float64) / (boundary - start)
    weights = 3.0 * progress**2 - 2.0 * progress**3
    desired_return = start_height + weights * (target_height - start_height)
    desired_tail = np.concatenate(
        [desired_return, np.full(len(action) - boundary - 1, target_height, dtype=np.float64)]
    )

    for relative_index, frame_index in enumerate(range(start, len(action))):
        reference = original_arm[frame_index]
        current_height = float(kinematics.pose(reference)[2, 3])
        desired_height = float(desired_tail[relative_index])
        if abs(current_height - desired_height) <= height_tolerance_m:
            solution = reference
        else:
            solution = kinematics.project_to_height(
                reference,
                desired_height,
                tolerance_m=height_tolerance_m / 4.0,
            )
        aligned[frame_index, arm] = solution

    raw_positions = kinematics.positions(original_arm[start:])
    aligned_positions = kinematics.positions(aligned[start:, arm])
    height_error = np.abs(aligned_positions[:, 2] - desired_tail)
    if float(height_error.max()) > height_tolerance_m:
        raise ValueError(f"TCP height residual {height_error.max():.6g} m exceeds tolerance")
    if _height_backtrack(aligned_positions[: boundary - start + 1, 2], target_height) > (
        height_tolerance_m * (boundary - start)
    ):
        raise ValueError("Aligned TCP height reverses away from the target")

    raw_step = _max_step(original_arm[start:])
    aligned_step = _max_step(aligned[start:, arm])
    allowed_step = max(float(max_step_rad), raw_step)
    step_validation_slack_rad = 1e-3
    if aligned_step > allowed_step + step_validation_slack_rad:
        raise ValueError(f"Aligned max step {aligned_step:.6f} rad exceeds allowed {allowed_step:.6f} rad")

    changed = aligned[:, arm] - original_arm
    xy_change = np.linalg.norm(aligned_positions[:, :2] - raw_positions[:, :2], axis=1)
    return aligned, {
        "return_start_frame": int(start),
        "final_stage_start_frame": int(boundary),
        "return_frames": int(boundary - start),
        "final_stage_frames": int(len(action) - boundary),
        "target_height_m": float(target_height),
        "action_start_height_m": start_height,
        "raw_action_boundary_height_m": float(raw_positions[boundary - start, 2]),
        "aligned_action_boundary_height_m": float(aligned_positions[boundary - start, 2]),
        "raw_return_height_backtrack_m": _height_backtrack(
            raw_positions[: boundary - start + 1, 2], target_height
        ),
        "aligned_return_height_backtrack_m": _height_backtrack(
            aligned_positions[: boundary - start + 1, 2], target_height
        ),
        "max_height_error_m": float(height_error.max()),
        "changed_frames": int(np.count_nonzero(np.any(np.abs(changed) > 1e-12, axis=1))),
        "rms_change_rad": float(np.sqrt(np.mean(np.square(changed)))),
        "max_abs_change_rad": float(np.abs(changed).max()),
        "max_tcp_xy_change_m": float(xy_change.max()),
        "raw_tail_max_step_rad": raw_step,
        "aligned_tail_max_step_rad": aligned_step,
        "allowed_tail_max_step_rad": allowed_step,
        "raw_tail_second_difference_rms": _second_difference_rms(original_arm[start:]),
        "aligned_tail_second_difference_rms": _second_difference_rms(aligned[start:, arm]),
    }


def _scan_dataset(
    root: Path,
    *,
    urdf_path: Path,
    target_statistic: str,
    min_group_episodes: int,
    height_tolerance_m: float,
    max_step_rad: float,
) -> tuple[dict[int, dict], dict[str, float], list[dict], list[dict], int]:
    tasks = _load_tasks(root)
    kinematics = {side: H10WDVT2ArmKinematics(urdf_path, side=side) for side in ARM_INDICES}
    candidates: dict[int, dict] = {}
    skipped: list[dict] = []
    total_frames = 0
    columns = ["episode_index", "task_index", STAGE_COLUMN, STATE_COLUMN, ACTION_COLUMN]

    for path in parquet_paths(root):
        table = pq.read_table(path, columns=columns)
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        task_values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        stage_values = np.asarray(table[STAGE_COLUMN].to_pylist(), dtype=np.int64)
        state_values = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float64)
        action_values = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float64)
        for episode_id in dict.fromkeys(episode_values.tolist()):
            positions = np.flatnonzero(episode_values == episode_id)
            total_frames += len(positions)
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
            group = f"{category}:{side}"
            arm = list(ARM_INDICES[side])
            boundary_state_height = float(kinematics[side].pose(state[boundary, arm])[2, 3])
            candidates[int(episode_id)] = {
                "episode_index": int(episode_id),
                "relative_path": str(path.relative_to(root / "data")),
                "task": task,
                "category": category,
                "active_side": side,
                "group": group,
                "start": start,
                "boundary": boundary,
                "boundary_state_height_m": boundary_state_height,
                "action": action,
            }

    targets: dict[str, float] = {}
    for group in sorted({item["group"] for item in candidates.values()}):
        values = np.asarray(
            [item["boundary_state_height_m"] for item in candidates.values() if item["group"] == group],
            dtype=np.float64,
        )
        if len(values) < min_group_episodes:
            for episode_id, item in list(candidates.items()):
                if item["group"] == group:
                    skipped.append(
                        {
                            "episode_index": episode_id,
                            "reason": f"group {group} has {len(values)} episodes; requires {min_group_episodes}",
                        }
                    )
                    del candidates[episode_id]
            continue
        targets[group] = float(np.median(values) if target_statistic == "median" else np.mean(values))

    episode_reports = []
    failed_ids = []
    for episode_id, item in sorted(candidates.items()):
        try:
            _, metrics = _align_episode_height(
                item["action"],
                start=item["start"],
                boundary=item["boundary"],
                target_height=targets[item["group"]],
                side=item["active_side"],
                kinematics=kinematics[item["active_side"]],
                height_tolerance_m=height_tolerance_m,
                max_step_rad=max_step_rad,
            )
        except ValueError as exc:
            skipped.append({"episode_index": episode_id, "reason": str(exc)})
            failed_ids.append(episode_id)
            continue
        episode_reports.append(
            {
                "episode_index": episode_id,
                "task": item["task"],
                "category": item["category"],
                "active_side": item["active_side"],
                "target_group": item["group"],
                "state_boundary_height_m": item["boundary_state_height_m"],
                "state_boundary_height_error_m": (item["boundary_state_height_m"] - targets[item["group"]]),
                **metrics,
            }
        )
    for episode_id in failed_ids:
        del candidates[episode_id]

    plans = {
        episode_id: {
            key: value for key, value in item.items() if key not in {"action", "boundary_state_height_m"}
        }
        for episode_id, item in candidates.items()
    }
    for plan in plans.values():
        plan["target_height_m"] = targets[plan["group"]]
    return (
        plans,
        targets,
        episode_reports,
        sorted(skipped, key=lambda row: row["episode_index"]),
        total_frames,
    )


def _rewrite_parquet_worker(args: tuple) -> dict[int, dict]:
    (
        src_text,
        dst_text,
        plan_rows,
        urdf_text,
        height_tolerance_m,
        max_step_rad,
    ) = args
    src, dst = Path(src_text), Path(dst_text)
    plans = {int(row["episode_index"]): row for row in plan_rows}
    kinematics = {side: H10WDVT2ArmKinematics(urdf_text, side=side) for side in ARM_INDICES}
    table = pq.read_table(src)
    episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float64)
    stats = {}
    for episode_id in dict.fromkeys(episode_values.tolist()):
        positions = np.flatnonzero(episode_values == episode_id)
        episode_action = action[positions]
        plan = plans.get(int(episode_id))
        if plan is not None:
            episode_action, _ = _align_episode_height(
                episode_action,
                start=int(plan["start"]),
                boundary=int(plan["boundary"]),
                target_height=float(plan["target_height_m"]),
                side=plan["active_side"],
                kinematics=kinematics[plan["active_side"]],
                height_tolerance_m=height_tolerance_m,
                max_step_rad=max_step_rad,
            )
            action[positions] = episode_action
        stats[int(episode_id)] = {ACTION_COLUMN: _column_stats(episode_action.astype(np.float32))}
    output = _replace_column(table, ACTION_COLUMN, action.astype(np.float32).tolist())
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, dst)
    return stats


def _summary_metrics(episode_reports: list[dict], total_frames: int) -> dict:
    changed_frames = sum(int(row["changed_frames"]) for row in episode_reports)
    if not episode_reports:
        return {"aligned_episodes": 0, "changed_frames": 0, "changed_frame_fraction": 0.0}
    return {
        "aligned_episodes": len(episode_reports),
        "changed_frames": changed_frames,
        "changed_frame_fraction": changed_frames / total_frames if total_frames else 0.0,
        "episode_rms_change_rad_median": float(np.median([row["rms_change_rad"] for row in episode_reports])),
        "episode_rms_change_rad_p90": float(
            np.quantile([row["rms_change_rad"] for row in episode_reports], 0.9)
        ),
        "max_abs_change_rad": float(max(row["max_abs_change_rad"] for row in episode_reports)),
        "max_tcp_xy_change_m_median": float(
            np.median([row["max_tcp_xy_change_m"] for row in episode_reports])
        ),
        "max_tcp_xy_change_m_p90": float(
            np.quantile([row["max_tcp_xy_change_m"] for row in episode_reports], 0.9)
        ),
        "max_tcp_xy_change_m": float(max(row["max_tcp_xy_change_m"] for row in episode_reports)),
        "max_height_error_m": float(max(row["max_height_error_m"] for row in episode_reports)),
        "raw_return_height_backtrack_m_total": float(
            sum(row["raw_return_height_backtrack_m"] for row in episode_reports)
        ),
        "aligned_return_height_backtrack_m_total": float(
            sum(row["aligned_return_height_backtrack_m"] for row in episode_reports)
        ),
        "aligned_tail_max_step_rad_max": float(
            max(row["aligned_tail_max_step_rad"] for row in episode_reports)
        ),
        "raw_tail_max_step_rad_max": float(max(row["raw_tail_max_step_rad"] for row in episode_reports)),
    }


def _validate_output(
    src_root: Path,
    out_root: Path,
    plans: dict[int, dict],
    *,
    urdf_path: Path,
    height_tolerance_m: float,
) -> dict:
    kinematics = {side: H10WDVT2ArmKinematics(urdf_path, side=side) for side in ARM_INDICES}
    validated_episodes = validated_frames = 0
    columns = ["episode_index", STATE_COLUMN, STAGE_COLUMN, ACTION_COLUMN]
    for src in parquet_paths(src_root):
        dst = out_root / "data" / src.relative_to(src_root / "data")
        if not dst.is_file():
            raise FileNotFoundError(f"Missing output parquet: {dst}")
        if pq.ParquetFile(src).schema_arrow != pq.ParquetFile(dst).schema_arrow:
            raise ValueError(f"Schema changed for {src.relative_to(src_root)}")
        source = pq.read_table(src, columns=columns)
        output = pq.read_table(dst, columns=columns)
        if source.num_rows != output.num_rows:
            raise ValueError(f"Row count changed for {src.relative_to(src_root)}")
        source_ids = np.asarray(source["episode_index"].to_pylist(), dtype=np.int64)
        output_ids = np.asarray(output["episode_index"].to_pylist(), dtype=np.int64)
        source_state = np.asarray(source[STATE_COLUMN].to_pylist())
        output_state = np.asarray(output[STATE_COLUMN].to_pylist())
        source_stages = np.asarray(source[STAGE_COLUMN].to_pylist())
        output_stages = np.asarray(output[STAGE_COLUMN].to_pylist())
        source_action = np.asarray(source[ACTION_COLUMN].to_pylist())
        output_action = np.asarray(output[ACTION_COLUMN].to_pylist())
        if not np.array_equal(source_ids, output_ids):
            raise ValueError(f"episode_index changed for {src.relative_to(src_root)}")
        if not np.array_equal(source_state, output_state):
            raise ValueError(f"state changed for {src.relative_to(src_root)}")
        if not np.array_equal(source_stages, output_stages):
            raise ValueError(f"stage labels changed for {src.relative_to(src_root)}")
        if not np.array_equal(
            source_action[:, list(DEFAULT_GRIPPER_INDICES)], output_action[:, list(DEFAULT_GRIPPER_INDICES)]
        ):
            raise ValueError(f"gripper actions changed for {src.relative_to(src_root)}")

        for episode_id in dict.fromkeys(source_ids.tolist()):
            positions = np.flatnonzero(source_ids == episode_id)
            validated_episodes += 1
            validated_frames += len(positions)
            plan = plans.get(int(episode_id))
            if plan is None:
                if not np.array_equal(source_action[positions], output_action[positions]):
                    raise ValueError(f"Skipped episode {episode_id} was modified")
                continue
            source_episode = source_action[positions]
            output_episode = output_action[positions]
            start, boundary = int(plan["start"]), int(plan["boundary"])
            side = plan["active_side"]
            active = list(ARM_INDICES[side])
            inactive = list(ARM_INDICES["right" if side == "left" else "left"])
            if not np.array_equal(source_episode[:start], output_episode[:start]):
                raise ValueError(f"Episode {episode_id} changed before return-stage start")
            if not np.array_equal(source_episode[:, inactive], output_episode[:, inactive]):
                raise ValueError(f"Episode {episode_id} changed inactive-arm actions")
            heights = kinematics[side].positions(output_episode[start:, active])[:, 2]
            target = float(plan["target_height_m"])
            if np.max(np.abs(heights[boundary - start :] - target)) > height_tolerance_m * 2.0:
                raise ValueError(f"Episode {episode_id} does not hold target TCP height")
            if _height_backtrack(heights[: boundary - start + 1], target) > (
                height_tolerance_m * (boundary - start)
            ):
                raise ValueError(f"Episode {episode_id} TCP height reverses away from target")
    return {"episodes": validated_episodes, "frames": validated_frames}


def run_stage_return_height_alignment(
    src_root: Path,
    out_root: Path | None = None,
    *,
    urdf_path: Path = DEFAULT_H10W_DVT2_URDF,
    target_statistic: str = "median",
    min_group_episodes: int = 5,
    height_tolerance_m: float = 1e-5,
    max_step_rad: float = 0.1,
    workers: int | None = 0,
    dry_run: bool = True,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    urdf_path = Path(urdf_path).resolve()
    if target_statistic not in {"median", "mean"}:
        raise ValueError("target_statistic must be 'median' or 'mean'")
    if min_group_episodes < 1 or height_tolerance_m <= 0 or max_step_rad <= 0:
        raise ValueError("min_group_episodes, height_tolerance_m, and max_step_rad must be positive")
    info = load_json(src_root / "meta" / "info.json")
    if info.get("robot_type") != "h10_w":
        raise ValueError(f"Expected robot_type 'h10_w', got {info.get('robot_type')!r}")
    for column in (ACTION_COLUMN, STATE_COLUMN, STAGE_COLUMN):
        if column not in (info.get("features") or {}):
            raise ValueError(f"Dataset metadata does not define {column!r}")

    target_root = ensure_output_root(
        out_root or default_preprocess_path(src_root, "stage_return_height"), dry_run
    )
    plans, targets, episode_reports, skipped, total_frames = _scan_dataset(
        src_root,
        urdf_path=urdf_path,
        target_statistic=target_statistic,
        min_group_episodes=min_group_episodes,
        height_tolerance_m=height_tolerance_m,
        max_step_rad=max_step_rad,
    )
    if not plans:
        raise ValueError(f"No eligible episodes to align; skipped={skipped}")

    group_counts = {
        group: sum(plan["group"] == group for plan in plans.values()) for group in sorted(targets)
    }
    group_errors = {
        group: [
            abs(row["state_boundary_height_error_m"])
            for row in episode_reports
            if row["target_group"] == group
        ]
        for group in sorted(targets)
    }
    summary = {
        "target_statistic": target_statistic,
        "target_source": "state TCP height at final-stage start",
        "target_groups": {
            group: {
                "episodes": group_counts.get(group, 0),
                "target_height_m": targets[group],
                "boundary_abs_error_median": float(np.median(group_errors[group])),
                "boundary_abs_error_p90": float(np.quantile(group_errors[group], 0.9)),
            }
            for group in sorted(targets)
            if group_counts.get(group, 0) > 0
        },
        "urdf_path": str(urdf_path),
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "height_frame": "Torso",
        "tcp_links": {"left": "LeftGripperTCP", "right": "RightGripperTCP"},
        "height_tolerance_m": height_tolerance_m,
        "max_step_rad": max_step_rad,
        "trajectory_method": "smoothstep_monotone_tcp_height_with_minimum_norm_joint_projection",
        "source_episodes": int(info.get("total_episodes") or len(plans) + len(skipped)),
        "source_frames": int(info.get("total_frames") or total_frames),
        "skipped_episodes": skipped,
        "preserved_fields": [STATE_COLUMN, "images", "gripper action", "inactive-arm action", STAGE_COLUMN],
        "category_intervals": {
            category: {"return_stage": stages[0], "final_stage": stages[1]}
            for category, stages in CATEGORY_STAGES.items()
        },
        **_summary_metrics(episode_reports, total_frames),
    }
    result = PreprocessResult(
        op="stage_return_height_alignment",
        src_roots=[src_root],
        out_root=target_root,
        repo_id=f"local/{target_root.name}",
        total_episodes=int(info.get("total_episodes") or len(plans) + len(skipped)),
        total_frames=int(info.get("total_frames") or total_frames),
        dry_run=dry_run,
        summary=summary,
    )
    emit(progress_callback, status="running", current=0, total=len(plans), message=f"Plan: {summary}")
    if dry_run:
        emit(progress_callback, status="done", current=0, total=len(plans), message="Dry run complete")
        return result

    paths = parquet_paths(src_root)
    worker_count = _resolve_workers(workers, len(paths))
    try:
        (target_root / "data").mkdir(parents=True)
        copy_meta_files(src_root, target_root)
        copy_sidecar_dirs(src_root, target_root)
        plans_by_path: dict[str, list[dict]] = {}
        for plan in plans.values():
            plans_by_path.setdefault(plan["relative_path"], []).append(plan)
        tasks = [
            (
                str(path),
                str(target_root / "data" / path.relative_to(src_root / "data")),
                plans_by_path.get(str(path.relative_to(src_root / "data")), []),
                str(urdf_path),
                height_tolerance_m,
                max_step_rad,
            )
            for path in paths
        ]
        stats_by_episode: dict[int, dict] = {}
        if worker_count == 1:
            for position, task in enumerate(tasks, start=1):
                stats_by_episode.update(_rewrite_parquet_worker(task))
                emit(
                    progress_callback,
                    status="running",
                    current=position,
                    total=len(paths),
                    message=f"Rewrote {position}/{len(paths)}",
                )
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_rewrite_parquet_worker, task) for task in tasks]
                for position, future in enumerate(as_completed(futures), start=1):
                    stats_by_episode.update(future.result())
                    emit(
                        progress_callback,
                        status="running",
                        current=position,
                        total=len(paths),
                        message=f"Rewrote {position}/{len(paths)} with {worker_count} workers",
                    )

        _rewrite_stats(target_root, stats_by_episode)
        write_jsonl(target_root / "meta" / STAGE_RETURN_HEIGHT_EPISODES, episode_reports)
        summary["validation"] = _validate_output(
            src_root,
            target_root,
            plans,
            urdf_path=urdf_path,
            height_tolerance_m=height_tolerance_m,
        )
        write_json(
            target_root / "meta" / STAGE_RETURN_HEIGHT_META,
            {
                "op": "stage_return_height_alignment",
                "source_root": str(src_root),
                "output_root": str(target_root),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                **summary,
            },
        )
    except Exception:
        shutil.rmtree(target_root, ignore_errors=True)
        raise

    emit(
        progress_callback,
        status="done",
        current=len(plans),
        total=len(plans),
        message=f"Complete: {target_root}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_H10W_DVT2_URDF)
    parser.add_argument("--target-statistic", choices=["median", "mean"], default="median")
    parser.add_argument("--min-group-episodes", type=int, default=5)
    parser.add_argument("--height-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--max-step-rad", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--apply", action="store_true", help="Write sibling output; default is dry-run")
    args = parser.parse_args()
    result = run_stage_return_height_alignment(
        args.src_root,
        args.out_root,
        urdf_path=args.urdf,
        target_statistic=args.target_statistic,
        min_group_episodes=args.min_group_episodes,
        height_tolerance_m=args.height_tolerance_m,
        max_step_rad=args.max_step_rad,
        workers=args.workers,
        dry_run=not args.apply,
    )
    print(json.dumps({"out_root": str(result.out_root), **result.summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
