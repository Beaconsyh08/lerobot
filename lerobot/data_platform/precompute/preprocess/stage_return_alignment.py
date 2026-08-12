"""Align final-stage return actions to robust, cohort-specific joint targets.

The source dataset is never modified.  For pick/place, arm actions move directly
from the stage 3 start pose to the target and are held there from stage 4 start.
Give uses the equivalent stage 4 -> stage 5 interval.  State, images, grippers,
and stage labels are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import cast_stats_to_numpy, serialize_dict
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    copy_meta_files,
    copy_sidecar_dirs,
    default_preprocess_path,
    emit,
    ensure_output_root,
    load_json,
    load_jsonl,
    parquet_paths,
    validate_dataset_root,
    write_json,
    write_jsonl,
)

ACTION_COLUMN = "action"
STATE_COLUMN = "state"
STAGE_COLUMN = "subtask_state"
STAGE_RETURN_ALIGNMENT_META = "preprocess_stage_return_alignment.json"
STAGE_RETURN_ALIGNMENT_EPISODES = "preprocess_stage_return_alignment_episodes.jsonl"
DEFAULT_ARM_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
DEFAULT_GRIPPER_INDICES = (7, 15)
CATEGORY_STAGES = {
    "pick": (3, 4),
    "place": (3, 4),
    "give": (4, 5),
}


def _replace_column(table: pa.Table, name: str, values) -> pa.Table:
    field = table.schema.field(name)
    return table.set_column(
        table.column_names.index(name),
        field,
        pa.array(values, type=field.type),
    )


def _load_tasks(root: Path) -> dict[int, str]:
    jsonl_path = root / "meta" / "tasks.jsonl"
    if jsonl_path.is_file():
        return {int(row["task_index"]): str(row["task"]) for row in load_jsonl(jsonl_path)}

    parquet_path = root / "meta" / "tasks.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError("Dataset has neither meta/tasks.jsonl nor meta/tasks.parquet")
    frame = pd.read_parquet(parquet_path)
    if "task" in frame.columns and "task_index" in frame.columns:
        return {
            int(task_index): str(task)
            for task_index, task in zip(frame["task_index"], frame["task"], strict=True)
        }
    if frame.index.name == "task" and "task_index" in frame.columns:
        return {
            int(task_index): str(task)
            for task, task_index in zip(frame.index, frame["task_index"], strict=True)
        }
    raise ValueError(f"Unsupported tasks.parquet schema: {list(frame.columns)}")


def _task_category(task: str) -> str | None:
    first_word = task.strip().split(maxsplit=1)[0].lower() if task.strip() else ""
    return first_word if first_word in CATEGORY_STAGES else None


def _active_side(action: np.ndarray) -> str:
    if action.ndim != 2 or action.shape[1] < 15:
        raise ValueError(f"Expected action shape [frames, >=15], got {action.shape}")
    left_motion = float(np.linalg.norm(np.ptp(action[:, :7], axis=0)))
    right_motion = float(np.linalg.norm(np.ptp(action[:, 8:15], axis=0)))
    if max(left_motion, right_motion) <= 0:
        raise ValueError("Cannot infer active arm from a stationary episode")
    return "left" if left_motion > right_motion else "right"


def _tail_interval(stages: np.ndarray, category: str) -> tuple[int, int]:
    return_stage, final_stage = CATEGORY_STAGES[category]
    final_positions = np.flatnonzero(stages == final_stage)
    if len(final_positions) == 0:
        raise ValueError(f"missing final stage {final_stage}")
    boundary = int(final_positions[0])
    if not np.all(stages[boundary:] == final_stage):
        raise ValueError(f"stage {final_stage} is not terminal")
    start = boundary
    while start > 0 and stages[start - 1] == return_stage:
        start -= 1
    if start == boundary:
        raise ValueError(f"stage {return_stage} does not directly precede stage {final_stage}")
    return start, boundary


def _max_step(values: np.ndarray) -> float:
    return float(np.abs(np.diff(values, axis=0)).max()) if len(values) > 1 else 0.0


def _second_difference_rms(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.diff(values, n=2, axis=0)))))


def _align_episode_action(
    action: np.ndarray,
    *,
    start: int,
    boundary: int,
    target: np.ndarray,
    arm_indices: tuple[int, ...],
    max_step_rad: float,
) -> tuple[np.ndarray, dict]:
    action = np.asarray(action, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not 0 <= start < boundary < len(action):
        raise ValueError(f"Invalid tail interval [{start}, {boundary}) for {len(action)} frames")
    if target.shape != (len(arm_indices),):
        raise ValueError(f"Target shape {target.shape} does not match {len(arm_indices)} arm joints")

    arm = np.asarray(arm_indices, dtype=np.int64)
    endpoint_delta = target - action[boundary, arm]
    start_pose = action[start, arm]
    direct_delta = target - start_pose
    progress = np.arange(boundary - start + 1, dtype=np.float64) / (boundary - start)
    linear_weight = progress
    cubic_weight = 3.0 * progress**2 - 2.0 * progress**3

    def _candidate(cubic_mix: float) -> np.ndarray:
        weight = linear_weight + cubic_mix * (cubic_weight - linear_weight)
        output = action.copy()
        output[start : boundary + 1, arm] = start_pose + weight[:, None] * direct_delta
        output[boundary:, arm] = target
        return output

    raw_tail_step = _max_step(action[start:, arm])
    allowed_step = max(float(max_step_rad), raw_tail_step)
    linear = _candidate(0.0)
    linear_step = _max_step(linear[start:, arm])
    if linear_step > allowed_step + 1e-9:
        raise ValueError(
            f"Cannot reach target within max step {allowed_step:.6f} rad "
            f"from stage start; linear ramp requires {linear_step:.6f} rad"
        )

    cubic = _candidate(1.0)
    if _max_step(cubic[start:, arm]) <= allowed_step + 1e-9:
        cubic_mix = 1.0
        aligned = cubic
    else:
        low, high = 0.0, 1.0
        for _ in range(40):
            middle = (low + high) / 2.0
            if _max_step(_candidate(middle)[start:, arm]) <= allowed_step + 1e-9:
                low = middle
            else:
                high = middle
        cubic_mix = low
        aligned = _candidate(cubic_mix)

    changed = aligned[:, arm] - action[:, arm]
    return aligned, {
        "return_start_frame": int(start),
        "final_stage_start_frame": int(boundary),
        "return_frames": int(boundary - start),
        "final_stage_frames": int(len(action) - boundary),
        "changed_frames": int(np.count_nonzero(np.any(np.abs(changed) > 1e-12, axis=1))),
        "rms_change_rad": float(np.sqrt(np.mean(np.square(changed)))),
        "max_abs_change_rad": float(np.abs(changed).max()),
        "target_l2_change_rad": float(np.linalg.norm(endpoint_delta)),
        "raw_tail_max_step_rad": raw_tail_step,
        "aligned_tail_max_step_rad": _max_step(aligned[start:, arm]),
        "raw_tail_second_difference_rms": _second_difference_rms(action[start:, arm]),
        "aligned_tail_second_difference_rms": _second_difference_rms(aligned[start:, arm]),
        "cubic_mix": float(cubic_mix),
    }


def _scan_dataset(
    root: Path,
    *,
    arm_indices: tuple[int, ...],
    target_statistic: str,
    min_group_episodes: int,
    max_step_rad: float,
) -> tuple[dict[int, dict], dict[str, np.ndarray], list[dict], list[dict], int]:
    tasks = _load_tasks(root)
    candidates: dict[int, dict] = {}
    skipped: list[dict] = []
    total_frames = 0
    columns = ["episode_index", "task_index", STAGE_COLUMN, ACTION_COLUMN]

    for path in parquet_paths(root):
        table = pq.read_table(path, columns=columns)
        episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        task_values = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        stage_values = np.asarray(table[STAGE_COLUMN].to_pylist(), dtype=np.int64)
        action_values = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float64)
        for episode_id in dict.fromkeys(episode_values.tolist()):
            positions = np.flatnonzero(episode_values == episode_id)
            total_frames += len(positions)
            if episode_id in candidates:
                raise ValueError(f"Episode {episode_id} spans multiple parquet files")
            episode_tasks = np.unique(task_values[positions])
            if len(episode_tasks) != 1:
                skipped.append({"episode_index": int(episode_id), "reason": "multiple task indices"})
                continue
            task_index = int(episode_tasks[0])
            task = tasks.get(task_index)
            category = _task_category(task or "")
            if category is None:
                skipped.append(
                    {
                        "episode_index": int(episode_id),
                        "reason": f"unsupported task {task!r}",
                    }
                )
                continue
            stages = stage_values[positions]
            action = action_values[positions]
            try:
                start, boundary = _tail_interval(stages, category)
                side = _active_side(action)
            except ValueError as exc:
                skipped.append({"episode_index": int(episode_id), "reason": str(exc)})
                continue
            group = f"{category}:{side}"
            candidates[int(episode_id)] = {
                "episode_index": int(episode_id),
                "relative_path": str(path.relative_to(root / "data")),
                "task": task,
                "category": category,
                "active_side": side,
                "group": group,
                "start": start,
                "boundary": boundary,
                "boundary_action": action[boundary, list(arm_indices)],
                "action": action,
            }

    groups = sorted({item["group"] for item in candidates.values()})
    targets: dict[str, np.ndarray] = {}
    for group in groups:
        values = np.stack([item["boundary_action"] for item in candidates.values() if item["group"] == group])
        if len(values) < min_group_episodes:
            for episode_id, item in list(candidates.items()):
                if item["group"] == group:
                    skipped.append(
                        {
                            "episode_index": episode_id,
                            "reason": (
                                f"group {group} has {len(values)} episodes; requires {min_group_episodes}"
                            ),
                        }
                    )
                    del candidates[episode_id]
            continue
        targets[group] = (
            np.median(values, axis=0) if target_statistic == "median" else np.mean(values, axis=0)
        )

    episode_reports = []
    for episode_id, item in sorted(candidates.items()):
        try:
            _, metrics = _align_episode_action(
                item["action"],
                start=item["start"],
                boundary=item["boundary"],
                target=targets[item["group"]],
                arm_indices=arm_indices,
                max_step_rad=max_step_rad,
            )
        except ValueError as exc:
            skipped.append({"episode_index": episode_id, "reason": str(exc)})
            continue
        episode_reports.append(
            {
                "episode_index": episode_id,
                "task": item["task"],
                "category": item["category"],
                "active_side": item["active_side"],
                "target_group": item["group"],
                **metrics,
            }
        )

    valid_ids = {row["episode_index"] for row in episode_reports}
    plans = {
        episode_id: {key: value for key, value in item.items() if key not in {"action", "boundary_action"}}
        for episode_id, item in candidates.items()
        if episode_id in valid_ids
    }
    for episode_id in plans:
        plans[episode_id]["target"] = targets[plans[episode_id]["group"]].tolist()
    return (
        plans,
        targets,
        episode_reports,
        sorted(skipped, key=lambda row: row["episode_index"]),
        total_frames,
    )


def _column_stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _rewrite_parquet_worker(args: tuple) -> dict[int, dict]:
    src_text, dst_text, plan_rows, arm_indices, max_step_rad = args
    src = Path(src_text)
    dst = Path(dst_text)
    plans = {int(row["episode_index"]): row for row in plan_rows}
    table = pq.read_table(src)
    episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float64)
    stats = {}
    for episode_id in dict.fromkeys(episode_values.tolist()):
        positions = np.flatnonzero(episode_values == episode_id)
        episode_action = action[positions]
        plan = plans.get(int(episode_id))
        if plan is not None:
            episode_action, _ = _align_episode_action(
                episode_action,
                start=int(plan["start"]),
                boundary=int(plan["boundary"]),
                target=np.asarray(plan["target"], dtype=np.float64),
                arm_indices=arm_indices,
                max_step_rad=max_step_rad,
            )
            action[positions] = episode_action
        stats[int(episode_id)] = {ACTION_COLUMN: _column_stats(episode_action.astype(np.float32))}

    output = _replace_column(table, ACTION_COLUMN, action.astype(np.float32).tolist())
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, dst)
    return stats


def _resolve_workers(workers: int | None, total: int) -> int:
    if total <= 1:
        return 1
    if workers is None or int(workers) == 0:
        return max(1, min(os.cpu_count() or 1, total, 4))
    if int(workers) < 1:
        raise ValueError("workers must be 0 for auto or >= 1")
    return min(int(workers), total)


def _rewrite_stats(root: Path, stats_by_episode: dict[int, dict]) -> None:
    episodes_stats_path = root / "meta" / "episodes_stats.jsonl"
    if episodes_stats_path.is_file():
        rows = load_jsonl(episodes_stats_path)
        for row in rows:
            episode_id = int(row["episode_index"])
            if episode_id in stats_by_episode:
                row.setdefault("stats", {}).update(stats_by_episode[episode_id])
        write_jsonl(episodes_stats_path, rows)
    else:
        from lerobot.data_platform.precompute.dataset_io import update_episode_metadata

        updates = {
            episode_id: {
                f"stats/{feature}/{stat_name}": value
                for feature, feature_stats in stats.items()
                for stat_name, value in feature_stats.items()
            }
            for episode_id, stats in stats_by_episode.items()
        }
        update_episode_metadata(root, updates)

    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file() or not stats_by_episode:
        return
    episode_ids = sorted(stats_by_episode)
    aggregated = aggregate_stats(
        [cast_stats_to_numpy(stats_by_episode[episode_id]) for episode_id in episode_ids],
        sample_ids=episode_ids,
        sample_label="episode_index",
    )
    global_stats = load_json(stats_path)
    global_stats[ACTION_COLUMN] = serialize_dict(aggregated)[ACTION_COLUMN]
    write_json(stats_path, global_stats)


def _summary_metrics(episode_reports: list[dict], total_frames: int) -> dict:
    if not episode_reports:
        return {
            "aligned_episodes": 0,
            "changed_frames": 0,
            "changed_frame_fraction": 0.0,
        }
    changed_frames = sum(int(row["changed_frames"]) for row in episode_reports)
    rms_values = np.asarray([row["rms_change_rad"] for row in episode_reports])
    max_values = np.asarray([row["max_abs_change_rad"] for row in episode_reports])
    step_values = np.asarray([row["aligned_tail_max_step_rad"] for row in episode_reports])
    return {
        "aligned_episodes": len(episode_reports),
        "changed_frames": changed_frames,
        "changed_frame_fraction": changed_frames / total_frames if total_frames else 0.0,
        "episode_rms_change_rad_median": float(np.median(rms_values)),
        "episode_rms_change_rad_p90": float(np.quantile(rms_values, 0.9)),
        "max_abs_change_rad": float(max_values.max()),
        "aligned_tail_max_step_rad_p90": float(np.quantile(step_values, 0.9)),
        "aligned_tail_max_step_rad_max": float(step_values.max()),
        "step_limited_episodes": sum(float(row["cubic_mix"]) < 1.0 - 1e-9 for row in episode_reports),
    }


def _validate_output(
    src_root: Path,
    out_root: Path,
    plans: dict[int, dict],
    *,
    arm_indices: tuple[int, ...],
    gripper_indices: tuple[int, ...],
) -> dict:
    validated_episodes = 0
    validated_frames = 0
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
        if gripper_indices and not np.array_equal(
            source_action[:, list(gripper_indices)], output_action[:, list(gripper_indices)]
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
            start = int(plan["start"])
            boundary = int(plan["boundary"])
            target = np.asarray(plan["target"], dtype=np.float32)
            source_episode = source_action[positions]
            output_episode = output_action[positions]
            if not np.array_equal(source_episode[:start], output_episode[:start]):
                raise ValueError(f"Episode {episode_id} changed before return-stage start")
            non_arm = sorted(set(range(source_episode.shape[1])) - set(arm_indices))
            if non_arm and not np.array_equal(source_episode[:, non_arm], output_episode[:, non_arm]):
                raise ValueError(f"Episode {episode_id} changed non-arm action dimensions")
            expected = np.broadcast_to(target, output_episode[boundary:, list(arm_indices)].shape)
            if not np.array_equal(output_episode[boundary:, list(arm_indices)], expected):
                raise ValueError(f"Episode {episode_id} does not hold the canonical target")
            aligned_return = output_episode[start : boundary + 1, list(arm_indices)]
            start_pose = source_episode[start, list(arm_indices)]
            lower = np.minimum(start_pose, target) - 1e-6
            upper = np.maximum(start_pose, target) + 1e-6
            if np.any(aligned_return < lower) or np.any(aligned_return > upper):
                raise ValueError(f"Episode {episode_id} overshoots the direct target path")
            direction = np.sign(target - start_pose)
            if np.any(np.diff(aligned_return, axis=0) * direction < -1e-6):
                raise ValueError(f"Episode {episode_id} reverses along the direct target path")
    return {"episodes": validated_episodes, "frames": validated_frames}


def run_stage_return_alignment(
    src_root: Path,
    out_root: Path | None = None,
    *,
    arm_indices: tuple[int, ...] = DEFAULT_ARM_INDICES,
    gripper_indices: tuple[int, ...] = DEFAULT_GRIPPER_INDICES,
    target_statistic: str = "median",
    min_group_episodes: int = 5,
    max_step_rad: float = 0.1,
    workers: int | None = 0,
    dry_run: bool = True,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    if target_statistic not in {"median", "mean"}:
        raise ValueError("target_statistic must be 'median' or 'mean'")
    if min_group_episodes < 1:
        raise ValueError("min_group_episodes must be >= 1")
    if max_step_rad <= 0:
        raise ValueError("max_step_rad must be positive")
    if not arm_indices or len(set(arm_indices)) != len(arm_indices):
        raise ValueError("arm_indices must be non-empty and unique")
    if set(arm_indices) & set(gripper_indices):
        raise ValueError("arm_indices and gripper_indices must not overlap")

    info = load_json(src_root / "meta" / "info.json")
    features = info.get("features") or {}
    for column in (ACTION_COLUMN, STATE_COLUMN, STAGE_COLUMN):
        if column not in features:
            raise ValueError(f"Dataset metadata does not define {column!r}")
    action_dim = int((features[ACTION_COLUMN].get("shape") or [0])[-1])
    if max((*arm_indices, *gripper_indices), default=-1) >= action_dim:
        raise ValueError(f"Configured joint indices do not fit action dimension {action_dim}")

    target_root = ensure_output_root(
        out_root or default_preprocess_path(src_root, "stage_return_aligned"), dry_run
    )
    plans, targets, episode_reports, skipped, total_frames = _scan_dataset(
        src_root,
        arm_indices=arm_indices,
        target_statistic=target_statistic,
        min_group_episodes=min_group_episodes,
        max_step_rad=max_step_rad,
    )
    if not plans:
        raise ValueError(f"No eligible episodes to align; skipped={skipped}")

    group_counts = {
        group: sum(plan["group"] == group for plan in plans.values()) for group in sorted(targets)
    }
    summary = {
        "target_statistic": target_statistic,
        "target_groups": {
            group: {
                "episodes": group_counts.get(group, 0),
                "arm_target": targets[group].tolist(),
            }
            for group in sorted(targets)
            if group_counts.get(group, 0) > 0
        },
        "arm_indices": list(arm_indices),
        "gripper_indices": list(gripper_indices),
        "max_step_rad": max_step_rad,
        "trajectory_method": "direct_monotone_start_to_target",
        "source_episodes": int(info.get("total_episodes") or len(plans) + len(skipped)),
        "source_frames": int(info.get("total_frames") or total_frames),
        "skipped_episodes": skipped,
        "preserved_fields": [STATE_COLUMN, "images", "gripper action", STAGE_COLUMN],
        "category_intervals": {
            category: {"return_stage": stages[0], "final_stage": stages[1]}
            for category, stages in CATEGORY_STAGES.items()
        },
        **_summary_metrics(episode_reports, total_frames),
    }
    result = PreprocessResult(
        op="stage_return_alignment",
        src_roots=[src_root],
        out_root=target_root,
        repo_id=f"local/{target_root.name}",
        total_episodes=int(info.get("total_episodes") or len(plans) + len(skipped)),
        total_frames=int(info.get("total_frames") or total_frames),
        dry_run=dry_run,
        summary=summary,
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(plans),
        message=f"Stage return alignment plan: {summary}",
    )
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
        tasks_to_run = [
            (
                str(path),
                str(target_root / "data" / path.relative_to(src_root / "data")),
                plans_by_path.get(str(path.relative_to(src_root / "data")), []),
                arm_indices,
                max_step_rad,
            )
            for path in paths
        ]
        stats_by_episode: dict[int, dict] = {}
        if worker_count == 1:
            for position, task in enumerate(tasks_to_run, start=1):
                stats_by_episode.update(_rewrite_parquet_worker(task))
                emit(
                    progress_callback,
                    status="running",
                    current=position,
                    total=len(paths),
                    message=f"Rewrote parquet {position}/{len(paths)}",
                )
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_rewrite_parquet_worker, task) for task in tasks_to_run]
                for position, future in enumerate(as_completed(futures), start=1):
                    stats_by_episode.update(future.result())
                    emit(
                        progress_callback,
                        status="running",
                        current=position,
                        total=len(paths),
                        message=f"Rewrote parquet {position}/{len(paths)} with {worker_count} workers",
                    )

        _rewrite_stats(target_root, stats_by_episode)
        write_jsonl(target_root / "meta" / STAGE_RETURN_ALIGNMENT_EPISODES, episode_reports)
        write_json(
            target_root / "meta" / STAGE_RETURN_ALIGNMENT_META,
            {
                "op": "stage_return_alignment",
                "source_root": str(src_root),
                "output_root": str(target_root),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                **summary,
            },
        )
        summary["validation"] = _validate_output(
            src_root,
            target_root,
            plans,
            arm_indices=arm_indices,
            gripper_indices=gripper_indices,
        )
        write_json(
            target_root / "meta" / STAGE_RETURN_ALIGNMENT_META,
            {
                "op": "stage_return_alignment",
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
        message=f"Stage return alignment complete: {target_root}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--target-statistic", choices=["median", "mean"], default="median")
    parser.add_argument("--min-group-episodes", type=int, default=5)
    parser.add_argument("--max-step-rad", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--apply", action="store_true", help="Write the sibling output; default is dry-run")
    args = parser.parse_args()
    result = run_stage_return_alignment(
        args.src_root,
        args.out_root,
        target_statistic=args.target_statistic,
        min_group_episodes=args.min_group_episodes,
        max_step_rad=args.max_step_rad,
        workers=args.workers,
        dry_run=not args.apply,
    )
    print(json.dumps({"out_root": str(result.out_root), **result.summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
