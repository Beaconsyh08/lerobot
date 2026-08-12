import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.preprocess.stage_return_alignment import (
    DEFAULT_ARM_INDICES,
    _align_episode_action,
    run_stage_return_alignment,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_dataset(root: Path) -> Path:
    fps = 30
    specs = [
        ("Pick up duck", "left", 0.6, [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]),
        ("Pick up duck", "left", 1.0, [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]),
        ("Pick up duck", "right", 0.8, [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]),
        ("Place object", "left", 0.7, [0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]),
        ("Give object", "right", 0.9, [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5]),
    ]
    features = {
        "state": {"dtype": "float32", "shape": [16], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [16], "names": ["actions"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": len(specs),
        "total_frames": sum(len(spec[3]) for spec in specs),
        "total_tasks": 3,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(specs)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    tasks = ["Pick up duck", "Place object", "Give object"]
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}) + "\n" for index, task in enumerate(tasks))
    )

    episode_rows = []
    episode_stats_rows = []
    global_index = 0
    for episode_id, (task, side, endpoint, stages) in enumerate(specs):
        task_index = tasks.index(task)
        rows = []
        frames = len(stages)
        for frame_index, stage in enumerate(stages):
            action = np.zeros(16, dtype=np.float32)
            action[:7] = -0.25
            action[8:15] = 0.25
            active = slice(0, 7) if side == "left" else slice(8, 15)
            action[active] = endpoint * frame_index / (frames - 1)
            if frame_index >= stages.index(max(stages)):
                action[active] += 0.02 * (frame_index - stages.index(max(stages)))
            action[7 if side == "left" else 15] = 1.0 if task.startswith("Pick") else 0.0
            state = action.copy()
            state[:7] += 0.001
            state[8:15] -= 0.001
            rows.append(
                {
                    "state": state.tolist(),
                    "action": action.tolist(),
                    "timestamp": frame_index / fps,
                    "frame_index": frame_index,
                    "episode_index": episode_id,
                    "index": global_index + frame_index,
                    "task_index": task_index,
                    "subtask_state": stage,
                }
            )
        path = root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        episode_rows.append({"episode_index": episode_id, "tasks": [task], "length": frames})
        episode_stats_rows.append({"episode_index": episode_id, "stats": {}})
        global_index += frames
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episode_rows))
    (root / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_stats_rows)
    )
    return root


def test_alignment_holds_target_and_respects_step_limit():
    action = np.zeros((11, 16), dtype=np.float64)
    action[:, :7] = np.linspace(0.0, 0.5, len(action))[:, None]
    action[:, 8:15] = np.linspace(0.0, 0.5, len(action))[:, None]
    target = np.full(len(DEFAULT_ARM_INDICES), 0.65)

    aligned, metrics = _align_episode_action(
        action,
        start=3,
        boundary=8,
        target=target,
        arm_indices=DEFAULT_ARM_INDICES,
        max_step_rad=0.1,
    )

    assert np.array_equal(aligned[:3], action[:3])
    assert np.allclose(aligned[8:, list(DEFAULT_ARM_INDICES)], target)
    assert metrics["aligned_tail_max_step_rad"] <= 0.1 + 1e-9


def test_alignment_goes_directly_to_target_without_overshoot():
    action = np.zeros((10, 16), dtype=np.float64)
    original_tail = np.array([-0.2, 0.0, 0.4, 0.9, 0.8, 0.6, 0.6])
    action[2:9, :7] = original_tail[:, None]
    action[2:9, 8:15] = original_tail[:, None]
    target = np.full(len(DEFAULT_ARM_INDICES), 0.2)

    aligned, _ = _align_episode_action(
        action,
        start=2,
        boundary=7,
        target=target,
        arm_indices=DEFAULT_ARM_INDICES,
        max_step_rad=0.3,
    )

    aligned_return = aligned[2:8, list(DEFAULT_ARM_INDICES)]
    assert np.all(np.diff(aligned_return, axis=0) >= -1e-12)
    assert np.all(aligned_return >= -0.2 - 1e-12)
    assert np.all(aligned_return <= 0.2 + 1e-12)
    assert np.allclose(aligned[7:, list(DEFAULT_ARM_INDICES)], target)


def test_stage_return_alignment_dry_run_then_writes_sibling_dataset(tmp_path: Path):
    src = _make_dataset(tmp_path / "src")
    out = tmp_path / "aligned"
    source_paths = sorted((src / "data").rglob("*.parquet"))
    source_digests = {path: _digest(path) for path in source_paths}

    dry_run = run_stage_return_alignment(
        src,
        out,
        min_group_episodes=1,
        workers=1,
    )

    assert dry_run.dry_run is True
    assert dry_run.summary["aligned_episodes"] == 5
    assert set(dry_run.summary["target_groups"]) == {
        "give:right",
        "pick:left",
        "pick:right",
        "place:left",
    }
    assert not out.exists()

    result = run_stage_return_alignment(
        src,
        out,
        min_group_episodes=1,
        workers=1,
        dry_run=False,
    )

    assert result.summary["validation"] == {"episodes": 5, "frames": 60}
    assert all(_digest(path) == source_digests[path] for path in source_paths)
    source = pq.read_table(source_paths[0])
    output = pq.read_table(out / "data" / "chunk-000" / "episode_000000.parquet")
    source_action = np.asarray(source["action"].to_pylist())
    output_action = np.asarray(output["action"].to_pylist())
    stages = np.asarray(source["subtask_state"].to_pylist())
    return_start = int(np.flatnonzero(stages == 3)[0])
    final_start = int(np.flatnonzero(stages == 4)[0])
    target = np.asarray(result.summary["target_groups"]["pick:left"]["arm_target"])

    assert np.array_equal(output_action[:return_start], source_action[:return_start])
    assert np.allclose(output_action[final_start:, list(DEFAULT_ARM_INDICES)], target)
    assert np.array_equal(output["state"].to_pylist(), source["state"].to_pylist())
    assert np.array_equal(output["subtask_state"].to_pylist(), source["subtask_state"].to_pylist())
    assert np.array_equal(output_action[:, [7, 15]], source_action[:, [7, 15]])

    provenance = json.loads((out / "meta" / "preprocess_stage_return_alignment.json").read_text())
    reports = [
        json.loads(line)
        for line in (out / "meta" / "preprocess_stage_return_alignment_episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    give_report = next(row for row in reports if row["category"] == "give")
    assert provenance["preserved_fields"] == ["state", "images", "gripper action", "subtask_state"]
    assert give_report["return_start_frame"] == 4
    assert give_report["final_stage_start_frame"] == 7
    stats_rows = [
        json.loads(line) for line in (out / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    ]
    assert all("action" in row["stats"] for row in stats_rows)
