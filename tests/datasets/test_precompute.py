import copy
import json
import sys
from pathlib import Path

import cv2
import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.data_platform import cli as prepare_script
from lerobot.data_platform.precompute.analysis import (
    build_dataset_analysis,
    infer_task_scene,
    parse_canonical_task,
)
from lerobot.data_platform.precompute.annotation import (
    assign_subtask_states,
    compute_quality_flags,
    compute_subtask_boundaries,
    get_columns_info,
    write_episode_csv,
)
from lerobot.data_platform.precompute.image_io import iter_image_bytes, read_image_bytes
from lerobot.data_platform.precompute.labeling.runner import _sync_missing_target_flags
from lerobot.data_platform.precompute.mutations import fix_episode_indices
from lerobot.data_platform.precompute.preprocess.delete_episodes import (
    _reindex_construction_plan,
    delete_episodes_inplace,
    reindex_static_after_episode_delete,
)
from lerobot.data_platform.precompute.preprocess.quality_flags import (
    apply_task_assignment_choice,
    list_multiple_task_assignments,
    list_task_assignment_choices,
    run_quality_flag_detection,
)
from lerobot.data_platform.precompute.timeseries import (
    DATA_VERSION_DVT2,
    normalize_gripper_columns,
    normalize_gripper_csv_value,
)


def _png_bytes(value: int) -> bytes:
    image = np.full((4, 4, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_analysis_prompt_prepositions_distinguish_absolute_and_relative():
    absolute = parse_canonical_task("pick up the yellow duck on the left")
    assert absolute["is_canonical"] is True
    assert absolute["task_type"] == "directional_pick"
    assert absolute["canonical_task"] == "Pick up the yellow duck on the left"

    relative = parse_canonical_task("pick up the yellow duck to the right of the brown dog")
    assert relative["is_canonical"] is True
    assert relative["task_type"] == "relational_pick"
    assert relative["canonical_task"] == "Pick up the yellow duck to the right of the brown dog"

    wrong_relative = parse_canonical_task("pick up the yellow duck on the right of the brown dog")
    assert wrong_relative["is_canonical"] is False
    assert wrong_relative["task_type"] == "relational_pick"
    assert wrong_relative["canonical_task"] == "Pick up the yellow duck to the right of the brown dog"
    assert infer_task_scene("pick up the yellow duck on the right of the brown dog") == "relational_pick"


def _build_episode_arrays(num_frames: int = 80, *, give: bool = False):
    timestamps = np.arange(num_frames, dtype=np.float32) / 10.0
    action = np.zeros((num_frames, 8), dtype=np.float32)
    state = np.zeros((num_frames, 10), dtype=np.float32)
    state[:, 1] = 0.0
    state[:, 9] = 0.0

    action[10:35, 0] = np.linspace(0.0, 1.0, 25, endpoint=False, dtype=np.float32)
    action[35:55, 0] = 1.0
    action[55:75, 0] = np.linspace(1.0, 0.0, 20, endpoint=False, dtype=np.float32)

    if give:
        action[30:65, 7] = 1.0
        state[33:68, 7] = 1.0
        task = "Give the cube to me"
    else:
        action[40:, 7] = 1.0
        state[43:, 7] = 1.0
        task = "Pick up the cube"

    return timestamps, action, state, task


def _list_array(matrix: np.ndarray) -> pa.Array:
    return pa.array([row.tolist() for row in matrix], type=pa.list_(pa.float32()))


def _int_list_array(matrix: np.ndarray) -> pa.Array:
    return pa.array([row.tolist() for row in matrix], type=pa.list_(pa.int64()))


def _write_episode_parquet(
    root: Path,
    episode_id: int,
    *,
    timestamps: np.ndarray,
    action: np.ndarray | None = None,
    state: np.ndarray | None = None,
    frame_index: list[int] | None = None,
    index: list[int] | None = None,
    image_rows: list[dict] | None = None,
    include_exist: bool = False,
    exist_label: np.ndarray | None = None,
    subtask_state: np.ndarray | None = None,
) -> Path:
    data = {
        "timestamp": pa.array(timestamps.tolist(), type=pa.float32()),
    }
    if action is not None:
        data["action"] = _list_array(action)
    if state is not None:
        data["state"] = _list_array(state)
    if frame_index is not None:
        data["frame_index"] = pa.array(frame_index, type=pa.int64())
    if index is not None:
        data["index"] = pa.array(index, type=pa.int64())
    if include_exist:
        data["exist"] = pa.array([1] * len(timestamps), type=pa.int32())
    if exist_label is not None:
        data["exist_label"] = _int_list_array(exist_label)
    if subtask_state is not None:
        data["subtask_state"] = pa.array(np.asarray(subtask_state, dtype=np.int32).tolist(), type=pa.int32())
    if image_rows is not None:
        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        data["front"] = pa.array(image_rows, type=image_type)

    table = pa.table(data)
    parquet_path = root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)
    return parquet_path


class DummyMeta:
    def __init__(self, root: Path, features: dict, episodes: dict[int, dict], fps: int = 10):
        self.root = root
        self.info = {"fps": fps, "features": features}
        self.episodes = episodes

    @property
    def features(self):
        return self.info["features"]

    @property
    def shapes(self):
        return {key: tuple(feature["shape"]) for key, feature in self.info["features"].items()}

    @property
    def fps(self):
        return self.info["fps"]

    def get_data_file_path(self, episode_id: int) -> Path:
        return Path(f"data/chunk-000/episode_{episode_id:06d}.parquet")


class DummyDataset:
    def __init__(self, root: Path, meta: DummyMeta):
        self.root = root
        self.meta = meta
        self.features = meta.features
        self.total_episodes = int(meta.info["total_episodes"])
        self.total_frames = int(meta.info["total_frames"])


def _numeric_stats(dim: int) -> dict:
    return {
        "min": [0.0] * dim,
        "max": [1.0] * dim,
        "mean": [0.0] * dim,
        "std": [0.0] * dim,
        "count": [80],
    }


def _prepare_script_dataset(root: Path, *, give: bool = False) -> DummyMeta:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    timestamps, action, state, task = _build_episode_arrays(give=give)
    _write_episode_parquet(
        root,
        0,
        timestamps=timestamps,
        action=action,
        state=state,
        include_exist=True,
    )

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "action": {"dtype": "float32", "shape": [8], "names": None},
        "state": {"dtype": "float32", "shape": [10], "names": None},
        "exist": {"dtype": "int32", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "dummy",
        "total_episodes": 1,
        "total_frames": len(timestamps),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    with jsonlines.open(root / "meta" / "episodes_stats.jsonl", mode="w") as writer:
        writer.write(
            {
                "episode_index": 0,
                "stats": {
                    "timestamp": _numeric_stats(1),
                    "action": _numeric_stats(8),
                    "state": _numeric_stats(10),
                    "exist": _numeric_stats(1),
                },
            }
        )

    episodes = {0: {"episode_index": 0, "tasks": [task], "length": len(timestamps)}}
    return DummyMeta(root, features, episodes)


def test_iter_image_bytes_supports_embedded_and_path_fallback(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    embedded = _png_bytes(32)
    fallback = _png_bytes(196)

    image_dir = dataset_root / "images" / "front"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_000001.png").write_bytes(fallback)

    parquet_path = _write_episode_parquet(
        dataset_root,
        0,
        timestamps=np.array([0.0, 0.1], dtype=np.float32),
        image_rows=[
            {"bytes": embedded, "path": None},
            {"bytes": None, "path": "frame_000001.png"},
        ],
    )

    all_bytes = list(iter_image_bytes(parquet_path, dataset_root, "front"))
    assert all_bytes == [embedded, fallback]
    assert read_image_bytes(parquet_path, dataset_root, "front", 1) == fallback


def test_compute_subtask_boundaries_for_pick_place():
    timestamps, action, state, task = _build_episode_arrays(give=False)
    boundaries, issues = compute_subtask_boundaries(
        timestamps, action, state, fps=10.0, episode_id=3, task=task
    )

    assert boundaries is not None
    assert (
        boundaries["stage0_end"]
        < boundaries["stage2_start"]
        <= boundaries["stage2_end"]
        < boundaries["stage4_start"]
    )
    assert not boundaries.get("is_give", False)
    assert issues == []


def test_compute_subtask_boundaries_for_give():
    timestamps, action, state, task = _build_episode_arrays(give=True)
    boundaries, issues = compute_subtask_boundaries(
        timestamps, action, state, fps=10.0, episode_id=4, task=task
    )

    assert boundaries is not None
    assert boundaries["is_give"] is True
    assert (
        boundaries["stage0_end"]
        < boundaries["stage2_start"]
        <= boundaries["stage2_end"]
        < boundaries["stage4_start"]
        <= boundaries["stage4_end"]
    )
    assert not any(issue["type"] == "multi_gripper" for issue in issues)


def test_compute_subtask_boundaries_for_direct_give_skips_grasp_stages():
    timestamps, action, state, task = _build_episode_arrays(give=True)
    action[:, 7] = 0.0
    state[:, 7] = 0.0
    action[:65, 7] = 1.0
    state[:68, 7] = 1.0

    boundaries, issues = compute_subtask_boundaries(
        timestamps, action, state, fps=10.0, episode_id=4, task=task
    )

    assert boundaries is not None
    assert boundaries["is_give"] is True
    assert boundaries["direct_give"] is True
    assert "stage2_start" not in boundaries
    assert "stage2_end" not in boundaries
    assert boundaries["stage0_end"] < boundaries["stage4_start"] <= boundaries["stage4_end"]
    assert not issues

    states = assign_subtask_states(timestamps, boundaries)
    assert set(states) == {0, 3, 4, 5}
    assert 1 not in states
    assert 2 not in states


def test_compute_subtask_boundaries_for_give_flags_more_than_two_gripper_transitions():
    timestamps, action, state, task = _build_episode_arrays(give=True)
    action[70:, 7] = 1.0
    state[72:, 7] = 1.0
    boundaries, issues = compute_subtask_boundaries(
        timestamps, action, state, fps=10.0, episode_id=4, task=task
    )

    assert boundaries is not None
    multi = [issue for issue in issues if issue["type"] == "multi_gripper"]
    assert len(multi) == 1
    assert len(multi[0]["frames"]) > 2


def test_compute_subtask_boundaries_returns_issue_when_transition_missing():
    timestamps, action, state, task = _build_episode_arrays(give=False)
    action[:, 7] = 0.0
    state[:, 7] = 0.0

    boundaries, issues = compute_subtask_boundaries(
        timestamps, action, state, fps=10.0, episode_id=5, task=task
    )

    assert boundaries is None
    assert issues and issues[0]["type"] == "error"


def test_compute_subtask_boundaries_uses_equal_time_stages_for_other_tasks():
    timestamps = np.linspace(0.0, 10.0, 11, dtype=np.float32)
    action = np.zeros((len(timestamps), 16), dtype=np.float32)

    boundaries, issues = compute_subtask_boundaries(
        timestamps,
        action,
        None,
        fps=1.0,
        episode_id=6,
        task="move the cube in a circle",
    )

    assert issues == []
    assert boundaries == {
        "equal_time": True,
        "num_stages": 5,
        "stage_boundaries": [2.0, 4.0, 6.0, 8.0],
    }
    assert assign_subtask_states(timestamps, boundaries) == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        4,
    ]


def test_compute_subtask_boundaries_allows_adjusting_equal_time_stage_count(tmp_path: Path):
    timestamps = np.linspace(0.0, 9.0, 10, dtype=np.float32)
    action = np.zeros((len(timestamps), 16), dtype=np.float32)

    boundaries, issues = compute_subtask_boundaries(
        timestamps,
        action,
        None,
        fps=1.0,
        task="rotate the cube",
        fallback_stage_count=3,
    )

    assert issues == []
    assert boundaries["stage_boundaries"] == [3.0, 6.0]
    assert assign_subtask_states(timestamps, boundaries) == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]

    prepare_script._write_subtask_annotations(tmp_path, {"0": boundaries}, True)
    annotations = json.loads((tmp_path / "subtask_annotations.json").read_text())
    assert annotations == {
        "0": [
            {"time": 3.0, "state": 1},
            {"time": 6.0, "state": 2},
        ]
    }


def test_compute_subtask_boundaries_rejects_invalid_equal_time_stage_count():
    timestamps = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    action = np.zeros((len(timestamps), 16), dtype=np.float32)

    with pytest.raises(ValueError, match="fallback_stage_count must be at least 2"):
        compute_subtask_boundaries(
            timestamps,
            action,
            None,
            fps=2.0,
            task="rotate the cube",
            fallback_stage_count=1,
        )


def test_compute_subtask_boundaries_accepts_new_gripper_0_100_encoding():
    timestamps, action, state, task = _build_episode_arrays(give=False)
    action[:, 7] *= 100.0
    state[:, 7] *= 100.0

    boundaries, issues = compute_subtask_boundaries(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=6,
        task=task,
        data_version=DATA_VERSION_DVT2,
    )

    assert boundaries is not None
    assert issues == []


def test_compute_subtask_boundaries_uses_tighter_dvt2_gripper_window():
    timestamps, action, state, task = _build_episode_arrays(give=False)

    dvt1_boundaries, _ = compute_subtask_boundaries(timestamps, action, state, fps=10.0, task=task)
    dvt2_boundaries, _ = compute_subtask_boundaries(
        timestamps,
        action,
        state,
        fps=10.0,
        task=task,
        data_version=DATA_VERSION_DVT2,
    )

    assert dvt1_boundaries is not None
    assert dvt2_boundaries is not None
    assert dvt2_boundaries["stage2_start"] > dvt1_boundaries["stage2_start"]
    assert dvt2_boundaries["stage2_end"] < dvt1_boundaries["stage2_end"]
    assert (
        dvt2_boundaries["stage2_end"] - dvt2_boundaries["stage2_start"]
        < dvt1_boundaries["stage2_end"] - dvt1_boundaries["stage2_start"]
    )


def test_compute_subtask_boundaries_dvt2_stage0_uses_start_pose():
    timestamps = np.arange(80, dtype=np.float32) / 10.0
    action = np.zeros((80, 19), dtype=np.float32)
    state = np.zeros((80, 18), dtype=np.float32)
    left_start_pose = np.deg2rad(
        np.array([-20.7723, 50.0851, -16.0979, -46.6055, -62.3497, 49.7502, -3.2868])
    )

    action[10:35, 0] = np.linspace(0.0, 1.0, 25, endpoint=False, dtype=np.float32)
    action[35:55, 0] = 1.0
    action[55:75, 0] = np.linspace(1.0, 0.0, 20, endpoint=False, dtype=np.float32)
    action[40:, 7] = 100.0
    state[43:, 7] = 100.0
    state[15:, :7] = left_start_pose

    boundaries, issues = compute_subtask_boundaries(
        timestamps,
        action,
        state,
        fps=10.0,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    assert boundaries is not None
    assert issues == []
    assert boundaries["stage0_end"] == float(timestamps[17])


def test_compute_subtask_boundaries_waits_for_initial_closed_gripper_to_open():
    timestamps = np.arange(90, dtype=np.float32) / 10.0
    action = np.zeros((90, 19), dtype=np.float32)
    state = np.zeros((90, 18), dtype=np.float32)
    left_start_pose = np.deg2rad(
        np.array([-20.7723, 50.0851, -16.0979, -46.6055, -62.3497, 49.7502, -3.2868])
    )

    action[10:35, 0] = np.linspace(0.0, 1.0, 25, endpoint=False, dtype=np.float32)
    action[35:65, 0] = 1.0
    action[65:85, 0] = np.linspace(1.0, 0.0, 20, endpoint=False, dtype=np.float32)
    state[15:, :7] = left_start_pose

    # The arm reaches the DVT2 start pose at frame 15, action opens at frame 22,
    # but observed state is only half-open at frame 25 and fully open at frame 28.
    # Stage 0 waits for fully-open state plus 0.1s.
    action[:22, 7] = 100.0
    state[:25, 7] = 100.0
    state[25:28, 7] = 40.0
    action[40:, 7] = 100.0
    state[43:, 7] = 100.0

    boundaries, issues = compute_subtask_boundaries(
        timestamps,
        action,
        state,
        fps=10.0,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    assert boundaries is not None
    assert not any(issue["type"] == "multi_gripper" for issue in issues)
    assert boundaries["stage0_end"] == float(timestamps[29])


def test_compute_subtask_boundaries_dvt2_stage4_uses_first_stable_window():
    timestamps = np.arange(100, dtype=np.float32) / 10.0
    action = np.zeros((100, 19), dtype=np.float32)
    state = np.zeros((100, 18), dtype=np.float32)
    state[:, 1] = 0.0
    state[:, 9] = 0.0

    action[10:30, 0] = np.linspace(0.0, 1.0, 20, endpoint=False, dtype=np.float32)
    action[30:, 0] = 1.0
    action[40:, 7] = 100.0
    state[43:, 7] = 100.0
    # A small late motion should not delay DVT2 stage4 once an earlier stable window exists.
    action[75:90, 0] = np.linspace(1.0, 1.25, 15, dtype=np.float32)
    action[90:, 0] = 1.25

    boundaries, issues = compute_subtask_boundaries(
        timestamps,
        action,
        state,
        fps=10.0,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    assert boundaries is not None
    assert issues == []
    assert boundaries["stage2_end"] < boundaries["stage4_start"] < float(timestamps[60])


def test_compute_subtask_boundaries_dvt2_place_uses_shorter_stage4_stable_window():
    timestamps = np.arange(100, dtype=np.float32) / 10.0
    action = np.zeros((100, 19), dtype=np.float32)
    state = np.zeros((100, 18), dtype=np.float32)
    state[:, 1] = 0.0
    state[:, 9] = 0.0

    action[10:30, 0] = np.linspace(0.0, 1.0, 20, endpoint=False, dtype=np.float32)
    action[30:, 0] = 1.0
    action[40:, 7] = 100.0
    state[43:, 7] = 100.0
    # Stable for ~0.2s but not 0.4s, then a small motion resumes.
    action[50:59, 0] = np.linspace(1.0, 1.25, 9, dtype=np.float32)
    action[59:, 0] = 1.25

    pick_boundaries, pick_issues = compute_subtask_boundaries(
        timestamps,
        action,
        state,
        fps=10.0,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )
    place_boundaries, place_issues = compute_subtask_boundaries(
        timestamps,
        action,
        state,
        fps=10.0,
        task="Place object",
        data_version=DATA_VERSION_DVT2,
    )

    assert pick_boundaries is not None
    assert place_boundaries is not None
    assert pick_issues == []
    assert place_issues == []
    assert place_boundaries["stage4_start"] < pick_boundaries["stage4_start"]


def test_compute_quality_flags_detects_abnormal_episode_start():
    timestamps = np.arange(80, dtype=np.float32) / 10.0
    action = np.zeros((80, 19), dtype=np.float32)
    state = np.zeros((80, 18), dtype=np.float32)
    action[1:, 7] = 100.0
    action[10:12, 7] = 0.0
    action[20:22, 7] = 100.0
    action[30:32, 7] = 0.0
    state[0, 0] = 0.0
    state[1, 0] = 1.2
    state[2, 3] = np.nan

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=9,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    reasons = {issue["reason"] for issue in issues}
    assert reasons == {"early_gripper_transition"}
    assert all(issue["type"] == "quality_flag" for issue in issues)
    assert all(issue["data_version"] == DATA_VERSION_DVT2 for issue in issues)


def test_compute_quality_flags_ignores_late_gripper_transitions_and_joint_spikes():
    timestamps = np.arange(80, dtype=np.float32) / 10.0
    action = np.zeros((80, 19), dtype=np.float32)
    state = np.zeros((80, 18), dtype=np.float32)
    action[10:12, 7] = 100.0
    action[20:22, 7] = 0.0
    action[30:32, 7] = 100.0
    state[1, 0] = 1.2
    state[2, 3] = np.nan

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=10,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    assert issues == []


def test_compute_quality_flags_detects_joint_zero_reset_spikes():
    timestamps = np.arange(80, dtype=np.float32) / 10.0
    action = np.zeros((80, 19), dtype=np.float32)
    state = np.zeros((80, 18), dtype=np.float32)
    for joint_idx, value in [(0, 0.8), (1, -0.9), (2, 0.7), (8, -0.8), (9, 0.9)]:
        state[:, joint_idx] = value
        state[10, joint_idx] = 0.0
        state[25, joint_idx] = 0.0

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=14,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    reset = [issue for issue in issues if issue["reason"] == "joint_zero_reset_spike"]
    assert len(reset) == 1
    assert reset[0]["frames"] == [10, 25]
    assert reset[0]["metrics"]["event_count"] == 10
    assert reset[0]["metrics"]["sync_frames"] == [10, 25]


def test_compute_quality_flags_detects_stuck_closed_left_gripper_without_action():
    timestamps = np.arange(50, dtype=np.float32) / 10.0
    action = np.zeros((50, 19), dtype=np.float32)
    state = np.zeros((50, 18), dtype=np.float32)
    state[:, 7] = 0.7

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=11,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    stuck = [issue for issue in issues if issue["reason"] == "stuck_closed_gripper_no_action"]
    assert len(stuck) == 1
    assert stuck[0]["metrics"]["side"] == "left"
    assert stuck[0]["metrics"]["gripper_index"] == 7
    assert stuck[0]["metrics"]["state_closed_ratio"] == 1.0


def test_compute_quality_flags_detects_stuck_closed_right_gripper_without_action():
    timestamps = np.arange(50, dtype=np.float32) / 10.0
    action = np.zeros((50, 19), dtype=np.float32)
    state = np.zeros((50, 18), dtype=np.float32)
    state[:, 15] = 0.8

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=12,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    stuck = [issue for issue in issues if issue["reason"] == "stuck_closed_gripper_no_action"]
    assert len(stuck) == 1
    assert stuck[0]["metrics"]["side"] == "right"
    assert stuck[0]["metrics"]["gripper_index"] == 15


def test_compute_quality_flags_does_not_flag_closed_gripper_with_action_command():
    timestamps = np.arange(50, dtype=np.float32) / 10.0
    action = np.zeros((50, 19), dtype=np.float32)
    state = np.zeros((50, 18), dtype=np.float32)
    state[:, 7] = 0.7
    action[10:20, 7] = 100.0

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=13,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    assert not any(issue["reason"] == "stuck_closed_gripper_no_action" for issue in issues)


def test_compute_quality_flags_detects_state_gripper_transition_without_action():
    timestamps = np.arange(30, dtype=np.float32) / 10.0
    action = np.zeros((30, 19), dtype=np.float32)
    state = np.zeros((30, 18), dtype=np.float32)
    state[10:, 7] = 0.8

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=15,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    missing = [issue for issue in issues if issue["reason"] == "state_gripper_transition_without_action"]
    assert len(missing) == 1
    assert missing[0]["frames"] == [10]
    assert missing[0]["metrics"]["events"][0]["gripper_index"] == 7
    assert missing[0]["metrics"]["events"][0]["from_state"] == 0
    assert missing[0]["metrics"]["events"][0]["to_state"] == 1


def test_compute_quality_flags_accepts_action_shortly_before_state_transition():
    timestamps = np.arange(30, dtype=np.float32) / 10.0
    action = np.zeros((30, 19), dtype=np.float32)
    state = np.zeros((30, 18), dtype=np.float32)
    action[9:, 7] = 100.0
    state[10:, 7] = 0.8

    issues = compute_quality_flags(
        timestamps,
        action,
        state,
        fps=10.0,
        episode_id=16,
        task="Pick up the cube",
        data_version=DATA_VERSION_DVT2,
    )

    assert not any(issue["reason"] == "state_gripper_transition_without_action" for issue in issues)


def test_run_quality_flag_detection_refreshes_quality_flags_only(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action0 = np.zeros((20, 19), dtype=np.float32)
    state0 = np.zeros((20, 18), dtype=np.float32)
    action0[1:, 7] = 100.0
    action1 = np.zeros((20, 19), dtype=np.float32)
    state1 = np.zeros((20, 18), dtype=np.float32)
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

    _write_episode_parquet(
        dataset_root, 0, timestamps=timestamps, action=action0, state=state0, subtask_state=subtask_state
    )
    _write_episode_parquet(
        dataset_root, 1, timestamps=timestamps, action=action1, state=state1, subtask_state=subtask_state
    )
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 2,
        "total_frames": 40,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the cube"], "length": 20})
        writer.write({"episode_index": 1, "tasks": ["Pick up the cube"], "length": 20})

    static_dir.mkdir(parents=True)
    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {"episode": 1, "type": "error", "reason": "existing_stage_issue"},
                {"episode": 1, "type": "quality_flag", "reason": "old_auto"},
            ]
        )
    )
    (static_dir / "quality_flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [1]}))
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [1, 3]}))

    result = run_quality_flag_detection(dataset_root, static_dir, data_version=DATA_VERSION_DVT2, workers=1)

    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    assert {"episode": 1, "type": "error", "reason": "existing_stage_issue"} in issues
    assert not any(issue.get("reason") == "old_auto" for issue in issues)
    assert any(issue.get("episode") == 0 and issue.get("type") == "quality_flag" for issue in issues)
    quality_flags = json.loads((static_dir / "quality_flagged_episodes.json").read_text())
    auto_flags = quality_flags["flagged_episodes"]
    all_flags = json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"]
    assert auto_flags == [0]
    assert quality_flags["flag_reasons"]["0"][0]["reason"] == "early_gripper_transition"
    assert all_flags == [0, 3]
    assert result.summary["quality_episode_count"] == 1


def test_run_quality_flag_detection_overwrite_can_clear_manual_flags(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action = np.zeros((20, 19), dtype=np.float32)
    state = np.zeros((20, 18), dtype=np.float32)
    action[1:, 7] = 100.0
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

    _write_episode_parquet(
        dataset_root, 0, timestamps=timestamps, action=action, state=state, subtask_state=subtask_state
    )
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 20,
                "fps": 10,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "action": {"dtype": "float32", "shape": [19], "names": None},
                    "state": {"dtype": "float32", "shape": [18], "names": None},
                    "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
                },
            }
        )
    )
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the cube"], "length": 20})

    static_dir.mkdir(parents=True)
    (static_dir / "manual_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [9],
                "flag_reasons": {"9": [{"type": "manual", "reason": "manual_flag"}]},
            }
        )
    )
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [9]}))

    result = run_quality_flag_detection(
        dataset_root,
        static_dir,
        data_version=DATA_VERSION_DVT2,
        workers=1,
        overwrite=True,
        clear_manual_flags=True,
    )

    manual_flags = json.loads((static_dir / "manual_flagged_episodes.json").read_text())
    all_flags = json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"]
    assert manual_flags == {"flagged_episodes": [], "flag_reasons": {}}
    assert all_flags == [0]
    assert result.summary["manual_flags_cleared"] == 1


def test_run_quality_flag_detection_syncs_prompt_action_mismatch_tags(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action = np.zeros((20, 19), dtype=np.float32)
    state = np.zeros((20, 18), dtype=np.float32)
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

    _write_episode_parquet(
        dataset_root, 0, timestamps=timestamps, action=action, state=state, subtask_state=subtask_state
    )
    _write_episode_parquet(
        dataset_root, 1, timestamps=timestamps, action=action, state=state, subtask_state=subtask_state
    )
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 2,
        "total_frames": 40,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 20})
        writer.write({"episode_index": 1, "tasks": ["Pick up the yellow duck"], "length": 20})

    tagging_dir = static_dir / "tagging"
    tagging_dir.mkdir(parents=True)
    (tagging_dir / "tags.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 1,
                "task": "Pick up the yellow duck",
                "tags": {"prompt_action_match": "mismatch"},
                "tag_details": {
                    "prompt_action_match": {
                        "observed_object": "brown dog",
                        "reason": "final frame shows dog",
                    }
                },
            }
        )
        + "\n"
    )

    result = run_quality_flag_detection(dataset_root, static_dir, data_version=DATA_VERSION_DVT2, workers=1)

    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    assert any(
        issue.get("episode") == 1
        and issue.get("type") == "tagging_prompt_behavior"
        and issue.get("reason") == "prompt_action_mismatch"
        for issue in issues
    )
    mismatch_flags = json.loads((static_dir / "tagging_prompt_mismatch_flagged_episodes.json").read_text())
    assert mismatch_flags["flagged_episodes"] == [1]
    assert mismatch_flags["flag_reasons"]["1"][0]["reason"] == "prompt_action_mismatch"
    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == [1]
    assert result.summary["prompt_action_mismatch_count"] == 1


def test_run_quality_flag_detection_preserves_unscanned_auto_flag_reasons(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action0 = np.zeros((20, 19), dtype=np.float32)
    state0 = np.zeros((20, 18), dtype=np.float32)
    action0[1:, 7] = 100.0
    action1 = np.zeros((20, 19), dtype=np.float32)
    state1 = np.zeros((20, 18), dtype=np.float32)
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

    _write_episode_parquet(
        dataset_root, 0, timestamps=timestamps, action=action0, state=state0, subtask_state=subtask_state
    )
    _write_episode_parquet(
        dataset_root, 1, timestamps=timestamps, action=action1, state=state1, subtask_state=subtask_state
    )
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "total_frames": 40,
                "fps": 10,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                    "action": {"dtype": "float32", "shape": [19], "names": None},
                    "state": {"dtype": "float32", "shape": [18], "names": None},
                    "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
                },
            }
        )
    )
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the cube"], "length": 20})
        writer.write({"episode_index": 1, "tasks": ["Pick up the cube"], "length": 20})

    static_dir.mkdir(parents=True)
    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {"episode": 1, "type": "quality_flag", "reason": "stuck_closed_gripper_no_action"},
            ]
        )
    )
    (static_dir / "quality_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [1],
                "flag_reasons": {
                    "1": [{"type": "quality_flag", "reason": "stuck_closed_gripper_no_action"}],
                },
            }
        )
    )
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [1]}))

    run_quality_flag_detection(
        dataset_root, static_dir, episodes=[0], data_version=DATA_VERSION_DVT2, workers=1
    )

    quality_flags = json.loads((static_dir / "quality_flagged_episodes.json").read_text())
    assert quality_flags["flagged_episodes"] == [0, 1]
    assert quality_flags["flag_reasons"]["0"][0]["reason"] == "early_gripper_transition"
    assert quality_flags["flag_reasons"]["1"][0]["reason"] == "stuck_closed_gripper_no_action"
    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == [0, 1]


def test_run_quality_flag_detection_flags_missing_subtask_state(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action = np.zeros((20, 19), dtype=np.float32)
    state = np.zeros((20, 18), dtype=np.float32)

    _write_episode_parquet(dataset_root, 0, timestamps=timestamps, action=action, state=state)
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 1,
        "total_frames": 20,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the cube"], "length": 20})

    run_quality_flag_detection(dataset_root, static_dir, data_version=DATA_VERSION_DVT2, workers=1)

    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    assert any(issue.get("reason") == "missing_subtask_state" for issue in issues)
    assert json.loads((static_dir / "quality_flagged_episodes.json").read_text())["flagged_episodes"] == [0]


def test_run_quality_flag_detection_flags_multi_sentence_prompt(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action = np.zeros((20, 19), dtype=np.float32)
    state = np.zeros((20, 18), dtype=np.float32)
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

    for episode_id in range(3):
        _write_episode_parquet(
            dataset_root,
            episode_id,
            timestamps=timestamps,
            action=action,
            state=state,
            subtask_state=subtask_state,
        )
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 3,
        "total_frames": 60,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the cube"], "length": 20})
        writer.write({"episode_index": 1, "tasks": ["Pick up the cube, pick up the duck"], "length": 20})
        writer.write({"episode_index": 2, "tasks": ["Pick up the cube."], "length": 20})

    run_quality_flag_detection(dataset_root, static_dir, data_version=DATA_VERSION_DVT2, workers=1)

    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    flagged_reasons = {
        issue["episode"]: issue["reason"]
        for issue in issues
        if issue.get("reason") == "multi_sentence_prompt"
    }
    assert flagged_reasons == {1: "multi_sentence_prompt", 2: "multi_sentence_prompt"}
    assert not any(
        issue.get("episode") == 0 and issue.get("reason") == "multi_sentence_prompt" for issue in issues
    )
    quality_flags = json.loads((static_dir / "quality_flagged_episodes.json").read_text())
    assert quality_flags["flagged_episodes"] == [1, 2]


def test_run_quality_flag_detection_checks_task_index_prompt(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action = np.zeros((20, 19), dtype=np.float32)
    state = np.zeros((20, 18), dtype=np.float32)
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

    parquet_path = _write_episode_parquet(
        dataset_root,
        0,
        timestamps=timestamps,
        action=action,
        state=state,
        subtask_state=subtask_state,
    )
    table = pq.read_table(parquet_path)
    table = table.append_column("task_index", pa.array([1] * len(timestamps), type=pa.int64()))
    pq.write_table(table, parquet_path)

    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 1,
        "total_frames": 20,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the cube"], "length": 20})
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "Pick up the cube"})
        writer.write({"task_index": 1, "task": "Pick up the yellow duck, pick up the brown dog"})

    run_quality_flag_detection(dataset_root, static_dir, data_version=DATA_VERSION_DVT2, workers=1)

    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    prompt_issues = [issue for issue in issues if issue.get("reason") == "multi_sentence_prompt"]
    assert len(prompt_issues) == 1
    assert prompt_issues[0]["metrics"]["prompts"] == ["Pick up the yellow duck, pick up the brown dog"]


def test_run_quality_flag_detection_flags_multiple_task_assignments(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action = np.zeros((20, 19), dtype=np.float32)
    state = np.zeros((20, 18), dtype=np.float32)
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)

    for episode_id in range(3):
        parquet_path = _write_episode_parquet(
            dataset_root,
            episode_id,
            timestamps=timestamps,
            action=action,
            state=state,
            subtask_state=subtask_state,
        )
        if episode_id == 2:
            table = pq.read_table(parquet_path)
            task_indices = [0] * 10 + [2] * 10
            table = table.append_column("task_index", pa.array(task_indices, type=pa.int64()))
            pq.write_table(table, parquet_path)

    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 3,
        "total_frames": 60,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the cube"], "length": 20})
        writer.write(
            {
                "episode_index": 1,
                "tasks": ["Pick up the cube", "Pick up the yellow duck"],
                "length": 20,
            }
        )
        writer.write({"episode_index": 2, "tasks": ["Pick up the cube"], "length": 20})
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "Pick up the cube"})
        writer.write({"task_index": 2, "task": "Pick up the yellow duck"})

    run_quality_flag_detection(dataset_root, static_dir, data_version=DATA_VERSION_DVT2, workers=1)

    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    multi_task_episodes = sorted(
        issue["episode"] for issue in issues if issue.get("reason") == "multiple_task_assignments"
    )
    assert multi_task_episodes == [1, 2]
    quality_flags = json.loads((static_dir / "quality_flagged_episodes.json").read_text())
    assert quality_flags["flagged_episodes"] == [1, 2]


def test_apply_task_assignment_choice_updates_metadata_parquet_and_flags(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(20, dtype=np.float32) / 10.0
    action = np.zeros((20, 19), dtype=np.float32)
    state = np.zeros((20, 18), dtype=np.float32)
    subtask_state = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)
    parquet_path = _write_episode_parquet(
        dataset_root,
        0,
        timestamps=timestamps,
        action=action,
        state=state,
        subtask_state=subtask_state,
    )
    table = pq.read_table(parquet_path)
    table = table.append_column("task_index", pa.array([0] * 10 + [1] * 10, type=pa.int64()))
    pq.write_table(table, parquet_path)

    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 1,
        "total_frames": 20,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "subtask_state": {"dtype": "int32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write(
            {"episode_index": 0, "tasks": ["Pick up the cube", "Pick up the yellow duck"], "length": 20}
        )
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "Pick up the cube"})
        writer.write({"task_index": 1, "task": "Pick up the yellow duck"})

    run_quality_flag_detection(dataset_root, static_dir, data_version=DATA_VERSION_DVT2, workers=1)
    records = list_multiple_task_assignments(dataset_root, static_dir)
    assert records[0]["candidates"] == ["Pick up the cube", "Pick up the yellow duck"]

    result = apply_task_assignment_choice(dataset_root, static_dir, 0, "Pick up the yellow duck")

    assert result["removed_issues"] == 1
    episodes = [
        json.loads(line) for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert episodes[0]["tasks"] == ["Pick up the yellow duck"]
    table = pq.read_table(parquet_path, columns=["task_index"])
    assert set(table["task_index"].to_pylist()) == {1}
    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    assert not any(issue.get("reason") == "multiple_task_assignments" for issue in issues)
    quality_flags = json.loads((static_dir / "quality_flagged_episodes.json").read_text())
    assert quality_flags["flagged_episodes"] == []


def test_prompt_action_mismatch_task_assignment_candidates_and_save(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(10, dtype=np.float32) / 10.0
    action = np.zeros((10, 19), dtype=np.float32)
    state = np.zeros((10, 18), dtype=np.float32)
    parquet_path = _write_episode_parquet(
        dataset_root,
        0,
        timestamps=timestamps,
        action=action,
        state=state,
    )
    table = pq.read_table(parquet_path)
    table = table.append_column("task_index", pa.array([0] * len(timestamps), type=pa.int64()))
    pq.write_table(table, parquet_path)

    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (static_dir).mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 1,
        "total_frames": 10,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 10})
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "Pick up the yellow duck"})
        writer.write({"task_index": 1, "task": "Pick up the brown dog"})
        writer.write({"task_index": 2, "task": "Pick up the green dinosaur"})

    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {
                    "episode": 0,
                    "type": "tagging_prompt_behavior",
                    "reason": "prompt_action_mismatch",
                    "task": "Pick up the yellow duck",
                    "metrics": {"observed_object": "brown dog"},
                }
            ]
        )
    )
    (static_dir / "tagging_prompt_mismatch_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [0],
                "flag_reasons": {
                    "0": [
                        {
                            "type": "tagging_prompt_behavior",
                            "reason": "prompt_action_mismatch",
                            "metrics": {"observed_object": "beige bear"},
                        }
                    ]
                },
            }
        )
    )
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0]}))

    records = list_task_assignment_choices(dataset_root, static_dir)
    assert records[0]["reason"] == "prompt_action_mismatch"
    assert records[0]["candidates"] == [
        "Pick up the yellow duck",
        "Pick up the brown dog",
        "Pick up the green dinosaur",
    ]

    result = apply_task_assignment_choice(dataset_root, static_dir, 0, "Pick up the brown dog")

    assert result["removed_issues"] == 1
    episodes = [
        json.loads(line) for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert episodes[0]["tasks"] == ["Pick up the brown dog"]
    table = pq.read_table(parquet_path, columns=["task_index"])
    assert set(table["task_index"].to_pylist()) == {1}
    assert json.loads((static_dir / "annotation_issues.json").read_text()) == []
    assert (
        json.loads((static_dir / "tagging_prompt_mismatch_flagged_episodes.json").read_text())[
            "flagged_episodes"
        ]
        == []
    )
    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == []


def test_manual_prompt_error_task_assignment_uses_prompt_repair_path(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(10, dtype=np.float32) / 10.0
    action = np.zeros((10, 19), dtype=np.float32)
    state = np.zeros((10, 18), dtype=np.float32)
    parquet_path = _write_episode_parquet(
        dataset_root,
        0,
        timestamps=timestamps,
        action=action,
        state=state,
    )
    table = pq.read_table(parquet_path)
    table = table.append_column("task_index", pa.array([0] * len(timestamps), type=pa.int64()))
    pq.write_table(table, parquet_path)

    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "total_episodes": 1,
        "total_frames": 10,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [19], "names": None},
            "state": {"dtype": "float32", "shape": [18], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 10})
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "Pick up the yellow duck"})
        writer.write({"task_index": 1, "task": "Pick up the brown dog"})
    (static_dir / "manual_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [0],
                "flag_reasons": {
                    "0": [{"type": "manual", "reason": "wrong_prompt"}],
                },
            }
        )
    )
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0]}))

    records = list_task_assignment_choices(dataset_root, static_dir)
    assert records[0]["reason"] == "wrong_prompt"
    assert records[0]["candidates"] == ["Pick up the yellow duck", "Pick up the brown dog"]

    result = apply_task_assignment_choice(
        dataset_root,
        static_dir,
        0,
        "Pick up the green dinosaur",
        reason="wrong_prompt",
    )

    assert result["task_index"] == 2
    episodes = [
        json.loads(line) for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert episodes[0]["tasks"] == ["Pick up the green dinosaur"]
    tasks = [json.loads(line) for line in (dataset_root / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks[-1] == {"task_index": 2, "task": "Pick up the green dinosaur"}
    table = pq.read_table(parquet_path, columns=["task_index"])
    assert set(table["task_index"].to_pylist()) == {2}
    assert json.loads((static_dir / "manual_flagged_episodes.json").read_text())["flagged_episodes"] == []
    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == []


def test_manual_prompt_error_candidates_preserve_prompt_form_and_case(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "data").mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 8,
                "fps": 10,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        )
    )
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["give the yellow duck to me"], "length": 8})
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "give the yellow duck to me"})
        writer.write({"task_index": 1, "task": "Pick up the brown dog"})
        writer.write({"task_index": 2, "task": "Give me the green dinosaur"})
        writer.write({"task_index": 3, "task": "PICK UP THE ORANGE LION"})
    (static_dir / "manual_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [0],
                "flag_reasons": {
                    "0": [
                        {
                            "type": "manual",
                            "reason": "wrong_prompt",
                            "metrics": {"candidates": ["Pick up the brown dog"]},
                        }
                    ],
                },
            }
        )
    )

    records = list_task_assignment_choices(dataset_root, static_dir)

    assert records[0]["candidates"] == [
        "give the yellow duck to me",
        "give the brown dog to me",
        "give the green dinosaur to me",
        "give the orange lion to me",
    ]


def test_prompt_action_mismatch_candidates_from_auto_flag_file(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    timestamps = np.arange(8, dtype=np.float32) / 10.0
    parquet_path = _write_episode_parquet(
        dataset_root,
        0,
        timestamps=timestamps,
        action=np.zeros((8, 19), dtype=np.float32),
        state=np.zeros((8, 18), dtype=np.float32),
    )
    table = pq.read_table(parquet_path)
    table = table.append_column("task_index", pa.array([0] * len(timestamps), type=pa.int64()))
    pq.write_table(table, parquet_path)

    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 8,
                "fps": 10,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {"task_index": {"dtype": "int64", "shape": [1], "names": None}},
            }
        )
    )
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 8})
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "Pick up the yellow duck"})
        writer.write({"task_index": 1, "task": "Pick up the brown dog"})

    (static_dir / "annotation_issues.json").write_text("[]")
    (static_dir / "tagging_prompt_mismatch_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [0],
                "flag_reasons": {
                    "0": [
                        {
                            "type": "tagging_prompt_behavior",
                            "reason": "prompt_action_mismatch",
                            "metrics": {"observed_object": "brown dog"},
                        }
                    ]
                },
            }
        )
    )
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0]}))

    records = list_task_assignment_choices(dataset_root, static_dir)

    assert records[0]["candidates"] == ["Pick up the yellow duck", "Pick up the brown dog"]

    apply_task_assignment_choice(dataset_root, static_dir, 0, "Pick up the brown dog")

    assert (
        json.loads((static_dir / "tagging_prompt_mismatch_flagged_episodes.json").read_text())[
            "flagged_episodes"
        ]
        == []
    )
    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == []


def test_prompt_action_mismatch_candidates_drop_unknown_objects(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "data").mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 8,
                "fps": 10,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        )
    )
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 8})
    with jsonlines.open(dataset_root / "meta" / "tasks.jsonl", mode="w") as writer:
        writer.write({"task_index": 0, "task": "Pick up the yellow duck"})
        writer.write({"task_index": 1, "task": "Pick up the brown dog"})

    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {
                    "episode": 0,
                    "type": "tagging_prompt_behavior",
                    "reason": "prompt_action_mismatch",
                    "task": "Pick up the yellow duck",
                    "metrics": {
                        "observed_object": "purple elephant",
                        "candidates": ["Pick up the yellow duck", "Pick up the purple elephant"],
                    },
                }
            ]
        )
    )

    records = list_task_assignment_choices(dataset_root, static_dir)

    assert records[0]["candidates"] == ["Pick up the yellow duck", "Pick up the brown dog"]


def test_prepare_annotation_issues_preserves_quality_flags(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    issues_path = static_dir / "annotation_issues.json"
    issues_path.write_text(
        json.dumps(
            [
                {"episode": 0, "type": "quality_flag", "reason": "early_gripper_transition"},
                {"episode": 0, "type": "error", "reason": "old prepare issue"},
                {"episode": 1, "type": "object_labeling", "reason": "missing_target_detection"},
            ]
        )
    )

    prepare_script._write_annotation_issues(
        static_dir,
        [{"episode": 0, "type": "multi_gripper", "reason": "new prepare issue", "frames": [3]}],
        scanned_episodes=[0],
    )

    issues = json.loads(issues_path.read_text())
    assert {"episode": 0, "type": "quality_flag", "reason": "early_gripper_transition"} in issues
    assert {"episode": 1, "type": "object_labeling", "reason": "missing_target_detection"} in issues
    assert not any(issue.get("reason") == "old prepare issue" for issue in issues)
    assert any(issue.get("reason") == "new prepare issue" for issue in issues)


def _make_delete_dataset(root: Path) -> DummyDataset:
    episodes = {
        episode_id: {
            "episode_index": episode_id,
            "tasks": ["Pick up the cube"],
            "length": 2,
        }
        for episode_id in range(3)
    }
    features = {
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    info = {
        "fps": 10,
        "features": features,
        "total_episodes": 3,
        "total_frames": 6,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with jsonlines.open(root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write_all(episodes.values())
    for episode_id in episodes:
        _write_episode_parquet(
            root,
            episode_id,
            timestamps=np.array([0.0, 0.1], dtype=np.float32),
            frame_index=[0, 1],
            index=[episode_id * 2, episode_id * 2 + 1],
        )
        parquet_path = root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        table = pq.read_table(parquet_path).append_column(
            "episode_index",
            pa.array([episode_id, episode_id], type=pa.int64()),
        )
        pq.write_table(table, parquet_path)

    meta = DummyMeta(root, features, episodes)
    meta.info = info
    return DummyDataset(root, meta)


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_delete_episodes_inplace_rolls_back_all_files_on_failure(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0, 2]}))
    source_videos = tmp_path / "source_videos"
    source_videos.mkdir()
    (static_dir / "videos").symlink_to(source_videos, target_is_directory=True)
    dataset = _make_delete_dataset(dataset_root)
    root_before = _file_snapshot(dataset_root)
    static_before = _file_snapshot(static_dir)
    info_before = copy.deepcopy(dataset.meta.info)
    episodes_before = copy.deepcopy(dataset.meta.episodes)

    def _fail_static_reindex(static_folder, *_args, **_kwargs):
        (Path(static_folder) / "flagged_episodes.json").write_text("partially rewritten")
        raise RuntimeError("injected static reindex failure")

    monkeypatch.setattr(
        "lerobot.data_platform.precompute.preprocess.delete_episodes.reindex_static_after_episode_delete",
        _fail_static_reindex,
    )

    with pytest.raises(RuntimeError, match="restored the pre-delete state"):
        delete_episodes_inplace(dataset, [0], static_folder=static_dir)

    assert _file_snapshot(dataset_root) == root_before
    assert _file_snapshot(static_dir) == static_before
    assert (static_dir / "videos").is_symlink()
    assert (static_dir / "videos").readlink() == source_videos
    assert dataset.meta.info == info_before
    assert dataset.meta.episodes == episodes_before
    assert dataset.total_episodes == 3
    assert dataset.total_frames == 6
    assert not list(tmp_path.glob(".dataset.delete-rollback-*"))


def test_delete_episodes_inplace_commits_and_discards_snapshot(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0, 2]}))
    dataset = _make_delete_dataset(dataset_root)

    result = delete_episodes_inplace(dataset, [1], static_folder=static_dir)

    assert result == {"deleted_episode_ids": [1], "new_total_episodes": 2, "next_episode": 1}
    assert sorted(path.name for path in (dataset_root / "data" / "chunk-000").glob("*.parquet")) == [
        "episode_000000.parquet",
        "episode_000001.parquet",
    ]
    table = pq.read_table(dataset_root / "data" / "chunk-000" / "episode_000001.parquet")
    assert table.column("episode_index").to_pylist() == [1, 1]
    assert table.column("index").to_pylist() == [2, 3]
    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == [0, 1]
    assert dataset.total_episodes == 2
    assert dataset.total_frames == 4
    assert not list(tmp_path.glob(".dataset.delete-rollback-*"))


def test_reindex_static_after_episode_delete_updates_feature_artifacts(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0, 2, 3]}))
    (static_dir / "quality_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [2, 3],
                "flag_reasons": {
                    "2": [{"reason": "early_gripper_transition"}],
                    "3": [{"reason": "missing_subtask_state"}],
                },
                "summary": {"quality_episode_count": 2, "flagged_episode_count": 2},
            }
        )
    )
    (static_dir / "trim_annotations.json").write_text(json.dumps({"2": {"start": 1.0}, "3": {"start": 2.0}}))
    (static_dir / "subtask_annotations.json").write_text(json.dumps({"0": {"stage": 0}, "2": {"stage": 1}}))
    (static_dir / "viewer_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "total_episodes": 4,
                "total_frames": 100,
                "episodes": [
                    {"episode_index": 0, "length": 10, "tasks": ["a"]},
                    {"episode_index": 1, "length": 20, "tasks": ["deleted"]},
                    {"episode_index": 2, "length": 30, "tasks": ["b"]},
                    {"episode_index": 3, "length": 40, "tasks": ["c"]},
                ],
            }
        )
    )
    (static_dir / "prompt_assignments_pending.json").write_text(
        json.dumps(
            {
                "version": 1,
                "assignments": [
                    {"episode_index": 1, "selected_task": "deleted"},
                    {"episode_index": 3, "selected_task": "keep"},
                ],
            }
        )
    )
    construction_plan_path = static_dir / "construction_plan.json"
    construction_plan_path.write_text(
        json.dumps(
            {
                "records": [
                    {"new_episode_index": 1, "new_task": "deleted"},
                    {"new_episode_index": 3, "new_task": "keep"},
                ],
            }
        )
    )
    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {"episode": 1, "type": "error", "reason": "deleted"},
                {"episode": 2, "type": "quality_flag", "reason": "kept"},
            ]
        )
    )

    labeling_dir = static_dir / "labeling"
    tagging_dir = static_dir / "tagging"
    labeling_dir.mkdir()
    tagging_dir.mkdir()
    (labeling_dir / "labels.jsonl").write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {"episode_index": 1, "task": "delete"},
                {"episode_index": 2, "task": "keep"},
            ]
        )
    )
    (tagging_dir / "tags_reviewed_trial.jsonl").write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {"episode_index": 0, "tags": {"arm": "left"}},
                {"episode_index": 3, "tags": {"arm": "right"}},
            ]
        )
    )
    for dirname in ["analysis", "embedding", "compare", "construction"]:
        directory = static_dir / dirname
        directory.mkdir()
        (directory / "stale.txt").write_text("stale")

    result = reindex_static_after_episode_delete(static_dir, {1}, {0: 0, 2: 1, 3: 2})

    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == [0, 1, 2]
    quality_flags = json.loads((static_dir / "quality_flagged_episodes.json").read_text())
    assert quality_flags["flagged_episodes"] == [1, 2]
    assert sorted(quality_flags["flag_reasons"].keys()) == ["1", "2"]
    assert quality_flags["summary"]["quality_episode_count"] == 2
    assert json.loads((static_dir / "trim_annotations.json").read_text()) == {
        "1": {"start": 1.0},
        "2": {"start": 2.0},
    }
    assert json.loads((static_dir / "subtask_annotations.json").read_text()) == {
        "0": {"stage": 0},
        "1": {"stage": 1},
    }
    manifest = json.loads((static_dir / "viewer_manifest.json").read_text())
    assert manifest["total_episodes"] == 3
    assert manifest["total_frames"] == 80
    assert [row["episode_index"] for row in manifest["episodes"]] == [0, 1, 2]
    assert [row["tasks"][0] for row in manifest["episodes"]] == ["a", "b", "c"]
    pending = json.loads((static_dir / "prompt_assignments_pending.json").read_text())
    assert pending["assignments"] == [{"episode_index": 2, "selected_task": "keep"}]
    assert _reindex_construction_plan(construction_plan_path, {1}, {0: 0, 2: 1, 3: 2})
    construction_plan = json.loads(construction_plan_path.read_text())
    assert construction_plan["records"] == [{"new_episode_index": 2, "new_task": "keep"}]
    assert json.loads((static_dir / "annotation_issues.json").read_text()) == [
        {"episode": 1, "type": "quality_flag", "reason": "kept"}
    ]
    labels = [json.loads(line) for line in (labeling_dir / "labels.jsonl").read_text().splitlines()]
    tags = [json.loads(line) for line in (tagging_dir / "tags_reviewed_trial.jsonl").read_text().splitlines()]
    assert labels == [{"episode_index": 1, "task": "keep"}]
    assert tags == [
        {"episode_index": 0, "tags": {"arm": "left"}},
        {"episode_index": 2, "tags": {"arm": "right"}},
    ]
    assert set(result["invalidated"]) == {"analysis", "embedding", "compare", "construction"}
    assert not (static_dir / "embedding").exists()


def test_labeling_missing_target_flags_replace_auto_flags_and_keep_manual(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {
                    "episode": 0,
                    "type": "object_labeling",
                    "reason": "missing_target_detection",
                    "task": "old",
                },
                {
                    "episode": 2,
                    "type": "object_labeling",
                    "reason": "missing_target_detection",
                    "task": "old",
                },
                {"episode": 2, "type": "error", "reason": "existing_stage_issue"},
            ]
        )
    )
    (static_dir / "labeling_flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0, 2]}))
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0, 2, 3]}))

    summary = _sync_missing_target_flags(
        static_dir,
        {
            0: {
                "episode_index": 0,
                "task": "Pick up the cube",
                "parsed": {"target": "cube", "direction": None, "reference": None},
                "detections_target": [],
            },
            2: {
                "episode_index": 2,
                "task": "Pick up the ball",
                "parsed": {"target": "ball", "direction": None, "reference": None},
                "detections_target": [{"bbox": {"left": 1, "top": 1, "right": 2, "bottom": 2}}],
            },
            4: {
                "episode_index": 4,
                "task": "Pick up the failed object",
                "parsed": {"target": "failed object", "direction": None, "reference": None},
                "detections_target": [],
                "error": "qwen_dashscope_failed",
            },
        },
        backend="grounding_dino",
        variant="grounding_dino",
        scanned_episodes={0, 2, 4},
    )

    issues = json.loads((static_dir / "annotation_issues.json").read_text())
    assert summary["missing_target_count"] == 1
    assert any(
        issue.get("episode") == 0 and issue.get("reason") == "missing_target_detection" for issue in issues
    )
    assert not any(
        issue.get("episode") == 2 and issue.get("reason") == "missing_target_detection" for issue in issues
    )
    assert not any(
        issue.get("episode") == 4 and issue.get("reason") == "missing_target_detection" for issue in issues
    )
    assert {"episode": 2, "type": "error", "reason": "existing_stage_issue"} in issues
    assert json.loads((static_dir / "labeling_flagged_episodes.json").read_text())["flagged_episodes"] == [0]
    assert json.loads((static_dir / "flagged_episodes.json").read_text())["flagged_episodes"] == [0, 3]


def test_assign_subtask_states_handles_five_and_six_stage_annotations():
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)

    pick_states = assign_subtask_states(
        timestamps,
        {
            "stage0_end": 0.75,
            "stage2_start": 1.25,
            "stage2_end": 2.0,
            "stage4_start": 3.0,
        },
    )
    give_states = assign_subtask_states(
        timestamps,
        {
            "stage0_end": 0.75,
            "stage2_start": 1.25,
            "stage2_end": 2.0,
            "stage4_start": 3.0,
            "stage4_end": 4.0,
            "is_give": True,
        },
    )

    assert pick_states == [0, 0, 1, 2, 3, 4, 4]
    assert give_states == [0, 0, 1, 2, 3, 4, 5]

    direct_give_states = assign_subtask_states(
        timestamps,
        {
            "stage0_end": 0.75,
            "stage4_start": 3.0,
            "stage4_end": 4.0,
            "is_give": True,
            "direct_give": True,
        },
    )
    assert direct_give_states == [0, 0, 3, 3, 3, 4, 5]


def test_gripper_normalization_handles_new_0_100_encoding():
    values = np.zeros((3, 18), dtype=np.float32)
    values[:, 7] = [0.0, 50.0, 100.0]
    values[:, 15] = [120.0, 50.0, 0.0]
    values[:, 16] = [10.0, 20.0, 30.0]

    normalized = normalize_gripper_columns(values, "action", DATA_VERSION_DVT2)

    np.testing.assert_allclose(normalized[:, 7], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(normalized[:, 15], [1.2, 0.5, 0.0])
    np.testing.assert_allclose(normalized[:, 16], [10.0, 20.0, 30.0])
    assert normalize_gripper_csv_value("state_7", "75", DATA_VERSION_DVT2) == "0.75"
    assert normalize_gripper_csv_value("state_7", "120", DATA_VERSION_DVT2) == "1.2"
    assert normalize_gripper_csv_value("state_7", "1.2", DATA_VERSION_DVT2) == "1.2"
    assert normalize_gripper_csv_value("state_7", "0.75", DATA_VERSION_DVT2) == "0.75"


def test_build_dataset_analysis_uses_exist_label_and_duration_buckets(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    csv_dir = static_dir / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / "episode_000000_ds1.csv").write_text(
        "timestamp,stage,yellow duck,brown dog,exist\n0,0,1,0,1\n2,2,1,0,1\n4,2,1,0,1\n6,1,1,0,1\n"
    )
    (csv_dir / "episode_000001_ds1.csv").write_text(
        "timestamp,stage,yellow duck,brown dog,exist\n0,0,0,1,1\n6,2,0,1,1\n12,1,0,1,1\n"
    )

    meta = DummyMeta(
        dataset_root,
        {
            "exist_label": {
                "dtype": "int32",
                "shape": [2],
                "names": ["yellow duck", "brown dog"],
            },
            "exist": {"dtype": "int32", "shape": [1], "names": None},
        },
        {
            0: {"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 2},
            1: {"episode_index": 1, "tasks": ["Give the brown dog to me"], "length": 2},
        },
    )

    analysis = build_dataset_analysis(dataset_root, meta, static_dir, episodes=[0, 1])

    exist_keys = {row["key"] for row in analysis["exist_distribution"]}
    assert exist_keys == {"yellow duck", "brown dog"}
    assert analysis["episodes"][0]["exist_label_source"] == "exist_label"
    assert analysis["episodes"][0]["duration_bucket"] == "5-10s"
    assert analysis["episodes"][1]["duration_bucket"] == "10-15s"
    assert analysis["duration_episode_count"] == 2
    assert [row["key"] for row in analysis["stage_distribution"]] == ["0", "2", "4", "5"]
    assert any(row["key"] == "5-10s" and row["count"] == 1 for row in analysis["duration_distribution"])
    assert any(row["key"] == "10-15s" and row["count"] == 1 for row in analysis["duration_distribution"])


def test_build_dataset_analysis_aliases_scalar_exist_label_0(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    static_dir = tmp_path / "static"
    csv_dir = static_dir / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / "episode_000000_ds1.csv").write_text("timestamp,exist_label_0\n0,0\n1,0\n")
    (csv_dir / "episode_000001_ds1.csv").write_text("timestamp,exist_label_0\n0,1\n1,1\n")

    meta = DummyMeta(
        dataset_root,
        {"exist_label": {"dtype": "int32", "shape": [1], "names": None}},
        {
            0: {"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 2},
            1: {"episode_index": 1, "tasks": ["Pick up the brown dog"], "length": 2},
        },
    )

    analysis = build_dataset_analysis(dataset_root, meta, static_dir, episodes=[0, 1])

    assert [row["key"] for row in analysis["exist_distribution"]] == ["exist_label"]
    assert analysis["exist_distribution"][0]["true"] == 2
    assert analysis["exist_distribution"][0]["false"] == 2
    assert analysis["episodes"][0]["exist_counts"]["exist_label"]["false"] == 2
    assert analysis["episodes"][1]["exist_counts"]["exist_label"]["true"] == 2


def test_get_columns_info_includes_int64_exist_label(tmp_path: Path):
    meta = DummyMeta(
        tmp_path / "dataset",
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "exist_label": {
                "dtype": "int64",
                "shape": [4],
                "names": ["yellow duck", "brown dog", "orange lion", "green dinosaur"],
            },
        },
        {},
    )

    columns, ignored_columns, selected_columns = get_columns_info(meta)

    assert "exist_label" in selected_columns
    assert "exist_label" not in ignored_columns
    assert next(row for row in columns if row["key"] == "exist_label")["value"] == [
        "yellow duck",
        "brown dog",
        "orange lion",
        "green dinosaur",
    ]


def test_get_columns_info_uses_scalar_exist_label_name(tmp_path: Path):
    meta = DummyMeta(
        tmp_path / "dataset",
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "exist_label": {"dtype": "int32", "shape": [1], "names": None},
        },
        {},
    )

    columns, ignored_columns, selected_columns = get_columns_info(meta)

    assert "exist_label" in selected_columns
    assert "exist_label" not in ignored_columns
    assert next(row for row in columns if row["key"] == "exist_label")["value"] == ["exist_label"]


def test_write_episode_csv_writes_exist_label_columns(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    timestamps = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    exist_label = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    _write_episode_parquet(dataset_root, 0, timestamps=timestamps, exist_label=exist_label)

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "exist_label": {
            "dtype": "int64",
            "shape": [4],
            "names": ["yellow duck", "brown dog", "orange lion", "green dinosaur"],
        },
    }
    meta = DummyMeta(
        dataset_root, features, {0: {"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 3}}
    )
    out_path = tmp_path / "static" / "csv" / "episode_000000_ds1.csv"

    ok, boundaries, issues = write_episode_csv(
        dataset_root,
        meta,
        0,
        out_path,
        max_frames=None,
        downsample=None,
        overwrite=True,
    )

    assert ok is True
    assert boundaries is None
    assert issues == []
    header = out_path.read_text().splitlines()[0].split(",")
    assert header == ["timestamp", "yellow duck", "brown dog", "orange lion", "green dinosaur"]


def test_write_episode_csv_writes_scalar_exist_label_column(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    timestamps = np.array([0.0, 0.1], dtype=np.float32)
    exist_label = np.array([[0], [1]], dtype=np.int64)
    _write_episode_parquet(dataset_root, 0, timestamps=timestamps, exist_label=exist_label)

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "exist_label": {"dtype": "int32", "shape": [1], "names": None},
    }
    meta = DummyMeta(
        dataset_root, features, {0: {"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 2}}
    )
    out_path = tmp_path / "static" / "csv" / "episode_000000_ds1.csv"

    ok, boundaries, issues = write_episode_csv(
        dataset_root,
        meta,
        0,
        out_path,
        max_frames=None,
        downsample=None,
        overwrite=True,
    )

    assert ok is True
    assert boundaries is None
    assert issues == []
    assert out_path.read_text().splitlines()[0].split(",") == ["timestamp", "exist_label"]


def test_fix_episode_indices_updates_parquet_and_metadata(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)

    timestamps0 = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    timestamps1 = np.array([2.0, 2.1], dtype=np.float32)
    _write_episode_parquet(
        dataset_root,
        0,
        timestamps=timestamps0,
        frame_index=[5, 6, 7],
        index=[9, 10, 11],
    )
    _write_episode_parquet(
        dataset_root,
        1,
        timestamps=timestamps1,
        frame_index=[2, 3],
        index=[99, 100],
    )

    (dataset_root / "meta" / "info.json").write_text(json.dumps({"total_frames": 999}, indent=2))
    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="w") as writer:
        writer.write({"episode_index": 0, "length": 99})
        writer.write({"episode_index": 1, "length": 98})

    meta = DummyMeta(
        dataset_root,
        {"timestamp": {"dtype": "float32", "shape": [1], "names": None}},
        {
            0: {"episode_index": 0, "tasks": ["Pick up the cube"], "length": 99},
            1: {"episode_index": 1, "tasks": ["Pick up the cube"], "length": 98},
        },
    )

    assert fix_episode_indices(dataset_root, meta, [0, 1]) is True

    table0 = pq.read_table(dataset_root / "data" / "chunk-000" / "episode_000000.parquet")
    table1 = pq.read_table(dataset_root / "data" / "chunk-000" / "episode_000001.parquet")
    assert table0.column("frame_index").to_pylist() == [0, 1, 2]
    assert np.allclose(table0.column("timestamp").to_pylist(), [0.0, 0.1, 0.2])
    assert table0.column("index").to_pylist() == [0, 1, 2]
    assert table1.column("frame_index").to_pylist() == [0, 1]
    assert table1.column("index").to_pylist() == [3, 4]

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    assert info["total_frames"] == 5

    with jsonlines.open(dataset_root / "meta" / "episodes.jsonl", mode="r") as reader:
        rows = list(reader)
    assert rows == [
        {"episode_index": 0, "length": 3},
        {"episode_index": 1, "length": 2},
    ]


def test_prepare_script_generates_csv_without_changing_cli(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dummy_meta = _prepare_script_dataset(dataset_root)

    monkeypatch.setattr(prepare_script, "LeRobotDatasetMetadata", lambda repo_id, root: dummy_meta)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_platform.py",
            "--root",
            str(dataset_root),
            "--prepare-videos",
            "0",
            "--run-visualize",
            "0",
        ],
    )

    prepare_script.main()

    csv_path = dataset_root.parent / "vis" / "local_vis_dataset" / "static" / "csv" / "episode_000000_ds1.csv"
    assert csv_path.is_file()
    header = csv_path.read_text().splitlines()[0]
    assert "stage" in header


def test_prepare_script_visualize_only_still_prepares_csv(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dummy_meta = _prepare_script_dataset(dataset_root)

    monkeypatch.setattr(prepare_script, "LeRobotDatasetMetadata", lambda repo_id, root: dummy_meta)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_platform.py",
            "--root",
            str(dataset_root),
            "--visualize-only",
            "1",
            "--prepare-videos",
            "0",
            "--run-visualize",
            "0",
        ],
    )

    prepare_script.main()

    csv_path = dataset_root.parent / "vis" / "local_vis_dataset" / "static" / "csv" / "episode_000000_ds1.csv"
    assert csv_path.is_file()


def test_prepare_script_overwrite_parquet_updates_meta_and_stats(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dummy_meta = _prepare_script_dataset(dataset_root)

    monkeypatch.setattr(prepare_script, "LeRobotDatasetMetadata", lambda repo_id, root: dummy_meta)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_platform.py",
            "--root",
            str(dataset_root),
            "--prepare-videos",
            "0",
            "--run-visualize",
            "0",
            "--overwrite-parquet",
            "1",
        ],
    )

    prepare_script.main()

    parquet_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(parquet_path)
    assert "subtask_state" in table.column_names
    assert "subtask" in table.column_names

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    assert "subtask_state" in info["features"]
    assert "subtask" in info["features"]

    with jsonlines.open(dataset_root / "meta" / "episodes_stats.jsonl", mode="r") as reader:
        rows = list(reader)
    assert "subtask_state" in rows[0]["stats"]
