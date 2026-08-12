import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from lerobot.data_platform.precompute.dataset_io import V3DatasetMetadata, read_episode_table
from lerobot.data_platform.precompute.preprocess.trajectory_cleanup import (
    _butterworth_zero_phase,
    _probe_video_frames,
    _rewrite_video_without_ranges,
    _video_encoder_args,
    run_trajectory_cleanup,
)


def _write_video(path: Path, frames: int, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((16, 16, 3), index * 8, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_v3_dataset(root: Path, *, with_video: bool) -> Path:
    fps = 30
    lengths = [6, 6, 6]
    features = {
        "state": {"dtype": "float32", "shape": [16], "names": ["state"], "fps": fps},
        "action": {"dtype": "float32", "shape": [16], "names": ["actions"], "fps": fps},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "subtask_state": {"dtype": "int32", "shape": [1], "names": None, "fps": fps},
    }
    if with_video:
        features["image"] = {
            "dtype": "video",
            "shape": [16, 16, 3],
            "names": ["height", "width", "channel"],
            "video.fps": fps,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.g": 2,
            "video.crf": 18,
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "total_episodes": 3,
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4" if with_video else None
        ),
        "features": features,
    }
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text("{}")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick duck"], name="task")).to_parquet(
        root / "meta" / "tasks.parquet"
    )

    rows = []
    episode_rows = []
    offset = 0
    for episode_index, length in enumerate(lengths):
        for frame_index in range(length):
            base = episode_index * 0.1 + frame_index * 0.005
            state = np.full(16, base, dtype=np.float32)
            action = state.copy()
            action[:7] += 0.01 if frame_index % 2 else -0.01
            action[8:15] += 0.01 if frame_index % 2 else -0.01
            action[7] = float(episode_index % 2)
            action[15] = float(episode_index % 2)
            state[7] = action[7]
            state[15] = action[15]
            if episode_index == 0 and frame_index == 0:
                state[:7] += 1.0
                state[8:15] += 1.0
                action[:7] += 1.0
                action[8:15] += 1.0
            rows.append(
                {
                    "state": state.tolist(),
                    "action": action.tolist(),
                    "timestamp": frame_index / fps,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": offset + frame_index,
                    "task_index": 0,
                    "subtask_state": 0 if frame_index < 3 else 1,
                }
            )
        episode = {
            "episode_index": episode_index,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": offset,
            "dataset_to_index": offset + length,
            "tasks": ["pick duck"],
            "length": length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        if with_video:
            episode.update(
                {
                    "videos/image/chunk_index": 0,
                    "videos/image/file_index": 0,
                    "videos/image/from_timestamp": offset / fps,
                    "videos/image/to_timestamp": (offset + length) / fps,
                }
            )
        episode_rows.append(episode)
        offset += length
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame(episode_rows).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    if with_video:
        _write_video(root / "videos" / "image" / "chunk-000" / "file-000.mp4", sum(lengths), fps)
    return root


def _make_v21_dataset(root: Path) -> Path:
    fps = 30
    lengths = [6, 6, 6]
    features = {
        "state": {"dtype": "float32", "shape": [16], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [16], "names": ["actions"]},
        "image": {
            "dtype": "image",
            "shape": [16, 16, 3],
            "names": ["height", "width", "channel"],
        },
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
        "total_episodes": 3,
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick duck"}) + "\n")

    episode_rows = []
    stats_rows = []
    offset = 0
    for episode_index, length in enumerate(lengths):
        rows = []
        for frame_index in range(length):
            base = episode_index * 0.1 + frame_index * 0.005
            state = np.full(16, base, dtype=np.float32)
            action = state.copy()
            action[:7] += 0.01 if frame_index % 2 else -0.01
            action[8:15] += 0.01 if frame_index % 2 else -0.01
            action[7] = float(episode_index % 2)
            action[15] = float(episode_index % 2)
            state[7] = action[7]
            state[15] = action[15]
            if episode_index == 0 and frame_index == 0:
                state[:7] += 1.0
                state[8:15] += 1.0
                action[:7] += 1.0
                action[8:15] += 1.0
            rows.append(
                {
                    "state": state.tolist(),
                    "action": action.tolist(),
                    "image": {"bytes": bytes([episode_index, frame_index]), "path": None},
                    "timestamp": frame_index / fps,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": offset + frame_index,
                    "task_index": 0,
                    "subtask_state": 0 if frame_index < 3 else 1,
                }
            )
        path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        episode_rows.append({"episode_index": episode_index, "tasks": ["pick duck"], "length": length})
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "image": {
                        "min": [[[0.0]], [[0.0]], [[0.0]]],
                        "max": [[[1.0]], [[1.0]], [[1.0]]],
                        "mean": [[[0.5]], [[0.5]], [[0.5]]],
                        "std": [[[0.1]], [[0.1]], [[0.1]]],
                        "count": [length],
                    }
                },
            }
        )
        offset += length
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episode_rows))
    (root / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stats_rows))
    return root


def test_butterworth_zero_phase_reduces_high_frequency_motion_and_preserves_endpoints():
    time = np.arange(120) / 30.0
    values = (0.2 * np.sin(2 * np.pi * time) + 0.02 * np.sin(2 * np.pi * 10 * time))[:, None]
    filtered = _butterworth_zero_phase(values, cutoff_hz=4.0, fps=30.0)

    assert np.array_equal(filtered[[0, -1]], values[[0, -1]])
    assert np.sqrt(np.mean(np.diff(filtered, n=2, axis=0) ** 2)) < 0.4 * np.sqrt(
        np.mean(np.diff(values, n=2, axis=0) ** 2)
    )


def test_av1_encoder_uses_an_available_ffmpeg_backend():
    args = _video_encoder_args(
        {
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.crf": 30,
            "video.preset": 12,
            "video.g": 2,
        }
    )
    assert args[0] == "-c:v"
    assert args[1] in {"libsvtav1", "libaom-av1"}


@pytest.mark.skipif(
    shutil.which("ffprobe") is None or "libsvtav1" not in av.codec.codecs_available,
    reason="ffprobe and PyAV libsvtav1 are required",
)
def test_av1_rewrite_uses_pyav_svt_and_preserves_requested_frame_count(tmp_path: Path):
    src = tmp_path / "input.mp4"
    dst = tmp_path / "output.mp4"
    _write_video(src, frames=12, fps=30)

    _rewrite_video_without_ranges(
        src,
        dst,
        [(0, 0), (5, 5)],
        {
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.crf": 30,
            "video.preset": 12,
            "video.g": 2,
        },
        30,
    )

    assert _probe_video_frames(dst) == 10


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are required",
)
def test_trajectory_cleanup_dry_run_then_rewrites_data_video_and_stats(tmp_path: Path):
    src = _make_v3_dataset(tmp_path / "src", with_video=True)
    out = tmp_path / "clean"

    dry_run = run_trajectory_cleanup(
        src,
        out,
        delete_episode_ids=(1,),
        initial_jump_threshold_rad=0.5,
        stabilize_final_stage=False,
    )
    assert dry_run.dry_run is True
    assert dry_run.summary["drop_first_frame_episodes"] == [0]
    assert dry_run.total_episodes == 2
    assert dry_run.total_frames == 11
    assert not out.exists()

    result = run_trajectory_cleanup(
        src,
        out,
        delete_episode_ids=(1,),
        initial_jump_threshold_rad=0.5,
        stabilize_final_stage=False,
        workers=1,
        dry_run=False,
    )

    assert result.total_episodes == 2
    meta = V3DatasetMetadata("local/clean", out)
    first = read_episode_table(out, meta, 0)
    second = read_episode_table(out, meta, 1)
    assert first.num_rows == 5
    assert second.num_rows == 6
    assert first["frame_index"].to_pylist() == list(range(5))
    assert first["index"].to_pylist() == list(range(5))
    assert second["index"].to_pylist() == list(range(5, 11))
    assert second["episode_index"].to_pylist() == [1] * 6
    assert first["action"].to_pylist()[0][7] == 0.0
    assert second["action"].to_pylist()[0][7] == 0.0
    assert meta.stats["action"]["count"] == [11]
    assert _probe_video_frames(out / "videos" / "image" / "chunk-000" / "file-000.mp4") == 11
    provenance = json.loads((out / "meta" / "preprocess_trajectory_cleanup.json").read_text())
    assert provenance["delete_episodes"] == [1]
    assert provenance["drop_first_frame_episodes"] == [0]
    assert provenance["rewritten_video_shards"] == 1

    assert pq.read_table(src / "data" / "chunk-000" / "file-000.parquet").num_rows == 18
    assert _probe_video_frames(src / "videos" / "image" / "chunk-000" / "file-000.mp4") == 18


def test_trajectory_cleanup_preserves_v21_layout_and_embedded_images(tmp_path: Path):
    src = _make_v21_dataset(tmp_path / "src_v21")
    out = tmp_path / "clean_v21"

    result = run_trajectory_cleanup(
        src,
        out,
        delete_episode_ids=(1,),
        initial_jump_threshold_rad=0.5,
        stabilize_final_stage=False,
        workers=2,
        dry_run=False,
    )

    assert result.summary["source_format"] == "v2.1"
    assert result.summary["drop_first_frame_episodes"] == [0]
    assert result.total_episodes == 2
    assert result.total_frames == 11
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 11
    episodes = [json.loads(line) for line in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in episodes] == [0, 1]
    assert [row["length"] for row in episodes] == [5, 6]
    first = pq.read_table(out / "data" / "chunk-000" / "episode_000000.parquet")
    second = pq.read_table(out / "data" / "chunk-000" / "episode_000001.parquet")
    assert first["frame_index"].to_pylist() == list(range(5))
    assert first["index"].to_pylist() == list(range(5))
    assert second["episode_index"].to_pylist() == [1] * 6
    assert second["index"].to_pylist() == list(range(5, 11))
    assert first["image"][0].as_py()["bytes"] == bytes([0, 1])
    assert second["image"][0].as_py()["bytes"] == bytes([2, 0])
    stats = [json.loads(line) for line in (out / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in stats] == [0, 1]
    assert stats[0]["stats"]["action"]["count"] == [5]
    assert (
        json.loads((out / "meta" / "preprocess_trajectory_cleanup.json").read_text())[
            "rewritten_video_shards"
        ]
        == 0
    )

    assert pq.read_table(src / "data" / "chunk-000" / "episode_000000.parquet").num_rows == 6
    assert (
        pq.read_table(src / "data" / "chunk-000" / "episode_000002.parquet")["episode_index"].to_pylist()
        == [2] * 6
    )
