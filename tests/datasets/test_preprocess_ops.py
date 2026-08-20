import io
import json
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from lerobot.data_platform.precompute import v3_viewer as v3_viewer_module
from lerobot.data_platform.precompute.compare.stats import action_stats, metadata_stats
from lerobot.data_platform.precompute.construction import run_construction
from lerobot.data_platform.precompute.construction.review import (
    finalize as finalize_construction,
)
from lerobot.data_platform.precompute.construction.review import (
    save_decision as save_construction_decision,
)
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    read_episode_frame_image,
    read_episode_table,
    replace_episode_column,
    update_episode_metadata,
)
from lerobot.data_platform.precompute.mutations import (
    fix_episode_indices,
    update_episode_stats_for_subtask_state,
    update_info_features,
    write_subtask_state_to_parquet,
    write_subtask_text_to_parquet,
)
from lerobot.data_platform.precompute.preprocess import dataset_version as dataset_version_module
from lerobot.data_platform.precompute.preprocess import (
    default_v3_path,
    delete_episodes_inplace,
    detect_dataset_version,
    repair_v3_video_timestamps,
    run_convert_action,
    run_convert_v3,
    run_drop_field,
    run_fix_prompt_prepositions,
    run_flag_fix,
    run_lowercase_prompts,
    run_merge,
    run_quality_flag_detection,
    run_smooth_action,
    run_split,
    run_standardize_dataset,
    run_subtract,
    validate_v3_dataset,
)
from lerobot.data_platform.precompute.preprocess.flag_fixes import (
    trim_v3_episode_inplace,
)
from lerobot.data_platform.precompute.preprocess.quality_flags import (
    apply_task_assignment_choice,
)
from lerobot.data_platform.precompute.tagging.runner import run_tagging


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_v3_viewer_clip_failure_preserves_existing_cache(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.mp4"
    target = tmp_path / "viewer.mp4"
    source.write_bytes(b"source")
    target.write_bytes(b"old-cache")
    commands = []

    def _run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"partial-cache")
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(v3_viewer_module.subprocess, "run", _run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        v3_viewer_module._clip_video(source, target, 0.0, 1.0)

    assert Path(commands[0][-1]) != target
    assert target.read_bytes() == b"old-cache"
    assert not list(tmp_path.glob(".viewer.*.mp4"))


def _make_dataset(root: Path, task: str = "pick duck", task_index: int = 0) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    info = {
        "robot_type": "dummy",
        "fps": 10,
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [17], "names": None},
            "state": {"dtype": "float32", "shape": [17], "names": None},
            "old_field": {"dtype": "float32", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": task_index, "task": task}])
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": [task], "length": 2},
            {"episode_index": 1, "tasks": [task], "length": 3},
        ],
    )
    _write_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {
                    "episode_index": {"min": [0], "max": [0], "mean": [0.0], "std": [0.0]},
                    "index": {"min": [0], "max": [1], "mean": [0.5], "std": [0.0]},
                    "action": {
                        "min": [[0.0] * 17],
                        "max": [[1.0] * 17],
                        "mean": [[0.5] * 17],
                        "std": [[0.1] * 17],
                    },
                    "state": {
                        "min": [[0.0] * 17],
                        "max": [[1.0] * 17],
                        "mean": [[0.5] * 17],
                        "std": [[0.1] * 17],
                    },
                    "old_field": {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.1]},
                },
            },
            {
                "episode_index": 1,
                "stats": {
                    "episode_index": {"min": [1], "max": [1], "mean": [1.0], "std": [0.0]},
                    "index": {"min": [2], "max": [4], "mean": [3.0], "std": [0.0]},
                    "action": {
                        "min": [[0.0] * 17],
                        "max": [[1.0] * 17],
                        "mean": [[0.5] * 17],
                        "std": [[0.1] * 17],
                    },
                    "state": {
                        "min": [[0.0] * 17],
                        "max": [[1.0] * 17],
                        "mean": [[0.5] * 17],
                        "std": [[0.1] * 17],
                    },
                    "old_field": {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.1]},
                },
            },
        ],
    )
    offset = 0
    for episode_index, length in [(0, 2), (1, 3)]:
        df = pd.DataFrame(
            {
                "episode_index": [episode_index] * length,
                "frame_index": list(range(length)),
                "index": list(range(offset, offset + length)),
                "timestamp": [i / 10 for i in range(length)],
                "task_index": [task_index] * length,
                "action": [[float(i)] * 17 for i in range(length)],
                "state": [[float(i)] * 17 for i in range(length)],
                "old_field": [float(i) for i in range(length)],
            }
        )
        df.to_parquet(root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet", index=False)
        offset += length


def _add_v21_stat_counts(root: Path) -> None:
    rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    lengths = {
        int(row["episode_index"]): int(row["length"])
        for row in [
            json.loads(line)
            for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
    }
    for row in rows:
        count = lengths[int(row["episode_index"])]
        for feature_stats in row["stats"].values():
            feature_stats["count"] = [count]
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", rows)


def _write_test_video(
    path: Path,
    value: int,
    frames: int = 2,
    fps: int = 10,
    frame_pts: list[int] | None = None,
) -> None:
    if frame_pts is not None and len(frame_pts) != frames:
        raise ValueError("frame_pts must contain one timestamp per frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        if frame_pts is not None:
            stream.time_base = Fraction(1, fps)
        for frame_index in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value, dtype=np.uint8),
                format="rgb24",
            )
            if frame_pts is not None:
                frame.pts = frame_pts[frame_index]
                frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _png_bytes(value: int) -> bytes:
    return _png_array_bytes(np.full((16, 16, 3), value, dtype=np.uint8))


def _png_array_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_convert_v21_to_v30_and_handle_existing_v30(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    (src / "README.md").write_text("preserve me")
    (src / "meta" / "custom.json").write_text('{"keep": true}')

    result = run_convert_v3(src, out)

    assert result.summary == {
        "source_version": "v2.1",
        "target_version": "v3.0",
        "baseline": "LeRobot 0.4.4",
        "already_v3": False,
        "action": "convert",
        "data_file_size_in_mb": 100,
        "video_file_size_in_mb": 200,
        "workers": 8,
        "tasks": 1,
        "image_video_mode": "rgb_lossless",
    }
    assert detect_dataset_version(src) == "v2.1"
    assert detect_dataset_version(out) == "v3.0"
    assert (src / "meta" / "episodes.jsonl").is_file()
    assert not (out / "meta" / "episodes.jsonl").exists()
    assert not (out / "meta" / "tasks.jsonl").exists()
    assert (out / "meta" / "tasks.parquet").is_file()
    assert (out / "meta" / "stats.json").is_file()
    assert (out / "data" / "chunk-000" / "file-000.parquet").is_file()
    assert (out / "README.md").read_text() == "preserve me"
    assert json.loads((out / "meta" / "custom.json").read_text()) == {"keep": True}

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert info["video_path"] is None
    assert "total_chunks" not in info
    assert info["features"]["action"]["fps"] == 10

    episode_meta = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert episode_meta["episode_index"].tolist() == [0, 1]
    assert episode_meta["data/chunk_index"].tolist() == [0, 0]
    assert episode_meta["data/file_index"].tolist() == [0, 0]
    assert episode_meta["dataset_from_index"].tolist() == [0, 2]
    assert episode_meta["dataset_to_index"].tolist() == [2, 5]
    assert "stats/action/mean" in episode_meta.columns
    assert validate_v3_dataset(out) == {
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
    }

    already_v3 = run_convert_v3(out, image_video_mode="lerobot_official")
    assert already_v3.out_root == out
    assert already_v3.summary["already_v3"] is True
    assert already_v3.summary["action"] == "already_v3"


def test_convert_v21_videos_to_v30(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    video_key = "observation.images.camera"
    info["features"][video_key] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channels"],
    }
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["total_videos"] = 2
    info_path.write_text(json.dumps(info))
    for episode_index, value in enumerate((32, 224)):
        _write_test_video(
            src / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4",
            value,
            frames=2 if episode_index == 0 else 3,
        )

    run_convert_v3(src, out)

    merged_video = out / "videos" / video_key / "chunk-000" / "file-000.mp4"
    assert merged_video.is_file()
    with av.open(str(merged_video)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 5
    episode_meta = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert episode_meta[f"videos/{video_key}/chunk_index"].tolist() == [0, 0]
    assert episode_meta[f"videos/{video_key}/file_index"].tolist() == [0, 0]
    assert episode_meta[f"videos/{video_key}/from_timestamp"].tolist() == [0.0, 0.2]
    assert episode_meta[f"videos/{video_key}/to_timestamp"].tolist() == [0.2, 0.5]
    assert validate_v3_dataset(out)["total_episodes"] == 2

    split = run_split(
        out,
        tmp_path / "v3_video_split",
        episode_range="1:2",
    )
    split_video = split.out_root / "videos" / video_key / "chunk-000" / "file-000.mp4"
    with av.open(str(split_video)) as container:
        stream = container.streams.video[0]
        assert stream.width == 16
        assert stream.height == 16
        assert sum(1 for _ in container.decode(video=0)) == 3
    assert validate_v3_dataset(split.out_root)["total_frames"] == 3


def test_convert_v21_parquet_images_to_three_v30_video_shards(
    tmp_path: Path,
    monkeypatch,
):
    src = tmp_path / "src"
    out = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    camera_keys = [
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for camera_key in camera_keys:
        info["features"][camera_key] = {
            "dtype": "image",
            "shape": [16, 16, 3],
            "names": ["height", "width", "channels"],
        }
    info_path.write_text(json.dumps(info))

    expected_frames = {camera_key: [] for camera_key in camera_keys}
    for episode_index in range(2):
        parquet_path = src / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        dataframe = pd.read_parquet(parquet_path)
        for camera_index, camera_key in enumerate(camera_keys):
            encoded_frames = []
            for frame_index in range(len(dataframe)):
                seed = episode_index * 50 + camera_index * 17 + frame_index * 7
                frame = (np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3) + seed) % 256
                frame = frame.astype(np.uint8)
                expected_frames[camera_key].append(frame)
                encoded_frames.append({"bytes": _png_array_bytes(frame), "path": None})
            dataframe[camera_key] = encoded_frames
        dataframe.to_parquet(parquet_path, index=False)

    active_encoders = 0
    peak_encoders = 0
    encoder_lock = threading.Lock()
    original_encode = dataset_version_module._encode_image_episode_video

    def tracked_encode(*args, **kwargs):
        nonlocal active_encoders, peak_encoders
        with encoder_lock:
            active_encoders += 1
            peak_encoders = max(peak_encoders, active_encoders)
        try:
            time.sleep(0.05)
            return original_encode(*args, **kwargs)
        finally:
            with encoder_lock:
                active_encoders -= 1

    monkeypatch.setattr(dataset_version_module, "_encode_image_episode_video", tracked_encode)
    run_convert_v3(src, out, workers=2)

    converted_info = json.loads((out / "meta" / "info.json").read_text())
    assert peak_encoders == 2
    assert converted_info["video_path"] == (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    for camera_key in camera_keys:
        feature = converted_info["features"][camera_key]
        assert feature["dtype"] == "video"
        assert feature["video.fps"] == 10
        assert feature["video.codec"] == "h264"
        assert feature["video.pix_fmt"] == "gbrp"
        assert feature["has_audio"] is False
        video_path = out / "videos" / camera_key / "chunk-000" / "file-000.mp4"
        assert video_path.is_file()
        with av.open(str(video_path)) as container:
            assert container.streams.video[0].codec_context.name == "h264"
            assert container.streams.video[0].codec_context.pix_fmt == "gbrp"
            frames = list(container.decode(video=0))
        assert len(frames) == 5
        for actual, expected in zip(frames, expected_frames[camera_key], strict=True):
            np.testing.assert_array_equal(actual.to_ndarray(format="rgb24"), expected)

    data_columns = pq.read_schema(out / "data" / "chunk-000" / "file-000.parquet").names
    assert not set(camera_keys) & set(data_columns)
    source_columns = pq.read_schema(src / "data" / "chunk-000" / "episode_000000.parquet").names
    assert set(camera_keys).issubset(source_columns)
    episode_meta = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for camera_key in camera_keys:
        assert episode_meta[f"videos/{camera_key}/chunk_index"].tolist() == [0, 0]
        assert episode_meta[f"videos/{camera_key}/file_index"].tolist() == [0, 0]
        assert episode_meta[f"videos/{camera_key}/from_timestamp"].tolist() == [0.0, 0.2]
        assert episode_meta[f"videos/{camera_key}/to_timestamp"].tolist() == [0.2, 0.5]
    assert validate_v3_dataset(out)["total_episodes"] == 2

    v3_meta = V3DatasetMetadata("local/v3", out)
    assert sorted(v3_meta.episodes) == [0, 1]
    assert v3_meta.tasks == {0: "pick duck"}
    assert read_episode_table(out, v3_meta, 1).num_rows == 3
    image, selected_key = read_episode_frame_image(
        out,
        v3_meta,
        1,
        0,
        image_key=camera_keys[0],
    )
    assert selected_key == camera_keys[0]
    np.testing.assert_array_equal(np.asarray(image), expected_frames[camera_keys[0]][2])
    assert update_episode_metadata(out, {1: {"tags": {"quality": True}}}) == [1]
    assert V3DatasetMetadata("local/v3", out).episodes[1]["tags"] == {"quality": True}


def test_convert_v21_parquet_images_with_lerobot_official_encoding(
    tmp_path: Path,
    capfd,
):
    src = tmp_path / "src"
    out = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    camera_key = "observation.images.front"
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][camera_key] = {
        "dtype": "image",
        "shape": [64, 96, 3],
        "names": ["height", "width", "channels"],
    }
    info_path.write_text(json.dumps(info))

    expected_frames = []
    for episode_index in range(2):
        parquet_path = src / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        dataframe = pd.read_parquet(parquet_path)
        encoded_frames = []
        for frame_index in range(len(dataframe)):
            seed = episode_index * 50 + frame_index * 7
            frame = (np.arange(64 * 96 * 3, dtype=np.uint16).reshape(64, 96, 3) + seed) % 256
            frame = frame.astype(np.uint8)
            expected_frames.append(frame)
            encoded_frames.append({"bytes": _png_array_bytes(frame), "path": None})
        dataframe[camera_key] = encoded_frames
        dataframe.to_parquet(parquet_path, index=False)

    result = run_convert_v3(src, out, image_video_mode="lerobot_official")

    assert "Svt[" not in capfd.readouterr().err
    assert result.summary["image_video_mode"] == "lerobot_official"
    converted_info = json.loads((out / "meta" / "info.json").read_text())
    feature = converted_info["features"][camera_key]
    assert feature["shape"] == [64, 96, 3]
    assert feature["video.codec"] == "av1"
    assert feature["video.pix_fmt"] == "yuv420p"
    assert feature["video.g"] == 2
    assert feature["video.crf"] == 30
    assert feature["video.preset"] == 12
    video_path = out / "videos" / camera_key / "chunk-000" / "file-000.mp4"
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.codec_tag == "av01"
        assert stream.codec_context.pix_fmt == "yuv420p"
        assert (stream.width, stream.height) == (96, 64)
        frames = list(container.decode(video=0))
    assert len(frames) == len(expected_frames)
    assert any(
        not np.array_equal(actual.to_ndarray(format="rgb24"), expected)
        for actual, expected in zip(frames, expected_frames, strict=True)
    )


def test_packed_video_normalizes_final_mp4_pts_at_episode_boundaries(tmp_path: Path):
    episode_paths = []
    for episode_index in range(20):
        path = tmp_path / "episodes" / f"episode_{episode_index:06d}.mp4"
        _write_test_video(path, episode_index, frames=7, fps=30)
        episode_paths.append(path)

    out = tmp_path / "v3"
    video_key = "observation.images.front"
    metadata = dataset_version_module._pack_episode_videos(
        episode_paths,
        [7] * len(episode_paths),
        out,
        video_key,
        1000,
        30,
    )

    merged_video = out / "videos" / video_key / "chunk-000" / "file-000.mp4"
    actual_timestamps = dataset_version_module._video_frame_timestamps_in_s(merged_video)
    start_key = f"videos/{video_key}/from_timestamp"
    end_key = f"videos/{video_key}/to_timestamp"
    assert actual_timestamps == pytest.approx(
        [frame_index / 30 for frame_index in range(20 * 7)],
        abs=1e-9,
    )
    assert metadata[19][start_key] == actual_timestamps[19 * 7]
    assert metadata[19][end_key] == pytest.approx(actual_timestamps[20 * 7 - 1] + 1 / 30)


def test_repair_v3_video_timestamps_remuxes_pts_without_reencoding(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    camera_key = "observation.images.front"
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][camera_key] = {
        "dtype": "image",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channels"],
    }
    info_path.write_text(json.dumps(info))
    for episode_index in range(2):
        parquet_path = src / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        dataframe = pd.read_parquet(parquet_path)
        dataframe[camera_key] = [
            {"bytes": _png_bytes(episode_index * 40 + frame_index), "path": None}
            for frame_index in range(len(dataframe))
        ]
        dataframe.to_parquet(parquet_path, index=False)

    run_convert_v3(src, out)
    video_path = out / "videos" / camera_key / "chunk-000" / "file-000.mp4"
    _write_test_video(
        video_path,
        value=32,
        frames=5,
        fps=10,
        frame_pts=[0, 1, 3, 4, 5],
    )
    with av.open(str(video_path)) as container:
        frames_before = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    video_bytes_before = video_path.read_bytes()
    stats_bytes = (out / "meta" / "stats.json").read_bytes()
    from_key = f"videos/{camera_key}/from_timestamp"
    to_key = f"videos/{camera_key}/to_timestamp"
    update_episode_metadata(
        out,
        {
            0: {from_key: 0.5, to_key: 0.7},
            1: {from_key: 0.7, to_key: 1.0},
        },
    )

    preview = repair_v3_video_timestamps(out, dry_run=True)
    assert preview.summary["video_shards_needing_remux"] == 1
    assert preview.summary["video_shards_remuxed"] == 0
    assert video_path.read_bytes() == video_bytes_before

    result = repair_v3_video_timestamps(out)

    timestamps = dataset_version_module._video_frame_timestamps_in_s(video_path)
    assert timestamps == pytest.approx([frame_index / 10 for frame_index in range(5)], abs=1e-9)
    with av.open(str(video_path)) as container:
        frames_after = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for before, after in zip(frames_before, frames_after, strict=True):
        np.testing.assert_array_equal(after, before)
    repaired = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert repaired[from_key].tolist() == [timestamps[0], timestamps[2]]
    assert repaired[to_key].tolist() == pytest.approx([timestamps[1] + 0.1, timestamps[4] + 0.1])
    assert result.summary["max_abs_shift_s"] >= 0.5
    assert result.summary["video_shards_needing_remux"] == 1
    assert result.summary["video_shards_remuxed"] == 1
    assert result.summary["videos_remuxed"] is True
    assert result.summary["videos_reencoded"] is False
    assert result.summary["norm_stats_recomputed"] is False
    assert video_path.read_bytes() != video_bytes_before
    assert (out / "meta" / "stats.json").read_bytes() == stats_bytes

    repaired_video_bytes = video_path.read_bytes()
    repeated = repair_v3_video_timestamps(out)
    assert repeated.summary["video_shards_needing_remux"] == 0
    assert repeated.summary["video_shards_remuxed"] == 0
    assert video_path.read_bytes() == repaired_video_bytes


def test_convert_v3_default_path_overwrite_and_failure_rollback(
    tmp_path: Path,
    monkeypatch,
):
    src = tmp_path / "source"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    expected_out = tmp_path / "source_v3"
    assert default_v3_path(src) == expected_out

    first = run_convert_v3(src)
    assert first.out_root == expected_out
    with pytest.raises(FileExistsError, match="already exists"):
        run_convert_v3(src)

    (expected_out / "old-output.txt").write_text("replace me")
    run_convert_v3(src, overwrite=True)
    assert not (expected_out / "old-output.txt").exists()
    (expected_out / "restore-on-failure.txt").write_text("keep me")

    def fail_write_tasks(*_args, **_kwargs):
        raise RuntimeError("injected conversion failure")

    monkeypatch.setattr(dataset_version_module, "_write_tasks", fail_write_tasks)
    with pytest.raises(RuntimeError, match="injected conversion failure"):
        run_convert_v3(src, overwrite=True)
    assert (expected_out / "restore-on-failure.txt").read_text() == "keep me"


def test_convert_v3_rejects_nonpositive_workers(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)
    _add_v21_stat_counts(src)

    with pytest.raises(ValueError, match="workers must be positive"):
        run_convert_v3(src, tmp_path / "v3", workers=0)


def test_lossless_rgb_h264_options_optimize_size_without_quantization():
    assert dataset_version_module.lossless_rgb_h264_options(30) == {
        "g": "30",
        "crf": "0",
        "preset": "slow",
    }
    assert dataset_version_module.lossless_rgb_h264_options(
        10,
        encoder_threads=1,
    ) == {
        "g": "10",
        "crf": "0",
        "preset": "slow",
        "threads": "1",
    }
    with pytest.raises(ValueError, match="fps must be positive"):
        dataset_version_module.lossless_rgb_h264_options(0)


def test_lerobot_official_av1_options_match_v044_defaults():
    assert dataset_version_module.lerobot_official_av1_options() == {
        "g": "2",
        "crf": "30",
        "preset": "12",
    }
    assert dataset_version_module.lerobot_official_av1_options(encoder_threads=1) == {
        "g": "2",
        "crf": "30",
        "preset": "12",
        "svtav1-params": "lp=1",
    }


def test_v3_rule_tagging_and_compare_stats(tmp_path: Path):
    src = tmp_path / "src"
    v3_root = tmp_path / "v3"
    static_dir = tmp_path / "static"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3_root)
    meta = V3DatasetMetadata("local/v3", v3_root)

    tagging = run_tagging(
        v3_root,
        meta,
        episodes=None,
        static_dir=static_dir,
        selected_tags=["arm"],
        workers=2,
    )

    tag_records = [json.loads(line) for line in tagging.tags_path.read_text().splitlines() if line.strip()]
    assert [record["episode_index"] for record in tag_records] == [0, 1]
    assert all("arm" in record["tags"] for record in tag_records)
    assert metadata_stats(v3_root, meta) == {
        "root": str(v3_root),
        "total_episodes": 2,
        "total_frames": 5,
        "fps": 10,
        "total_tasks": 1,
        "image_keys": [],
        "action_shape": [17],
    }
    stats = action_stats(v3_root, meta)
    assert stats["available"] is True
    assert stats["dims"] == 17
    assert stats["min"] == [0.0] * 17
    assert stats["max"] == [2.0] * 17


def test_v3_subtask_and_index_mutations_preserve_sibling_rows(tmp_path: Path):
    src = tmp_path / "src"
    v3_root = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3_root)
    meta = V3DatasetMetadata("local/v3", v3_root)

    episode_stats = write_subtask_state_to_parquet(
        v3_root,
        meta,
        {
            0: {
                "stage0_end": 0.05,
                "stage2_start": 0.08,
                "stage2_end": 0.12,
                "stage4_start": 0.15,
            }
        },
    )
    update_info_features(
        v3_root,
        {"subtask_state": {"dtype": "int32", "shape": [1], "names": None}},
    )
    assert update_episode_stats_for_subtask_state(v3_root, episode_stats) is True
    assert write_subtask_text_to_parquet(v3_root, meta, [0]) == 1

    refreshed = V3DatasetMetadata("local/v3", v3_root)
    assert read_episode_table(v3_root, refreshed, 0)["subtask_state"].to_pylist() == [0, 2]
    assert read_episode_table(v3_root, refreshed, 1)["subtask_state"].to_pylist() == [
        None,
        None,
        None,
    ]
    assert all(
        value is not None for value in read_episode_table(v3_root, refreshed, 0)["subtask"].to_pylist()
    )
    assert read_episode_table(v3_root, refreshed, 1)["subtask"].to_pylist() == [
        None,
        None,
        None,
    ]
    assert refreshed.episodes_stats[0]["subtask_state"]["count"] == [2]

    replace_episode_column(v3_root, refreshed, 1, "index", [20, 21, 22])
    assert fix_episode_indices(v3_root, refreshed, [0, 1]) is True
    repaired = V3DatasetMetadata("local/v3", v3_root)
    assert read_episode_table(v3_root, repaired, 0)["index"].to_pylist() == [0, 1]
    assert read_episode_table(v3_root, repaired, 1)["index"].to_pylist() == [2, 3, 4]


def test_trim_v3_episode_arbitrary_range_rebuilds_shared_shards(tmp_path: Path):
    src = tmp_path / "src"
    v3_root = tmp_path / "v3"
    static_dir = tmp_path / "static"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3_root)
    static_dir.mkdir()

    class Dataset:
        root = v3_root
        repo_id = "local/v3"
        meta = V3DatasetMetadata(repo_id, root)
        features = meta.features
        fps = meta.fps
        codebase_version = "v3.0"
        total_episodes = meta.total_episodes
        total_frames = meta.total_frames

    result = trim_v3_episode_inplace(
        Dataset,
        static_dir,
        episode_id=1,
        start_frame=1,
        end_frame=2,
        workers=2,
    )

    assert result["original_length"] == 3
    assert result["new_length"] == 2
    assert validate_v3_dataset(v3_root)["total_frames"] == 4
    refreshed = V3DatasetMetadata("local/v3", v3_root)
    assert read_episode_table(v3_root, refreshed, 0)["action"].to_pylist() == [
        [0.0] * 17,
        [1.0] * 17,
    ]
    assert read_episode_table(v3_root, refreshed, 1)["action"].to_pylist() == [
        [1.0] * 17,
        [2.0] * 17,
    ]
    assert refreshed.episodes[1]["length"] == 2
    assert refreshed.episodes_stats[1]["action"]["count"] == [2]


def test_prepare_v3_viewer_cache_in_current_environment(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    v3_root = tmp_path / "v3"
    viewer_output = tmp_path / "vis" / "local_vis_v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    video_key = "observation.images.camera"
    info["features"][video_key] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channels"],
    }
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["total_videos"] = 2
    info_path.write_text(json.dumps(info))
    for episode_index, value in enumerate((32, 224)):
        _write_test_video(
            src / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4",
            value,
            frames=2 if episode_index == 0 else 3,
        )
    run_convert_v3(src, v3_root)
    source_snapshot = {
        path.relative_to(v3_root): path.read_bytes() for path in v3_root.rglob("*") if path.is_file()
    }

    from flask import Flask

    from lerobot.data_platform import viewer as viewer_module

    captured = {}
    monkeypatch.setattr(Flask, "run", lambda self, **_kwargs: captured.update(app=self))
    console_static = tmp_path / "vis" / "_console" / "static"
    console_static.mkdir(parents=True)
    viewer_module.run_server(
        dataset=None,
        episodes=None,
        max_frames=None,
        prepare_videos=False,
        downsample=None,
        precompute_csv=False,
        precomputed_only=True,
        host="127.0.0.1",
        port=0,
        static_folder=console_static,
        template_folder=Path(viewer_module.__file__).parent / "templates",
        datasets_root=tmp_path,
    )
    client = captured["app"].test_client()
    assert client.get("/").status_code == 200
    register_response = client.post("/api/datasets/register", json={"root": str(v3_root)})
    assert register_response.status_code == 200
    assert register_response.get_json()["dataset"]["dataset_format_version"] == "v3.0"

    precompute_response = client.post(
        "/api/precompute/start",
        json={
            "dataset_key": "local/v3",
            "options": {
                "prepare_videos": True,
                "prepare_csv": True,
                "prepare_workers": 2,
                "precomputed_only": True,
                "data_version": "DVT2",
            },
        },
    )
    assert precompute_response.status_code == 200
    job_id = precompute_response.get_json()["job"]["id"]
    job = None
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").get_json()["job"]
        if job["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert job is not None
    assert job["status"] == "done", job.get("error")

    csv_path = viewer_output / "static" / "csv" / "episode_000000_ds1.csv"
    csv_rows = csv_path.read_text().splitlines()
    assert csv_rows[0].startswith("timestamp,action_0")
    assert len(csv_rows) == 3
    clip_path = viewer_output / "static" / "videos" / video_key / "episode_000001_h264.mp4"
    with av.open(str(clip_path)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3

    csv_path.write_bytes(b"")
    clip_path.write_bytes(b"")
    v3_viewer_module.run_v3_viewer_precompute(
        root=v3_root,
        repo_id="local/v3",
        output_dir=viewer_output,
        episodes=[0, 1],
        prepare_videos=True,
        prepare_csv=True,
        workers=2,
        data_version="DVT2",
    )
    assert csv_path.stat().st_size > 0
    with av.open(str(clip_path)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3

    manifest = json.loads((viewer_output / "static" / "viewer_manifest.json").read_text())
    assert manifest["codebase_version"] == "v3.0"
    assert manifest["image_keys"] == [video_key]
    assert [episode["length"] for episode in manifest["episodes"]] == [2, 3]
    assert detect_dataset_version(v3_root) == "v3.0"
    assert {
        path.relative_to(v3_root): path.read_bytes() for path in v3_root.rglob("*") if path.is_file()
    } == source_snapshot

    def fail_hub_load(*_args, **_kwargs):
        raise AssertionError("v3 viewer must not use Hub loading")

    monkeypatch.setattr(viewer_module, "get_dataset_info", fail_hub_load)
    assert client.get("/api/datasets/local/v3?full=1").status_code == 200
    preload_response = client.post(
        "/api/viewer/preload",
        json={"dataset_key": "local/v3", "episode_id": 0, "data_version": "DVT2"},
    )
    assert preload_response.status_code == 200
    preload_job_id = preload_response.get_json()["job"]["id"]
    preload_job = None
    for _ in range(100):
        preload_job = client.get(f"/api/jobs/{preload_job_id}").get_json()["job"]
        if preload_job["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert preload_job is not None
    assert preload_job["status"] == "done", preload_job.get("error")

    assert client.get("/local/v3/episode_0?direct=1").status_code == 200
    assert client.get("/local/v3/episode_0/data.csv").status_code == 200
    assert client.get(f"/assets/local/v3/videos/{video_key}/episode_000000_h264.mp4").status_code == 200

    # The persisted registry and cache must remain sufficient after restarting
    # the Data Platform process; the legacy metadata reader still cannot load v3.
    viewer_module.run_server(
        dataset=None,
        episodes=None,
        max_frames=None,
        prepare_videos=False,
        downsample=None,
        precompute_csv=False,
        precomputed_only=True,
        host="127.0.0.1",
        port=0,
        static_folder=console_static,
        template_folder=Path(viewer_module.__file__).parent / "templates",
        datasets_root=tmp_path,
    )
    restarted_client = captured["app"].test_client()
    restarted_datasets = restarted_client.get("/api/datasets").get_json()["datasets"]
    restarted_v3 = next(dataset for dataset in restarted_datasets if dataset["key"] == "local/v3")
    assert restarted_v3["cache"]["status"] == "cached"
    assert restarted_client.get("/local/v3/episode_0?direct=1").status_code == 200


def test_convert_v3_dry_run_and_report_existing_v3_without_copy(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "v3"
    clone = tmp_path / "v3_clone"
    _make_dataset(src)
    _add_v21_stat_counts(src)

    dry_run = run_convert_v3(src, out, dry_run=True)
    assert dry_run.dry_run is True
    assert not out.exists()

    run_convert_v3(src, out)
    already_v3 = run_convert_v3(out, clone)
    assert already_v3.out_root == out
    assert already_v3.summary["action"] == "already_v3"
    assert not clone.exists()


def test_convert_action_and_drop_field(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)

    converted = run_convert_action(src, tmp_path / "action16")
    info = json.loads((converted.out_root / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [16]
    assert info["features"]["state"]["shape"] == [16]
    row = pd.read_parquet(converted.out_root / "data" / "chunk-000" / "episode_000000.parquet").iloc[0]
    assert len(row["action"]) == 16

    dropped = run_drop_field(src, tmp_path / "drop_old", field_name="old_field")
    info = json.loads((dropped.out_root / "meta" / "info.json").read_text())
    assert "old_field" not in info["features"]
    assert (
        "old_field"
        not in pd.read_parquet(dropped.out_root / "data" / "chunk-000" / "episode_000000.parquet").columns
    )


def test_smooth_action_rewrites_action_and_stats(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)

    smoothed = run_smooth_action(src, tmp_path / "smooth", window=3, workers=2)
    df = pd.read_parquet(smoothed.out_root / "data" / "chunk-000" / "episode_000001.parquet")
    assert abs(df.iloc[0]["action"][0] - (1 / 3)) < 1e-6
    assert abs(df.iloc[1]["action"][0] - 1.0) < 1e-6
    assert abs(df.iloc[2]["action"][0] - (5 / 3)) < 1e-6
    assert abs(df.iloc[0]["state"][0] - (1 / 3)) < 1e-6
    stats_rows = [
        json.loads(line)
        for line in (smoothed.out_root / "meta" / "episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ep1_stats = next(row for row in stats_rows if row["episode_index"] == 1)["stats"]["action"]
    assert abs(ep1_stats["min"][0] - (1 / 3)) < 1e-6
    assert ep1_stats["count"] == [3]
    smooth_meta = json.loads((smoothed.out_root / "meta" / "preprocess_smooth_action.json").read_text())
    assert smooth_meta["source_root"] == str(src)
    assert smooth_meta["window"] == 3
    assert smooth_meta["workers"] == 2
    assert smooth_meta["fields"] == ["action", "state"]
    assert smooth_meta["smooth_state"] is True


def test_v3_action_transform_drop_and_smooth_preserve_layout_and_episode_boundaries(
    tmp_path: Path,
):
    src = tmp_path / "src"
    v3 = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3)

    converted = run_convert_action(v3, tmp_path / "v3_action16")
    assert detect_dataset_version(converted.out_root) == "v3.0"
    assert validate_v3_dataset(converted.out_root)["total_episodes"] == 2
    converted_info = json.loads((converted.out_root / "meta" / "info.json").read_text())
    assert converted_info["features"]["action"]["shape"] == [16]
    converted_meta = V3DatasetMetadata("local/v3_action16", converted.out_root)
    assert len(read_episode_table(converted.out_root, converted_meta, 1)["action"][0]) == 16
    assert len(converted_meta.episodes_stats[1]["action"]["mean"][0]) == 16
    assert len(converted_meta.stats["action"]["mean"][0]) == 16

    dropped = run_drop_field(v3, tmp_path / "v3_drop_old", field_name="old_field")
    assert validate_v3_dataset(dropped.out_root)["total_episodes"] == 2
    dropped_meta = V3DatasetMetadata("local/v3_drop_old", dropped.out_root)
    assert "old_field" not in dropped_meta.features
    assert "old_field" not in dropped_meta.stats
    assert "old_field" not in dropped_meta.episodes_stats[0]
    assert (
        "old_field"
        not in read_episode_table(
            dropped.out_root,
            dropped_meta,
            0,
        ).column_names
    )

    smoothed = run_smooth_action(
        v3,
        tmp_path / "v3_smooth",
        window=3,
        workers=1,
    )
    assert validate_v3_dataset(smoothed.out_root)["total_episodes"] == 2
    smoothed_meta = V3DatasetMetadata("local/v3_smooth", smoothed.out_root)
    episode_zero = read_episode_table(smoothed.out_root, smoothed_meta, 0)
    episode_one = read_episode_table(smoothed.out_root, smoothed_meta, 1)
    assert abs(episode_zero["action"][1].as_py()[0] - (2 / 3)) < 1e-6
    assert abs(episode_one["action"][0].as_py()[0] - (1 / 3)) < 1e-6
    assert abs(episode_one["action"][2].as_py()[0] - (5 / 3)) < 1e-6
    assert smoothed_meta.episodes_stats[1]["action"]["count"] == [3]
    assert abs(smoothed_meta.episodes_stats[1]["action"]["min"][0] - (1 / 3)) < 1e-6
    assert len(smoothed_meta.stats["action"]["mean"]) == 17


def test_v3_standardize_prompt_split_merge_and_subtract(tmp_path: Path):
    src = tmp_path / "src"
    v3 = tmp_path / "v3"
    _make_dataset(src, task="Pick Duck")
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3)

    lowercase_result = run_lowercase_prompts(v3)
    assert lowercase_result.summary["changed_task_rows"] == 1
    lowercase_meta = V3DatasetMetadata("local/v3", v3)
    assert lowercase_meta.tasks == {0: "pick duck"}
    assert lowercase_meta.episodes[0]["tasks"] == ["pick duck"]

    standardized = run_standardize_dataset(
        v3,
        tmp_path / "v3_standardized",
        workers=1,
    )
    assert validate_v3_dataset(standardized.out_root)["total_episodes"] == 2
    standardized_meta = V3DatasetMetadata(
        "local/v3_standardized",
        standardized.out_root,
    )
    standardized_episode = read_episode_table(
        standardized.out_root,
        standardized_meta,
        1,
    )
    assert len(standardized_episode["action"][0]) == 16
    assert standardized_episode["exist_label"].to_pylist() == [1, 1, 1]
    assert standardized_meta.episodes_stats[1]["exist_label"]["count"] == [3]

    split = run_split(v3, tmp_path / "v3_split", episode_range="1:2")
    assert validate_v3_dataset(split.out_root) == {
        "total_episodes": 1,
        "total_frames": 3,
        "total_tasks": 1,
    }
    split_meta = V3DatasetMetadata("local/v3_split", split.out_root)
    assert read_episode_table(split.out_root, split_meta, 0).num_rows == 3

    merged = run_merge(
        [split.out_root, split.out_root],
        tmp_path / "v3_merge",
        workers=2,
    )
    assert validate_v3_dataset(merged.out_root)["total_episodes"] == 2

    subtracted = run_subtract(
        v3,
        [split.out_root],
        tmp_path / "v3_subtract",
        workers=2,
    )
    assert validate_v3_dataset(subtracted.out_root) == {
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 1,
    }


def test_v3_quality_scan_and_prompt_assignment_only_rewrite_selected_episode(
    tmp_path: Path,
):
    src = tmp_path / "src"
    v3 = tmp_path / "v3"
    static_dir = tmp_path / "static"
    _make_dataset(src, task="Pick Duck")
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3)

    scan = run_quality_flag_detection(
        v3,
        static_dir,
        data_version="DVT2",
        workers=2,
    )
    assert scan.summary["episodes_scanned"] == 2

    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {
                    "episode": 1,
                    "type": "quality_flag",
                    "reason": "wrong_prompt",
                    "metrics": {},
                }
            ]
        )
    )
    (static_dir / "quality_flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [1]}))
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [1]}))

    assignment = apply_task_assignment_choice(
        v3,
        static_dir,
        1,
        "Pick Duck Corrected",
        reason="wrong_prompt",
    )
    assert assignment["task_index"] == 1
    meta = V3DatasetMetadata("local/v3", v3)
    assert meta.tasks == {0: "Pick Duck", 1: "Pick Duck Corrected"}
    assert meta.episodes[1]["tasks"] == ["Pick Duck Corrected"]
    assert read_episode_table(v3, meta, 0)["task_index"].to_pylist() == [0, 0]
    assert read_episode_table(v3, meta, 1)["task_index"].to_pylist() == [1, 1, 1]


def test_v3_delete_episode_rebuilds_shards_and_reindexes(tmp_path: Path):
    src = tmp_path / "src"
    v3 = tmp_path / "v3"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3)

    class Dataset:
        repo_id = "local/v3"

        def __init__(self, root: Path):
            self.root = root
            self.meta = V3DatasetMetadata(self.repo_id, root)
            self.features = self.meta.features
            self.fps = self.meta.fps
            self.total_episodes = self.meta.total_episodes
            self.total_frames = self.meta.total_frames

    dataset = Dataset(v3)
    result = delete_episodes_inplace(
        dataset,
        [0],
        static_folder=tmp_path / "static",
    )

    assert result == {
        "deleted_episode_ids": [0],
        "new_total_episodes": 1,
        "next_episode": 0,
    }
    assert validate_v3_dataset(v3) == {
        "total_episodes": 1,
        "total_frames": 3,
        "total_tasks": 1,
    }
    assert dataset.total_episodes == 1
    assert read_episode_table(v3, dataset.meta, 0)["episode_index"].to_pylist() == [
        0,
        0,
        0,
    ]


@pytest.mark.parametrize("dataset_format_version", ["v2.1", "v3.0"])
def test_viewer_single_episode_delete_refreshes_registered_episode_state(
    tmp_path: Path,
    monkeypatch,
    dataset_format_version: str,
):
    from flask import Flask

    from lerobot.data_platform import viewer as viewer_module

    root = tmp_path / "source"
    if dataset_format_version == "v3.0":
        legacy_root = tmp_path / "legacy_source"
        _make_dataset(legacy_root)
        _add_v21_stat_counts(legacy_root)
        run_convert_v3(legacy_root, root)
    else:
        _make_dataset(root)
        _add_v21_stat_counts(root)

    captured = {}
    monkeypatch.setattr(Flask, "run", lambda self, **_kwargs: captured.update(app=self))
    console_static = tmp_path / "vis" / "_console" / "static"
    console_static.mkdir(parents=True)
    viewer_module.run_server(
        dataset=None,
        episodes=None,
        max_frames=None,
        prepare_videos=False,
        downsample=None,
        precompute_csv=False,
        precomputed_only=True,
        host="127.0.0.1",
        port=0,
        static_folder=console_static,
        template_folder=Path(viewer_module.__file__).parent / "templates",
        annotate=True,
        datasets_root=tmp_path,
    )
    client = captured["app"].test_client()
    registered = client.post("/api/datasets/register", json={"root": str(root)}).get_json()["dataset"]
    dataset_key = registered["key"]
    assert client.get(f"/api/datasets/{dataset_key}?full=1").get_json()["dataset"]["episodes"] == [0, 1]

    response = client.post(
        f"/{dataset_key}/delete_episode",
        json={"episode_id": 0},
        buffered=True,
    )

    assert response.status_code == 200
    assert b"event: done" in response.data
    refreshed = client.get(f"/api/datasets/{dataset_key}?full=1").get_json()["dataset"]
    assert refreshed["episodes"] == [0]
    assert refreshed["episode_count"] == 1


def test_v3_construction_and_review_finalize_keep_v3_layout(tmp_path: Path):
    src = tmp_path / "src"
    v3 = tmp_path / "v3"
    out = tmp_path / "constructed_v3"
    labeling_dir = tmp_path / "labeling"
    _make_dataset(src, task="Pick up the yellow duck")
    _add_v21_stat_counts(src)
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = 2
    info_path.write_text(json.dumps(info))
    _write_jsonl(
        src / "meta" / "tasks.jsonl",
        [
            {"task_index": 0, "task": "Pick up the yellow duck"},
            {"task_index": 1, "task": "Pick up the brown dog"},
        ],
    )
    episode_rows = [
        {"episode_index": 0, "tasks": ["Pick up the yellow duck"], "length": 2},
        {"episode_index": 1, "tasks": ["Pick up the brown dog"], "length": 3},
    ]
    _write_jsonl(src / "meta" / "episodes.jsonl", episode_rows)
    episode_one_path = src / "data" / "chunk-000" / "episode_000001.parquet"
    episode_one = pd.read_parquet(episode_one_path)
    episode_one["task_index"] = 1
    episode_one.to_parquet(episode_one_path, index=False)
    run_convert_v3(src, v3)

    labeling_dir.mkdir()
    label_rows = []
    for episode_index, task, target in [
        (0, "Pick up the yellow duck", "yellow duck"),
        (1, "Pick up the brown dog", "brown dog"),
    ]:
        label_rows.append(
            {
                "episode_index": episode_index,
                "task": task,
                "parsed": {
                    "action": "pick",
                    "target": target,
                    "direction": None,
                    "reference": None,
                },
                "selected": {
                    "bbox": {"left": 0, "top": 0, "right": 10, "bottom": 10},
                    "confidence": 0.9,
                },
                "relation_satisfied": True,
                "detections_target": [
                    {
                        "bbox": {
                            "left": 0,
                            "top": 0,
                            "right": 10,
                            "bottom": 10,
                        },
                        "confidence": 0.9,
                    }
                ],
                "detections_ref": [],
            }
        )
    _write_jsonl(labeling_dir / "labels.jsonl", label_rows)

    meta = V3DatasetMetadata("local/v3", v3)
    result = run_construction(
        v3,
        meta,
        labeling_dir,
        out,
        {
            "uncertainty_threshold": 50,
            "per_scenario_counts": {"single_pick": 1},
            "include_positives": True,
        },
    )
    assert len(result.plans) == 1
    assert validate_v3_dataset(out)["total_episodes"] == 3
    plan_episode = result.plans[0].new_episode_index
    save_construction_decision(out, plan_episode, "reject", "test")
    finalized = finalize_construction(out)
    assert finalized == {"removed": 1, "remaining": 2}
    assert validate_v3_dataset(out)["total_episodes"] == 2
    assert not (out / "meta" / "episodes.jsonl").exists()


def test_v3_flag_fixes_preserve_other_episode_rows_and_repack_trim(tmp_path: Path):
    src = tmp_path / "src"
    v3 = tmp_path / "v3"
    static_dir = tmp_path / "static"
    _make_dataset(src)
    _add_v21_stat_counts(src)
    run_convert_v3(src, v3)
    static_dir.mkdir()

    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {
                    "episode": 1,
                    "type": "quality_flag",
                    "reason": "stuck_closed_gripper_no_action",
                    "metrics": {"gripper_index": 7},
                }
            ]
        )
    )
    run_flag_fix(
        v3,
        static_dir,
        "fix_stuck_closed_action",
        data_version="DVT1",
    )
    meta = V3DatasetMetadata("local/v3", v3)
    assert [row[7] for row in read_episode_table(v3, meta, 0)["action"].to_pylist()] == [0.0, 1.0]
    assert [row[7] for row in read_episode_table(v3, meta, 1)["action"].to_pylist()] == [100.0, 100.0, 100.0]

    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {
                    "episode": 1,
                    "type": "quality_flag",
                    "reason": "early_gripper_transition",
                    "metrics": {},
                }
            ]
        )
    )
    run_flag_fix(
        v3,
        static_dir,
        "trim_early_gripper_first_frame",
        data_version="DVT1",
    )
    assert validate_v3_dataset(v3)["total_frames"] == 4
    trimmed_meta = V3DatasetMetadata("local/v3", v3)
    assert read_episode_table(v3, trimmed_meta, 0).num_rows == 2
    assert read_episode_table(v3, trimmed_meta, 1).num_rows == 2


def test_standardize_dataset_normalizes_trims_and_drops_depth(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [20]
    info["features"]["state"]["shape"] = [19]
    info["features"]["head_depth"] = {"dtype": "float32", "shape": [1], "names": None}
    info["features"]["observation.images.left_wrist_depth"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    info_path.write_text(json.dumps(info))

    for parquet_path in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        actions = []
        states = []
        for frame_idx in range(len(df)):
            action = [float(frame_idx)] * 20
            action[7] = 50.0
            action[15] = 120.0
            state = [float(frame_idx)] * 19
            state[7] = 25.0
            state[15] = 75.0
            actions.append(action)
            states.append(state)
        df["action"] = actions
        df["state"] = states
        df["head_depth"] = [1.0] * len(df)
        df["observation.images.left_wrist_depth"] = [2.0] * len(df)
        df.to_parquet(parquet_path, index=False)

    result = run_standardize_dataset(src, tmp_path / "standardized")
    out_info = json.loads((result.out_root / "meta" / "info.json").read_text())
    assert out_info["features"]["action"]["shape"] == [16]
    assert out_info["features"]["state"]["shape"] == [16]
    assert out_info["features"]["exist_label"] == {"dtype": "int32", "shape": [1], "names": None}
    assert "head_depth" not in out_info["features"]
    assert "observation.images.left_wrist_depth" not in out_info["features"]

    out_df = pd.read_parquet(result.out_root / "data" / "chunk-000" / "episode_000000.parquet")
    assert len(out_df.iloc[0]["action"]) == 16
    assert len(out_df.iloc[0]["state"]) == 16
    assert abs(out_df.iloc[0]["action"][7] - 0.5) < 1e-6
    assert abs(out_df.iloc[0]["action"][15] - 1.2) < 1e-6
    assert abs(out_df.iloc[0]["state"][7] - 0.25) < 1e-6
    assert abs(out_df.iloc[0]["state"][15] - 0.75) < 1e-6
    assert out_df["exist_label"].tolist() == [1, 1]
    assert "head_depth" not in out_df.columns
    assert "observation.images.left_wrist_depth" not in out_df.columns
    stats_rows = [
        json.loads(line)
        for line in (result.out_root / "meta" / "episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert stats_rows[0]["stats"]["exist_label"]["min"] == [1.0]


def test_standardize_dataset_preserves_existing_exist_label(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)
    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["exist_label"] = {"dtype": "int64", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))

    for parquet_path in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        df["exist_label"] = [idx % 2 for idx in range(len(df))]
        df.to_parquet(parquet_path, index=False)

    result = run_standardize_dataset(src, tmp_path / "standardized")
    out_info = json.loads((result.out_root / "meta" / "info.json").read_text())
    assert out_info["features"]["exist_label"] == {"dtype": "int64", "shape": [1], "names": None}

    out_df = pd.read_parquet(result.out_root / "data" / "chunk-000" / "episode_000001.parquet")
    assert out_df["exist_label"].tolist() == [0, 1, 0]


def test_flag_fix_adds_action_lead_for_state_only_gripper_transition(tmp_path: Path):
    src = tmp_path / "src"
    static_dir = tmp_path / "static"
    _make_dataset(src)

    info_path = src / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 23
    info_path.write_text(json.dumps(info))
    _write_jsonl(
        src / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick duck"], "length": 20},
            {"episode_index": 1, "tasks": ["pick duck"], "length": 3},
        ],
    )
    frames = list(range(20))
    action = [[0.0] * 17 for _ in frames]
    state = [[0.0] * 17 for _ in frames]
    for idx in range(10, 20):
        state[idx][7] = 1.0
    pd.DataFrame(
        {
            "episode_index": [0] * 20,
            "frame_index": frames,
            "index": frames,
            "timestamp": [idx / 10 for idx in frames],
            "task_index": [0] * 20,
            "action": action,
            "state": state,
            "old_field": [0.0] * 20,
        }
    ).to_parquet(src / "data" / "chunk-000" / "episode_000000.parquet", index=False)
    (static_dir / "csv").mkdir(parents=True)
    (static_dir / "csv" / "episode_000000_ds1.csv").write_text("stale")
    (static_dir / "annotation_issues.json").write_text(
        json.dumps(
            [
                {
                    "episode": 0,
                    "type": "quality_flag",
                    "reason": "state_gripper_transition_without_action",
                    "frames": [10],
                    "metrics": {
                        "events": [
                            {
                                "gripper_index": 7,
                                "frame": 10,
                                "from_state": 0,
                                "to_state": 1,
                            }
                        ]
                    },
                }
            ]
        )
    )
    (static_dir / "quality_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [0],
                "flag_reasons": {
                    "0": [{"type": "quality_flag", "reason": "state_gripper_transition_without_action"}]
                },
            }
        )
    )
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0]}))

    result = run_flag_fix(src, static_dir, "fix_state_gripper_transition_action", data_version="DVT1")

    fixed = pd.read_parquet(src / "data" / "chunk-000" / "episode_000000.parquet")
    assert fixed.iloc[8]["action"][7] == 0.0
    assert fixed.iloc[9]["action"][7] == 1.0
    assert fixed.iloc[10]["action"][7] == 1.0
    assert not (static_dir / "csv" / "episode_000000_ds1.csv").exists()
    assert json.loads((static_dir / "annotation_issues.json").read_text()) == []
    assert json.loads((static_dir / "flagged_episodes.json").read_text()) == {"flagged_episodes": []}
    backup_manifest = Path(result.summary["backup_manifest"])
    assert backup_manifest.is_file()
    backup = json.loads(backup_manifest.read_text())
    assert backup["episodes"] == [0]
    assert Path(backup["details"][0]["backup_path"]).is_file()


def test_fix_prompt_prepositions_rewrites_absolute_and_relative_metadata(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src, task="pick up the yellow duck to the left")
    _write_jsonl(
        src / "meta" / "tasks.jsonl",
        [
            {"task_index": 0, "task": "pick up the yellow duck to the left"},
            {"task_index": 1, "task": "pick up the brown dog on the right of the green dinosaur"},
        ],
    )
    _write_jsonl(
        src / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick up the yellow duck to the left"], "length": 2},
            {
                "episode_index": 1,
                "tasks": ["pick up the brown dog on the right of the green dinosaur"],
                "length": 3,
            },
        ],
    )

    dry_run = run_fix_prompt_prepositions(src, dry_run=True)
    assert dry_run.summary["total_replacements"] == 4

    result = run_fix_prompt_prepositions(src)
    assert result.summary["total_replacements"] == 4
    tasks = [json.loads(line)["task"] for line in (src / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks == [
        "pick up the yellow duck on the left",
        "pick up the brown dog to the right of the green dinosaur",
    ]
    episodes = [
        json.loads(line)["tasks"][0] for line in (src / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert episodes == tasks
    assert list((src / "meta" / "prompt_rewrite_backups").glob("*"))


def test_lowercase_prompts_rewrites_task_and_episode_metadata(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src, task="Pick up the Yellow Duck")
    _write_jsonl(
        src / "meta" / "tasks.jsonl",
        [
            {"task_index": 0, "task": "Pick up the Yellow Duck"},
            {"task_index": 1, "task": "pick up the green dinosaur"},
        ],
    )
    _write_jsonl(
        src / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["Pick up the Yellow Duck"], "length": 2},
            {"episode_index": 1, "tasks": ["pick up the green dinosaur"], "length": 3},
        ],
    )

    dry_run = run_lowercase_prompts(src, dry_run=True)
    assert dry_run.summary["total_replacements"] == 2
    assert (
        json.loads((src / "meta" / "tasks.jsonl").read_text().splitlines()[0])["task"]
        == "Pick up the Yellow Duck"
    )

    result = run_lowercase_prompts(src)
    assert result.summary["total_replacements"] == 2
    tasks = [json.loads(line)["task"] for line in (src / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks == ["pick up the yellow duck", "pick up the green dinosaur"]
    episodes = [
        json.loads(line)["tasks"][0] for line in (src / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert episodes == tasks
    assert list((src / "meta" / "prompt_rewrite_backups").glob("*"))


def test_lowercase_prompts_remaps_task_index_and_pending_viewer_prompts(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src, task="Pick up the Yellow Duck")
    _write_jsonl(
        src / "meta" / "tasks.jsonl",
        [
            {"task_index": 0, "task": "Pick up the Yellow Duck"},
            {"task_index": 1, "task": "pick up the yellow duck"},
        ],
    )
    _write_jsonl(
        src / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["Pick up the Yellow Duck"], "length": 2},
            {"episode_index": 1, "tasks": ["pick up the yellow duck"], "length": 3},
        ],
    )
    parquet_path = src / "data" / "chunk-000" / "episode_000001.parquet"
    table = pq.read_table(parquet_path)
    task_field = table.schema.field("task_index")
    table = table.set_column(
        table.column_names.index("task_index"),
        task_field,
        pa.array([1] * table.num_rows, type=task_field.type),
    )
    pq.write_table(table, parquet_path)

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "prompt_assignments_pending.json").write_text(
        json.dumps(
            {
                "version": 1,
                "assignments": [
                    {
                        "episode_index": 0,
                        "selected_task": "Pick up the Yellow Duck",
                        "updated_at": 1,
                        "source": "viewer_cache_only",
                    }
                ],
            }
        )
    )

    result = run_lowercase_prompts(src, static_dir=static_dir)

    assert result.summary["removed_duplicate_task_rows"] == 1
    assert result.summary["parquet_task_index_files_changed"] == 1
    assert result.summary["parquet_task_index_values_changed"] == 3
    assert result.summary["pending_prompt_assignments_changed"] == 1
    tasks = [json.loads(line) for line in (src / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": "pick up the yellow duck"}]
    info = json.loads((src / "meta" / "info.json").read_text())
    assert info["total_tasks"] == 1
    table = pq.read_table(parquet_path, columns=["task_index"])
    assert set(table["task_index"].to_pylist()) == {0}
    pending = json.loads((static_dir / "prompt_assignments_pending.json").read_text())
    assert pending["assignments"][0]["selected_task"] == "pick up the yellow duck"


def test_split_and_merge_reindex_metadata(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)

    split = run_split(src, tmp_path / "split", episode_range="1:2")
    info = json.loads((split.out_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 3
    df = pd.read_parquet(split.out_root / "data" / "chunk-000" / "episode_000000.parquet")
    assert df["episode_index"].tolist() == [0, 0, 0]
    assert df["index"].tolist() == [0, 1, 2]

    merged = run_merge([split.out_root, split.out_root], tmp_path / "merge")
    info = json.loads((merged.out_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 6
    df = pd.read_parquet(merged.out_root / "data" / "chunk-000" / "episode_000001.parquet")
    assert df["episode_index"].tolist() == [1, 1, 1]
    assert df["index"].tolist() == [3, 4, 5]


def test_merge_rejects_action_shape_mismatch_before_writing(tmp_path: Path):
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    _make_dataset(src_a)
    _make_dataset(src_b)
    info_path = src_b / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [18]
    info_path.write_text(json.dumps(info))

    try:
        run_merge([src_a, src_b], tmp_path / "merge_bad")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("run_merge should reject mismatched action shapes")

    assert "action shape mismatch" in message
    assert "please standardize datasets first" in message
    assert not (tmp_path / "merge_bad").exists()


def test_merge_can_exclude_source_episodes_before_reindexing(tmp_path: Path):
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    _make_dataset(src_a, task="pick duck", task_index=0)
    _make_dataset(src_b, task="pick dog", task_index=0)

    merged = run_merge(
        [src_a, src_b],
        tmp_path / "merge_excluding",
        exclude_episodes=[[0], [1]],
        workers=2,
    )
    info = json.loads((merged.out_root / "meta" / "info.json").read_text())
    episodes = [
        json.loads(line) for line in (merged.out_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    first = pd.read_parquet(merged.out_root / "data" / "chunk-000" / "episode_000000.parquet")
    second = pd.read_parquet(merged.out_root / "data" / "chunk-000" / "episode_000001.parquet")

    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5
    assert [row["episode_index"] for row in episodes] == [0, 1]
    assert first["episode_index"].tolist() == [0, 0, 0]
    assert first["index"].tolist() == [0, 1, 2]
    assert second["episode_index"].tolist() == [1, 1]
    assert second["index"].tolist() == [3, 4]
    assert merged.summary["deleted_source_episodes"] == {str(src_a): [0], str(src_b): [1]}


def test_subtract_matches_split_subset_by_content_fingerprint(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)
    split = run_split(src, tmp_path / "split", episode_range="1:2")

    subtracted = run_subtract(src, [split.out_root], tmp_path / "subtract")
    info = json.loads((subtracted.out_root / "meta" / "info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (subtracted.out_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    out_df = pd.read_parquet(subtracted.out_root / "data" / "chunk-000" / "episode_000000.parquet")
    src_df = pd.read_parquet(src / "data" / "chunk-000" / "episode_000001.parquet")

    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert episodes == [{"episode_index": 0, "tasks": ["pick duck"], "length": 2}]
    assert out_df["episode_index"].tolist() == [0, 0]
    assert out_df["index"].tolist() == [0, 1]
    assert src_df["episode_index"].tolist() == [1, 1, 1]
    assert subtracted.summary["removed_base_episodes"] == [1]
    assert subtracted.summary["kept_base_episode_count"] == 1


def test_subtract_carries_static_artifacts_for_kept_episodes(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)
    split = run_split(src, tmp_path / "split", episode_range="0:1")
    static = tmp_path / "static"
    out_static = tmp_path / "out_static"

    (static / "csv").mkdir(parents=True)
    (static / "csv" / "episode_000000_ds1.csv").write_text(
        "timestamp,episode_index,frame_index,index,stage\n0.0,0,0,0,delete\n"
    )
    (static / "csv" / "episode_000001_ds1.csv").write_text(
        "timestamp,episode_index,frame_index,index,stage\n0.0,1,0,2,keep\n0.1,1,1,3,keep\n"
    )
    (static / "labeling").mkdir(parents=True)
    _write_jsonl(static / "labeling" / "labels.jsonl", [{"episode_index": 0, "task": "delete"}])
    _write_jsonl(static / "labeling" / "labels_reviewed.jsonl", [{"episode_index": 1, "task": "keep"}])
    (static / "tagging").mkdir(parents=True)
    _write_jsonl(static / "tagging" / "tags.jsonl", [{"episode_index": 1, "tags": {"arm": "right"}}])
    (static / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0, 1]}))
    (static / "annotation_issues.json").write_text(json.dumps([{"episode": 1, "type": "quality_flag"}]))
    (static / "trim_annotations.json").write_text(json.dumps({"0": {"start": 0.0}, "1": {"start": 1.0}}))
    (static / "subtask_annotations.json").write_text(json.dumps({"1": {"stage": "keep"}}))
    (static / "prompt_assignments_pending.json").write_text(
        json.dumps(
            {
                "version": 1,
                "assignments": [
                    {"episode_index": 0, "selected_task": "delete"},
                    {"episode_index": 1, "selected_task": "keep"},
                ],
            }
        )
    )

    result = run_subtract(
        src,
        [split.out_root],
        tmp_path / "subtract_artifacts",
        src_static_dir=static,
        out_static_dir=out_static,
    )

    assert result.summary["artifacts"]["csv_files"] == 1
    assert (out_static / "csv" / "episode_000000_ds1.csv").read_text().splitlines() == [
        "timestamp,episode_index,frame_index,index,stage",
        "0.0,0,0,0,keep",
        "0.1,0,1,1,keep",
    ]
    reviewed = [
        json.loads(line)
        for line in (out_static / "labeling" / "labels_reviewed.jsonl").read_text().splitlines()
    ]
    tags = [json.loads(line) for line in (out_static / "tagging" / "tags.jsonl").read_text().splitlines()]
    flags = json.loads((out_static / "flagged_episodes.json").read_text())
    issues = json.loads((out_static / "annotation_issues.json").read_text())
    pending = json.loads((out_static / "prompt_assignments_pending.json").read_text())

    assert not (out_static / "labeling" / "labels.jsonl").exists()
    assert reviewed == [{"episode_index": 0, "task": "keep"}]
    assert tags == [{"episode_index": 0, "tags": {"arm": "right"}}]
    assert flags["flagged_episodes"] == [0]
    assert issues == [{"episode": 0, "type": "quality_flag"}]
    assert json.loads((out_static / "trim_annotations.json").read_text()) == {"0": {"start": 1.0}}
    assert json.loads((out_static / "subtask_annotations.json").read_text()) == {"0": {"stage": "keep"}}
    assert pending["assignments"] == [{"episode_index": 0, "selected_task": "keep"}]


def test_subtract_dry_run_reports_removals_without_writing(tmp_path: Path):
    src = tmp_path / "src"
    _make_dataset(src)
    split = run_split(src, tmp_path / "split", episode_range="1:2")
    out_root = tmp_path / "subtract_dry"

    result = run_subtract(src, [split.out_root], out_root, dry_run=True)

    assert not out_root.exists()
    assert result.dry_run is True
    assert result.summary["removed_base_episodes"] == [1]
    assert result.summary["removed_base_episode_count"] == 1
    assert result.summary["unmatched_subtract_fingerprint_count"] == 0


def test_merge_preserves_depth_array_columns(tmp_path: Path):
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    _make_dataset(src_a)
    _make_dataset(src_b)
    for src in (src_a, src_b):
        info_path = src / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["features"]["head_depth"] = {"dtype": "float32", "shape": [2], "names": None}
        info_path.write_text(json.dumps(info))
        for parquet_path in sorted((src / "data").rglob("*.parquet")):
            table = pq.read_table(parquet_path)
            values = [[1.0, 2.0] for _ in range(table.num_rows)]
            table = table.append_column(
                pa.field("head_depth", pa.list_(pa.float32(), 2)),
                pa.array(values, type=pa.list_(pa.float32(), 2)),
            )
            pq.write_table(table, parquet_path)

    merged = run_merge([src_a, src_b], tmp_path / "merge_depth")
    out_table = pq.read_table(merged.out_root / "data" / "chunk-000" / "episode_000000.parquet")

    assert "head_depth" in out_table.column_names
    assert out_table.schema.field("head_depth").type == pa.list_(pa.float32(), 2)
    assert out_table["head_depth"][0].as_py() == [1.0, 2.0]


def test_merge_carries_static_artifacts_with_episode_remap(tmp_path: Path):
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    _make_dataset(src_a, task="pick duck", task_index=0)
    _make_dataset(src_b, task="pick dog", task_index=0)
    static_a = tmp_path / "static_a"
    static_b = tmp_path / "static_b"

    (static_a / "csv").mkdir(parents=True)
    (static_a / "csv" / "episode_000001_ds1.csv").write_text(
        "timestamp,episode_index,frame_index,index,stage\n0.0,1,0,2,0\n0.1,1,1,3,1\n"
    )
    (static_b / "videos" / "front").mkdir(parents=True)
    (static_b / "videos" / "front" / "episode_000000_h264.mp4").write_bytes(b"video")

    (static_a / "labeling").mkdir(parents=True)
    _write_jsonl(
        static_a / "labeling" / "labels.jsonl",
        [{"episode_index": 1, "task": "pick duck", "selected": {"left": 1}}],
    )
    _write_jsonl(
        static_a / "labeling" / "labels_reviewed.jsonl",
        [{"episode_index": 1, "task": "pick duck", "selected": {"left": 2}}],
    )
    (static_a / "labeling" / "source.json").write_text(json.dumps({"backend": "grounding_dino"}))
    (static_a / "labeling" / "vis").mkdir(parents=True)
    (static_a / "labeling" / "vis" / "episode_000001.png").write_bytes(b"png")

    (static_b / "tagging").mkdir(parents=True)
    _write_jsonl(static_b / "tagging" / "tags.jsonl", [{"episode_index": 0, "tags": {"background": "sofa"}}])
    (static_b / "tagging" / "source.json").write_text(json.dumps({"output_variant": "latest"}))
    (static_b / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [0]}))
    (static_b / "quality_flagged_episodes.json").write_text(
        json.dumps(
            {
                "flagged_episodes": [0],
                "flag_reasons": {"0": [{"type": "quality_flag", "reason": "early_gripper_transition"}]},
            }
        )
    )
    (static_b / "annotation_issues.json").write_text(
        json.dumps([{"episode": 0, "type": "quality_flag", "reason": "early_gripper_transition"}])
    )
    (static_a / "trim_annotations.json").write_text(json.dumps({"1": {"start": 1.0}}))
    (static_b / "subtask_annotations.json").write_text(json.dumps({"0": {"stage": "place"}}))
    (static_b / "prompt_assignments_pending.json").write_text(
        json.dumps({"version": 1, "assignments": [{"episode_index": 0, "selected_task": "pick dog"}]})
    )

    out_static = tmp_path / "out_static"
    result = run_merge(
        [src_a, src_b],
        tmp_path / "merge_artifacts",
        src_static_dirs=[static_a, static_b],
        out_static_dir=out_static,
    )

    assert result.summary["artifacts"]["csv_files"] == 1
    assert result.summary["artifacts"]["video_files"] == 1
    csv_text = (out_static / "csv" / "episode_000001_ds1.csv").read_text()
    assert "0.0,1,0,2,0" in csv_text
    assert (out_static / "videos" / "front" / "episode_000002_h264.mp4").read_bytes() == b"video"
    labels = [
        json.loads(line) for line in (out_static / "labeling" / "labels.jsonl").read_text().splitlines()
    ]
    assert labels == [{"episode_index": 1, "task": "pick duck", "selected": {"left": 1}}]
    reviewed = [
        json.loads(line)
        for line in (out_static / "labeling" / "labels_reviewed.jsonl").read_text().splitlines()
    ]
    assert reviewed[0]["episode_index"] == 1
    tags = [json.loads(line) for line in (out_static / "tagging" / "tags.jsonl").read_text().splitlines()]
    assert tags == [{"episode_index": 2, "tags": {"background": "sofa"}}]
    assert (out_static / "labeling" / "vis" / "episode_000001.png").read_bytes() == b"png"
    flags = json.loads((out_static / "flagged_episodes.json").read_text())
    assert flags["flagged_episodes"] == [2]
    quality_flags = json.loads((out_static / "quality_flagged_episodes.json").read_text())
    assert quality_flags["flagged_episodes"] == [2]
    assert quality_flags["flag_reasons"]["2"][0]["reason"] == "early_gripper_transition"
    issues = json.loads((out_static / "annotation_issues.json").read_text())
    assert issues == [{"episode": 2, "type": "quality_flag", "reason": "early_gripper_transition"}]
    assert json.loads((out_static / "trim_annotations.json").read_text()) == {"1": {"start": 1.0}}
    assert json.loads((out_static / "subtask_annotations.json").read_text()) == {"2": {"stage": "place"}}
    pending = json.loads((out_static / "prompt_assignments_pending.json").read_text())
    assert pending["assignments"] == [{"episode_index": 2, "selected_task": "pick dog"}]
