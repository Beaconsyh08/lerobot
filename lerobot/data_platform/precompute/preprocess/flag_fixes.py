from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.utils import serialize_dict
from lerobot.data_platform.precompute.annotation import QUALITY_FLAG_TYPE
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    load_episode_records,
    read_episode_table,
    replace_episode_column,
)
from lerobot.data_platform.precompute.mutations import fix_episode_indices
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    emit,
    format_data_path,
    format_video_path,
    load_json,
    load_jsonl,
    validate_dataset_root,
    write_json,
    write_jsonl,
)
from lerobot.data_platform.precompute.preprocess.dataset_version import (
    LOSSLESS_IMAGE_VIDEO_CODEC,
    lossless_rgb_h264_options,
)
from lerobot.data_platform.precompute.preprocess.quality_flags import QUALITY_FLAGGED_EPISODES
from lerobot.data_platform.precompute.timeseries import DATA_VERSION_DVT2, infer_data_version_from_features

FLAG_FIX_TRIM_EARLY_GRIPPER = "trim_early_gripper_first_frame"
FLAG_FIX_STUCK_CLOSED_ACTION = "fix_stuck_closed_action"
FLAG_FIX_STATE_GRIPPER_TRANSITION_ACTION = "fix_state_gripper_transition_action"
FLAG_FIX_DELETE_ALL_FLAGGED = "delete_all_flagged"
FLAG_FIX_STATE_ACTION_LEAD_SECONDS = 0.1


class _MetaLite:
    def __init__(self, info: dict, episodes: dict[int, dict]):
        self.info = info
        self.episodes = episodes

    def get_data_file_path(self, episode_id: int) -> Path:
        return format_data_path(self.info, episode_id)


def _load_json_any(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _issue_episode(issue: dict) -> int | None:
    try:
        return int(issue.get("episode"))
    except (TypeError, ValueError):
        return None


def _load_quality_issues(static_dir: Path, *, reason: str | None = None) -> list[dict]:
    issues = _load_json_any(Path(static_dir) / "annotation_issues.json")
    if not isinstance(issues, list):
        return []
    out = []
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("type") != QUALITY_FLAG_TYPE:
            continue
        if reason is not None and issue.get("reason") != reason:
            continue
        if _issue_episode(issue) is not None:
            out.append(issue)
    return out


def _flag_set(path: Path) -> set[int]:
    data = _load_json_any(path)
    values = (
        data.get("flagged_episodes") if isinstance(data, dict) else data if isinstance(data, list) else []
    )
    out = set()
    for value in values or []:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def load_flagged_episode_ids(static_dir: Path) -> list[int]:
    return sorted(_flag_set(Path(static_dir) / "flagged_episodes.json"))


def _delete_episode_cache(static_dir: Path, episode_id: int) -> None:
    static_dir = Path(static_dir)
    csv_dir = static_dir / "csv"
    if csv_dir.is_dir():
        for path in csv_dir.glob(f"episode_{episode_id:06d}_ds*.csv"):
            path.unlink()
    videos_dir = static_dir / "videos"
    if videos_dir.is_dir():
        for path in videos_dir.glob(f"*/episode_{episode_id:06d}_h264.mp4"):
            path.unlink()


def _replace_column(table: pa.Table, name: str, values, value_type: pa.DataType | None = None) -> pa.Table:
    field = table.schema.field(name)
    idx = table.column_names.index(name)
    return table.set_column(idx, field, pa.array(values, type=value_type or field.type))


def _trim_first_frame(root: Path, info: dict, episodes_by_id: dict[int, dict], episode_id: int) -> int:
    parquet_path = root / format_data_path(info, episode_id)
    if not parquet_path.is_file():
        return 0
    table = pq.read_table(parquet_path)
    if table.num_rows <= 1:
        return 0
    table = table.slice(1)
    new_rows = int(table.num_rows)
    if "timestamp" in table.column_names and new_rows:
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        timestamps = timestamps - float(timestamps[0])
        table = _replace_column(table, "timestamp", timestamps.tolist())
    if "frame_index" in table.column_names:
        table = _replace_column(table, "frame_index", list(range(new_rows)))
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(parquet_path)
    for video_key, feature in (info.get("features") or {}).items():
        if feature.get("dtype") != "video":
            continue
        video_path = root / format_video_path(info, episode_id, video_key)
        if video_path.is_file():
            _trim_video_first_frame(video_path, int(info["fps"]))
    if episode_id in episodes_by_id:
        episodes_by_id[episode_id]["length"] = new_rows
    return 1


def _trim_video_first_frame(path: Path, fps: int) -> None:
    _trim_video_frames(path, fps, 1, None)


def _trim_video_frames(
    path: Path,
    fps: int,
    start_frame: int,
    end_frame: int | None,
) -> None:
    with av.open(str(path)) as container:
        frames = [
            frame.to_image().convert("RGB")
            for frame_index, frame in enumerate(container.decode(video=0))
            if frame_index >= start_frame and (end_frame is None or frame_index <= end_frame)
        ]
    if not frames:
        raise ValueError(f"Trim range contains no video frames: {path}")
    temporary = path.with_suffix(".trim.mp4")
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream(
                LOSSLESS_IMAGE_VIDEO_CODEC,
                rate=fps,
                options=lossless_rgb_h264_options(fps),
            )
            stream.width, stream.height = frames[0].size
            stream.pix_fmt = "rgb24"
            for image in frames:
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _recompute_v21_stats(root: Path, info: dict) -> None:
    old_stats = {
        int(row["episode_index"]): row.get("stats") or {}
        for row in load_jsonl(root / "meta" / "episodes_stats.jsonl")
    }
    features = info.get("features") or {}
    episode_stats = []
    raw_stats = []
    for episode in load_jsonl(root / "meta" / "episodes.jsonl"):
        episode_index = int(episode["episode_index"])
        table = pq.read_table(root / format_data_path(info, episode_index))
        episode_data = {}
        for key, feature in features.items():
            if feature.get("dtype") in {"string", "image", "video"}:
                continue
            if key in table.column_names:
                episode_data[key] = np.asarray(table[key].to_pylist())
        stats = compute_episode_stats(episode_data, features)
        for key, value in old_stats.get(episode_index, {}).items():
            if key not in stats and features.get(key, {}).get("dtype") in {"image", "video"}:
                stats[key] = value
        for key, value in stats.items():
            if features.get(key, {}).get("dtype") in {"image", "video"}:
                value["count"] = np.asarray([table.num_rows])
        raw_stats.append(stats)
        episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": serialize_dict(stats),
            }
        )
    write_jsonl(root / "meta" / "episodes_stats.jsonl", episode_stats)
    write_json(root / "meta" / "stats.json", serialize_dict(aggregate_stats(raw_stats)))


def _trim_v21_episode_frames(
    root: Path,
    episode_id: int,
    start_frame: int,
    end_frame: int,
) -> dict:
    info = load_json(root / "meta" / "info.json")
    episodes_by_id = {int(row["episode_index"]): row for row in load_jsonl(root / "meta" / "episodes.jsonl")}
    if episode_id not in episodes_by_id:
        raise ValueError(f"episode {episode_id} not found")
    parquet_path = root / format_data_path(info, episode_id)
    table = pq.read_table(parquet_path)
    original_length = int(table.num_rows)
    if start_frame < 0 or end_frame >= original_length or end_frame < start_frame:
        raise ValueError(f"Invalid trim range {start_frame}:{end_frame} for episode length {original_length}")
    if start_frame == 0 and end_frame == original_length - 1:
        return {
            "episode_id": episode_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "original_length": original_length,
            "new_length": original_length,
            "dropped_frames": 0,
        }

    table = table.slice(start_frame, end_frame - start_frame + 1)
    new_length = int(table.num_rows)
    if "timestamp" in table.column_names:
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        table = _replace_column(table, "timestamp", (timestamps - timestamps[0]).tolist())
    if "frame_index" in table.column_names:
        table = _replace_column(table, "frame_index", list(range(new_length)))
    temporary = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(parquet_path)

    for video_key, feature in (info.get("features") or {}).items():
        if feature.get("dtype") != "video":
            continue
        video_path = root / format_video_path(info, episode_id, video_key)
        if video_path.is_file():
            _trim_video_frames(
                video_path,
                int(info["fps"]),
                start_frame,
                end_frame,
            )

    episodes_by_id[episode_id]["length"] = new_length
    _write_meta_lengths(root, info, episodes_by_id)
    fix_episode_indices(root, _MetaLite(info, episodes_by_id), sorted(episodes_by_id))
    info = load_json(root / "meta" / "info.json")
    _recompute_v21_stats(root, info)
    return {
        "episode_id": episode_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "original_length": original_length,
        "new_length": new_length,
        "dropped_frames": original_length - new_length,
    }


def trim_v3_episode_inplace(
    dataset,
    static_dir: Path,
    episode_id: int,
    start_frame: int,
    end_frame: int,
    *,
    workers: int = 8,
) -> dict:
    """Trim an arbitrary v3 episode range and rebuild shared data/video shards losslessly."""
    root = validate_dataset_root(Path(dataset.root))
    if not is_v3_dataset(root):
        raise ValueError("trim_v3_episode_inplace requires a v3.0 dataset")
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.trim-v3-",
        dir=root.parent,
    ) as temp_dir:
        from lerobot.data_platform.precompute.preprocess.dataset_version import (
            materialize_v21_from_v3,
            run_convert_v3,
        )

        temp_root = Path(temp_dir)
        legacy_root = materialize_v21_from_v3(
            root,
            temp_root / "legacy",
            workers=workers,
        )
        result = _trim_v21_episode_frames(
            legacy_root,
            int(episode_id),
            int(start_frame),
            int(end_frame),
        )
        rebuilt_root = temp_root / "rebuilt"
        run_convert_v3(legacy_root, rebuilt_root, workers=workers)

        backup_root = temp_root / "backup"
        backup_root.mkdir()
        replaced = []
        try:
            for name in ("data", "meta", "videos"):
                current = root / name
                if current.exists():
                    current.rename(backup_root / name)
                replaced.append(name)
                generated = rebuilt_root / name
                if generated.exists():
                    generated.rename(current)
        except Exception:
            for name in reversed(replaced):
                current = root / name
                if current.is_dir():
                    shutil.rmtree(current)
                backup = backup_root / name
                if backup.exists():
                    backup.rename(current)
            raise

    new_meta = V3DatasetMetadata(getattr(dataset, "repo_id", f"local/{root.name}"), root)
    dataset.meta = new_meta
    dataset.features = new_meta.features
    dataset.fps = new_meta.fps
    dataset.codebase_version = new_meta.info.get("codebase_version", "v3.0")
    dataset.total_episodes = new_meta.total_episodes
    dataset.total_frames = new_meta.total_frames
    _delete_episode_cache(static_dir, int(episode_id))
    return result


def _raw_closed_action_value(action_array: np.ndarray, gripper_index: int, data_version: str) -> float:
    if str(data_version).upper() == DATA_VERSION_DVT2 and action_array.shape[1] >= 19:
        return 100.0
    if gripper_index < action_array.shape[1]:
        finite = action_array[:, gripper_index][np.isfinite(action_array[:, gripper_index])]
        if finite.size and float(np.nanmax(np.abs(finite))) > 1.5:
            return 100.0
    return 1.0


def _raw_gripper_action_value(
    action_array: np.ndarray, gripper_index: int, closed: int, data_version: str
) -> float:
    return _raw_closed_action_value(action_array, gripper_index, data_version) if int(closed) else 0.0


def _backup_parquet(root: Path, parquet_path: Path, backup_dir: Path) -> Path:
    rel_path = parquet_path.relative_to(root)
    backup_path = backup_dir / rel_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parquet_path, backup_path)
    return backup_path


def _fix_stuck_action(
    root: Path,
    info: dict,
    issue: dict,
    data_version: str,
    meta: V3DatasetMetadata | None = None,
) -> bool:
    episode_id = int(issue["episode"])
    metrics = issue.get("metrics") or {}
    gripper_index = int(metrics.get("gripper_index", 7))
    parquet_path = (
        root / meta.get_data_file_path(episode_id)
        if meta is not None
        else root / format_data_path(info, episode_id)
    )
    if not parquet_path.is_file():
        return False
    table = read_episode_table(root, meta, episode_id) if meta is not None else pq.read_table(parquet_path)
    if "action" not in table.column_names:
        return False
    values = table["action"].to_pylist()
    action = np.asarray(values, dtype=np.float64)
    if action.ndim != 2 or gripper_index >= action.shape[1]:
        return False
    action[:, gripper_index] = _raw_closed_action_value(action, gripper_index, data_version)
    if meta is not None:
        replace_episode_column(
            root,
            meta,
            episode_id,
            "action",
            action.tolist(),
        )
    else:
        field = table.schema.field("action")
        table = _replace_column(table, "action", action.tolist(), field.type)
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)
    return True


def _fix_state_transition_action(
    root: Path,
    info: dict,
    issue: dict,
    data_version: str,
    backup_dir: Path,
    meta: V3DatasetMetadata | None = None,
) -> dict:
    episode_id = int(issue["episode"])
    metrics = issue.get("metrics") or {}
    events = [event for event in metrics.get("events") or [] if isinstance(event, dict)]
    if not events:
        events = [
            {
                "frame": frame,
                "gripper_index": 7,
                "from_state": 0,
                "to_state": 1,
            }
            for frame in issue.get("frames") or []
        ]
    parquet_path = (
        root / meta.get_data_file_path(episode_id)
        if meta is not None
        else root / format_data_path(info, episode_id)
    )
    if not parquet_path.is_file():
        return {"fixed": False, "reason": "missing_parquet"}
    table = read_episode_table(root, meta, episode_id) if meta is not None else pq.read_table(parquet_path)
    if "action" not in table.column_names:
        return {"fixed": False, "reason": "missing_action"}
    values = table["action"].to_pylist()
    action = np.asarray(values, dtype=np.float64)
    if action.ndim != 2 or action.shape[0] == 0:
        return {"fixed": False, "reason": "invalid_action"}

    fps = float(info.get("fps") or 0)
    lead_frames = max(1, int(round(fps * FLAG_FIX_STATE_ACTION_LEAD_SECONDS))) if fps > 0 else 1
    changed = False
    applied = []
    events_by_gripper: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        try:
            gripper_index = int(event.get("gripper_index", 7))
            frame = int(event["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if gripper_index < 0 or gripper_index >= action.shape[1] or frame <= 0:
            continue
        copied = dict(event)
        copied["gripper_index"] = gripper_index
        copied["frame"] = min(frame, action.shape[0] - 1)
        events_by_gripper[gripper_index].append(copied)

    for gripper_index, gripper_events in events_by_gripper.items():
        gripper_events.sort(key=lambda item: int(item["frame"]))
        starts = [max(0, int(event["frame"]) - lead_frames) for event in gripper_events]
        for idx, event in enumerate(gripper_events):
            frame = int(event["frame"])
            start = starts[idx]
            end = starts[idx + 1] if idx + 1 < len(starts) else action.shape[0]
            if end <= start:
                end = min(action.shape[0], start + 1)
            from_state = int(event.get("from_state", 0))
            to_state = int(event.get("to_state", 1))
            pre_start = max(0, start - lead_frames)
            if pre_start < start:
                old_value = _raw_gripper_action_value(action, gripper_index, from_state, data_version)
                if not np.allclose(action[pre_start:start, gripper_index], old_value, equal_nan=True):
                    action[pre_start:start, gripper_index] = old_value
                    changed = True
            new_value = _raw_gripper_action_value(action, gripper_index, to_state, data_version)
            if not np.allclose(action[start:end, gripper_index], new_value, equal_nan=True):
                action[start:end, gripper_index] = new_value
                changed = True
            applied.append(
                {
                    "gripper_index": int(gripper_index),
                    "state_frame": int(frame),
                    "action_frame": int(start),
                    "end_frame": int(end),
                    "from_state": int(from_state),
                    "to_state": int(to_state),
                }
            )

    if not changed:
        return {"fixed": False, "reason": "no_change", "applied": applied}

    backup_path = _backup_parquet(root, parquet_path, backup_dir)
    if meta is not None:
        replace_episode_column(
            root,
            meta,
            episode_id,
            "action",
            action.tolist(),
        )
    else:
        field = table.schema.field("action")
        table = _replace_column(table, "action", action.tolist(), field.type)
        tmp_path = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(parquet_path)
    return {
        "fixed": True,
        "backup_path": str(backup_path),
        "lead_frames": int(lead_frames),
        "lead_seconds": float(FLAG_FIX_STATE_ACTION_LEAD_SECONDS),
        "applied": applied,
    }


def _write_meta_lengths(root: Path, info: dict, episodes_by_id: dict[int, dict]) -> None:
    rows = [episodes_by_id[idx] for idx in sorted(episodes_by_id)]
    write_jsonl(root / "meta" / "episodes.jsonl", rows)
    total_frames = sum(int(row.get("length") or 0) for row in rows)
    info["total_frames"] = total_frames
    write_json(root / "meta" / "info.json", info)


def _remove_resolved_quality_issues(static_dir: Path, *, reason: str, episode_ids: set[int]) -> dict:
    static_dir = Path(static_dir)
    issues_path = static_dir / "annotation_issues.json"
    issues = _load_json_any(issues_path)
    if not isinstance(issues, list):
        issues = []
    retained = []
    removed = []
    for issue in issues:
        episode = _issue_episode(issue) if isinstance(issue, dict) else None
        if (
            isinstance(issue, dict)
            and issue.get("type") == QUALITY_FLAG_TYPE
            and issue.get("reason") == reason
            and episode in episode_ids
        ):
            removed.append(issue)
            continue
        retained.append(issue)
    write_json(issues_path, retained)

    remaining_quality = [
        issue for issue in retained if isinstance(issue, dict) and issue.get("type") == QUALITY_FLAG_TYPE
    ]
    next_auto = {_issue_episode(issue) for issue in remaining_quality}
    next_auto = {episode for episode in next_auto if episode is not None}
    previous_auto = _flag_set(static_dir / QUALITY_FLAGGED_EPISODES)
    existing_flagged = _flag_set(static_dir / "flagged_episodes.json")
    manual_or_other_auto = existing_flagged - previous_auto
    combined = manual_or_other_auto | next_auto

    reason_map: dict[str, list[dict]] = defaultdict(list)
    for issue in remaining_quality:
        episode = _issue_episode(issue)
        if episode is None:
            continue
        reason_item = {
            "type": str(issue.get("type") or QUALITY_FLAG_TYPE),
            "reason": str(issue.get("reason") or "unknown"),
        }
        if "frames" in issue:
            reason_item["frames"] = issue.get("frames") or []
        if "metrics" in issue:
            reason_item["metrics"] = issue.get("metrics") or {}
        reason_map[str(episode)].append(reason_item)

    reason_counts = Counter(str(issue.get("reason") or "unknown") for issue in remaining_quality)
    write_json(
        static_dir / QUALITY_FLAGGED_EPISODES,
        {
            "flagged_episodes": sorted(next_auto),
            "flag_reasons": dict(sorted(reason_map.items())),
            "summary": {
                "quality_episode_count": len(next_auto),
                "quality_issue_count": len(remaining_quality),
                "reason_counts": dict(sorted(reason_counts.items())),
            },
        },
    )
    write_json(static_dir / "flagged_episodes.json", {"flagged_episodes": sorted(combined)})
    return {"removed_issues": len(removed), "remaining_quality_episodes": len(next_auto)}


def run_flag_fix(
    root: Path,
    static_dir: Path,
    fix_kind: str,
    *,
    episodes: list[int] | None = None,
    data_version: str | None = None,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    root = validate_dataset_root(Path(root))
    static_dir = Path(static_dir).expanduser()
    info = load_json(root / "meta" / "info.json")
    source_is_v3 = is_v3_dataset(root)
    if source_is_v3 and fix_kind == FLAG_FIX_TRIM_EARLY_GRIPPER:
        from lerobot.data_platform.precompute.preprocess.dataset_version import (
            materialize_v21_from_v3,
            run_convert_v3,
        )

        with tempfile.TemporaryDirectory(
            prefix=f".{root.name}.flag-fix-",
            dir=root.parent,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            legacy_root = materialize_v21_from_v3(
                root,
                temp_root / "legacy",
                workers=8,
                progress_callback=progress_callback,
            )
            legacy_result = run_flag_fix(
                legacy_root,
                static_dir,
                fix_kind,
                episodes=episodes,
                data_version=data_version,
                progress_callback=progress_callback,
            )
            _recompute_v21_stats(
                legacy_root,
                load_json(legacy_root / "meta" / "info.json"),
            )
            rebuilt_root = temp_root / "rebuilt"
            run_convert_v3(
                legacy_root,
                rebuilt_root,
                workers=8,
                progress_callback=progress_callback,
            )
            backup_root = temp_root / "backup"
            backup_root.mkdir()
            replaced = []
            try:
                for name in ("data", "meta", "videos"):
                    current = root / name
                    if current.exists():
                        current.rename(backup_root / name)
                    replaced.append(name)
                    generated = rebuilt_root / name
                    if generated.exists():
                        generated.rename(current)
            except Exception:
                for name in reversed(replaced):
                    current = root / name
                    if current.is_dir():
                        shutil.rmtree(current)
                    backup = backup_root / name
                    if backup.exists():
                        backup.rename(current)
                raise
        return PreprocessResult(
            op=legacy_result.op,
            src_roots=[root],
            out_root=root,
            repo_id=root.name,
            total_episodes=legacy_result.total_episodes,
            total_frames=int(load_json(root / "meta" / "info.json").get("total_frames") or 0),
            dry_run=False,
            summary={**legacy_result.summary, "dataset_format": "v3.0"},
        )
    selected_data_version = str(
        data_version or infer_data_version_from_features(info.get("features") or {})
    ).upper()
    episode_rows = load_episode_records(root)
    episodes_by_id = {int(row["episode_index"]): row for row in episode_rows}
    meta = V3DatasetMetadata(f"local/{root.name}", root) if source_is_v3 else None
    allowed = {int(ep) for ep in episodes} if episodes else None

    if fix_kind == FLAG_FIX_TRIM_EARLY_GRIPPER:
        reason = "early_gripper_transition"
        issues = _load_quality_issues(static_dir, reason=reason)
        episode_ids = sorted(
            {int(issue["episode"]) for issue in issues if allowed is None or int(issue["episode"]) in allowed}
        )
        emit(
            progress_callback,
            status="running",
            current=0,
            total=len(episode_ids),
            message=f"Trimming first frame for {len(episode_ids)} episodes",
        )
        fixed = 0
        fixed_episode_ids: set[int] = set()
        for idx, episode_id in enumerate(episode_ids, start=1):
            did_fix = _trim_first_frame(root, info, episodes_by_id, episode_id)
            fixed += did_fix
            if did_fix:
                fixed_episode_ids.add(episode_id)
                _delete_episode_cache(static_dir, episode_id)
            emit(
                progress_callback,
                status="running",
                current=idx,
                total=len(episode_ids),
                episode=episode_id,
                message=f"Trimmed episode {episode_id}",
            )
        if fixed_episode_ids:
            _write_meta_lengths(root, info, episodes_by_id)
            fix_episode_indices(root, _MetaLite(info, episodes_by_id), sorted(episodes_by_id))
        cleanup = _remove_resolved_quality_issues(static_dir, reason=reason, episode_ids=fixed_episode_ids)
        summary = {
            "fix_kind": fix_kind,
            "episodes": sorted(fixed_episode_ids),
            "attempted_episodes": episode_ids,
            "fixed": fixed,
            **cleanup,
        }

    elif fix_kind == FLAG_FIX_STUCK_CLOSED_ACTION:
        reason = "stuck_closed_gripper_no_action"
        issues = _load_quality_issues(static_dir, reason=reason)
        selected_issues = [issue for issue in issues if allowed is None or int(issue["episode"]) in allowed]
        emit(
            progress_callback,
            status="running",
            current=0,
            total=len(selected_issues),
            message=f"Fixing stuck gripper action for {len(selected_issues)} issues",
        )
        fixed_episodes: set[int] = set()
        fixed = 0
        for idx, issue in enumerate(selected_issues, start=1):
            episode_id = int(issue["episode"])
            if _fix_stuck_action(
                root,
                info,
                issue,
                selected_data_version,
                meta,
            ):
                fixed += 1
                fixed_episodes.add(episode_id)
                _delete_episode_cache(static_dir, episode_id)
            emit(
                progress_callback,
                status="running",
                current=idx,
                total=len(selected_issues),
                episode=episode_id,
                message=f"Updated episode {episode_id}",
            )
        cleanup = _remove_resolved_quality_issues(static_dir, reason=reason, episode_ids=fixed_episodes)
        summary = {"fix_kind": fix_kind, "episodes": sorted(fixed_episodes), "fixed": fixed, **cleanup}

    elif fix_kind == FLAG_FIX_STATE_GRIPPER_TRANSITION_ACTION:
        reason = "state_gripper_transition_without_action"
        issues = _load_quality_issues(static_dir, reason=reason)
        selected_issues = [issue for issue in issues if allowed is None or int(issue["episode"]) in allowed]
        emit(
            progress_callback,
            status="running",
            current=0,
            total=len(selected_issues),
            message=f"Adding gripper action lead signals for {len(selected_issues)} issues",
        )
        backup_dir = (
            static_dir
            / "flag_fix_backups"
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{FLAG_FIX_STATE_GRIPPER_TRANSITION_ACTION}"
        )
        fixed_episodes: set[int] = set()
        fixed = 0
        details = []
        for idx, issue in enumerate(selected_issues, start=1):
            episode_id = int(issue["episode"])
            result = _fix_state_transition_action(
                root,
                info,
                issue,
                selected_data_version,
                backup_dir,
                meta,
            )
            details.append({"episode": episode_id, **result})
            if result.get("fixed"):
                fixed += 1
                fixed_episodes.add(episode_id)
                _delete_episode_cache(static_dir, episode_id)
            emit(
                progress_callback,
                status="running",
                current=idx,
                total=len(selected_issues),
                episode=episode_id,
                message=f"Updated episode {episode_id}",
            )
        backup_manifest = None
        if fixed_episodes:
            backup_manifest = backup_dir / "manifest.json"
            write_json(
                backup_manifest,
                {
                    "fix_kind": fix_kind,
                    "source_root": str(root),
                    "data_version": selected_data_version,
                    "episodes": sorted(fixed_episodes),
                    "lead_seconds": FLAG_FIX_STATE_ACTION_LEAD_SECONDS,
                    "details": details,
                    "restore_note": "To roll back manually, copy each backed up parquet over the same relative path under source_root.",
                },
            )
        cleanup = _remove_resolved_quality_issues(static_dir, reason=reason, episode_ids=fixed_episodes)
        summary = {
            "fix_kind": fix_kind,
            "episodes": sorted(fixed_episodes),
            "fixed": fixed,
            "backup_manifest": str(backup_manifest) if backup_manifest else None,
            "details": details,
            **cleanup,
        }
    else:
        raise ValueError(f"Unsupported flag fix: {fix_kind}")

    emit(
        progress_callback,
        status="done",
        current=summary.get("fixed", 0),
        total=max(1, summary.get("fixed", 0)),
        message="Flag fix complete",
    )
    return PreprocessResult(
        op=f"flag_fix:{fix_kind}",
        src_roots=[root],
        out_root=root,
        repo_id=root.name,
        total_episodes=len(summary.get("episodes", [])),
        total_frames=int(info.get("total_frames") or 0),
        dry_run=False,
        summary=summary,
    )
