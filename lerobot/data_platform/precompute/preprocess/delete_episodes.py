import copy
import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    STATS_PATH,
    load_episodes,
    load_episodes_stats,
    load_info,
    serialize_dict,
    write_info,
    write_jsonlines,
    write_stats,
)
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    load_episode_records,
)
from lerobot.data_platform.precompute.mutations import fix_episode_indices

LogCallback = Callable[[str], None] | None


def _link_or_copy(source: str, destination: str) -> str:
    """Create a cheap snapshot when possible, falling back to a real copy."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _snapshot_static_file(source: str, destination: str) -> str:
    """Copy mutable control files; hard-link only large cache payloads."""
    cache_suffixes = {".avi", ".csv", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".png", ".webp"}
    if Path(source).suffix.lower() in cache_suffixes:
        return _link_or_copy(source, destination)
    return shutil.copy2(source, destination)


class _DeleteRollbackSnapshot:
    """Snapshot files touched by episode deletion for best-effort rollback."""

    def __init__(self, root: Path, static_folder: Path | None) -> None:
        self.root = root
        self.backup_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.delete-rollback-", dir=root.parent))
        self.entries: list[tuple[Path, Path, bool]] = []
        try:
            self._snapshot(root / "data", self.backup_root / "dataset" / "data", _link_or_copy)
            self._snapshot(root / "videos", self.backup_root / "dataset" / "videos", _link_or_copy)
            self._snapshot(root / "meta", self.backup_root / "dataset" / "meta", shutil.copy2)
            if static_folder is not None:
                self._snapshot(Path(static_folder), self.backup_root / "static", _snapshot_static_file)
        except Exception:
            shutil.rmtree(self.backup_root, ignore_errors=True)
            raise

    def _snapshot(self, target: Path, backup: Path, copy_function) -> None:
        existed = target.exists()
        self.entries.append((target, backup, existed))
        if not existed:
            return
        if not target.is_dir():
            raise ValueError(f"rollback target is not a directory: {target}")
        shutil.copytree(target, backup, copy_function=copy_function, symlinks=True)

    def rollback(self) -> None:
        for target, backup, existed in reversed(self.entries):
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            if existed:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(backup, target, copy_function=shutil.copy2, symlinks=True)
        self.discard()

    def discard(self) -> None:
        shutil.rmtree(self.backup_root, ignore_errors=True)


def _dataset_delete_state(dataset) -> dict:
    state = {}
    if hasattr(dataset.meta, "info"):
        state["meta_info"] = copy.deepcopy(dataset.meta.info)
    if hasattr(dataset.meta, "episodes"):
        state["meta_episodes"] = copy.deepcopy(dataset.meta.episodes)
    for attribute in ["total_episodes", "total_frames"]:
        if hasattr(dataset, attribute):
            state[attribute] = getattr(dataset, attribute)
    return state


def _restore_dataset_delete_state(dataset, state: dict) -> None:
    if "meta_info" in state:
        dataset.meta.info.clear()
        dataset.meta.info.update(state["meta_info"])
    if "meta_episodes" in state:
        dataset.meta.episodes.clear()
        dataset.meta.episodes.update(state["meta_episodes"])
    for attribute in ["total_episodes", "total_frames"]:
        if attribute in state:
            setattr(dataset, attribute, state[attribute])


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _episode_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _remap_episode(old_episode: int, delete_set: set[int], old_to_new: dict[int, int]) -> int | None:
    if old_episode in delete_set:
        return None
    return int(old_to_new.get(old_episode, old_episode))


def _episode_sort_key(item: dict, field: str) -> int:
    value = _episode_int(item.get(field))
    return value if value is not None else -1


def _reindex_episode_list(values, delete_set: set[int], old_to_new: dict[int, int]) -> list[int]:
    episodes: set[int] = set()
    for value in values if isinstance(values, list) else []:
        old_episode = _episode_int(value)
        if old_episode is None:
            continue
        new_episode = _remap_episode(old_episode, delete_set, old_to_new)
        if new_episode is not None:
            episodes.add(new_episode)
    return sorted(episodes)


def _reindex_flag_file(path: Path, delete_set: set[int], old_to_new: dict[int, int]) -> bool:
    if not path.is_file():
        return False
    data = _load_json(path, None)
    if data is None:
        return False
    if isinstance(data, list):
        _write_json(path, _reindex_episode_list(data, delete_set, old_to_new))
        return True
    if not isinstance(data, dict):
        return False

    updated = dict(data)
    flagged_episodes = _reindex_episode_list(data.get("flagged_episodes", []), delete_set, old_to_new)
    updated["flagged_episodes"] = flagged_episodes
    reasons = data.get("flag_reasons")
    if isinstance(reasons, dict):
        updated_reasons = {}
        for key, value in reasons.items():
            old_episode = _episode_int(key)
            if old_episode is None:
                continue
            new_episode = _remap_episode(old_episode, delete_set, old_to_new)
            if new_episode is not None:
                updated_reasons[str(new_episode)] = value
        updated["flag_reasons"] = updated_reasons
    summary = data.get("summary")
    if isinstance(summary, dict):
        updated_summary = dict(summary)
        for key in [
            "flagged_episode_count",
            "quality_episode_count",
            "missing_target_count",
            "prompt_action_mismatch_count",
        ]:
            if key in updated_summary:
                updated_summary[key] = len(flagged_episodes)
        updated["summary"] = updated_summary
    _write_json(path, updated)
    return True


def _reindex_keyed_episode_json(path: Path, delete_set: set[int], old_to_new: dict[int, int]) -> bool:
    data = _load_json(path, None)
    if not isinstance(data, dict):
        return False
    updated = {}
    for key, value in data.items():
        old_episode = _episode_int(key)
        if old_episode is None:
            updated[key] = value
            continue
        new_episode = _remap_episode(old_episode, delete_set, old_to_new)
        if new_episode is not None:
            updated[str(new_episode)] = value
    _write_json(path, updated)
    return True


def _reindex_annotation_issues(path: Path, delete_set: set[int], old_to_new: dict[int, int]) -> bool:
    issues = _load_json(path, None)
    if not isinstance(issues, list):
        return False
    updated = []
    for issue in issues:
        if not isinstance(issue, dict):
            updated.append(issue)
            continue
        old_episode = _episode_int(issue.get("episode"))
        if old_episode is None:
            updated.append(issue)
            continue
        new_episode = _remap_episode(old_episode, delete_set, old_to_new)
        if new_episode is None:
            continue
        new_issue = dict(issue)
        new_issue["episode"] = new_episode
        updated.append(new_issue)
    _write_json(path, updated)
    return True


def _reindex_viewer_manifest(path: Path, delete_set: set[int], old_to_new: dict[int, int]) -> bool:
    data = _load_json(path, None)
    if not isinstance(data, dict):
        return False
    raw_episodes = data.get("episodes")
    if not isinstance(raw_episodes, list):
        return False

    updated_episodes = []
    changed = False
    for episode in raw_episodes:
        if not isinstance(episode, dict):
            updated_episodes.append(episode)
            continue
        old_episode = _episode_int(episode.get("episode_index"))
        if old_episode is None:
            updated_episodes.append(episode)
            continue
        new_episode = _remap_episode(old_episode, delete_set, old_to_new)
        if new_episode is None:
            changed = True
            continue
        updated_episode = dict(episode)
        if new_episode != old_episode:
            updated_episode["episode_index"] = new_episode
            changed = True
        updated_episodes.append(updated_episode)

    if not changed:
        return False
    updated_episodes.sort(
        key=lambda item: _episode_sort_key(item, "episode_index") if isinstance(item, dict) else -1
    )
    updated = dict(data)
    updated["episodes"] = updated_episodes
    updated["total_episodes"] = len([item for item in updated_episodes if isinstance(item, dict)])
    lengths = [
        int(item.get("length") or 0)
        for item in updated_episodes
        if isinstance(item, dict) and _episode_int(item.get("length")) is not None
    ]
    if lengths:
        updated["total_frames"] = sum(lengths)
    _write_json(path, updated)
    return True


def _reindex_pending_prompt_assignments(path: Path, delete_set: set[int], old_to_new: dict[int, int]) -> bool:
    data = _load_json(path, None)
    raw_items = data.get("assignments") if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        return False

    updated_items = []
    changed = False
    for item in raw_items:
        if not isinstance(item, dict):
            updated_items.append(item)
            continue
        old_episode = _episode_int(item.get("episode_index"))
        if old_episode is None:
            updated_items.append(item)
            continue
        new_episode = _remap_episode(old_episode, delete_set, old_to_new)
        if new_episode is None:
            changed = True
            continue
        updated_item = dict(item)
        if new_episode != old_episode:
            updated_item["episode_index"] = new_episode
            changed = True
        updated_items.append(updated_item)

    if not changed:
        return False
    updated_items.sort(
        key=lambda item: _episode_sort_key(item, "episode_index") if isinstance(item, dict) else -1
    )
    if isinstance(data, dict):
        updated = dict(data)
        updated["assignments"] = updated_items
    else:
        updated = {"version": 1, "assignments": updated_items}
    _write_json(path, updated)
    return True


def _reindex_construction_plan(path: Path, delete_set: set[int], old_to_new: dict[int, int]) -> bool:
    data = _load_json(path, None)
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return False

    updated_records = []
    changed = False
    for record in records:
        if not isinstance(record, dict):
            updated_records.append(record)
            continue
        old_episode = _episode_int(record.get("new_episode_index"))
        if old_episode is None:
            updated_records.append(record)
            continue
        new_episode = _remap_episode(old_episode, delete_set, old_to_new)
        if new_episode is None:
            changed = True
            continue
        updated_record = dict(record)
        if new_episode != old_episode:
            updated_record["new_episode_index"] = new_episode
            changed = True
        updated_records.append(updated_record)

    if not changed:
        return False
    updated_records.sort(
        key=lambda item: _episode_sort_key(item, "new_episode_index") if isinstance(item, dict) else -1
    )
    if isinstance(data, dict):
        updated = dict(data)
        updated["records"] = updated_records
    else:
        updated = updated_records
    _write_json(path, updated)
    return True


def _jsonl_paths(static_folder: Path) -> list[Path]:
    paths: set[Path] = set()
    for subdir, patterns in {
        "labeling": ["labels*.jsonl", "labels_reviewed*.jsonl"],
        "tagging": ["tags*.jsonl", "tags_reviewed*.jsonl"],
    }.items():
        directory = static_folder / subdir
        if not directory.is_dir():
            continue
        for pattern in patterns:
            paths.update(path for path in directory.glob(pattern) if path.is_file())
    return sorted(paths)


def _reindex_episode_jsonl(path: Path, delete_set: set[int], old_to_new: dict[int, int]) -> bool:
    records = []
    changed = False
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # Preserve unreadable lines by leaving the file untouched.
            return False
        if not isinstance(record, dict):
            records.append(record)
            continue
        old_episode = _episode_int(record.get("episode_index"))
        if old_episode is None:
            records.append(record)
            continue
        new_episode = _remap_episode(old_episode, delete_set, old_to_new)
        if new_episode is None:
            changed = True
            continue
        if new_episode != old_episode:
            record = dict(record)
            record["episode_index"] = new_episode
            changed = True
        records.append(record)
    if not changed:
        return False
    records.sort(key=lambda item: item.get("episode_index", -1) if isinstance(item, dict) else -1)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    tmp.replace(path)
    return True


def _invalidate_derived_static(static_folder: Path) -> list[str]:
    invalidated: list[str] = []
    for dirname in ["analysis", "embedding", "compare", "construction"]:
        path = static_folder / dirname
        if path.is_dir():
            shutil.rmtree(path)
            invalidated.append(dirname)
    return invalidated


def reindex_static_after_episode_delete(
    static_folder: Path | None,
    delete_set: set[int],
    old_to_new: dict[int, int],
    *,
    invalidate_derived: bool = True,
    raise_on_error: bool = False,
    log: LogCallback = None,
) -> dict:
    """Reindex static per-episode artifacts after deleting episodes.

    Files with explicit episode ids are rewritten in place. Derived caches whose
    row order or reducer state depends on the old episode set are invalidated.
    """

    if static_folder is None:
        return {"reindexed_files": [], "invalidated": []}
    static_folder = Path(static_folder)
    if not static_folder.exists():
        return {"reindexed_files": [], "invalidated": []}

    reindexed: list[str] = []
    try:
        flag_paths = {static_folder / "flagged_episodes.json"}
        flag_paths.update(path for path in static_folder.glob("*_flagged_episodes.json") if path.is_file())
        for path in sorted(flag_paths):
            if _reindex_flag_file(path, delete_set, old_to_new):
                reindexed.append(path.name)

        for filename in ["trim_annotations.json", "subtask_annotations.json"]:
            path = static_folder / filename
            if path.is_file() and _reindex_keyed_episode_json(path, delete_set, old_to_new):
                reindexed.append(filename)

        issues_path = static_folder / "annotation_issues.json"
        if issues_path.is_file() and _reindex_annotation_issues(issues_path, delete_set, old_to_new):
            reindexed.append(issues_path.name)

        manifest_path = static_folder / "viewer_manifest.json"
        if manifest_path.is_file() and _reindex_viewer_manifest(manifest_path, delete_set, old_to_new):
            reindexed.append(manifest_path.name)

        pending_prompt_path = static_folder / "prompt_assignments_pending.json"
        if pending_prompt_path.is_file() and _reindex_pending_prompt_assignments(
            pending_prompt_path,
            delete_set,
            old_to_new,
        ):
            reindexed.append(pending_prompt_path.name)

        for path in _jsonl_paths(static_folder):
            if _reindex_episode_jsonl(path, delete_set, old_to_new):
                reindexed.append(str(path.relative_to(static_folder)))

        invalidated = _invalidate_derived_static(static_folder) if invalidate_derived else []
        if log and (reindexed or invalidated):
            log(f"Reindexed {len(reindexed)} static files; invalidated {len(invalidated)} derived caches")
        return {"reindexed_files": reindexed, "invalidated": invalidated}
    except Exception:
        logging.exception("Failed to reindex static artifacts after deleting episodes")
        if raise_on_error:
            raise
        return {"reindexed_files": reindexed, "invalidated": []}


def _feature_keys(dataset, dtype: str) -> list[str]:
    return [key for key, feature in getattr(dataset, "features", {}).items() if feature.get("dtype") == dtype]


def _clear_empty_dirs(root: Path) -> None:
    for base in [root / "data", root / "videos"]:
        if not base.is_dir():
            continue
        for child in sorted(base.rglob("*"), reverse=True):
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()


def _reindex_cached_files(
    static_folder: Path | None,
    image_keys: list[str],
    video_keys: list[str],
    old_idx: int,
    new_idx: int,
) -> None:
    if static_folder is None:
        return
    for key in image_keys:
        old_cached = static_folder / "videos" / key / f"episode_{old_idx:06d}_h264.mp4"
        new_cached = static_folder / "videos" / key / f"episode_{new_idx:06d}_h264.mp4"
        if old_cached.is_file():
            new_cached.parent.mkdir(parents=True, exist_ok=True)
            old_cached.rename(new_cached)
    for video_key in video_keys:
        old_cached = static_folder / "videos" / video_key / f"episode_{old_idx:06d}_h264.mp4"
        new_cached = static_folder / "videos" / video_key / f"episode_{new_idx:06d}_h264.mp4"
        if old_cached.is_file():
            new_cached.parent.mkdir(parents=True, exist_ok=True)
            old_cached.rename(new_cached)
    cache_dir = static_folder / "csv"
    if cache_dir.is_dir():
        for csv_file in cache_dir.glob(f"episode_{old_idx:06d}_ds*.csv"):
            new_name = csv_file.name.replace(f"episode_{old_idx:06d}", f"episode_{new_idx:06d}")
            csv_file.rename(cache_dir / new_name)


def _delete_cached_files(
    static_folder: Path | None,
    image_keys: list[str],
    video_keys: list[str],
    episode_id: int,
) -> None:
    if static_folder is None:
        return
    for key in image_keys:
        cached_video = static_folder / "videos" / key / f"episode_{episode_id:06d}_h264.mp4"
        if cached_video.is_file():
            cached_video.unlink()
    for video_key in video_keys:
        cached_video = static_folder / "videos" / video_key / f"episode_{episode_id:06d}_h264.mp4"
        if cached_video.is_file():
            cached_video.unlink()
    cache_dir = static_folder / "csv"
    if cache_dir.is_dir():
        for csv_file in cache_dir.glob(f"episode_{episode_id:06d}_ds*.csv"):
            csv_file.unlink()


def _delete_episodes_inplace_unprotected(
    dataset,
    episode_ids: list[int],
    *,
    static_folder: Path | None = None,
    log: LogCallback = None,
) -> dict:
    """Apply episode deletion. The public wrapper owns snapshot and rollback handling."""

    def _log(message: str) -> None:
        if log:
            log(message)

    delete_set = {int(idx) for idx in episode_ids}
    if not delete_set:
        raise ValueError("no episodes selected")

    root = Path(dataset.root)
    chunks_size = int(dataset.meta.info.get("chunks_size", 1000) or 1000)
    image_keys = _feature_keys(dataset, "image")
    video_keys = (
        list(dataset.meta.video_keys)
        if hasattr(dataset.meta, "video_keys")
        else _feature_keys(dataset, "video")
    )

    eps_data = load_episodes(root)
    all_indices = sorted(int(idx) for idx in eps_data)
    missing = sorted(delete_set - set(all_indices))
    if missing:
        raise ValueError(f"episodes not found: {missing}")

    ep_lengths: dict[int, int] = {}
    for episode_id in sorted(delete_set):
        ep_length = int(eps_data.get(episode_id, {}).get("length", 0) or 0)
        if ep_length == 0:
            parquet_path = root / dataset.meta.get_data_file_path(episode_id)
            if parquet_path.is_file():
                ep_length = pq.read_table(parquet_path, columns=["timestamp"]).num_rows
        ep_lengths[episode_id] = ep_length

    _log(f"Deleting files for episodes: {sorted(delete_set)}")
    for episode_id in sorted(delete_set):
        parquet_path = root / dataset.meta.get_data_file_path(episode_id)
        if parquet_path.is_file():
            parquet_path.unlink()
        for video_key in video_keys:
            video_path = root / dataset.meta.get_video_file_path(episode_id, video_key)
            if video_path.is_file():
                video_path.unlink()
        _delete_cached_files(static_folder, image_keys, video_keys, episode_id)

    remaining_old_indices = [idx for idx in all_indices if idx not in delete_set]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_old_indices)}

    _log(f"Reindexing {len(remaining_old_indices)} remaining episodes once")
    for old_idx, new_idx in old_to_new.items():
        if old_idx == new_idx:
            continue
        old_chunk = old_idx // chunks_size
        new_chunk = new_idx // chunks_size

        old_pq = root / dataset.meta.info["data_path"].format(episode_chunk=old_chunk, episode_index=old_idx)
        new_pq = root / dataset.meta.info["data_path"].format(episode_chunk=new_chunk, episode_index=new_idx)
        if old_pq.is_file():
            new_pq.parent.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(old_pq)
            if "episode_index" in table.schema.names:
                field = table.schema.field("episode_index")
                column_idx = table.schema.get_field_index("episode_index")
                table = table.set_column(
                    column_idx,
                    field,
                    pa.array([new_idx] * table.num_rows, type=field.type),
                )
            tmp_pq = new_pq.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp_pq)
            old_pq.unlink()
            tmp_pq.rename(new_pq)

        if dataset.meta.info.get("video_path"):
            for video_key in video_keys:
                old_video = root / dataset.meta.info["video_path"].format(
                    episode_chunk=old_chunk, video_key=video_key, episode_index=old_idx
                )
                new_video = root / dataset.meta.info["video_path"].format(
                    episode_chunk=new_chunk, video_key=video_key, episode_index=new_idx
                )
                if old_video.is_file():
                    new_video.parent.mkdir(parents=True, exist_ok=True)
                    old_video.rename(new_video)

        _reindex_cached_files(static_folder, image_keys, video_keys, old_idx, new_idx)

    _clear_empty_dirs(root)

    new_eps_data = {}
    for old_idx, new_idx in old_to_new.items():
        entry = dict(eps_data[old_idx])
        entry["episode_index"] = new_idx
        new_eps_data[new_idx] = entry
    write_jsonlines([new_eps_data[idx] for idx in sorted(new_eps_data.keys())], root / EPISODES_PATH)
    if hasattr(dataset.meta, "episodes"):
        dataset.meta.episodes.clear()
        dataset.meta.episodes.update(new_eps_data)

    info = load_info(root)
    new_total_eps = len(new_eps_data)
    info["total_episodes"] = new_total_eps
    info["total_frames"] = max(0, int(info.get("total_frames", 0) or 0) - sum(ep_lengths.values()))
    if video_keys:
        info["total_videos"] = max(
            0,
            int(info.get("total_videos", 0) or 0) - len(delete_set) * len(video_keys),
        )
    if "splits" in info:
        for split_name in info["splits"]:
            info["splits"][split_name] = f"0:{new_total_eps}"
    info["total_chunks"] = math.ceil(new_total_eps / chunks_size) if new_total_eps > 0 else 0
    write_info(info, root)
    if hasattr(dataset.meta, "info"):
        dataset.meta.info.update(info)
    if hasattr(dataset, "total_episodes"):
        dataset.total_episodes = new_total_eps
    if hasattr(dataset, "total_frames"):
        dataset.total_frames = info["total_frames"]

    episodes_stats_path = root / EPISODES_STATS_PATH
    if episodes_stats_path.is_file():
        old_stats = load_episodes_stats(root)
        new_stats = {
            new_idx: old_stats[old_idx] for old_idx, new_idx in old_to_new.items() if old_idx in old_stats
        }
        write_jsonlines(
            [
                {"episode_index": idx, "stats": serialize_dict(new_stats[idx])}
                for idx in sorted(new_stats.keys())
            ],
            episodes_stats_path,
        )
        global_stats_path = root / STATS_PATH
        if global_stats_path.is_file():
            if new_stats:
                write_stats(aggregate_stats(list(new_stats.values())), root)
            else:
                global_stats_path.unlink()

    remaining_episode_ids = sorted(new_eps_data.keys())
    fix_episode_indices(root, dataset.meta, remaining_episode_ids)
    construction_plan_path = root / "meta" / "construction_plan.json"
    if construction_plan_path.is_file():
        _reindex_construction_plan(construction_plan_path, delete_set, old_to_new)
    reindex_static_after_episode_delete(
        static_folder,
        delete_set,
        old_to_new,
        raise_on_error=True,
        log=log,
    )
    _log("Updated metadata and repaired frame_index/timestamp/index after batch delete")

    return {
        "deleted_episode_ids": sorted(delete_set),
        "new_total_episodes": len(remaining_episode_ids),
        "next_episode": (
            min(sorted(delete_set)[0], len(remaining_episode_ids) - 1) if remaining_episode_ids else None
        ),
    }


def delete_episodes_inplace(
    dataset,
    episode_ids: list[int],
    *,
    static_folder: Path | None = None,
    log: LogCallback = None,
) -> dict:
    """Delete and reindex episodes, restoring the pre-delete state if any step fails.

    The rollback snapshot uses hard links for large immutable/replace-only files
    when the filesystem supports them, so successful deletion does not require a
    second full copy of every video. This is not an isolated filesystem transaction:
    concurrent readers can observe intermediate paths, and concurrent writers are not
    supported. If rollback itself fails, the recovery snapshot is retained.
    """

    delete_set = {int(idx) for idx in episode_ids}
    if not delete_set:
        raise ValueError("no episodes selected")

    root = Path(dataset.root)
    source_is_v3 = is_v3_dataset(root)
    all_indices = (
        {int(row["episode_index"]) for row in load_episode_records(root)}
        if source_is_v3
        else set(load_episodes(root))
    )
    missing = sorted(delete_set - all_indices)
    if missing:
        raise ValueError(f"episodes not found: {missing}")
    if len(delete_set) == len(all_indices):
        raise ValueError("episode deletion would remove the entire dataset")

    snapshot = _DeleteRollbackSnapshot(root, static_folder)
    dataset_state = _dataset_delete_state(dataset)
    if log:
        log("Created rollback snapshot before episode deletion")

    try:
        if source_is_v3:
            from lerobot.data_platform.precompute.preprocess.dataset_merge import run_merge

            remaining_old_indices = sorted(all_indices - delete_set)
            old_to_new = {old_index: new_index for new_index, old_index in enumerate(remaining_old_indices)}
            with tempfile.TemporaryDirectory(
                prefix=f".{root.name}.delete-v3-",
                dir=root.parent,
            ) as temp_dir:
                rebuilt_root = Path(temp_dir) / "dataset"
                run_merge(
                    [root],
                    out_root=rebuilt_root,
                    workers=8,
                    exclude_episodes=[delete_set],
                    _allow_single_source=True,
                    _op="delete_episodes",
                    _default_op="delete_episodes",
                )

                generated_meta = rebuilt_root / "meta"
                for child in (root / "meta").iterdir():
                    if child.name in {
                        "episodes",
                        "episodes.jsonl",
                        "episodes_stats.jsonl",
                        "info.json",
                        "stats.json",
                        "tasks.jsonl",
                        "tasks.parquet",
                    }:
                        continue
                    destination = generated_meta / child.name
                    if destination.exists():
                        continue
                    if child.is_dir():
                        shutil.copytree(child, destination, symlinks=True)
                    else:
                        shutil.copy2(child, destination, follow_symlinks=False)

                for name in ("data", "meta", "videos"):
                    target = root / name
                    if target.is_dir():
                        shutil.rmtree(target)
                    source = rebuilt_root / name
                    if source.exists():
                        shutil.move(str(source), target)

            image_keys = _feature_keys(dataset, "image")
            video_keys = (
                list(dataset.meta.video_keys)
                if hasattr(dataset.meta, "video_keys")
                else _feature_keys(dataset, "video")
            )
            for episode_index in sorted(delete_set):
                _delete_cached_files(
                    static_folder,
                    image_keys,
                    video_keys,
                    episode_index,
                )
            for old_index, new_index in old_to_new.items():
                if old_index != new_index:
                    _reindex_cached_files(
                        static_folder,
                        image_keys,
                        video_keys,
                        old_index,
                        new_index,
                    )
            construction_plan_path = root / "meta" / "construction_plan.json"
            if construction_plan_path.is_file():
                _reindex_construction_plan(
                    construction_plan_path,
                    delete_set,
                    old_to_new,
                )
            reindex_static_after_episode_delete(
                static_folder,
                delete_set,
                old_to_new,
                raise_on_error=True,
                log=log,
            )
            new_meta = V3DatasetMetadata(
                getattr(dataset, "repo_id", f"local/{root.name}"),
                root,
            )
            dataset.meta = new_meta
            dataset.features = new_meta.features
            dataset.fps = new_meta.fps
            dataset.codebase_version = new_meta.info.get(
                "codebase_version",
                "v3.0",
            )
            dataset.total_episodes = new_meta.total_episodes
            dataset.total_frames = new_meta.total_frames
            result = {
                "deleted_episode_ids": sorted(delete_set),
                "new_total_episodes": new_meta.total_episodes,
                "next_episode": min(
                    min(delete_set),
                    new_meta.total_episodes - 1,
                ),
            }
        else:
            result = _delete_episodes_inplace_unprotected(
                dataset,
                sorted(delete_set),
                static_folder=static_folder,
                log=log,
            )
    except Exception as exc:
        logging.exception("Episode deletion failed; restoring rollback snapshot")
        try:
            snapshot.rollback()
            _restore_dataset_delete_state(dataset, dataset_state)
        except Exception as rollback_exc:
            logging.exception("Episode deletion rollback failed; backup retained at %s", snapshot.backup_root)
            raise RuntimeError(
                f"episode deletion and rollback failed; recovery snapshot kept at {snapshot.backup_root}"
            ) from rollback_exc
        if log:
            log("Episode deletion failed; restored the dataset to its pre-delete state")
        raise RuntimeError("episode deletion failed; restored the pre-delete state") from exc

    snapshot.discard()
    if log:
        log("Episode deletion completed; removed rollback snapshot")
    return result
