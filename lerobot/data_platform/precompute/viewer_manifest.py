#!/usr/bin/env python

"""Small manifest that lets the HTML viewer run from prepared cache only."""

import json
from pathlib import Path
from typing import Any

from lerobot.data_platform.precompute.data_profile import resolve_data_profile

VIEWER_MANIFEST = "viewer_manifest.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def build_viewer_manifest(
    *,
    root: Path,
    repo_id: str,
    meta,
    episodes: list[int],
    image_keys: list[str],
    data_version: str,
    downsample: int | None = None,
) -> dict:
    episode_rows = []
    total_frames = 0
    for episode_id in episodes:
        ep_info = meta.episodes.get(episode_id) or meta.episodes.get(str(episode_id)) or {}
        length = int(ep_info.get("length") or 0)
        total_frames += length
        episode_rows.append(
            {
                "episode_index": int(episode_id),
                "length": length,
                "tasks": list(ep_info.get("tasks") or []),
            }
        )

    features = dict(getattr(meta, "features", {}) or {})
    data_profile = resolve_data_profile(
        root,
        features,
        data_version_override=data_version,
    )
    video_keys = list(getattr(meta, "video_keys", []) or [])
    return {
        "version": 1,
        "repo_id": repo_id,
        "root": str(Path(root).expanduser()),
        "data_version": str(data_version),
        "data_profile": data_profile.to_dict(),
        "fps": int(getattr(meta, "fps", 0) or 0),
        "total_episodes": len(episode_rows),
        "total_frames": int(getattr(meta, "total_frames", 0) or total_frames),
        "episodes": episode_rows,
        "features": _json_safe(features),
        "image_keys": list(image_keys),
        "video_keys": video_keys,
        "downsample": int(downsample) if downsample and downsample > 1 else 1,
    }


def write_viewer_manifest(
    *,
    root: Path,
    repo_id: str,
    meta,
    episodes: list[int],
    image_keys: list[str],
    static_dir: Path,
    data_version: str,
    downsample: int | None = None,
) -> Path:
    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_viewer_manifest(
        root=Path(root),
        repo_id=repo_id,
        meta=meta,
        episodes=episodes,
        image_keys=image_keys,
        data_version=data_version,
        downsample=downsample,
    )
    path = static_dir / VIEWER_MANIFEST
    path.write_text(json.dumps(manifest, indent=2))
    return path


def load_viewer_manifest(static_dir: Path) -> dict | None:
    path = Path(static_dir) / VIEWER_MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def manifest_episode_ids(manifest: dict) -> list[int]:
    ids = []
    for episode in manifest.get("episodes") or []:
        try:
            ids.append(int(episode.get("episode_index")))
        except (TypeError, ValueError):
            continue
    return ids


def manifest_task_episode_map(manifest: dict) -> dict[str, list[int]]:
    task_map: dict[str, list[int]] = {}
    for episode in manifest.get("episodes") or []:
        try:
            episode_id = int(episode.get("episode_index"))
        except (TypeError, ValueError):
            continue
        for task in episode.get("tasks") or []:
            task_map.setdefault(str(task), []).append(episode_id)
    return task_map


def manifest_episode_info(manifest: dict, episode_id: int) -> dict:
    for episode in manifest.get("episodes") or []:
        try:
            if int(episode.get("episode_index")) == int(episode_id):
                return episode
        except (TypeError, ValueError):
            continue
    return {}
