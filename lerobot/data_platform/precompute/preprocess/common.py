import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[dict], None] | None


@dataclass
class PreprocessResult:
    op: str
    src_roots: list[Path]
    out_root: Path
    repo_id: str
    total_episodes: int = 0
    total_frames: int = 0
    dry_run: bool = False
    summary: dict = field(default_factory=dict)
    episode_lineage: list[dict] = field(default_factory=list)


def emit(progress_callback: ProgressCallback, **payload) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def default_preprocess_path(src_root: Path, op: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_op = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in op).strip("_")
    return Path(src_root).expanduser().parent / f"{Path(src_root).name}_{safe_op}_{timestamp}"


def validate_dataset_root(root: Path) -> Path:
    root = Path(root).expanduser()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {root / 'meta' / 'info.json'}")
    if not (root / "data").is_dir():
        raise FileNotFoundError(f"Missing LeRobot data directory: {root / 'data'}")
    return root


def ensure_output_root(out_root: Path, dry_run: bool = False) -> Path:
    out_root = Path(out_root).expanduser()
    if out_root.exists() and not dry_run:
        raise FileExistsError(f"Output dataset already exists: {out_root}")
    return out_root


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_meta_files(src_root: Path, out_root: Path) -> None:
    src_meta = Path(src_root) / "meta"
    out_meta = Path(out_root) / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)
    for path in src_meta.iterdir():
        if path.is_dir():
            shutil.copytree(path, out_meta / path.name)
        else:
            shutil.copy2(path, out_meta / path.name)


def copy_sidecar_dirs(src_root: Path, out_root: Path, skip: set[str] | None = None) -> None:
    skip = {"data", "meta"} | set(skip or set())
    for child in Path(src_root).iterdir():
        if not child.is_dir() or child.name in skip:
            continue
        dst = Path(out_root) / child.name
        if dst.exists():
            continue
        shutil.copytree(child, dst, symlinks=True)


def parquet_paths(root: Path) -> list[Path]:
    return sorted((Path(root) / "data").rglob("*.parquet"))


def format_data_path(info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    template = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    return Path(
        template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    )


def format_video_path(info: dict, episode_index: int, video_key: str) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    template = (
        info.get("video_path")
        or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    return Path(
        template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def video_feature_keys(info: dict) -> list[str]:
    return [key for key, feature in (info.get("features") or {}).items() if feature.get("dtype") == "video"]


def update_info_counts(
    info: dict, total_episodes: int, total_frames: int, total_tasks: int | None = None
) -> dict:
    updated = dict(info)
    chunks_size = int(updated.get("chunks_size") or 1000)
    updated["total_episodes"] = int(total_episodes)
    updated["total_frames"] = int(total_frames)
    if total_tasks is not None:
        updated["total_tasks"] = int(total_tasks)
    updated["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    updated["splits"] = {"train": f"0:{total_episodes}"}
    updated["total_videos"] = len(video_feature_keys(updated)) * int(total_episodes)
    return updated


def copy_episode_videos(
    episode_map: list[tuple[Path, int, int]],
    info: dict,
    out_root: Path,
) -> None:
    video_keys = video_feature_keys(info)
    if not video_keys:
        return
    for src_root, old_idx, new_idx in episode_map:
        for video_key in video_keys:
            src = Path(src_root) / format_video_path(info, old_idx, video_key)
            dst = Path(out_root) / format_video_path(info, new_idx, video_key)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
