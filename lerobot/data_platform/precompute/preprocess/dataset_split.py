import tempfile
from pathlib import Path

import pandas as pd

from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    ProgressCallback,
    copy_episode_videos,
    default_preprocess_path,
    emit,
    ensure_output_root,
    format_data_path,
    load_json,
    load_jsonl,
    update_info_counts,
    validate_dataset_root,
    write_json,
    write_jsonl,
)
from lerobot.data_platform.precompute.preprocess.dataset_version import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_V3_CONVERT_WORKERS,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    V30,
    detect_dataset_version,
    materialize_v21_from_v3,
    run_convert_v3,
)


def parse_episode_range(value: str | tuple[int, int] | None) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, tuple):
        return value
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        raise ValueError("episode_range must be START:END")
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end <= start:
        raise ValueError("episode_range must satisfy 0 <= START < END")
    return start, end


def _task_filter_texts(task_filter, tasks: list[dict]) -> set[str]:
    if task_filter in (None, "", []):
        return set()
    if isinstance(task_filter, str):
        tokens = [token.strip() for token in task_filter.replace("\n", ",").split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in task_filter if str(token).strip()]
    idx_to_text = {int(row["task_index"]): row["task"] for row in tasks}
    texts = set()
    for token in tokens:
        if token.isdigit() and int(token) in idx_to_text:
            texts.add(idx_to_text[int(token)])
        else:
            texts.add(token)
    return texts


def _select_episodes(episodes: list[dict], episode_range, task_texts: set[str]) -> list[dict]:
    selected = episodes
    if episode_range is not None:
        start, end = episode_range
        selected = [row for row in selected if start <= int(row["episode_index"]) < end]
    if task_texts:
        selected = [row for row in selected if any(task in task_texts for task in row.get("tasks", []))]
    return sorted(selected, key=lambda row: int(row["episode_index"]))


def _build_tasks(selected: list[dict], tasks: list[dict]) -> tuple[list[dict], dict[int, int]]:
    text_to_old = {row["task"]: int(row["task_index"]) for row in tasks}
    old_to_text = {int(row["task_index"]): row["task"] for row in tasks}
    text_to_new = {}
    new_tasks = []
    for episode in selected:
        for text in episode.get("tasks", []):
            if text not in text_to_new:
                new_idx = len(text_to_new)
                text_to_new[text] = new_idx
                new_tasks.append({"task_index": new_idx, "task": text})
    old_to_new = {}
    for old_idx, text in old_to_text.items():
        if text in text_to_new:
            old_to_new[old_idx] = text_to_new[text]
    for text in text_to_new:
        if text not in text_to_old:
            raise ValueError(f"Episode references task missing from tasks.jsonl: {text}")
    return new_tasks, old_to_new


def _build_episode_records(
    selected: list[dict], stats: list[dict]
) -> tuple[list[tuple[int, int]], list[dict], list[dict], int]:
    stats_by_idx = {int(row["episode_index"]): row for row in stats}
    mapping = []
    new_episodes = []
    new_stats = []
    cumulative_frames = 0
    for new_idx, episode in enumerate(selected):
        old_idx = int(episode["episode_index"])
        length = int(episode["length"])
        mapping.append((old_idx, new_idx))
        new_episodes.append({"episode_index": new_idx, "tasks": episode.get("tasks", []), "length": length})
        if old_idx in stats_by_idx:
            row = dict(stats_by_idx[old_idx])
            row["episode_index"] = new_idx
            stats_payload = row.get("stats") or {}
            if "episode_index" in stats_payload:
                stats_payload["episode_index"].update(
                    {"min": [new_idx], "max": [new_idx], "mean": [float(new_idx)], "std": [0.0]}
                )
            if "index" in stats_payload:
                stats_payload["index"].update(
                    {
                        "min": [cumulative_frames],
                        "max": [cumulative_frames + length - 1],
                        "mean": [cumulative_frames + (length - 1) / 2.0],
                    }
                )
            new_stats.append(row)
        cumulative_frames += length
    return mapping, new_episodes, new_stats, cumulative_frames


def run_split(
    src_root: Path,
    out_root: Path | None = None,
    episode_range: str | tuple[int, int] | None = None,
    task_filter=None,
    dry_run: bool = False,
    progress_callback: ProgressCallback = None,
) -> PreprocessResult:
    src_root = validate_dataset_root(src_root)
    out_root = ensure_output_root(out_root or default_preprocess_path(src_root, "split"), dry_run)
    if detect_dataset_version(src_root) == V30:
        source_info = load_json(src_root / "meta" / "info.json")
        with tempfile.TemporaryDirectory(
            prefix=".lerobot-v3-split-",
            dir=out_root.parent,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            legacy_source = materialize_v21_from_v3(
                src_root,
                temp_root / "source",
                include_data=not dry_run,
                include_videos=not dry_run,
                workers=DEFAULT_V3_CONVERT_WORKERS,
                progress_callback=progress_callback,
            )
            legacy_output = temp_root / "output"
            legacy_result = run_split(
                legacy_source,
                legacy_output,
                episode_range=episode_range,
                task_filter=task_filter,
                dry_run=dry_run,
                progress_callback=progress_callback,
            )
            summary = {
                **legacy_result.summary,
                "source_format": V30,
                "output_format": V30,
            }
            if not dry_run:
                run_convert_v3(
                    legacy_output,
                    out_root,
                    data_file_size_in_mb=int(
                        source_info.get("data_files_size_in_mb") or DEFAULT_DATA_FILE_SIZE_IN_MB
                    ),
                    video_file_size_in_mb=int(
                        source_info.get("video_files_size_in_mb") or DEFAULT_VIDEO_FILE_SIZE_IN_MB
                    ),
                    workers=DEFAULT_V3_CONVERT_WORKERS,
                    progress_callback=progress_callback,
                )
            return PreprocessResult(
                op="split",
                src_roots=[src_root],
                out_root=out_root,
                repo_id=f"local/{out_root.name}",
                total_episodes=legacy_result.total_episodes,
                total_frames=legacy_result.total_frames,
                dry_run=dry_run,
                summary=summary,
            )
    info = load_json(src_root / "meta" / "info.json")
    episodes = load_jsonl(src_root / "meta" / "episodes.jsonl")
    tasks = load_jsonl(src_root / "meta" / "tasks.jsonl")
    stats = load_jsonl(src_root / "meta" / "episodes_stats.jsonl")
    parsed_range = parse_episode_range(episode_range)
    task_texts = _task_filter_texts(task_filter, tasks)
    selected = _select_episodes(episodes, parsed_range, task_texts)
    if not selected:
        raise ValueError("No episodes match split filters")
    new_tasks, old_to_new = _build_tasks(selected, tasks)
    mapping, new_episodes, new_stats, total_frames = _build_episode_records(selected, stats)
    result = PreprocessResult(
        op="split",
        src_roots=[src_root],
        out_root=out_root,
        repo_id=f"local/{out_root.name}",
        total_episodes=len(new_episodes),
        total_frames=total_frames,
        dry_run=dry_run,
        summary={"selected_episodes": len(new_episodes), "tasks": len(new_tasks)},
    )
    emit(
        progress_callback,
        status="running",
        current=0,
        total=len(mapping),
        message=f"Planning split: {result.summary}",
    )
    if dry_run:
        emit(progress_callback, status="done", current=0, total=len(mapping), message="Dry run complete")
        return result

    write_json(
        out_root / "meta" / "info.json",
        update_info_counts(info, len(new_episodes), total_frames, len(new_tasks)),
    )
    write_jsonl(out_root / "meta" / "tasks.jsonl", new_tasks)
    write_jsonl(out_root / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(out_root / "meta" / "episodes_stats.jsonl", new_stats)

    for idx, (old_idx, new_idx) in enumerate(mapping, start=1):
        src = src_root / format_data_path(info, old_idx)
        dst = out_root / format_data_path(info, new_idx)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source parquet: {src}")
        df = pd.read_parquet(src)
        df["episode_index"] = new_idx
        if "task_index" in df.columns:
            mapped = df["task_index"].map(old_to_new)
            if mapped.isna().any():
                raise ValueError(f"Episode {old_idx} has task_index values outside selected tasks")
            df["task_index"] = mapped.astype(df["task_index"].dtype)
        if "index" in df.columns and "frame_index" in df.columns:
            frame_offset = sum(int(ep["length"]) for ep in new_episodes[:new_idx])
            df["index"] = frame_offset + df["frame_index"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        emit(
            progress_callback,
            status="running",
            current=idx,
            total=len(mapping),
            message=f"Split episode {old_idx} -> {new_idx}",
        )

    copy_episode_videos([(src_root, old_idx, new_idx) for old_idx, new_idx in mapping], info, out_root)
    emit(
        progress_callback,
        status="done",
        current=len(mapping),
        total=len(mapping),
        message=f"Split complete: {out_root}",
    )
    return result
