#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local data platform for dataset preprocessing, visualization, and analysis."""

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.utils.utils import init_logging
from lerobot.data_platform.operation_log import audit_cli_main
from lerobot.data_platform.precompute.annotation import (
    DEFAULT_FALLBACK_STAGE_COUNT,
    write_episode_csv,
)
from lerobot.data_platform.precompute.compare import CompareResult, run_compare_build
from lerobot.data_platform.precompute.construction import (
    ConstructionResult,
    default_synthetic_path,
    run_construction,
)
from lerobot.data_platform.precompute.data_profile import resolve_data_profile
from lerobot.data_platform.precompute.dataset_io import V3DatasetMetadata, is_v3_dataset
from lerobot.data_platform.precompute.embedding import EmbeddingResult, run_embedding
from lerobot.data_platform.precompute.labeling import (
    DEFAULT_BACKEND,
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_DASHSCOPE_MODEL,
    DEFAULT_MODEL_ID,
    DEFAULT_QWEN_ENDPOINT,
    DEFAULT_QWEN_MODEL,
)
from lerobot.data_platform.precompute.labeling.review import merge_reviewed_labels_to_metadata
from lerobot.data_platform.precompute.labeling.runner import (
    LabelingResult,
    run_labeling,
    sample_episodes_by_task_type,
)
from lerobot.data_platform.precompute.mutations import (
    fix_episode_indices,
    update_episode_stats_for_subtask_state,
    update_info_features,
    write_subtask_state_to_parquet,
    write_subtask_text_to_parquet,
)
from lerobot.data_platform.precompute.preprocess import (
    DEFAULT_V3_CONVERT_WORKERS,
    IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL,
    IMAGE_VIDEO_MODES,
    default_v3_path,
    repair_v3_video_timestamps,
    run_convert_action,
    run_convert_v3,
    run_drop_field,
    run_merge,
    run_smooth_action,
    run_split,
    run_standardize_dataset,
    run_subtract,
)
from lerobot.data_platform.precompute.tagging import (
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MODEL,
    TaggingResult,
    merge_tags_to_metadata,
    run_tagging,
)
from lerobot.data_platform.precompute.timeseries import (
    DATA_VERSION_DVT1,
    DATA_VERSION_DVT2,
)
from lerobot.data_platform.precompute.v3_viewer import run_v3_viewer_precompute
from lerobot.data_platform.precompute.video import encode_episode_video
from lerobot.data_platform.precompute.viewer_manifest import write_viewer_manifest


@dataclass
class PrecomputeResult:
    root: Path
    repo_id: str
    output_dir: Path
    static_dir: Path
    episodes: list[int]
    image_keys: list[str]
    prepared: bool
    annotation_issues: list[dict]
    subtask_boundaries: dict[str, dict]
    labeling_result: LabelingResult | None = None
    label_merge_result: dict | None = None
    construction_result: ConstructionResult | None = None
    tagging_result: TaggingResult | None = None
    tag_merge_result: dict | None = None
    embedding_result: EmbeddingResult | None = None
    compare_result: CompareResult | None = None


def get_default_output_dir(root: Path) -> Path:
    dataset_name = root.name or "dataset"
    return root.parent / "vis" / f"local_vis_{dataset_name}"


def infer_data_version_from_root(root: Path) -> str:
    return resolve_data_profile(root).legacy_data_version


def load_platform_metadata(root: Path, repo_id: str):
    root = Path(root)
    if is_v3_dataset(root):
        return V3DatasetMetadata(repo_id, root)
    return LeRobotDatasetMetadata(repo_id, root=root)


def _all_precomputed_files_exist(
    static_dir: Path,
    episodes: list[int],
    image_keys: list[str],
    prepare_videos: bool,
    prepare_csv: bool,
    downsample: int | None,
) -> bool:
    if prepare_videos:
        for image_key in image_keys:
            cam_dir = static_dir / "videos" / image_key
            try:
                video_sizes: dict[str, int] = {}
                with os.scandir(cam_dir) as it:
                    for entry in it:
                        if entry.is_file():
                            try:
                                video_sizes[entry.name] = entry.stat().st_size
                            except OSError:
                                video_sizes[entry.name] = 0
            except FileNotFoundError:
                return False
            for episode_id in episodes:
                size = video_sizes.get(f"episode_{episode_id:06d}_h264.mp4")
                if not size or size <= 0:
                    return False

    if prepare_csv:
        downsample_value = downsample if downsample and downsample > 1 else 1
        csv_dir = static_dir / "csv"
        try:
            csv_sizes: dict[str, int] = {}
            with os.scandir(csv_dir) as it:
                for entry in it:
                    if entry.is_file():
                        try:
                            csv_sizes[entry.name] = entry.stat().st_size
                        except OSError:
                            csv_sizes[entry.name] = 0
        except FileNotFoundError:
            return False
        for episode_id in episodes:
            size = csv_sizes.get(f"episode_{episode_id:06d}_ds{downsample_value}.csv")
            if not size or size <= 0:
                return False

    return True


def _issue_episode(issue: dict) -> int | None:
    try:
        return int(issue.get("episode"))
    except (TypeError, ValueError):
        return None


def _is_precompute_issue(issue: dict) -> bool:
    return isinstance(issue, dict) and issue.get("type") in {"error", "multi_gripper"}


def _write_annotation_issues(
    static_dir: Path, all_issues: list[dict], scanned_episodes: list[int] | None = None
) -> None:
    issues_path = static_dir / "annotation_issues.json"
    existing_issues = []
    if issues_path.is_file():
        try:
            loaded = json.loads(issues_path.read_text())
            existing_issues = loaded if isinstance(loaded, list) else []
        except (json.JSONDecodeError, OSError):
            existing_issues = []
    scanned = {int(ep) for ep in scanned_episodes} if scanned_episodes is not None else None
    retained_issues = []
    for issue in existing_issues:
        if not isinstance(issue, dict):
            continue
        episode = _issue_episode(issue)
        if _is_precompute_issue(issue) and (scanned is None or episode in scanned):
            continue
        retained_issues.append(issue)
    merged_issues = retained_issues + sorted(
        all_issues,
        key=lambda issue: (
            int(issue.get("episode", -1)),
            str(issue.get("type", "")),
            str(issue.get("reason", "")),
        ),
    )
    issues_path.write_text(json.dumps(merged_issues, indent=2))
    if all_issues:
        logging.info(
            "Saved %d precompute annotation issues to %s (total retained+new: %d, errors: %d, multi_gripper: %d)",
            len(all_issues),
            issues_path,
            len(merged_issues),
            sum(1 for issue in all_issues if issue["type"] == "error"),
            sum(1 for issue in all_issues if issue["type"] == "multi_gripper"),
        )
    else:
        logging.info(
            "No precompute annotation issues for scanned episodes — retained %d existing issues in %s",
            len(retained_issues),
            issues_path,
        )


def _write_subtask_annotations(
    static_dir: Path,
    all_boundaries: dict[str, dict],
    overwrite_csv: bool,
) -> None:
    ann_path = static_dir / "subtask_annotations.json"
    existing = {}
    if ann_path.is_file():
        try:
            existing = json.loads(ann_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    for episode_key, bounds in all_boundaries.items():
        if overwrite_csv or episode_key not in existing:
            if bounds.get("equal_time"):
                transitions = [
                    {"time": boundary_time, "state": stage_index}
                    for stage_index, boundary_time in enumerate(
                        bounds["stage_boundaries"],
                        start=1,
                    )
                ]
            elif bounds.get("direct_give"):
                transitions = [{"time": bounds["stage0_end"], "state": 3}]
            else:
                transitions = [{"time": bounds["stage0_end"], "state": 1}]
            if not bounds.get("equal_time"):
                if "stage2_start" in bounds and "stage2_end" in bounds:
                    transitions.extend(
                        [
                            {"time": bounds["stage2_start"], "state": 2},
                            {"time": bounds["stage2_end"], "state": 3},
                        ]
                    )
                transitions.append({"time": bounds["stage4_start"], "state": 4})
                if bounds.get("is_give"):
                    transitions.append({"time": bounds["stage4_end"], "state": 5})
            existing[episode_key] = transitions

    ann_path.write_text(json.dumps(existing, indent=2))
    logging.info(
        "Saved auto subtask annotations for %d episodes to %s",
        len(all_boundaries),
        ann_path,
    )


def _emit_progress(progress_callback: Callable[[dict], None] | None, **payload) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)


def run_precompute(
    root: Path,
    repo_id: str | None = None,
    episodes: list[int] | None = None,
    image_keys: list[str] | None = None,
    output_dir: Path | None = None,
    prepare_videos: bool = True,
    prepare_csv: bool = True,
    prepare_workers: int = 8,
    max_frames: int | None = None,
    downsample: int | None = None,
    overwrite: bool = False,
    overwrite_csv: bool = False,
    fix_episode_indices_enabled: bool = False,
    annotate: bool = False,
    write_parquet: bool = False,
    force_recompute_stage: bool = False,
    fallback_stage_count: int = DEFAULT_FALLBACK_STAGE_COUNT,
    write_subtask: bool = False,
    overwrite_parquet: bool = False,
    overwrite_subtask_text: bool = False,
    visualize_only: bool = False,
    data_version: str | None = None,
    label_bbox: bool = False,
    label_backend: str = DEFAULT_BACKEND,
    label_model: str = DEFAULT_MODEL_ID,
    label_endpoint: str = DEFAULT_QWEN_ENDPOINT,
    label_qwen_model: str = DEFAULT_QWEN_MODEL,
    label_qwen_token: str | None = None,
    label_min_pixels: int = 1024,
    label_max_pixels: int = 9800,
    label_workers: int = 8,
    label_devices: str | None = None,
    label_run_mode: str = "missing",
    label_trial: bool = False,
    label_trial_per_type: int = 20,
    label_trial_seed: int | None = None,
    label_vis: bool = False,
    merge_labels: bool = False,
    construct_data: bool = False,
    construct_config: dict | None = None,
    construct_out: Path | None = None,
    auto_tag: bool = False,
    tag_names: list[str] | None = None,
    tag_vlm_backend: str = DEFAULT_VLM_BACKEND,
    tag_vlm_model: str = DEFAULT_VLM_MODEL,
    tag_vlm_endpoint: str | None = None,
    tag_vlm_token: str | None = None,
    tag_workers: int = 8,
    tag_trial: bool = False,
    tag_trial_per_type: int = 20,
    tag_trial_seed: int | None = None,
    merge_tags: bool = False,
    embed_policy: Path | None = None,
    embed_layer: str = "pi_prefix",
    embed_config: str | None = None,
    embed_workers: int | None = None,
    embed_devices: str | None = None,
    embed_refit: bool = False,
    compare_with: Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    show_progress: bool = True,
) -> PrecomputeResult:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Local dataset root does not exist: {root}")

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata at: {info_path}")

    repo_id = repo_id or f"local/{root.name or 'dataset'}"
    source_is_v3 = is_v3_dataset(root)
    meta = load_platform_metadata(root, repo_id)
    data_version = resolve_data_profile(
        root,
        meta.features,
        data_version_override=data_version,
    ).legacy_data_version
    if data_version not in {DATA_VERSION_DVT1, DATA_VERSION_DVT2}:
        raise ValueError(f"Unsupported data_version: {data_version}")

    if image_keys is None:
        image_keys = [
            key
            for key, feature in meta.features.items()
            if feature["dtype"] in ({"image", "video"} if source_is_v3 else {"image"})
        ]

    episodes = sorted(meta.episodes.keys()) if episodes is None else episodes
    output_dir = get_default_output_dir(root) if output_dir is None else Path(output_dir)
    prepare_workers = max(1, int(prepare_workers or 1))
    fallback_stage_count = int(fallback_stage_count)
    if fallback_stage_count < 2:
        raise ValueError("fallback_stage_count must be at least 2")

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    if visualize_only:
        logging.info(
            "--visualize-only batch safe mode: disabling fix-episode-indices, write-parquet, "
            "write-subtask, annotate, and parquet overwrite modes. "
            "Video/CSV caches may still be prepared or overwritten for visualization."
        )
        fix_episode_indices_enabled = False
        write_parquet = False
        write_subtask = False
        annotate = False
        overwrite_parquet = False
        overwrite_subtask_text = False

    csv_dir = static_dir / "csv"
    if prepare_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    overwrite_video = bool(overwrite)
    overwrite_csv = bool(overwrite) or bool(overwrite_csv)
    if overwrite_subtask_text:
        write_subtask = True
        logging.info("--overwrite-subtask-text: regenerating subtask text from existing subtask_state.")
    if overwrite_parquet:
        force_recompute_stage = True
        write_parquet = True
        write_subtask = True
        overwrite_csv = True
        logging.info(
            "--overwrite-parquet: enabling --force-recompute-stage, --write-parquet, "
            "--write-subtask, --overwrite-csv."
        )
    if write_parquet and not annotate:
        logging.warning("--write-parquet requires --annotate 1. Enabling --annotate automatically.")
        annotate = True
    if force_recompute_stage or write_parquet:
        if not prepare_csv:
            logging.info("Stage computation requires CSV preparation; enabling it automatically.")
        prepare_csv = True
        overwrite_csv = True
        csv_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        progress_callback,
        status="running",
        step="check_cache",
        current=0,
        total=len(episodes),
        message=f"Checking existing video/CSV cache for {len(episodes)} episodes",
    )
    needs_prepare = (
        overwrite_video
        or overwrite_csv
        or not _all_precomputed_files_exist(
            static_dir=static_dir,
            episodes=episodes,
            image_keys=image_keys,
            prepare_videos=prepare_videos,
            prepare_csv=prepare_csv,
            downsample=downsample,
        )
    )
    prepared = bool(needs_prepare)
    _emit_progress(
        progress_callback,
        status="running",
        step="init",
        current=0,
        total=len(episodes),
        message=f"Loaded {len(episodes)} episodes from {root}",
    )

    if fix_episode_indices_enabled:
        _emit_progress(
            progress_callback, status="running", step="fix_indices", message="Checking episode indices"
        )
        indices_fixed = fix_episode_indices(root, meta, episodes)
        if indices_fixed:
            meta = load_platform_metadata(root, repo_id)
            overwrite_csv = True
            needs_prepare = True
            logging.info("Parquet indices were fixed — forcing CSV overwrite")

    all_boundaries: dict[str, dict] = {}
    all_issues: list[dict] = []

    if source_is_v3 and needs_prepare and (prepare_videos or prepare_csv):
        # v3.0 data/video files can contain multiple episodes in one shard. Let
        # the v3 viewer executor prepare videos and its manifest, then use the
        # shared episode-aware CSV writer below for both plot data and Stage.
        run_v3_viewer_precompute(
            root=root,
            repo_id=repo_id,
            episodes=episodes,
            output_dir=output_dir,
            prepare_videos=prepare_videos,
            prepare_csv=False,
            workers=prepare_workers,
            downsample=downsample,
            overwrite_videos=overwrite_video,
            overwrite_csv=False,
            data_version=data_version,
            progress_callback=progress_callback,
        )
        needs_prepare = prepare_csv

    if needs_prepare:
        if prepare_videos or prepare_csv:
            progress = (
                tqdm(total=len(episodes), desc="Precomputing", unit="episode", dynamic_ncols=True)
                if show_progress
                else None
            )

            def _prepare_episode(episode_id: int) -> tuple[int, dict | None, list[dict]]:
                if prepare_videos and not source_is_v3:
                    for image_key in image_keys:
                        video_path = encode_episode_video(
                            root,
                            meta,
                            episode_id,
                            image_key,
                            static_dir,
                            max_frames,
                            overwrite_video,
                        )
                        if video_path is None:
                            raise RuntimeError(
                                f"Failed to prepare video cache for episode {episode_id}, "
                                f"image key {image_key!r}"
                            )

                if prepare_csv:
                    downsample_value = downsample if downsample and downsample > 1 else 1
                    csv_path = csv_dir / f"episode_{episode_id:06d}_ds{downsample_value}.csv"
                    _, boundaries, episode_issues = write_episode_csv(
                        root,
                        meta,
                        episode_id,
                        csv_path,
                        max_frames,
                        downsample,
                        overwrite_csv,
                        force_recompute_stage=bool(force_recompute_stage),
                        data_version=data_version,
                        fallback_stage_count=fallback_stage_count,
                    )
                    return episode_id, boundaries, episode_issues

                return episode_id, None, []

            try:
                if prepare_workers == 1 or len(episodes) <= 1:
                    for idx, episode_id in enumerate(episodes, start=1):
                        _emit_progress(
                            progress_callback,
                            status="running",
                            step="prepare",
                            current=idx - 1,
                            total=len(episodes),
                            episode=episode_id,
                            message=f"Preparing episode {episode_id}",
                        )
                        _, boundaries, episode_issues = _prepare_episode(episode_id)
                        if boundaries is not None:
                            all_boundaries[str(episode_id)] = boundaries
                        all_issues.extend(episode_issues)
                        if progress is not None:
                            progress.update(1)
                        _emit_progress(
                            progress_callback,
                            status="running",
                            step="prepare",
                            current=idx,
                            total=len(episodes),
                            episode=episode_id,
                            message=f"Prepared episode {episode_id}",
                        )
                else:
                    _emit_progress(
                        progress_callback,
                        status="running",
                        step="prepare",
                        current=0,
                        total=len(episodes),
                        message=f"Preparing {len(episodes)} episodes with {prepare_workers} workers",
                    )
                    with ThreadPoolExecutor(max_workers=min(prepare_workers, len(episodes))) as executor:
                        futures = {
                            executor.submit(_prepare_episode, episode_id): episode_id
                            for episode_id in episodes
                        }
                        for idx, future in enumerate(as_completed(futures), start=1):
                            episode_id, boundaries, episode_issues = future.result()
                            if boundaries is not None:
                                all_boundaries[str(episode_id)] = boundaries
                            all_issues.extend(episode_issues)
                            if progress is not None:
                                progress.update(1)
                            _emit_progress(
                                progress_callback,
                                status="running",
                                step="prepare",
                                current=idx,
                                total=len(episodes),
                                episode=episode_id,
                                message=f"Prepared episode {episode_id}",
                            )
            finally:
                if progress is not None:
                    progress.close()
        else:
            logging.info("No precompute tasks selected. Skip prepare stage.")
            _emit_progress(
                progress_callback, status="running", step="skip", message="No precompute tasks selected"
            )
    else:
        if visualize_only:
            logging.info("Visualize-only mode: skip prepare stage.")
        else:
            logging.info("Detected all precomputed files in '%s'. Skip prepare stage.", static_dir)
        _emit_progress(
            progress_callback,
            status="running",
            step="cached",
            current=len(episodes),
            total=len(episodes),
            message="Detected cached video/CSV files; skipped prepare stage",
        )

    if needs_prepare and prepare_csv:
        _write_annotation_issues(static_dir, all_issues, scanned_episodes=episodes)

    if all_boundaries and write_parquet:
        _emit_progress(
            progress_callback, status="running", step="write_parquet", message="Writing subtask_state"
        )
        episode_stats = write_subtask_state_to_parquet(root, meta, all_boundaries)
        update_info_features(
            root,
            {
                "subtask_state": {
                    "dtype": "int32",
                    "shape": [1],
                    "names": None,
                }
            },
        )
        update_episode_stats_for_subtask_state(root, episode_stats)

    if all_boundaries:
        _write_subtask_annotations(static_dir, all_boundaries, overwrite_csv)

    if write_subtask:
        _emit_progress(
            progress_callback, status="running", step="write_subtask", message="Writing subtask text"
        )
        written = write_subtask_text_to_parquet(root, meta, episodes)
        logging.info("Wrote subtask text for %d episodes", written)
        if written:
            update_info_features(
                root,
                {
                    "subtask": {
                        "dtype": "string",
                        "shape": [1],
                        "names": None,
                    }
                },
            )

    if not source_is_v3:
        try:
            write_viewer_manifest(
                root=root,
                repo_id=repo_id,
                meta=meta,
                episodes=episodes,
                image_keys=image_keys,
                static_dir=static_dir,
                data_version=data_version,
                downsample=downsample,
            )
        except OSError as exc:
            logging.warning("Could not write viewer manifest to %s: %s", static_dir, exc)

    labeling_result = None
    if label_bbox:
        label_episodes = episodes
        output_variant = None
        if label_trial:
            sample_result = sample_episodes_by_task_type(
                root,
                meta,
                per_type=label_trial_per_type,
                episodes=episodes,
                seed=label_trial_seed,
            )
            label_episodes = sample_result.episodes
            output_variant = f"{label_backend}_trial"
            logging.info(
                "Trial object labeling sampled %s episodes with counts %s from available %s",
                len(label_episodes),
                sample_result.counts,
                sample_result.available_counts,
            )
            logging.info("Trial object labeling seed: %s", sample_result.seed)
            if not label_episodes:
                raise ValueError("No supported episodes found for trial labeling.")
        _emit_progress(
            progress_callback,
            status="running",
            step="labeling",
            current=0,
            total=len(label_episodes),
            message=(
                f"Starting trial object labeling ({len(label_episodes)} episodes)"
                if label_trial
                else "Starting object labeling"
            ),
        )
        effective_label_qwen_model = (
            DEFAULT_DASHSCOPE_MODEL
            if label_backend == "qwen_dashscope" and label_qwen_model == DEFAULT_QWEN_MODEL
            else label_qwen_model
        )
        effective_label_endpoint = (
            DEFAULT_DASHSCOPE_BASE_URL
            if label_backend == "qwen_dashscope" and label_endpoint == DEFAULT_QWEN_ENDPOINT
            else label_endpoint
        )
        labeling_result = run_labeling(
            root=root,
            meta=meta,
            episodes=label_episodes,
            static_dir=static_dir,
            backend=label_backend,
            model_id=label_model,
            endpoint=effective_label_endpoint,
            qwen_model=effective_label_qwen_model,
            qwen_token=label_qwen_token,
            min_pixels=label_min_pixels,
            max_pixels=label_max_pixels,
            workers=label_workers,
            devices=label_devices,
            output_variant=output_variant,
            run_mode=label_run_mode,
            save_vis=label_vis,
            progress_callback=progress_callback,
            show_progress=show_progress,
        )

    label_merge_result = None
    if merge_labels:
        _emit_progress(
            progress_callback,
            status="running",
            step="merge_labels",
            message="Merging reviewed labels",
        )
        label_merge_result = merge_reviewed_labels_to_metadata(root, static_dir / "labeling")
        logging.info(
            "Merged reviewed object labels for %d episodes into %s",
            label_merge_result["merged"],
            label_merge_result["episodes_path"],
        )

    construction_result = None
    if construct_data:
        _emit_progress(
            progress_callback,
            status="running",
            step="construction",
            current=0,
            total=1,
            message="Starting data construction",
        )
        construction_result = run_construction(
            src_root=root,
            meta=meta,
            labeling_dir=static_dir / "labeling",
            out_root=construct_out or default_synthetic_path(root),
            config=construct_config or {},
            progress_callback=progress_callback,
        )
        logging.info("Constructed dataset at %s", construction_result.out_root)

    tagging_result = None
    tag_merge_result = None
    if auto_tag:
        tag_episodes = episodes
        tag_output_variant = None
        if tag_trial:
            sample_result = sample_episodes_by_task_type(
                root,
                meta,
                per_type=tag_trial_per_type,
                episodes=episodes,
                seed=tag_trial_seed,
            )
            tag_episodes = sample_result.episodes
            tag_output_variant = "trial"
            logging.info(
                "Trial auto-tagging sampled %s episodes with counts %s from available %s",
                len(tag_episodes),
                sample_result.counts,
                sample_result.available_counts,
            )
            logging.info("Trial auto-tagging seed: %s", sample_result.seed)
            if not tag_episodes:
                raise ValueError("No supported episodes found for trial tagging.")
        _emit_progress(
            progress_callback,
            status="running",
            step="tagging",
            current=0,
            total=len(tag_episodes),
            message=(
                f"Starting trial auto-tagging ({len(tag_episodes)} episodes)"
                if tag_trial
                else "Starting auto-tagging"
            ),
        )
        tagging_result = run_tagging(
            root=root,
            meta=meta,
            episodes=tag_episodes,
            static_dir=static_dir,
            selected_tags=tag_names,
            vlm_backend=tag_vlm_backend,
            vlm_model=tag_vlm_model,
            vlm_endpoint=tag_vlm_endpoint,
            vlm_token=tag_vlm_token,
            output_variant=tag_output_variant,
            workers=tag_workers,
            progress_callback=progress_callback,
        )
        logging.info("Wrote auto-tags to %s", tagging_result.tags_path)

    if merge_tags:
        tag_merge_result = merge_tags_to_metadata(root, static_dir / "tagging")
        logging.info(
            "Merged tags for %d episodes into %s",
            tag_merge_result["merged"],
            tag_merge_result["episodes_path"],
        )

    embedding_result = None
    if embed_policy is not None:
        embedding_result = run_embedding(
            root=root,
            meta=meta,
            episodes=episodes,
            static_dir=static_dir,
            ckpt_path=embed_policy,
            layer_hook=embed_layer,
            openpi_config=embed_config,
            workers=embed_workers,
            devices=embed_devices,
            refit=embed_refit,
            progress_callback=progress_callback,
        )
        logging.info(
            "Wrote embeddings for %d episodes to %s", embedding_result.points, embedding_result.embedding_dir
        )

    compare_result = None
    if compare_with is not None:
        compare_root = Path(compare_with).expanduser()
        compare_meta = load_platform_metadata(
            compare_root,
            f"local/{compare_root.name or 'dataset'}",
        )
        compare_static = get_default_output_dir(compare_root) / "static"
        compare_static.mkdir(parents=True, exist_ok=True)
        compare_result = run_compare_build(
            root_a=root,
            meta_a=meta,
            static_a=static_dir,
            repo_id_a=repo_id,
            root_b=compare_root,
            meta_b=compare_meta,
            static_b=compare_static,
            repo_id_b=compare_meta.repo_id,
            progress_callback=progress_callback,
        )
        logging.info("Built compare cache at %s", compare_result.out_dir)

    _emit_progress(
        progress_callback,
        status="done",
        step="done",
        current=len(episodes),
        total=len(episodes),
        message="Precompute complete",
    )
    return PrecomputeResult(
        root=root,
        repo_id=repo_id,
        output_dir=output_dir,
        static_dir=static_dir,
        episodes=episodes,
        image_keys=image_keys,
        prepared=prepared,
        annotation_issues=all_issues,
        subtask_boundaries=all_boundaries,
        labeling_result=labeling_result,
        label_merge_result=label_merge_result,
        construction_result=construction_result,
        tagging_result=tagging_result,
        tag_merge_result=tag_merge_result,
        embedding_result=embedding_result,
        compare_result=compare_result,
    )


@audit_cli_main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory containing one or more local datasets. The home console lists datasets under it.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repo id for naming outputs when using --output-dir (e.g. lerobot/pusht).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to prepare. Default: all episodes.",
    )
    parser.add_argument(
        "--image-keys",
        type=str,
        nargs="*",
        default=None,
        help="Image keys to encode (default: all image features).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output dir used by visualize_dataset_html (will write to <output-dir>/static/videos). "
            "Default: <root parent>/vis/local_vis_<root name>."
        ),
    )
    parser.add_argument(
        "--prepare-videos",
        type=int,
        default=1,
        help="Prepare mp4 videos from image streams.",
    )
    parser.add_argument(
        "--prepare-csv",
        type=int,
        default=1,
        help="Prepare downsampled CSV for time-series plots.",
    )
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=8,
        help="Parallel episode workers for video/CSV cache preparation. Default: 8.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames per episode (for quick tests).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Downsample time series by keeping one every N frames (e.g. 5). Default: no downsampling.",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        choices=[DATA_VERSION_DVT1, DATA_VERSION_DVT2],
        default=None,
        help=(
            "Override the robot/Stage profile. By default the portable dataset profile is used; "
            "dimensions are only a legacy fallback."
        ),
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="Overwrite all existing precomputed files (videos + csv).",
    )
    parser.add_argument(
        "--overwrite-csv",
        type=int,
        default=0,
        help="Overwrite only existing precomputed CSV files (not videos).",
    )
    parser.add_argument(
        "--fix-episode-indices",
        type=int,
        default=0,
        help=(
            "Check and fix per-episode frame_index/timestamp so they start from 0, "
            "and make global index contiguous across episodes."
        ),
    )
    parser.add_argument(
        "--annotate",
        type=int,
        default=0,
        help="Enable interactive subtask annotation editing (press A to advance stage in frontend).",
    )
    parser.add_argument(
        "--write-parquet",
        type=int,
        default=0,
        help="Write computed subtask_state back into original parquet files (requires --annotate 1).",
    )
    parser.add_argument(
        "--force-recompute-stage",
        type=int,
        default=0,
        help="Force recompute subtask stage even if subtask_state exists in parquet (default: read from parquet).",
    )
    parser.add_argument(
        "--fallback-stage-count",
        type=int,
        default=DEFAULT_FALLBACK_STAGE_COUNT,
        help="Number of equal-duration stages for tasks without pick/place/give rules. Default: 5.",
    )
    parser.add_argument(
        "--write-subtask",
        type=int,
        default=0,
        help="Write subtask text column into parquet files based on task + subtask_state.",
    )
    parser.add_argument(
        "--overwrite-parquet",
        type=int,
        default=0,
        help=(
            "Force overwrite existing subtask_state and subtask columns in parquet "
            "(implies --force-recompute-stage 1, --write-parquet 1, --write-subtask 1)."
        ),
    )
    parser.add_argument(
        "--overwrite-subtask-text",
        type=int,
        default=0,
        help="Regenerate subtask text from existing subtask_state in parquet (does not recompute stage).",
    )
    parser.add_argument(
        "--preprocess-convert-action",
        type=int,
        default=0,
        help="Create a sibling dataset with action/state vectors trimmed to --preprocess-target-dim.",
    )
    parser.add_argument(
        "--preprocess-convert-v3",
        type=int,
        default=0,
        help=(
            "Convert a v2.1 dataset to the LeRobot 0.4.4 v3.0 layout. "
            "Parquet image cameras use the selected official or RGB-lossless MP4 encoding. "
            "An existing v3.0 dataset is reported unchanged."
        ),
    )
    parser.add_argument(
        "--preprocess-v3-out",
        type=Path,
        default=None,
        help="Output root for v2.1 conversion. Defaults to a new sibling directory.",
    )
    parser.add_argument(
        "--preprocess-v3-data-file-size-mb",
        type=int,
        default=100,
        help="Maximum target parquet file size for --preprocess-convert-v3.",
    )
    parser.add_argument(
        "--preprocess-v3-video-file-size-mb",
        type=int,
        default=200,
        help="Maximum target video file size for --preprocess-convert-v3.",
    )
    parser.add_argument(
        "--preprocess-v3-workers",
        type=int,
        default=DEFAULT_V3_CONVERT_WORKERS,
        help="Parallel episode workers for Parquet image-to-MP4 encoding. Default: 8.",
    )
    parser.add_argument(
        "--preprocess-v3-image-video-mode",
        choices=sorted(IMAGE_VIDEO_MODES),
        default=IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL,
        help=(
            "Encoding for Parquet image cameras: lerobot_official uses LeRobot 0.4.4 "
            "AV1/yuv420p/CRF 30 defaults; rgb_lossless uses H.264 RGB/CRF 0. "
            "Both preserve the source resolution. Default: lerobot_official."
        ),
    )
    parser.add_argument(
        "--preprocess-v3-overwrite",
        type=int,
        default=0,
        help=(
            "Overwrite an existing v3 output directory. Interactive CLI runs ask for "
            "confirmation when this is 0."
        ),
    )
    parser.add_argument(
        "--preprocess-repair-v3-video-timestamps",
        type=int,
        default=0,
        help=(
            "Stream-copy remux v3 MP4 shards onto the dataset FPS grid and rewrite "
            "episode video from/to timestamps. Encoded frames and norm stats are unchanged."
        ),
    )
    parser.add_argument(
        "--preprocess-target-dim",
        type=int,
        default=16,
        help="Target trailing vector dimension for --preprocess-convert-action.",
    )
    parser.add_argument(
        "--preprocess-convert-out",
        type=Path,
        default=None,
        help="Output root for --preprocess-convert-action.",
    )
    parser.add_argument(
        "--preprocess-drop-field",
        type=str,
        default=None,
        help="Create a sibling dataset with this parquet/meta field removed.",
    )
    parser.add_argument(
        "--preprocess-drop-out",
        type=Path,
        default=None,
        help="Output root for --preprocess-drop-field.",
    )
    parser.add_argument(
        "--preprocess-smooth-action",
        type=int,
        default=0,
        help="Create a sibling dataset with action vectors smoothed over time.",
    )
    parser.add_argument(
        "--preprocess-smooth-window",
        type=int,
        default=5,
        help="Odd centered moving-average window for --preprocess-smooth-action.",
    )
    parser.add_argument(
        "--preprocess-smooth-workers",
        type=int,
        default=0,
        help="Parallel workers for --preprocess-smooth-action. 0 auto-selects up to 8.",
    )
    parser.add_argument(
        "--preprocess-smooth-state",
        type=int,
        default=1,
        help="Also smooth state when a state feature exists.",
    )
    parser.add_argument(
        "--preprocess-smooth-out",
        type=Path,
        default=None,
        help="Output root for --preprocess-smooth-action.",
    )
    parser.add_argument(
        "--preprocess-standardize",
        type=int,
        default=0,
        help=(
            "Create a sibling standardized dataset: cache if needed, normalize DVT2 grippers, "
            "trim action/state to 16D, drop depth fields, write stage/subtask, and repair indices."
        ),
    )
    parser.add_argument(
        "--preprocess-standardize-out",
        type=Path,
        default=None,
        help="Output root for --preprocess-standardize. Default: <src>_preprocessed.",
    )
    parser.add_argument(
        "--preprocess-standardize-overwrite",
        type=int,
        default=0,
        help="Overwrite --preprocess-standardize output root if it already exists.",
    )
    parser.add_argument(
        "--preprocess-split-episodes",
        type=str,
        default=None,
        help="Create a sibling split dataset with episode range START:END.",
    )
    parser.add_argument(
        "--preprocess-split-tasks",
        type=str,
        default=None,
        help="Comma-separated task texts or task indices for split filtering.",
    )
    parser.add_argument(
        "--preprocess-split-out",
        type=Path,
        default=None,
        help="Output root for --preprocess-split-*.",
    )
    parser.add_argument(
        "--preprocess-merge-with",
        type=str,
        default=None,
        help="Comma-separated additional dataset roots to merge with --root.",
    )
    parser.add_argument(
        "--preprocess-merge-out",
        type=Path,
        default=None,
        help="Output root for --preprocess-merge-with.",
    )
    parser.add_argument(
        "--preprocess-subtract-with",
        type=str,
        default=None,
        help="Comma-separated dataset roots to subtract from --root by exact episode fingerprint.",
    )
    parser.add_argument(
        "--preprocess-subtract-out",
        type=Path,
        default=None,
        help="Output root for --preprocess-subtract-with.",
    )
    parser.add_argument(
        "--preprocess-dry-run",
        type=int,
        default=0,
        help="Plan preprocess operation(s) without writing output datasets.",
    )
    parser.add_argument(
        "--label-bbox",
        type=int,
        default=0,
        help="Run object bbox labeling and write <output-dir>/static/labeling/labels.jsonl.",
    )
    parser.add_argument(
        "--label-backend",
        choices=["grounding_dino", "qwen_remote", "qwen_dashscope"],
        default=DEFAULT_BACKEND,
        help="Object labeling backend.",
    )
    parser.add_argument(
        "--label-model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace object detection model id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--label-endpoint",
        type=str,
        default=DEFAULT_QWEN_ENDPOINT,
        help=f"Remote Qwen gradio endpoint (default: {DEFAULT_QWEN_ENDPOINT}).",
    )
    parser.add_argument(
        "--label-qwen-model",
        type=str,
        default=DEFAULT_QWEN_MODEL,
        help=f"Remote Qwen model name (default: {DEFAULT_QWEN_MODEL}).",
    )
    parser.add_argument(
        "--label-qwen-token",
        type=str,
        default=None,
        help="ModelScope SDK token for Remote Qwen. If omitted, reads MODELSCOPE_SDK_TOKEN/QWEN_REMOTE_TOKEN.",
    )
    parser.add_argument(
        "--label-min-pixels",
        type=int,
        default=1024,
        help="Remote Qwen min_pixels value.",
    )
    parser.add_argument(
        "--label-max-pixels",
        type=int,
        default=9800,
        help="Remote Qwen max_pixels value.",
    )
    parser.add_argument(
        "--label-workers",
        type=int,
        default=8,
        help="Parallel object labeling workers. GroundingDINO workers load one model each.",
    )
    parser.add_argument(
        "--label-devices",
        type=str,
        default="",
        help="Comma/space separated GroundingDINO devices, e.g. cuda:0,cuda:1 or 0,1,2,3. Empty auto-selects CUDA devices.",
    )
    parser.add_argument(
        "--label-run-mode",
        choices=["missing", "full"],
        default="missing",
        help="Object labeling run mode: missing keeps existing labels/reviews and runs only missing episodes; full replaces selected labels.",
    )
    parser.add_argument(
        "--label-trial",
        type=int,
        default=0,
        help="Run trial object labeling by sampling episodes per task type.",
    )
    parser.add_argument(
        "--label-trial-per-type",
        type=int,
        default=20,
        help="Episodes to sample per task type when --label-trial 1.",
    )
    parser.add_argument(
        "--label-trial-seed",
        type=int,
        default=None,
        help="Random seed for trial object labeling. Default: current time.",
    )
    parser.add_argument(
        "--label-vis",
        type=int,
        default=0,
        help="Save rendered bbox images under <output-dir>/static/labeling/vis.",
    )
    parser.add_argument(
        "--merge-labels",
        type=int,
        default=0,
        help="Merge existing labels_reviewed.jsonl into v2.1 or v3 episode metadata.",
    )
    parser.add_argument(
        "--construct-data",
        type=int,
        default=0,
        help="Run Data Construction from existing <output-dir>/static/labeling/labels.jsonl.",
    )
    parser.add_argument(
        "--construct-config",
        type=Path,
        default=None,
        help=(
            "JSON config for data construction: "
            "{uncertainty_threshold, per_scenario_counts, oversample_factor}."
        ),
    )
    parser.add_argument(
        "--construct-out",
        type=Path,
        default=None,
        help=(
            "Output dataset root for data construction. "
            "The legacy default path is <src>_synthetic_<timestamp>."
        ),
    )
    parser.add_argument(
        "--construct-uncertainty-threshold",
        type=int,
        default=50,
        help="Fallback object labeling uncertainty threshold when --construct-config is not specified.",
    )
    parser.add_argument(
        "--construct-positives",
        type=int,
        default=0,
        help="Whether to copy source episodes as positive exist_label=1 examples. Default: 0.",
    )
    parser.add_argument(
        "--construct-oversample-factor",
        type=float,
        default=1.0,
        help="Candidate buffer multiplier for Data Construction. 2.0 generates twice the requested negatives for review.",
    )
    parser.add_argument(
        "--auto-tag",
        type=int,
        default=0,
        help="Run Auto-tagging and write <output-dir>/static/tagging/tags.jsonl.",
    )
    parser.add_argument(
        "--tag-names",
        nargs="*",
        default=None,
        help="Tag names to run. Default: all schema tags.",
    )
    parser.add_argument(
        "--tag-vlm-backend",
        choices=["qwen_dashscope", "qwen_remote"],
        default=DEFAULT_VLM_BACKEND,
        help=f"VLM backend for VLM-backed tags (default: {DEFAULT_VLM_BACKEND}).",
    )
    parser.add_argument(
        "--tag-vlm-model",
        type=str,
        default=DEFAULT_VLM_MODEL,
        help=f"VLM model id for VLM-backed tags (default: {DEFAULT_VLM_MODEL}).",
    )
    parser.add_argument(
        "--tag-vlm-endpoint",
        type=str,
        default=None,
        help="VLM endpoint/base_url for Auto-tagging. DashScope default uses compatible-mode/v1.",
    )
    parser.add_argument(
        "--tag-vlm-token",
        type=str,
        default=None,
        help="VLM token/API key for Auto-tagging. If omitted, reads backend-specific environment variables.",
    )
    parser.add_argument(
        "--tag-workers",
        type=int,
        default=8,
        help="Number of parallel workers for Auto-tagging. Remote Qwen VLM tags benefit from >1.",
    )
    parser.add_argument(
        "--tag-trial",
        type=int,
        default=0,
        help="Run trial Auto-tagging: randomly sample --tag-trial-per-type episodes per task type.",
    )
    parser.add_argument(
        "--tag-trial-per-type",
        type=int,
        default=20,
        help="Episodes per task type for --tag-trial.",
    )
    parser.add_argument(
        "--tag-trial-seed",
        type=int,
        default=None,
        help="Random seed for --tag-trial. Default: current time.",
    )
    parser.add_argument(
        "--merge-tags",
        type=int,
        default=0,
        help="Merge current tagging results into v2.1 or v3 episode metadata.",
    )
    parser.add_argument(
        "--embed-policy",
        type=Path,
        default=None,
        help="Local PI/policy checkpoint path. Non-empty triggers per-episode embedding extraction.",
    )
    parser.add_argument(
        "--embed-layer",
        type=str,
        default="pi_prefix",
        help="Embedding layer hook: pi_prefix, vision_encoder, pi_prefix_prompt, or episode_stats_fallback.",
    )
    parser.add_argument(
        "--embed-config",
        type=str,
        default=None,
        help="OpenPI training config name. Default: read train_config_full.json:name from --embed-policy.",
    )
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=None,
        help="Parallel OpenPI embedding workers. Default: auto-detect CUDA devices, up to 8.",
    )
    parser.add_argument(
        "--embed-devices",
        type=str,
        default=None,
        help="Comma/space separated CUDA device ids for embedding workers, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--embed-refit",
        type=int,
        default=0,
        help="Force refit reducer and allow old 2D coordinates to move.",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Another local dataset root; non-empty triggers pairwise compare cache build.",
    )
    parser.add_argument(
        "--visualize-only",
        type=int,
        default=0,
        help=(
            "Batch safe mode for --run-visualize 0: disable repair/writeback/parquet overwrite modes. "
            "Video/CSV cache overwrite flags still apply. "
            "This is not the home-console mode; use --mode visualize for that."
        ),
    )
    parser.add_argument(
        "--run-visualize",
        type=int,
        default=1,
        help="Launch the web home console. Set 0 to run batch precompute on --root as a single dataset root.",
    )
    parser.add_argument(
        "--mode",
        dest="console_mode",
        choices=["full", "visualize"],
        default="full",
        help=(
            "Home console feature set. 'full' exposes Data Platform and Data Curation; "
            "'visualize' keeps the reduced review surface. Unlock legacy in-place mutations "
            "from the page with the local administrator password."
        ),
    )
    parser.add_argument(
        "--console-mode",
        dest="console_mode",
        choices=["full", "visualize"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enable-legacy-mutations",
        type=int,
        choices=[0, 1],
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--protected-source-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Protect every dataset below this path from in-place mutation. "
            "Repeat the option to configure multiple source roots."
        ),
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server when visualize_dataset_html runs.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by visualize_dataset_html.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9091,
        help="Web port used by visualize_dataset_html.",
    )

    args = parser.parse_args()
    init_logging()

    if args.run_visualize:
        from lerobot.data_platform.viewer import visualize_dataset_html

        datasets_root = args.root.expanduser() if args.root is not None else None
        if datasets_root is not None and not datasets_root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {datasets_root}")
        console_output_dir = args.output_dir
        if console_output_dir is None and datasets_root is not None:
            console_output_dir = datasets_root / "vis" / "_console"
        logging.info(
            "Launching preprocessing home console%s (mode=%s)",
            f" for {datasets_root}" if datasets_root else "",
            args.console_mode,
        )
        visualize_dataset_html(
            dataset=None,
            episodes=None,
            output_dir=console_output_dir,
            serve=bool(args.serve),
            host=args.host,
            port=args.port,
            max_frames=None,
            prepare_videos=False,
            downsample=args.downsample,
            data_version=args.data_version,
            precompute_csv=False,
            precomputed_only=True,
            annotate=False,
            datasets_root=datasets_root,
            console_mode=args.console_mode,
            legacy_mutations_enabled=bool(args.enable_legacy_mutations),
            protected_source_roots=args.protected_source_root,
        )
        return

    if args.root is None:
        raise ValueError("--root is required when --run-visualize 0.")

    preprocess_ran = False
    dry_run = bool(args.preprocess_dry_run)
    if args.preprocess_convert_v3:
        overwrite_v3 = bool(args.preprocess_v3_overwrite)
        target_v3_root = args.preprocess_v3_out or default_v3_path(args.root)
        if not is_v3_dataset(args.root) and target_v3_root.exists() and not dry_run and not overwrite_v3:
            if not sys.stdin.isatty():
                raise FileExistsError(
                    f"Output dataset already exists: {target_v3_root}. "
                    "Pass --preprocess-v3-overwrite 1 to confirm replacement."
                )
            answer = input(f"Output dataset already exists: {target_v3_root}. Overwrite? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                raise RuntimeError("v3 conversion cancelled")
            overwrite_v3 = True
        result = run_convert_v3(
            args.root,
            out_root=args.preprocess_v3_out,
            data_file_size_in_mb=args.preprocess_v3_data_file_size_mb,
            video_file_size_in_mb=args.preprocess_v3_video_file_size_mb,
            workers=args.preprocess_v3_workers,
            image_video_mode=args.preprocess_v3_image_video_mode,
            overwrite=overwrite_v3,
            dry_run=dry_run,
        )
        logging.info("Preprocess convert_v3 result: %s", result)
        preprocess_ran = True
    if args.preprocess_repair_v3_video_timestamps:
        result = repair_v3_video_timestamps(
            args.root,
            dry_run=dry_run,
        )
        logging.info("Preprocess repair_v3_video_timestamps result: %s", result)
        preprocess_ran = True
    if args.preprocess_convert_action:
        result = run_convert_action(
            args.root,
            out_root=args.preprocess_convert_out,
            target_dim=args.preprocess_target_dim,
            dry_run=dry_run,
        )
        logging.info("Preprocess convert_action result: %s", result)
        preprocess_ran = True
    if args.preprocess_drop_field:
        result = run_drop_field(
            args.root,
            out_root=args.preprocess_drop_out,
            field_name=args.preprocess_drop_field,
            dry_run=dry_run,
        )
        logging.info("Preprocess drop_field result: %s", result)
        preprocess_ran = True
    if args.preprocess_smooth_action:
        result = run_smooth_action(
            args.root,
            out_root=args.preprocess_smooth_out,
            window=args.preprocess_smooth_window,
            workers=args.preprocess_smooth_workers,
            smooth_state=bool(args.preprocess_smooth_state),
            dry_run=dry_run,
        )
        logging.info("Preprocess smooth_action result: %s", result)
        preprocess_ran = True
    if args.preprocess_standardize:
        if dry_run:
            standardize_data_version = args.data_version or DATA_VERSION_DVT2
            result = run_standardize_dataset(
                args.root,
                out_root=args.preprocess_standardize_out,
                data_version=standardize_data_version,
                overwrite=bool(args.preprocess_standardize_overwrite),
                dry_run=True,
            )
            logging.info("Preprocess standardize dry-run result: %s", result)
            preprocess_ran = True
        else:
            source_data_version = args.data_version or DATA_VERSION_DVT2
            run_precompute(
                root=args.root,
                output_dir=args.output_dir,
                prepare_videos=True,
                prepare_csv=True,
                prepare_workers=args.prepare_workers,
                downsample=args.downsample,
                data_version=source_data_version,
                show_progress=True,
            )
            result = run_standardize_dataset(
                args.root,
                out_root=args.preprocess_standardize_out,
                data_version=source_data_version,
                overwrite=bool(args.preprocess_standardize_overwrite),
                dry_run=False,
            )
            run_precompute(
                root=result.out_root,
                repo_id=result.repo_id,
                prepare_videos=True,
                prepare_csv=True,
                prepare_workers=args.prepare_workers,
                downsample=args.downsample,
                fix_episode_indices_enabled=True,
                annotate=True,
                write_parquet=True,
                force_recompute_stage=True,
                write_subtask=True,
                overwrite_csv=True,
                data_version=source_data_version,
                show_progress=True,
            )
            logging.info("Preprocess standardize result: %s", result)
            preprocess_ran = True
    if args.preprocess_split_episodes or args.preprocess_split_tasks:
        result = run_split(
            args.root,
            out_root=args.preprocess_split_out,
            episode_range=args.preprocess_split_episodes,
            task_filter=args.preprocess_split_tasks,
            dry_run=dry_run,
        )
        logging.info("Preprocess split result: %s", result)
        preprocess_ran = True
    if args.preprocess_merge_with:
        merge_roots = [args.root] + [
            Path(item).expanduser()
            for item in str(args.preprocess_merge_with).replace("\n", ",").split(",")
            if item.strip()
        ]
        result = run_merge(
            merge_roots,
            out_root=args.preprocess_merge_out,
            dry_run=dry_run,
            src_static_dirs=[get_default_output_dir(root) / "static" for root in merge_roots],
            out_static_dir=(get_default_output_dir(args.preprocess_merge_out) / "static")
            if args.preprocess_merge_out is not None
            else None,
        )
        logging.info("Preprocess merge result: %s", result)
        preprocess_ran = True
    if args.preprocess_subtract_with:
        subtract_roots = [
            Path(item).expanduser()
            for item in str(args.preprocess_subtract_with).replace("\n", ",").split(",")
            if item.strip()
        ]
        result = run_subtract(
            args.root,
            subtract_roots,
            out_root=args.preprocess_subtract_out,
            dry_run=dry_run,
            src_static_dir=get_default_output_dir(args.root) / "static",
            out_static_dir=(get_default_output_dir(args.preprocess_subtract_out) / "static")
            if args.preprocess_subtract_out is not None
            else None,
        )
        logging.info("Preprocess subtract result: %s", result)
        preprocess_ran = True

    if preprocess_ran:
        return

    construct_config = None
    if args.construct_config is not None:
        construct_config = json.loads(args.construct_config.expanduser().read_text())
    elif args.construct_data:
        construct_config = {
            "uncertainty_threshold": args.construct_uncertainty_threshold,
            "per_scenario_counts": {},
            "include_positives": bool(args.construct_positives),
            "oversample_factor": args.construct_oversample_factor,
        }

    run_precompute(
        root=args.root,
        repo_id=args.repo_id,
        episodes=args.episodes,
        image_keys=args.image_keys,
        output_dir=args.output_dir,
        prepare_videos=bool(args.prepare_videos),
        prepare_csv=bool(args.prepare_csv),
        prepare_workers=args.prepare_workers,
        max_frames=args.max_frames,
        downsample=args.downsample,
        data_version=args.data_version,
        overwrite=bool(args.overwrite),
        overwrite_csv=bool(args.overwrite_csv),
        fix_episode_indices_enabled=bool(args.fix_episode_indices),
        annotate=bool(args.annotate),
        write_parquet=bool(args.write_parquet),
        force_recompute_stage=bool(args.force_recompute_stage),
        fallback_stage_count=args.fallback_stage_count,
        write_subtask=bool(args.write_subtask),
        overwrite_parquet=bool(args.overwrite_parquet),
        overwrite_subtask_text=bool(args.overwrite_subtask_text),
        visualize_only=bool(args.visualize_only),
        label_bbox=bool(args.label_bbox),
        label_backend=args.label_backend,
        label_model=args.label_model,
        label_endpoint=args.label_endpoint,
        label_qwen_model=args.label_qwen_model,
        label_qwen_token=args.label_qwen_token,
        label_min_pixels=args.label_min_pixels,
        label_max_pixels=args.label_max_pixels,
        label_workers=args.label_workers,
        label_devices=args.label_devices,
        label_run_mode=args.label_run_mode,
        label_trial=bool(args.label_trial),
        label_trial_per_type=args.label_trial_per_type,
        label_trial_seed=args.label_trial_seed,
        label_vis=bool(args.label_vis),
        merge_labels=bool(args.merge_labels),
        construct_data=bool(args.construct_data),
        construct_config=construct_config,
        construct_out=args.construct_out,
        auto_tag=bool(args.auto_tag),
        tag_names=args.tag_names,
        tag_vlm_backend=args.tag_vlm_backend,
        tag_vlm_model=args.tag_vlm_model,
        tag_vlm_endpoint=args.tag_vlm_endpoint,
        tag_vlm_token=args.tag_vlm_token,
        tag_workers=args.tag_workers,
        tag_trial=bool(args.tag_trial),
        tag_trial_per_type=args.tag_trial_per_type,
        tag_trial_seed=args.tag_trial_seed,
        merge_tags=bool(args.merge_tags),
        embed_policy=args.embed_policy,
        embed_layer=args.embed_layer,
        embed_config=args.embed_config,
        embed_workers=args.embed_workers,
        embed_devices=args.embed_devices,
        embed_refit=bool(args.embed_refit),
        compare_with=args.compare_with,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
