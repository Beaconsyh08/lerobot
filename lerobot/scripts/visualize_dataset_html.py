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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize a local dataset by its root path:
```bash
local$ python lerobot/scripts/visualize_dataset_html.py \
    --root /home/yuhao.song/nfs-share/yuhao.song/datasets/MERGED_04_17_clean

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import csv
import gc
import json
import logging
import re
import shutil
import tempfile
import traceback
from io import StringIO
from pathlib import Path
from functools import lru_cache
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import shutil
import subprocess
from werkzeug.serving import WSGIRequestHandler

from lerobot import available_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import IterableNamespace
from lerobot.common.utils.utils import init_logging


class MetaOnlyDataset:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.meta = LeRobotDatasetMetadata(
            repo_id=repo_id, root=root, revision=revision, force_cache_sync=force_cache_sync
        )
        self.root = self.meta.root
        self.features = self.meta.features
        self.fps = self.meta.fps
        self.codebase_version = self.meta.info.get("codebase_version", "unknown")
        self.total_frames = self.meta.total_frames
        self.total_episodes = self.meta.total_episodes


@lru_cache(maxsize=8)
def _get_parquet_file(path: str) -> pq.ParquetFile:
    return pq.ParquetFile(path)


@lru_cache(maxsize=8)
def _get_row_group_offsets(path: str) -> list[int]:
    pf = _get_parquet_file(path)
    offsets: list[int] = []
    total = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        total += pf.metadata.row_group(rg_idx).num_rows
        offsets.append(total)
    return offsets


def _find_row_group(offsets: list[int], row_index: int) -> tuple[int, int]:
    for rg_idx, end in enumerate(offsets):
        if row_index < end:
            start = 0 if rg_idx == 0 else offsets[rg_idx - 1]
            return rg_idx, row_index - start
    raise IndexError(f"Row index out of range: {row_index}")


def _read_image_bytes(
    parquet_path: Path, dataset_root: Path, image_key: str, frame_index: int
) -> bytes | None:
    pf = _get_parquet_file(str(parquet_path))
    offsets = _get_row_group_offsets(str(parquet_path))
    rg_idx, local_idx = _find_row_group(offsets, frame_index)
    table = pf.read_row_group(rg_idx, columns=[image_key])
    if len(table) == 0:
        return None
    value = table[image_key][local_idx].as_py()
    if isinstance(value, dict):
        if value.get("bytes"):
            return value["bytes"]
        # Fall back to path if bytes are not stored
        if value.get("path"):
            candidates = [
                dataset_root / value["path"],
                dataset_root / "images" / value["path"],
                dataset_root / "images" / image_key / value["path"],
            ]
            for img_path in candidates:
                if img_path.is_file():
                    return img_path.read_bytes()
    return None


@lru_cache(maxsize=256)
def _cached_image_bytes(
    parquet_path: str, dataset_root: str, image_key: str, frame_index: int
) -> bytes | None:
    return _read_image_bytes(Path(parquet_path), Path(dataset_root), image_key, frame_index)


def _iter_image_bytes(
    parquet_path: Path,
    dataset_root: Path,
    image_key: str,
    max_frames: int | None = None,
):
    pf = _get_parquet_file(str(parquet_path))
    total = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[image_key])
        col = table[image_key]
        for i in range(len(col)):
            if max_frames is not None and total >= max_frames:
                return
            value = col[i].as_py()
            img_bytes = None
            if isinstance(value, dict):
                img_bytes = value.get("bytes")
                if img_bytes is None and value.get("path"):
                    candidates = [
                        dataset_root / value["path"],
                        dataset_root / "images" / value["path"],
                        dataset_root / "images" / image_key / value["path"],
                    ]
                    for img_path in candidates:
                        if img_path.is_file():
                            img_bytes = img_path.read_bytes()
                            break
            if img_bytes is not None:
                yield img_bytes
                total += 1


def _prepare_episode_videos(
    dataset: LeRobotDataset,
    episode_id: int,
    image_keys: list[str],
    static_dir: Path,
    max_frames: int | None = None,
) -> list[dict]:
    videos_info: list[dict] = []
    if not image_keys:
        return videos_info
    parquet_path = dataset.root / dataset.meta.get_data_file_path(episode_id)
    if not parquet_path.is_file():
        return videos_info

    ffmpeg_path = shutil.which("ffmpeg")

    def _encode_with_ffmpeg(
        out_path: Path,
        frame_iter,
        width: int,
        height: int,
        fps: int,
    ) -> bool:
        if not ffmpeg_path:
            return False
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        for frame in frame_iter:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.stdin = None
        _, err = proc.communicate()
        if proc.returncode != 0:
            logging.warning("ffmpeg failed to encode video: %s", err.decode(errors="ignore"))
            return False
        return True

    for image_key in image_keys:
        rel_path = Path("videos") / image_key / f"episode_{episode_id:06d}_h264.mp4"
        out_path = static_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.is_file():
            frame_iter = _iter_image_bytes(parquet_path, dataset.root, image_key, max_frames=max_frames)
            first_bytes = next(frame_iter, None)
            if first_bytes is None:
                continue
            first_arr = np.frombuffer(first_bytes, dtype=np.uint8)
            first_img = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
            if first_img is None:
                continue
            height, width = first_img.shape[:2]
            target_w = width - (width % 2)
            target_h = height - (height % 2)
            if target_w != width or target_h != height:
                first_img = cv2.resize(first_img, (target_w, target_h))
                width, height = target_w, target_h

            def frames_rgb():
                yield cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
                for img_bytes in frame_iter:
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height))
                    yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if not _encode_with_ffmpeg(out_path, frames_rgb(), width, height, dataset.fps):
                logging.warning("Failed to encode video with ffmpeg; video may be unplayable.")

        if out_path.is_file() and out_path.stat().st_size > 0:
            videos_info.append(
                {
                    "url": url_for("static", filename=rel_path.as_posix()),
                    "filename": image_key,
                }
            )

    return videos_info


def _find_prepared_videos(static_dir: Path, image_keys: list[str], episode_id: int) -> list[dict]:
    videos_info: list[dict] = []
    for image_key in image_keys:
        rel_path = Path("videos") / image_key / f"episode_{episode_id:06d}_h264.mp4"
        out_path = static_dir / rel_path
        if out_path.is_file() and out_path.stat().st_size > 0:
            videos_info.append(
                {
                    "url": url_for("static", filename=rel_path.as_posix()),
                    "filename": image_key,
                }
            )
    return videos_info


def _read_parquet_head(parquet_path: Path, columns: list[str], max_rows: int) -> pd.DataFrame:
    pf = _get_parquet_file(str(parquet_path))
    tables = []
    remaining = max_rows
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=columns)
        if len(table) > remaining:
            table = table.slice(0, remaining)
            tables.append(table)
            break
        tables.append(table)
        remaining -= len(table)
        if remaining <= 0:
            break
    if not tables:
        return pd.DataFrame(columns=columns)
    return pa.concat_tables(tables).to_pandas()


def _find_any_cached_csv(cache_dir: Path, episode_id: int) -> Path | None:
    pattern = f"episode_{episode_id:06d}_ds*.csv"
    candidates = list(cache_dir.glob(pattern))
    if not candidates:
        return None

    def _ds_value(path: Path) -> int:
        match = re.search(r"_ds(\d+)\\.csv$", path.name)
        return int(match.group(1)) if match else 10**9

    return min(candidates, key=_ds_value)


def _get_csv_cache_path(
    cache_dir: Path, episode_id: int, downsample: int | None, precomputed_only: bool
) -> Path | None:
    ds = downsample if downsample and downsample > 1 else 1
    preferred = cache_dir / f"episode_{episode_id:06d}_ds{ds}.csv"
    if preferred.is_file():
        return preferred
    if precomputed_only:
        return _find_any_cached_csv(cache_dir, episode_id)
    return preferred


def _columns_from_csv_header(csv_path: Path) -> list[dict]:
    try:
        with csv_path.open("r") as f:
            header_line = f.readline().strip()
    except OSError:
        return []
    if not header_line:
        return []
    fields = [field.strip() for field in header_line.split(",") if field.strip()]
    if not fields:
        return []
    if fields[0] == "timestamp":
        fields = fields[1:]
    columns: dict[str, list[str]] = {}
    order: list[str] = []
    for name in fields:
        base = name
        if "_" in name:
            prefix, suffix = name.rsplit("_", 1)
            if suffix.isdigit():
                base = prefix
        if base not in columns:
            columns[base] = []
            order.append(base)
        columns[base].append(name)
    return [{"key": key, "value": columns[key]} for key in order]


def _video_rank(filename: str) -> int:
    normalized = re.sub(r"[./-]+", "_", (filename or "").lower()).strip("_")
    tokens = {token for token in normalized.split("_") if token}
    is_left_wrist = ("left" in tokens and "wrist" in tokens) or "left_wrist" in normalized
    is_right_wrist = ("right" in tokens and "wrist" in tokens) or "right_wrist" in normalized
    is_main_image = "image" in tokens and not is_left_wrist and not is_right_wrist

    if is_left_wrist:
        return 0
    if is_main_image:
        return 1
    if is_right_wrist:
        return 2
    return 3


def _sort_videos_info(videos_info: list[dict]) -> list[dict]:
    # Stable sort: keep original order inside each rank bucket.
    return sorted(videos_info, key=lambda info: _video_rank(info.get("filename", "")))


def run_server(
    dataset: LeRobotDataset | IterableNamespace | MetaOnlyDataset | None,
    episodes: list[int] | None,
    max_frames: int | None,
    prepare_videos: bool,
    downsample: int | None,
    precompute_csv: bool,
    precomputed_only: bool,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
):
    class QuietRequestHandler(WSGIRequestHandler):
        def log_request(self, code="-", size="-"):
            return

    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.INFO)
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.logger.setLevel(logging.ERROR)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    @app.route("/")
    def hommepage(dataset=dataset):
        if dataset:
            dataset_namespace, dataset_name = dataset.repo_id.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )

        dataset_param, episode_param = None, None
        all_params = request.args
        if "dataset" in all_params:
            dataset_param = all_params["dataset"]
        if "episode" in all_params:
            episode_param = int(all_params["episode"])

        if dataset_param:
            dataset_namespace, dataset_name = dataset_param.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=episode_param if episode_param is not None else 0,
                )
            )

        featured_datasets = [
            "lerobot/aloha_static_cups_open",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/taco_play",
        ]
        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=featured_datasets,
            lerobot_datasets=available_datasets,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>")
    def show_first_episode(dataset_namespace, dataset_name):
        first_episode_id = 0
        return redirect(
            url_for(
                "show_episode",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id, dataset=dataset, episodes=episodes):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        try:
            if dataset is None:
                dataset = get_dataset_info(repo_id)
        except FileNotFoundError:
            return (
                "Make sure to convert your LeRobotDataset to v2 & above. See how to convert your dataset at https://github.com/huggingface/lerobot/pull/461",
                400,
            )
        dataset_version = (
            str(dataset.meta._version) if isinstance(dataset, LeRobotDataset) else dataset.codebase_version
        )
        match = re.search(r"v(\d+)\.", dataset_version)
        if match:
            major_version = int(match.group(1))
            if major_version < 2:
                return "Make sure to convert your LeRobotDataset to v2 & above."

        cache_dir = static_folder / "csv"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_csv = _get_csv_cache_path(cache_dir, episode_id, downsample, precomputed_only)
        if precomputed_only and cached_csv and cached_csv.is_file():
            columns = _columns_from_csv_header(cached_csv)
            ignored_columns = []
            if not columns:
                columns, ignored_columns, _ = get_columns_info(dataset)
        else:
            columns, ignored_columns, _ = get_columns_info(dataset)
        episode_data_csv_str = ""
        data_len = None
        dataset_info = {
            "repo_id": f"{dataset_namespace}/{dataset_name}",
            "num_samples": dataset.num_frames
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_frames,
            "num_episodes": dataset.num_episodes
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_episodes,
            "fps": dataset.fps,
        }
        is_local = isinstance(dataset, LeRobotDataset) or hasattr(dataset, "meta")
        if is_local:
            video_paths = [
                dataset.meta.get_video_file_path(episode_id, key) for key in dataset.meta.video_keys
            ]
            videos_info = [
                {
                    "url": url_for("static", filename=str(video_path).replace("\\", "/")),
                    "filename": video_path.parent.name,
                }
                for video_path in video_paths
            ]
            tasks = dataset.meta.episodes[episode_id]["tasks"]
        else:
            video_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "video"]
            videos_info = [
                {
                    "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
                    + dataset.video_path.format(
                        episode_chunk=int(episode_id) // dataset.chunks_size,
                        video_key=video_key,
                        episode_index=episode_id,
                    ),
                    "filename": video_key,
                }
                for video_key in video_keys
            ]

            response = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/episodes.jsonl", timeout=5
            )
            response.raise_for_status()
            # Split into lines and parse each line as JSON
            tasks_jsonl = [json.loads(line) for line in response.text.splitlines() if line.strip()]

            filtered_tasks_jsonl = [row for row in tasks_jsonl if row["episode_index"] == episode_id]
            tasks = filtered_tasks_jsonl[0]["tasks"]

        language_instruction = tasks if tasks else None

        image_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "image"]
        prepared_videos = False
        if not videos_info and image_keys:
            prepared = _find_prepared_videos(static_folder, image_keys, episode_id)
            if prepared:
                videos_info = prepared
                image_keys = []
                prepared_videos = True

        if (
            is_local
            and prepare_videos
            and not precomputed_only
            and not videos_info
            and image_keys
        ):
            videos_info = _prepare_episode_videos(
                dataset,
                episode_id,
                image_keys,
                static_folder,
                max_frames=max_frames,
            )
            if videos_info:
                image_keys = []
                prepared_videos = True

        videos_info = _sort_videos_info(videos_info)

        if videos_info:
            video_codec_hint = "h264" if prepared_videos else "av1"
        else:
            video_codec_hint = "none"
        episode_length = data_len if data_len is not None else None
        if isinstance(dataset, LeRobotDataset):
            ep_from = dataset.episode_data_index["from"][episode_id].item()
            ep_to = dataset.episode_data_index["to"][episode_id].item()
            if episode_length is None:
                episode_length = ep_to - ep_from
            elif max_frames is not None:
                episode_length = min(episode_length, ep_to - ep_from)
        elif is_local and hasattr(dataset.meta, "episodes"):
            if episode_length is None:
                episode_length = dataset.meta.episodes[episode_id]["length"]

        if episodes is None:
            episodes = list(
                range(dataset.num_episodes if isinstance(dataset, LeRobotDataset) else dataset.total_episodes)
            )

        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_namespace=dataset_namespace,
            dataset_name=dataset_name,
            dataset_info=dataset_info,
            videos_info=videos_info,
            image_keys=image_keys,
            episode_length=episode_length,
            video_codec_hint=video_codec_hint,
            language_instruction=language_instruction,
            episode_data_csv_str=episode_data_csv_str,
            csv_url=url_for(
                "get_episode_csv",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=episode_id,
            ),
            columns=columns,
            ignored_columns=ignored_columns,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/image")
    def get_image(dataset_namespace, dataset_name, dataset=dataset):
        if not isinstance(dataset, LeRobotDataset):
            abort(404)
        key = request.args.get("key")
        if not key:
            abort(400)
        try:
            episode_id = int(request.args.get("episode", 0))
            frame_index = int(request.args.get("frame", 0))
        except (TypeError, ValueError):
            abort(400)
        if key not in dataset.features or dataset.features[key]["dtype"] != "image":
            abort(404)
        if episode_id < 0 or episode_id >= dataset.num_episodes:
            abort(404)
        ep_from = dataset.episode_data_index["from"][episode_id].item()
        ep_to = dataset.episode_data_index["to"][episode_id].item()
        episode_length = ep_to - ep_from
        if frame_index < 0 or frame_index >= episode_length:
            abort(404)
        parquet_path = dataset.root / dataset.meta.get_data_file_path(episode_id)
        img_bytes = _cached_image_bytes(str(parquet_path), str(dataset.root), key, frame_index)
        if img_bytes is None:
            abort(404)
        return send_file(BytesIO(img_bytes), mimetype="image/png")

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>/data.csv")
    def get_episode_csv(dataset_namespace, dataset_name, episode_id, dataset=dataset):
        try:
            if dataset is None:
                abort(404)
            cache_dir = static_folder / "csv"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = _get_csv_cache_path(cache_dir, episode_id, downsample, precomputed_only)
            if precomputed_only:
                if cache_path and cache_path.is_file():
                    return send_file(cache_path, mimetype="text/csv")
                return (
                    "CSV cache not found. Please precompute with prepare_videos_from_images.py --prepare-csv 1.",
                    404,
                )
            if cache_path is None or not cache_path.is_file():
                csv_string, _, _, _ = get_episode_data(
                    dataset, episode_id, max_frames=max_frames, downsample=downsample
                )
                ds = downsample if downsample and downsample > 1 else 1
                cache_path = cache_dir / f"episode_{episode_id:06d}_ds{ds}.csv"
                cache_path.write_text(csv_string)
            return send_file(cache_path, mimetype="text/csv")
        except Exception:
            logging.exception("Failed to serve CSV for episode %s", episode_id)
            return (f"Failed to serve CSV for episode {episode_id}:\n\n{traceback.format_exc()}", 500)

    if precompute_csv and dataset is not None and not precomputed_only:
        cache_dir = static_folder / "csv"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_episodes = episodes
        if target_episodes is None:
            total_eps = dataset.num_episodes if isinstance(dataset, LeRobotDataset) else dataset.total_episodes
            target_episodes = range(total_eps)
        ds = downsample if downsample and downsample > 1 else 1
        for ep_id in target_episodes:
            cache_path = cache_dir / f"episode_{ep_id:06d}_ds{ds}.csv"
            if cache_path.is_file():
                continue
            csv_string, _, _, _ = get_episode_data(
                dataset, ep_id, max_frames=max_frames, downsample=downsample
            )
            cache_path.write_text(csv_string)
            del csv_string
            gc.collect()
            logging.info("CSV cached: %s", cache_path)

    app.run(host=host, port=port, request_handler=QuietRequestHandler)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def _get_feature_shape(feature) -> tuple:
    if isinstance(feature, dict):
        shape = feature.get("shape")
        return tuple(shape) if shape is not None else ()
    if hasattr(feature, "shape"):
        return tuple(feature.shape)
    return ()


def get_columns_info(dataset: LeRobotDataset | IterableNamespace | MetaOnlyDataset):
    columns = []
    selected_columns = [col for col, ft in dataset.features.items() if ft["dtype"] in ["float32", "int32"]]
    if "timestamp" in selected_columns:
        selected_columns.remove("timestamp")

    ignored_columns = []
    filtered_columns = []
    for column_name in selected_columns:
        shape = dataset.features[column_name]["shape"]
        shape_dim = len(shape)
        if shape_dim > 1:
            ignored_columns.append(column_name)
        else:
            filtered_columns.append(column_name)
    selected_columns = filtered_columns

    for column_name in selected_columns:
        if isinstance(dataset, LeRobotDataset):
            dim_state = dataset.meta.shapes[column_name][0]
        else:
            shape = _get_feature_shape(dataset.features[column_name])
            dim_state = shape[0] if shape else 0

        if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
            column_names = dataset.features[column_name]["names"]
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
            if not isinstance(column_names, list) or len(column_names) != dim_state:
                column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

    selected_columns.insert(0, "timestamp")
    return columns, ignored_columns, selected_columns


def get_episode_data(
    dataset: LeRobotDataset | IterableNamespace | MetaOnlyDataset,
    episode_index: int,
    max_frames: int | None = None,
    downsample: int | None = None,
):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    columns, ignored_columns, selected_columns = get_columns_info(dataset)

    if isinstance(dataset, LeRobotDataset):
        parquet_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
        if parquet_path.is_file():
            if max_frames is not None:
                data = _read_parquet_head(parquet_path, selected_columns, max_frames)
            else:
                data = pd.read_parquet(parquet_path, columns=selected_columns)
        else:
            from_idx = dataset.episode_data_index["from"][episode_index]
            to_idx = dataset.episode_data_index["to"][episode_index]
            data = (
                dataset.hf_dataset.select(range(from_idx, to_idx))
                .select_columns(selected_columns)
                .with_format("pandas")
            )
    else:
        repo_id = dataset.repo_id

        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/" + dataset.data_path.format(
            episode_chunk=int(episode_index) // dataset.chunks_size, episode_index=episode_index
        )
        df = pd.read_parquet(url)
        data = df[selected_columns]  # Select specific columns
        if max_frames is not None:
            data = data.head(max_frames)

    if downsample is not None and downsample > 1:
        data = data.iloc[::downsample].reset_index(drop=True)

    def _get_len(item) -> int:
        if item is None:
            return 0
        try:
            return len(item)
        except TypeError:
            return 0

    def _normalize_series(series, dim: int) -> np.ndarray:
        rows = []
        for item in series:
            row = [np.nan] * dim
            if item is not None:
                values = list(item)
                if len(values) > dim:
                    values = values[:dim]
                row[: len(values)] = values
            rows.append(row)
        return np.asarray(rows)

    data_arrays = []
    for col in selected_columns[1:]:
        if isinstance(dataset, LeRobotDataset):
            fallback_dim = dataset.meta.shapes[col][0]
        else:
            shape = _get_feature_shape(dataset.features[col])
            fallback_dim = shape[0] if shape else 0
        series = data[col]
        actual_dim = max((_get_len(item) for item in series), default=0)
        dim = actual_dim if actual_dim > 0 else fallback_dim

        # Ensure column names match actual data length
        col_entry = next((c for c in columns if c["key"] == col), None)
        if col_entry is not None and len(col_entry["value"]) != dim:
            col_entry["value"] = [f"{col}_{i}" for i in range(dim)]

        data_arrays.append(_normalize_series(series, dim))

    # Refresh header in case we resized columns to match actual data
    header = ["timestamp"]
    for col_entry in columns:
        header += col_entry["value"]

    rows = np.hstack((np.expand_dims(data["timestamp"], axis=1), *data_arrays)).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Write header
    csv_writer.writerow(header)
    # Write data rows
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string, columns, ignored_columns, len(data)


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()
    return [
        dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"]
        for key in dataset.meta.video_keys
    ]


def get_episode_language_instruction(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # check if the dataset has language instructions
    if "language_instruction" not in dataset.features:
        return None

    # get first frame index
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()

    language_instruction = dataset.hf_dataset[first_frame_idx]["language_instruction"]
    # TODO (michel-aractingi) hack to get the sentence, some strings in openx are badly stored
    # with the tf.tensor appearing in the string
    return language_instruction.removeprefix("tf.Tensor(b'").removesuffix("', shape=(), dtype=string)")


def get_dataset_info(repo_id: str) -> IterableNamespace:
    response = requests.get(
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json", timeout=5
    )
    response.raise_for_status()  # Raises an HTTPError for bad responses
    dataset_info = response.json()
    dataset_info["repo_id"] = repo_id
    return IterableNamespace(dataset_info)


def visualize_dataset_html(
    dataset: LeRobotDataset | MetaOnlyDataset | IterableNamespace | None,
    episodes: list[int] | None = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
    max_frames: int | None = None,
    prepare_videos: int | bool = False,
    downsample: int | None = None,
    precompute_csv: int | bool = False,
    precomputed_only: int | bool = False,
) -> Path | None:
    init_logging()

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    if output_dir is None:
        # Create a temporary directory that will be automatically cleaned up
        output_dir = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")

    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        if serve:
            run_server(
                dataset=None,
                episodes=None,
                max_frames=max_frames,
                prepare_videos=False,
                downsample=downsample,
                precompute_csv=False,
                precomputed_only=bool(precomputed_only),
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
            )
    else:
        # Create a simlink from the dataset video folder containing mp4 files to the output directory
        # so that the http server can get access to the mp4 files.
        if isinstance(dataset, LeRobotDataset) or hasattr(dataset, "meta"):
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                if len(dataset.meta.video_keys) > 0:
                    ln_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())
                else:
                    ln_videos_dir.mkdir(parents=True, exist_ok=True)

        if serve:
            run_server(
                dataset=dataset,
                episodes=episodes,
                max_frames=max_frames,
                prepare_videos=bool(prepare_videos),
                downsample=downsample,
                precompute_csv=bool(precompute_csv),
                precomputed_only=bool(precomputed_only),
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Root directory for a dataset stored locally (e.g. `--root data`). "
            "If used without `--repo-id`, the dataset is loaded directly from this path."
        ),
    )
    parser.add_argument(
        "--load-from-hf-hub",
        type=int,
        default=0,
        help="Load videos and parquet files from HF Hub rather than local system.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write html files and kickoff a web server. By default write them to 'outputs/visualize_dataset/REPO_ID'.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="Delete the output directory if it exists already.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of frames loaded per episode (useful for quick debugging).",
    )
    parser.add_argument(
        "--prepare-videos",
        type=int,
        default=0,
        help="Precompute mp4 videos from image streams and serve them.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Downsample time series by keeping one every N frames (e.g. 5).",
    )
    parser.add_argument(
        "--precompute-csv",
        type=int,
        default=0,
        help="Precompute CSV cache for selected episodes on startup.",
    )
    parser.add_argument(
        "--precomputed-only",
        type=int,
        default=0,
        help="Only use precomputed CSV/videos. Do not generate on the fly.",
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    load_from_hf_hub = kwargs.pop("load_from_hf_hub")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    dataset = None
    precomputed_only = bool(kwargs.get("precomputed_only", False))
    if root is not None and not repo_id:
        if load_from_hf_hub:
            raise ValueError("--load-from-hf-hub requires --repo-id and cannot be used with a local root.")
        if not root.exists():
            raise FileNotFoundError(f"Local dataset root does not exist: {root}")
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing dataset metadata at: {info_path}")
        repo_id = f"local/{root.name or 'dataset'}"
        dataset = (
            MetaOnlyDataset(repo_id, root=root)
            if precomputed_only
            else LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
        )
    elif repo_id:
        if precomputed_only:
            dataset = MetaOnlyDataset(repo_id, root=root)
        else:
            dataset = (
                LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
                if not load_from_hf_hub
                else get_dataset_info(repo_id)
            )

    visualize_dataset_html(dataset, **kwargs)


if __name__ == "__main__":
    main()
