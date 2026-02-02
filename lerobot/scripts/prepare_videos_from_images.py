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
"""Precompute mp4 videos from image streams for faster visualization."""

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.utils.utils import init_logging


@lru_cache(maxsize=8)
def _get_parquet_file(path: str) -> pq.ParquetFile:
    return pq.ParquetFile(path)


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


def _encode_with_ffmpeg(
    out_path: Path,
    frame_iter,
    width: int,
    height: int,
    fps: int,
) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logging.error("ffmpeg not found in PATH.")
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


def _encode_episode_key(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    episode_id: int,
    image_key: str,
    static_dir: Path,
    max_frames: int | None,
    overwrite: bool,
) -> Path | None:
    parquet_path = dataset_root / meta.get_data_file_path(episode_id)
    if not parquet_path.is_file():
        return None

    rel_path = Path("videos") / image_key / f"episode_{episode_id:06d}_h264.mp4"
    out_path = static_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    frame_iter = _iter_image_bytes(parquet_path, dataset_root, image_key, max_frames=max_frames)
    first_bytes = next(frame_iter, None)
    if first_bytes is None:
        return None
    first_arr = np.frombuffer(first_bytes, dtype=np.uint8)
    first_img = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
    if first_img is None:
        return None
    height, width = first_img.shape[:2]
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        return None
    if (first_img.shape[1], first_img.shape[0]) != (width, height):
        first_img = cv2.resize(first_img, (width, height))

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

    if not _encode_with_ffmpeg(out_path, frames_rgb(), width, height, meta.info["fps"]):
        return None

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    return None


def _get_columns_info(meta: LeRobotDatasetMetadata):
    columns = []
    selected_columns = [col for col, ft in meta.features.items() if ft["dtype"] in ["float32", "int32"]]
    if "timestamp" in selected_columns:
        selected_columns.remove("timestamp")

    ignored_columns = []
    filtered_columns = []
    for column_name in selected_columns:
        shape = meta.features[column_name]["shape"]
        if len(shape) > 1:
            ignored_columns.append(column_name)
        else:
            filtered_columns.append(column_name)
    selected_columns = filtered_columns

    for column_name in selected_columns:
        dim_state = meta.shapes[column_name][0]
        names = meta.features[column_name].get("names")
        if names:
            column_names = names
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
            if not isinstance(column_names, list) or len(column_names) != dim_state:
                column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

    selected_columns.insert(0, "timestamp")
    return columns, ignored_columns, selected_columns


def _write_episode_csv(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    episode_id: int,
    out_path: Path,
    max_frames: int | None,
    downsample: int | None,
    overwrite: bool,
) -> bool:
    if out_path.exists() and not overwrite:
        return True
    parquet_path = dataset_root / meta.get_data_file_path(episode_id)
    if not parquet_path.is_file():
        return False

    columns, _, selected_columns = _get_columns_info(meta)
    data = pd.read_parquet(parquet_path, columns=selected_columns)
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
        fallback_dim = meta.shapes[col][0]
        series = data[col]
        actual_dim = max((_get_len(item) for item in series), default=0)
        dim = actual_dim if actual_dim > 0 else fallback_dim

        col_entry = next((c for c in columns if c["key"] == col), None)
        if col_entry is not None and len(col_entry["value"]) != dim:
            col_entry["value"] = [f"{col}_{i}" for i in range(dim)]

        data_arrays.append(_normalize_series(series, dim))

    header = ["timestamp"]
    for col_entry in columns:
        header += col_entry["value"]

    rows = np.hstack((np.expand_dims(data["timestamp"], axis=1), *data_arrays)).tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory for a dataset stored locally.",
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
        help="Output dir used by visualize_dataset_html (will write to <output-dir>/static/videos).",
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
        default=0,
        help="Prepare downsampled CSV for time-series plots.",
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
        help="Downsample time series by keeping one every N frames (e.g. 5).",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="Overwrite existing mp4 files.",
    )

    args = parser.parse_args()
    init_logging()

    if not args.root.exists():
        raise FileNotFoundError(f"Local dataset root does not exist: {args.root}")
    info_path = args.root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata at: {info_path}")

    repo_id = args.repo_id or f"local/{args.root.name or 'dataset'}"
    meta = LeRobotDatasetMetadata(repo_id, root=args.root)

    image_keys = args.image_keys
    if image_keys is None:
        image_keys = [key for key, ft in meta.features.items() if ft["dtype"] == "image"]

    if args.episodes is None:
        episodes = sorted(meta.episodes.keys())
    else:
        episodes = args.episodes

    if args.output_dir is None:
        safe_repo = repo_id.replace("/", "_")
        output_dir = Path("outputs") / "visualize_dataset" / safe_repo
    else:
        output_dir = args.output_dir

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = static_dir / "csv"
    if args.prepare_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)

    total = len(episodes) * len(image_keys) if args.prepare_videos else len(episodes)
    done = 0

    def print_progress(done_count: int) -> None:
        if total == 0:
            return
        width = 30
        filled = int(width * done_count / total)
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[{bar}] {done_count}/{total}")
        sys.stdout.flush()

    print_progress(done)
    for ep_id in episodes:
        if args.prepare_videos:
            for image_key in image_keys:
                out_path = _encode_episode_key(
                    args.root,
                    meta,
                    ep_id,
                    image_key,
                    static_dir,
                    args.max_frames,
                    bool(args.overwrite),
                )
                done += 1
                print_progress(done)
                if out_path is not None:
                    logging.info("Prepared %s", out_path)
        if args.prepare_csv:
            ds = args.downsample if args.downsample and args.downsample > 1 else 1
            csv_path = csv_dir / f"episode_{ep_id:06d}_ds{ds}.csv"
            if _write_episode_csv(
                args.root,
                meta,
                ep_id,
                csv_path,
                args.max_frames,
                args.downsample,
                bool(args.overwrite),
            ):
                logging.info("Prepared %s", csv_path)
            if not args.prepare_videos:
                done += 1
                print_progress(done)
    if total > 0:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
