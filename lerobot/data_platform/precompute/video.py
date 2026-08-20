from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.data_platform.precompute.image_io import iter_image_bytes


def temporary_output_path(target: Path) -> Path:
    """Return a unique sibling path that preserves the target suffix."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def encode_with_ffmpeg(
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

    temporary = temporary_output_path(out_path)
    cmd = [
        ffmpeg_path,
        "-loglevel",
        "error",
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
        str(temporary),
    ]
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
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
        if not temporary.is_file() or temporary.stat().st_size == 0:
            logging.warning("ffmpeg produced an empty video: %s", temporary)
            return False
        temporary.replace(out_path)
        return True
    except Exception:
        if proc is not None and hasattr(proc, "poll") and proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def encode_episode_video(
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
    if not overwrite:
        try:
            if out_path.is_file() and out_path.stat().st_size > 0:
                return out_path
        except OSError:
            pass

    frame_iter = iter_image_bytes(parquet_path, dataset_root, image_key, max_frames=max_frames)
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
        for image_bytes in frame_iter:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
            yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fps = meta.fps if hasattr(meta, "fps") else meta.info["fps"]
    if not encode_with_ffmpeg(out_path, frames_rgb(), width, height, fps):
        return None

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    return None
