from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pyarrow as pa

from lerobot.data_platform.precompute.dataset_io import read_episode_frame_image

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # optional dependency guard
    torch = None
    _TORCH_AVAILABLE = False

DEFAULT_OPENPI_CONFIG = "pi05_h10w_dual_full_finetune_0417_ALL_80k"
DEFAULT_OPENPI_CHECKPOINT = Path(
    "/DATA/disk0/yuhao.song/checkpoints/pi05_h10w_dual_full_finetune_0417_ALL/0417_ALL/29999"
)
TRAIN_CONFIG_FILENAME = "train_config_full.json"
FALLBACK_LAYER = "episode_stats_fallback"
OPENPI_LAYER_OPTIONS = ["pi_prefix", "vision_encoder", "pi_prefix_prompt"]
OPENPI_EMBED_MODE = os.environ.get("OPENPI_EMBED_MODE", "uv").strip().lower() or "uv"


class OpenPIWorkerError(RuntimeError):
    pass


def _lerobot_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "lerobot").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).resolve().parents[5]


def _normalize_openpi_src(path: Path) -> Path:
    path = Path(path).expanduser()
    if (path / "src" / "openpi").is_dir():
        return path / "src"
    if path.name == "src" and (path / "openpi").is_dir():
        return path
    return path


def _normalize_openpi_root(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.name == "src" and (path / "openpi").is_dir():
        return path.parent
    return path


def _unique_paths(candidates: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _openpi_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("OPENPI_ROOT") or os.environ.get("OPENPI_SRC") or os.environ.get("OPENPI_PATH")
    if env_root:
        candidates.append(_normalize_openpi_root(Path(env_root)))
    repo_root = _lerobot_repo_root()
    candidates.append(repo_root.parent / "openpi")
    candidates.append(Path("/home/peng/yuhao.song/Codes/openpi"))
    return _unique_paths(candidates)


def _openpi_src_candidates() -> list[Path]:
    candidates = [_normalize_openpi_src(root) for root in _openpi_root_candidates()]
    return _unique_paths(candidates)


def _valid_openpi_root(root: Path) -> bool:
    return (root / "pyproject.toml").is_file() and (root / "src" / "openpi").is_dir()


def _default_openpi_root() -> Path:
    for root in _openpi_root_candidates():
        if _valid_openpi_root(root):
            return root
    return _openpi_root_candidates()[0]


DEFAULT_OPENPI_ROOT = _default_openpi_root()
DEFAULT_OPENPI_SRC = _normalize_openpi_src(DEFAULT_OPENPI_ROOT)


def _first_openpi_root() -> Path:
    for root in _openpi_root_candidates():
        if _valid_openpi_root(root):
            return root
    attempted = [str(path) for path in _openpi_root_candidates()]
    raise OpenPIWorkerError(
        f"Could not find sibling OpenPI repo. Tried: {attempted}. "
        "Set OPENPI_ROOT=/path/to/openpi or OPENPI_SRC=/path/to/openpi/src."
    )


def _uv_bin() -> str:
    uv = os.environ.get("OPENPI_UV") or shutil.which("uv")
    if not uv:
        raise OpenPIWorkerError("Could not find uv. Install uv or set OPENPI_UV=/path/to/uv.")
    return uv


def _detect_gpu_count() -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 0
    try:
        proc = subprocess.run([nvidia_smi, "-L"], check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return 0
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.strip().startswith("GPU "))


def _split_devices(devices: str | None) -> list[str]:
    if not devices:
        return []
    return [part.strip() for part in devices.replace(",", " ").split() if part.strip()]


def default_openpi_devices() -> str:
    env_devices = os.environ.get("OPENPI_EMBED_DEVICES")
    if env_devices is not None:
        return env_devices.strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return visible.strip()
    gpu_count = _detect_gpu_count()
    if gpu_count <= 0:
        return ""
    return ",".join(str(i) for i in range(min(gpu_count, 8)))


def default_openpi_workers() -> int:
    env_workers = os.environ.get("OPENPI_EMBED_WORKERS")
    if env_workers:
        try:
            return max(1, int(env_workers))
        except ValueError:
            return 1
    devices = _split_devices(default_openpi_devices())
    return max(1, len(devices))


def _ensure_openpi_importable():
    try:
        import openpi  # noqa: F401

        return
    except ModuleNotFoundError as exc:
        if exc.name != "openpi":
            raise
        first_exc = exc

    attempted = []
    for src in _openpi_src_candidates():
        attempted.append(str(src))
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    try:
        import openpi  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "openpi":
            raise ModuleNotFoundError(
                "No module named 'openpi'. Tried adding OpenPI source paths: "
                f"{attempted}. Set OPENPI_SRC=/path/to/openpi/src if needed."
            ) from first_exc
        raise


def _openpi_status() -> tuple[bool, str | None]:
    if OPENPI_EMBED_MODE != "inprocess":
        try:
            _uv_bin()
            _first_openpi_root()
            return True, None
        except Exception as exc:
            return False, str(exc)
    try:
        _ensure_openpi_importable()
        from openpi.training import config as _config  # noqa: F401

        return True, None
    except Exception as exc:
        return False, str(exc)


def get_capabilities() -> dict:
    gpu = False
    if _TORCH_AVAILABLE:
        try:
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
    openpi_available, openpi_error = _openpi_status()
    return {
        "torch_available": bool(_TORCH_AVAILABLE),
        "torch_gpu": gpu,
        "openpi_available": openpi_available,
        "openpi_error": openpi_error,
        "openpi_execution_mode": OPENPI_EMBED_MODE,
        "openpi_root": str(DEFAULT_OPENPI_ROOT),
        "openpi_src": str(DEFAULT_OPENPI_SRC),
        "default_openpi_config": DEFAULT_OPENPI_CONFIG,
        "default_checkpoint_path": str(DEFAULT_OPENPI_CHECKPOINT),
        "default_workers": default_openpi_workers(),
        "default_devices": default_openpi_devices(),
        "default_layer_options": [*OPENPI_LAYER_OPTIONS, FALLBACK_LAYER],
        "default_layer": "pi_prefix",
        "policy_adapter": "openpi_prefix" if OPENPI_EMBED_MODE == "inprocess" else "openpi_prefix_uv",
    }


def checkpoint_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Embedding checkpoint path does not exist: {path}")
    if path.is_dir():
        items = sorted(p for p in path.rglob("*") if p.is_file())[:32]
        h = hashlib.sha256()
        for item in items:
            h.update(str(item.relative_to(path)).encode())
            h.update(str(item.stat().st_size).encode())
        return h.hexdigest()[:16]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def resolve_openpi_config(ckpt_path: Path | None, explicit_config: str | None = None) -> str:
    explicit_config = (explicit_config or "").strip()
    if explicit_config:
        return explicit_config
    if ckpt_path is not None:
        config_path = Path(ckpt_path).expanduser() / TRAIN_CONFIG_FILENAME
        if config_path.is_file():
            try:
                name = str(json.loads(config_path.read_text()).get("name") or "").strip()
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"Could not read OpenPI config name from {config_path}: {exc}") from exc
            if name:
                return name
            raise ValueError(f"Missing 'name' field in {config_path}")
    return DEFAULT_OPENPI_CONFIG


def _summarize_timing_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    summary = {"count": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if values.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
            "sum": float(np.sum(values)),
        }
    return summary


def _raw_value_to_payload(key: str, value, arrays: dict[str, np.ndarray], manifest: dict[str, dict]) -> None:
    if isinstance(value, np.ndarray):
        arr_key = f"arr_{len(arrays)}"
        arrays[arr_key] = value
        manifest[key] = {"type": "array", "name": arr_key}
        return
    if isinstance(value, np.generic):
        manifest[key] = {"type": "scalar", "value": value.item()}
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        manifest[key] = {"type": "scalar", "value": value}
        return
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
        except Exception:
            return
        if arr.dtype != object:
            arr_key = f"arr_{len(arrays)}"
            arrays[arr_key] = arr
            manifest[key] = {"type": "array", "name": arr_key}
        return
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return
    manifest[key] = {"type": "scalar", "value": value}


def _write_raw_payload(raws: list[dict], payload_path: Path) -> Path:
    arrays: dict[str, np.ndarray] = {}
    manifest = {"frames": []}
    for raw in raws:
        frame_manifest: dict[str, dict] = {}
        for key, value in raw.items():
            _raw_value_to_payload(str(key), value, arrays, frame_manifest)
        manifest["frames"].append(frame_manifest)
    np.savez_compressed(payload_path, **arrays)
    manifest_path = payload_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class OpenPISubprocessWorker:
    def __init__(self, ckpt_path: Path, config_name: str, layer_hook: str, device: str | None = None):
        self.ckpt_path = Path(ckpt_path).expanduser()
        self.config_name = config_name
        self.layer_hook = layer_hook
        self.device = device
        self.proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self.timings: list[dict] = []
        self._start()

    def _start(self) -> None:
        openpi_root = _first_openpi_root()
        worker_path = Path(__file__).with_name("openpi_uv_worker.py")
        env = os.environ.copy()
        openpi_src = openpi_root / "src"
        pythonpath = str(openpi_src)
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONPATH": pythonpath})
        if self.device:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        cmd = [
            _uv_bin(),
            "run",
            "python",
            str(worker_path),
            "--ckpt",
            str(self.ckpt_path),
            "--config",
            self.config_name,
            "--layer-hook",
            self.layer_hook,
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=openpi_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="openpi-embedding-stderr", daemon=True
        )
        self._stderr_thread.start()
        ready = self._read_response()
        if ready.get("status") != "ready":
            raise OpenPIWorkerError(f"OpenPI worker did not become ready: {ready}")

    def _drain_stderr(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            logging.info("openpi embedding worker: %s", line.rstrip())

    def _read_response(self) -> dict:
        if self.proc is None or self.proc.stdout is None:
            raise OpenPIWorkerError("OpenPI worker is not running")
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.poll()
            raise OpenPIWorkerError(f"OpenPI worker exited before response. returncode={code}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpenPIWorkerError(f"Invalid OpenPI worker response: {line[:500]}") from exc
        if response.get("status") == "error":
            raise OpenPIWorkerError(str(response.get("error") or "OpenPI worker failed"))
        return response

    def embed(self, raw: dict) -> np.ndarray:
        return self.embed_many([raw])[0]

    def embed_many(self, raws: list[dict]) -> np.ndarray:
        if self.proc is None or self.proc.stdin is None:
            raise OpenPIWorkerError("OpenPI worker is not running")
        request_start = time.perf_counter()
        with tempfile.NamedTemporaryFile(prefix="openpi_embed_", suffix=".npz", delete=False) as f:
            payload_path = Path(f.name)
        serialize_start = time.perf_counter()
        manifest_path = _write_raw_payload(raws, payload_path)
        serialize_s = time.perf_counter() - serialize_start
        try:
            roundtrip_start = time.perf_counter()
            self.proc.stdin.write(
                json.dumps({"path": str(payload_path), "manifest": str(manifest_path), "count": len(raws)})
                + "\n"
            )
            self.proc.stdin.flush()
            response = self._read_response()
            roundtrip_s = time.perf_counter() - roundtrip_start
            timing = {
                "host_serialize_s": serialize_s,
                "worker_roundtrip_s": roundtrip_s,
                "host_request_s": time.perf_counter() - request_start,
                "frames": float(len(raws)),
                "device": self.device,
                **(response.get("timing") or {}),
            }
            self.timings.append(timing)
            return np.asarray(response["vectors"], dtype=np.float32)
        finally:
            payload_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

    def timing_summary(self) -> dict:
        return _summarize_timing_rows(self.timings)

    def close(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        with suppress(Exception):
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"stop": True}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()

    def __del__(self):
        self.close()


def _column_to_2d(column: pa.ChunkedArray) -> np.ndarray:
    rows = []
    for value in column.to_pylist():
        if value is None or isinstance(value, (bytes, bytearray, dict)):
            continue
        if isinstance(value, (list, tuple)):
            try:
                rows.append([float(v) for v in value])
            except (TypeError, ValueError):
                continue
        else:
            try:
                rows.append([float(value)])
            except (TypeError, ValueError):
                continue
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    width = max(len(row) for row in rows)
    padded = [row + [0.0] * (width - len(row)) for row in rows]
    return np.asarray(padded, dtype=np.float32)


class PolicyEmbedder:
    def __init__(
        self,
        ckpt_path: Path | None,
        layer_hook: str = "pi_prefix",
        dim: int = 64,
        openpi_config: str | None = None,
        workers: int | None = None,
        devices: str | None = None,
    ):
        self.ckpt_path = Path(ckpt_path).expanduser() if ckpt_path else None
        self.layer_hook = layer_hook or "pi_prefix"
        self.dim = int(dim)
        self.ckpt_hash = checkpoint_hash(self.ckpt_path) if self.ckpt_path is not None else None
        self.openpi_config = resolve_openpi_config(self.ckpt_path, openpi_config)
        self.policy = None
        self.worker = None
        self.workers: list[OpenPISubprocessWorker] = []
        self._worker_queue: queue.Queue[OpenPISubprocessWorker] = queue.Queue()
        if self.layer_hook != FALLBACK_LAYER:
            if OPENPI_EMBED_MODE == "inprocess":
                self.policy = self._load_openpi_policy()
            else:
                if self.ckpt_path is None:
                    raise ValueError(
                        "OpenPI embedding requires a checkpoint path. Use episode_stats_fallback for stats-only embeddings."
                    )
                worker_count = max(1, int(workers or default_openpi_workers()))
                device_list = _split_devices(devices if devices is not None else default_openpi_devices())
                for idx in range(worker_count):
                    device = device_list[idx % len(device_list)] if device_list else None
                    worker = OpenPISubprocessWorker(
                        self.ckpt_path, self.openpi_config, self.layer_hook, device
                    )
                    self.workers.append(worker)
                    self._worker_queue.put(worker)
                self.worker = self.workers[0]

    @property
    def adapter_name(self) -> str:
        if self.layer_hook == FALLBACK_LAYER:
            return FALLBACK_LAYER
        return "openpi_prefix" if OPENPI_EMBED_MODE == "inprocess" else "openpi_prefix_uv"

    @property
    def parallelism(self) -> int:
        return max(1, len(self.workers))

    def _acquire_worker(self) -> OpenPISubprocessWorker:
        if not self.workers:
            raise RuntimeError("OpenPI workers are not loaded")
        return self._worker_queue.get()

    def _release_worker(self, worker: OpenPISubprocessWorker) -> None:
        self._worker_queue.put(worker)

    def close(self) -> None:
        for worker in self.workers:
            worker.close()

    def timing_summary(self) -> dict:
        worker_rows = [row for worker in self.workers for row in worker.timings]
        per_worker = [
            {"device": worker.device, "profile": worker.timing_summary()} for worker in self.workers
        ]
        return {
            "workers": len(self.workers),
            "overall": _summarize_timing_rows(worker_rows),
            "per_worker": per_worker,
        }

    def _load_openpi_policy(self):
        if self.ckpt_path is None:
            raise ValueError(
                "OpenPI embedding requires a checkpoint path. Use episode_stats_fallback for stats-only embeddings."
            )
        _ensure_openpi_importable()
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        train_config = _config.get_config(self.openpi_config)
        return _policy_config.create_trained_policy(train_config, self.ckpt_path)

    def embed_table(self, table) -> np.ndarray:
        features = []
        for name in table.column_names:
            if name.startswith("observation.image") or name.startswith("observation.video"):
                continue
            arr = _column_to_2d(table[name])
            if arr.size == 0:
                continue
            features.extend(np.nanmean(arr, axis=0).tolist())
            features.extend(np.nanstd(arr, axis=0).tolist())
            features.extend(np.nanmin(arr, axis=0).tolist())
            features.extend(np.nanmax(arr, axis=0).tolist())
        if not features:
            features = [0.0]
        vec = np.asarray(features, dtype=np.float32)
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        elif vec.size > self.dim:
            vec = vec[: self.dim]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec

    def embed_episode(self, *, table, root: Path, meta, episode_index: int) -> np.ndarray:
        if self.layer_hook == FALLBACK_LAYER:
            return self.embed_table(table)
        parquet_path = Path(root) / meta.get_data_file_path(episode_index)
        raws = [
            self._raw_frame(
                table=table,
                root=Path(root),
                parquet_path=parquet_path,
                meta=meta,
                episode_index=episode_index,
                frame_index=frame_index,
            )
            for frame_index in self._sample_frame_indices(table.num_rows)
        ]
        if self.worker is not None:
            worker = self._acquire_worker()
            try:
                frame_vectors = worker.embed_many([self._openpi_raw(raw) for raw in raws])
            finally:
                self._release_worker(worker)
        else:
            frame_vectors = [self._embed_openpi_frame(raw) for raw in raws]
        if len(frame_vectors) == 0:
            raise ValueError(f"Episode {episode_index} has no frames to embed")
        vec = np.mean(np.asarray(frame_vectors, dtype=np.float32), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec

    @staticmethod
    def _sample_frame_indices(length: int) -> list[int]:
        if length <= 0:
            return []
        return sorted({0, length // 2, length - 1})

    @staticmethod
    def _as_numpy_value(value):
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return value
        if isinstance(value, list):
            try:
                return np.asarray(value)
            except Exception:
                return value
        if value is None:
            return value
        try:
            return np.asarray(value)
        except Exception:
            return value

    @staticmethod
    def _image_keys(meta) -> list[str]:
        features = getattr(meta, "features", {}) or {}
        return [
            key
            for key, spec in features.items()
            if isinstance(spec, dict) and spec.get("dtype") in {"image", "video"}
        ]

    @staticmethod
    def _image_array(
        root: Path,
        meta,
        episode_index: int,
        image_key: str,
        frame_index: int,
    ) -> np.ndarray:
        image, _ = read_episode_frame_image(
            root,
            meta,
            episode_index,
            frame_index,
            image_key=image_key,
        )
        return np.asarray(image)

    @staticmethod
    def _episode_task(meta, row: dict, episode_index: int) -> tuple[str, int]:
        task_index = row.get("task_index")
        try:
            task_index_int = int(np.asarray(task_index).item())
        except Exception:
            task_index_int = 0
        tasks_by_index = getattr(meta, "tasks", {}) or {}
        task_text = str(tasks_by_index.get(task_index_int, ""))
        episode = (getattr(meta, "episodes", {}) or {}).get(episode_index) or {}
        episode_tasks = episode.get("tasks") or []
        if episode_tasks:
            task_text = str(episode_tasks[0])
            for idx, text in tasks_by_index.items():
                if str(text) == task_text:
                    task_index_int = int(idx)
                    break
        if not task_text:
            task_text = str(row.get("prompt") or row.get("task") or "")
        return task_text, task_index_int

    def _raw_frame(
        self, *, table, root: Path, parquet_path: Path, meta, episode_index: int, frame_index: int
    ) -> dict:
        row = table.slice(frame_index, 1).to_pylist()[0]
        raw = {key: self._as_numpy_value(value) for key, value in row.items()}
        for image_key in self._image_keys(meta):
            raw[image_key] = self._image_array(
                root,
                meta,
                episode_index,
                image_key,
                frame_index,
            )
        self._add_image_aliases(raw)
        self._add_state_action_aliases(raw)
        task_text, task_index = self._episode_task(meta, row, episode_index)
        raw.setdefault("prompt", task_text)
        raw["task_index"] = np.asarray(task_index, dtype=np.int32)
        raw["episode_index"] = np.asarray(episode_index, dtype=np.int32)
        raw["frame_index"] = np.asarray(frame_index, dtype=np.int32)
        raw["episode_length"] = np.asarray(table.num_rows, dtype=np.int32)
        raw.setdefault("timestamp", np.asarray(float(frame_index), dtype=np.float32))
        return raw

    @staticmethod
    def _first_existing(raw: dict, keys: tuple[str, ...]):
        for key in keys:
            if key in raw:
                return raw[key]
        return None

    def _add_image_aliases(self, raw: dict) -> None:
        base = self._first_existing(
            raw,
            (
                "observation/image",
                "image",
                "observation.images.camera_01",
                "observation.images.camera_top",
                "observation.images.top",
                "observation.images.cam_high",
                "observation.images.cam_low",
                "observation.image",
                "observation.images.front",
            ),
        )
        left = self._first_existing(
            raw,
            (
                "observation/wrist_image_left",
                "left_wrist_image",
                "observation.images.camera_left_wrist",
                "observation.wrist_image_left",
                "observation.images.left_wrist",
            ),
        )
        right = self._first_existing(
            raw,
            (
                "observation/wrist_image_right",
                "right_wrist_image",
                "observation.images.camera_right_wrist",
                "observation.wrist_image_right",
                "observation.images.right_wrist",
            ),
        )
        if base is None:
            base = left if left is not None else right
        if left is None:
            left = base
        if right is None:
            right = base
        if base is not None:
            raw.setdefault("observation/image", base)
            raw.setdefault("image", base)
            raw.setdefault("observation.images.camera_01", base)
        if left is not None:
            raw.setdefault("observation/wrist_image_left", left)
            raw.setdefault("left_wrist_image", left)
            raw.setdefault("observation.images.camera_left_wrist", left)
        if right is not None:
            raw.setdefault("observation/wrist_image_right", right)
            raw.setdefault("right_wrist_image", right)
            raw.setdefault("observation.images.camera_right_wrist", right)

    def _add_state_action_aliases(self, raw: dict) -> None:
        state = self._first_existing(raw, ("state", "observation/state", "observation.state"))
        action = self._first_existing(raw, ("action", "actions"))
        if state is not None:
            raw.setdefault("state", state)
            raw.setdefault("observation/state", state)
            raw.setdefault("observation.state", state)
        if action is not None:
            raw.setdefault("action", action)
            raw.setdefault("actions", action)

    @staticmethod
    def _openpi_raw(raw: dict) -> dict:
        raw = dict(raw)
        raw.pop("actions", None)
        return raw

    def _embed_openpi_frame(self, raw: dict) -> np.ndarray:
        raw = self._openpi_raw(raw)
        if self.worker is not None:
            worker = self._acquire_worker()
            try:
                return worker.embed(raw)
            finally:
                self._release_worker(worker)
        if self.policy is None:
            raise RuntimeError("OpenPI policy was not loaded")
        if getattr(self.policy, "_is_pytorch_model", False):
            return self._embed_openpi_frame_torch(raw)
        return self._embed_openpi_frame_jax(raw)

    def _pool_prefix(
        self, prefix_out: np.ndarray, prefix_mask: np.ndarray, prompt_len: int = 0, stage_tokens: int = 0
    ) -> np.ndarray:
        prefix_out = np.asarray(prefix_out, dtype=np.float32)
        prefix_mask = np.asarray(prefix_mask).astype(bool)
        if prefix_out.ndim == 3:
            prefix_out = prefix_out[0]
        if prefix_mask.ndim == 2:
            prefix_mask = prefix_mask[0]
        if self.layer_hook == "vision_encoder" and prompt_len:
            end = max(prefix_out.shape[0] - prompt_len - stage_tokens, 1)
            prefix_out = prefix_out[:end]
            prefix_mask = prefix_mask[:end]
        elif self.layer_hook == "pi_prefix_prompt" and prompt_len:
            start = max(prefix_out.shape[0] - prompt_len - stage_tokens, 0)
            end = max(prefix_out.shape[0] - stage_tokens, start + 1)
            prefix_out = prefix_out[start:end]
            prefix_mask = prefix_mask[start:end]
        weights = prefix_mask.astype(np.float32)
        denom = max(float(weights.sum()), 1.0)
        return ((prefix_out * weights[:, None]).sum(axis=0) / denom).astype(np.float32)

    def _embed_openpi_frame_jax(self, raw: dict) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        inputs = self.policy._input_transform(dict(raw))
        inputs.pop("prompt_text", None)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.policy._model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.policy._model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        prompt_len = (
            int(observation.tokenized_prompt.shape[1]) if observation.tokenized_prompt is not None else 0
        )
        stage_tokens = (
            4
            if getattr(self.policy._model, "use_stage_fusion", False)
            and observation.subtask_state is not None
            else 0
        )
        return self._pool_prefix(np.asarray(prefix_out), np.asarray(prefix_mask), prompt_len, stage_tokens)

    def _embed_openpi_frame_torch(self, raw: dict) -> np.ndarray:
        import jax
        from openpi.models import model as _model
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        inputs = self.policy._input_transform(dict(raw))
        inputs.pop("prompt_text", None)
        device = self.policy._pytorch_device
        torch_inputs = dict(inputs)
        torch_inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
            torch_inputs,
        )
        observation = _model.Observation.from_dict(torch_inputs)
        model = self.policy._model
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(
                observation, train=False
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            att_4d = model._prepare_attention_masks_4d(att_2d)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            outputs_embeds, _ = model.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            prefix_out = outputs_embeds[0].detach().float().cpu().numpy()
            prefix_mask = prefix_pad_masks.detach().cpu().numpy()
        prompt_len = int(lang_tokens.shape[1]) if lang_tokens is not None else 0
        return self._pool_prefix(prefix_out, prefix_mask, prompt_len, 0)
