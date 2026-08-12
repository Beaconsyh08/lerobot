from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.data_platform.precompute.dataset_io import read_episode_table
from lerobot.data_platform.precompute.embedding.policy_backend import PolicyEmbedder, get_capabilities
from lerobot.data_platform.precompute.embedding.reducer import (
    fit_reducer,
    load_reducer,
    reducer_name,
    save_reducer,
    transform,
)


@dataclass
class EmbeddingResult:
    root: Path
    repo_id: str
    static_dir: Path
    embedding_dir: Path
    episodes: list[int]
    points: int


def _projection_options(
    *,
    method: str = "auto",
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> dict:
    return {
        "method": (method or "auto").strip().lower(),
        "seed": int(seed),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": (metric or "euclidean").strip().lower(),
    }


def _emit(progress_callback: Callable[[dict], None] | None, **payload) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _load_existing(embedding_dir: Path) -> tuple[list[int], np.ndarray] | None:
    path = embedding_dir / "embeddings.npz"
    if not path.is_file():
        return None
    data = np.load(path)
    return data["episode_index"].astype(int).tolist(), data["embeddings"].astype(np.float32)


def _load_source(embedding_dir: Path) -> dict:
    path = embedding_dir / "source.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_projection(
    embedding_dir: Path,
    episode_indices: list[int],
    embeddings: np.ndarray,
    options: dict,
) -> dict:
    reducer = fit_reducer(
        embeddings,
        seed=options["seed"],
        method=options["method"],
        n_neighbors=options["n_neighbors"],
        min_dist=options["min_dist"],
        metric=options["metric"],
    )
    save_reducer(embedding_dir / "reducer.pkl", reducer)
    coords = transform(reducer, embeddings)
    np.savez_compressed(
        embedding_dir / "coords_2d.npz",
        episode_index=np.asarray(episode_indices, dtype=np.int64),
        coords=coords,
    )
    return {**options, "actual_method": reducer_name(reducer), "points": int(len(episode_indices))}


def project_existing_embeddings(
    static_dir: Path,
    *,
    method: str = "auto",
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> dict:
    embedding_dir = Path(static_dir) / "embedding"
    existing = _load_existing(embedding_dir)
    if existing is None:
        raise FileNotFoundError(f"Missing existing embeddings at {embedding_dir / 'embeddings.npz'}")
    episode_indices, embeddings = existing
    options = _projection_options(
        method=method,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    projection = _save_projection(embedding_dir, episode_indices, embeddings, options)
    source = _load_source(embedding_dir)
    source["projection"] = projection
    (embedding_dir / "source.json").write_text(json.dumps(source, indent=2))
    return projection


def _source_matches(source: dict, embedder: PolicyEmbedder, layer_hook: str) -> bool:
    return (
        source.get("adapter") == embedder.adapter_name
        and source.get("layer_hook") == layer_hook
        and source.get("openpi_config") == embedder.openpi_config
        and source.get("ckpt_hash") == embedder.ckpt_hash
    )


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


def run_embedding(
    root: Path,
    meta,
    episodes: list[int] | None,
    static_dir: Path,
    ckpt_path: Path | None,
    layer_hook: str = "pi_prefix",
    openpi_config: str | None = None,
    refit: bool = False,
    workers: int | None = None,
    devices: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> EmbeddingResult:
    root = Path(root)
    static_dir = Path(static_dir)
    embedding_dir = static_dir / "embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    selected_episodes = sorted(getattr(meta, "episodes", {}).keys()) if episodes is None else list(episodes)
    embedder = PolicyEmbedder(
        ckpt_path, layer_hook, openpi_config=openpi_config, workers=workers, devices=devices
    )

    source = _load_source(embedding_dir)
    source_compatible = _source_matches(source, embedder, layer_hook)
    existing = None if refit or not source_compatible else _load_existing(embedding_dir)
    vectors_by_episode: dict[int, np.ndarray] = {}
    if existing is not None:
        old_episode_indices, old_vectors = existing
        vectors_by_episode.update({int(ep): old_vectors[i] for i, ep in enumerate(old_episode_indices)})

    to_compute = [
        episode_index for episode_index in selected_episodes if int(episode_index) not in vectors_by_episode
    ]
    episode_timings: list[dict] = []

    def _compute_episode(episode_index: int) -> tuple[int, np.ndarray, dict]:
        start = time.perf_counter()
        read_start = time.perf_counter()
        table = read_episode_table(root, meta, episode_index)
        read_s = time.perf_counter() - read_start
        embed_start = time.perf_counter()
        vector = embedder.embed_episode(
            table=table,
            root=root,
            meta=meta,
            episode_index=int(episode_index),
        )
        embed_s = time.perf_counter() - embed_start
        timing = {
            "episode_index": int(episode_index),
            "parquet_read_s": read_s,
            "embed_episode_s": embed_s,
            "episode_total_s": time.perf_counter() - start,
        }
        return int(episode_index), vector, timing

    completed = len(selected_episodes) - len(to_compute)
    _emit(
        progress_callback,
        status="running",
        step="embedding",
        current=completed,
        total=len(selected_episodes),
        message=f"Starting embedding extraction with {embedder.parallelism} worker(s)",
    )
    try:
        if to_compute and embedder.parallelism > 1:
            with ThreadPoolExecutor(max_workers=embedder.parallelism) as executor:
                futures = {
                    executor.submit(_compute_episode, int(episode_index)): int(episode_index)
                    for episode_index in to_compute
                }
                for future in as_completed(futures):
                    episode_index, vector, timing = future.result()
                    vectors_by_episode[episode_index] = vector
                    episode_timings.append(timing)
                    completed += 1
                    _emit(
                        progress_callback,
                        status="running",
                        step="embedding",
                        current=completed,
                        total=len(selected_episodes),
                        message=(
                            f"Embedded episode {episode_index} "
                            f"(total {timing['episode_total_s']:.2f}s, read {timing['parquet_read_s']:.2f}s, "
                            f"embed {timing['embed_episode_s']:.2f}s)"
                        ),
                    )
        else:
            for episode_index in to_compute:
                episode_index, vector, timing = _compute_episode(int(episode_index))
                vectors_by_episode[episode_index] = vector
                episode_timings.append(timing)
                completed += 1
                _emit(
                    progress_callback,
                    status="running",
                    step="embedding",
                    current=completed,
                    total=len(selected_episodes),
                    message=(
                        f"Embedded episode {episode_index} "
                        f"(total {timing['episode_total_s']:.2f}s, read {timing['parquet_read_s']:.2f}s, "
                        f"embed {timing['embed_episode_s']:.2f}s)"
                    ),
                )
    finally:
        worker_profile = embedder.timing_summary()
        embedder.close()

    profile = {
        "episodes": _summarize_timing_rows(episode_timings),
        "workers": worker_profile,
        "computed_episodes": len(to_compute),
        "cached_episodes": len(selected_episodes) - len(to_compute),
    }

    all_episodes = sorted(vectors_by_episode)
    embeddings = np.stack([vectors_by_episode[episode_index] for episode_index in all_episodes]).astype(
        np.float32
    )
    reducer_path = embedding_dir / "reducer.pkl"
    if reducer_path.is_file() and not refit and source_compatible:
        reducer = load_reducer(reducer_path)
    else:
        reducer = fit_reducer(embeddings)
        save_reducer(reducer_path, reducer)
    coords = transform(reducer, embeddings)
    projection = {
        "method": "auto",
        "actual_method": reducer_name(reducer),
        "seed": 42,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "points": int(len(all_episodes)),
    }

    np.savez_compressed(
        embedding_dir / "embeddings.npz",
        episode_index=np.asarray(all_episodes, dtype=np.int64),
        embeddings=embeddings,
    )
    np.savez_compressed(
        embedding_dir / "coords_2d.npz", episode_index=np.asarray(all_episodes, dtype=np.int64), coords=coords
    )
    (embedding_dir / "embedding_profile.json").write_text(json.dumps(profile, indent=2))
    (embedding_dir / "source.json").write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "repo_id": getattr(meta, "repo_id", f"local/{root.name}"),
                "ckpt_path": str(Path(ckpt_path).expanduser()) if ckpt_path else None,
                "ckpt_hash": embedder.ckpt_hash,
                "layer_hook": layer_hook,
                "openpi_config": embedder.openpi_config,
                "adapter": embedder.adapter_name,
                "workers": embedder.parallelism,
                "devices": devices,
                "refit": bool(refit),
                "projection": projection,
                "profile": profile,
                "capabilities": get_capabilities(),
            },
            indent=2,
        )
    )
    _emit(
        progress_callback,
        status="done",
        step="embedding_done",
        current=len(all_episodes),
        total=len(all_episodes),
        message="Embedding complete",
    )
    return EmbeddingResult(
        root=root,
        repo_id=getattr(meta, "repo_id", f"local/{root.name}"),
        static_dir=static_dir,
        embedding_dir=embedding_dir,
        episodes=all_episodes,
        points=len(all_episodes),
    )
