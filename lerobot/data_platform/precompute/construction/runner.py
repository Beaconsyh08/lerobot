from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.data_platform.precompute.construction.selector import select_sources, summarize_candidates
from lerobot.data_platform.precompute.construction.types import ConstructionPlan
from lerobot.data_platform.precompute.construction.vocab import build_vocab
from lerobot.data_platform.precompute.construction.writer import write_synthetic_dataset
from lerobot.data_platform.precompute.dataset_io import V3DatasetMetadata, is_v3_dataset
from lerobot.data_platform.precompute.labeling.review import labels_path, load_labels_jsonl, reviewed_path
from lerobot.data_platform.precompute.preprocess.dataset_version import (
    DEFAULT_V3_CONVERT_WORKERS,
    materialize_v21_from_v3,
    run_convert_v3,
)
from lerobot.data_platform.precompute.tagging.review import current_tags


@dataclass
class ConstructionResult:
    src_root: Path
    out_root: Path
    repo_id: str
    vocab: list[str]
    plans: list[ConstructionPlan]
    include_positives: bool
    preview: dict


def default_synthetic_path(src_root: Path) -> Path:
    """Return the timestamped output path, preserving the legacy ``_synthetic_`` suffix."""
    src_root = Path(src_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return src_root.parent / f"{src_root.name}_synthetic_{timestamp}"


def _emit(progress_callback: Callable[[dict], None] | None, **payload) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def load_current_labels(labeling_dir: Path) -> dict[int, dict]:
    labeling_dir = Path(labeling_dir)
    labels_file = labels_path(labeling_dir)
    if not labels_file.is_file():
        raise FileNotFoundError("object_labeling_required")
    labels = load_labels_jsonl(labels_file)
    reviewed = load_labels_jsonl(reviewed_path(labeling_dir))
    labels.update(reviewed)
    return labels


def load_current_tags_for_construction(labeling_dir: Path) -> dict[int, dict]:
    tagging_dir = Path(labeling_dir).parent / "tagging"
    try:
        return current_tags(tagging_dir)
    except FileNotFoundError:
        return {}


def preview_construction(
    meta,
    labeling_dir: Path,
    uncertainty_threshold: int = 50,
    allow_pick_to_give: bool = False,
) -> dict:
    vocab = build_vocab(meta)
    labels = load_current_labels(labeling_dir)
    tags = load_current_tags_for_construction(labeling_dir)
    return {
        "vocab": sorted(vocab),
        "scenarios": summarize_candidates(
            labels,
            vocab,
            int(uncertainty_threshold),
            tags_by_episode=tags,
            allow_pick_to_give=allow_pick_to_give,
        ),
        "tagging_object_count_ready": bool(tags),
    }


def _config_value(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def run_construction(
    src_root: Path,
    meta,
    labeling_dir: Path,
    out_root: Path | None,
    config: dict | None,
    progress_callback: Callable[[dict], None] | None = None,
) -> ConstructionResult:
    config = config or {}
    src_root = Path(src_root)
    out_root = Path(out_root) if out_root is not None else default_synthetic_path(src_root)
    threshold = int(_config_value(config, "uncertainty_threshold", 50))
    per_scenario_counts = {
        str(key): int(value or 0)
        for key, value in dict(_config_value(config, "per_scenario_counts", {})).items()
    }
    include_positives = bool(_config_value(config, "include_positives", False))
    oversample_factor = float(_config_value(config, "oversample_factor", 1.0))
    allow_pick_to_give = bool(_config_value(config, "allow_pick_to_give", False))

    _emit(
        progress_callback,
        status="running",
        step="construction_preview",
        current=0,
        total=1,
        message="Building construction preview",
    )
    vocab = build_vocab(meta)
    labels = load_current_labels(labeling_dir)
    tags = load_current_tags_for_construction(labeling_dir)
    preview = {
        "vocab": sorted(vocab),
        "scenarios": summarize_candidates(
            labels,
            vocab,
            threshold,
            tags_by_episode=tags,
            allow_pick_to_give=allow_pick_to_give,
        ),
        "tagging_object_count_ready": bool(tags),
    }
    plans = select_sources(
        labels,
        vocab,
        threshold,
        per_scenario_counts,
        oversample_factor=oversample_factor,
        tags_by_episode=tags,
        allow_pick_to_give=allow_pick_to_give,
    )
    _emit(
        progress_callback,
        status="running",
        step="construction_select",
        current=0,
        total=max(1, len(plans)),
        message=f"Selected {len(plans)} constructed negatives",
    )

    source_repo_id = getattr(meta, "repo_id", f"local/{src_root.name}")
    if is_v3_dataset(src_root):
        with tempfile.TemporaryDirectory(
            prefix=".lerobot-v3-construction-",
            dir=out_root.parent,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            legacy_source = materialize_v21_from_v3(
                src_root,
                temp_root / "source",
                workers=DEFAULT_V3_CONVERT_WORKERS,
                progress_callback=progress_callback,
            )
            legacy_output = temp_root / "output"
            write_result = write_synthetic_dataset(
                legacy_source,
                plans,
                legacy_output,
                include_positives,
                progress_callback=progress_callback,
                source_repo_id=source_repo_id,
            )
            run_convert_v3(
                legacy_output,
                out_root,
                workers=DEFAULT_V3_CONVERT_WORKERS,
                progress_callback=progress_callback,
            )
        plan_path = out_root / "meta" / "construction_plan.json"
        if plan_path.is_file():
            plan_doc = json.loads(plan_path.read_text())
            plan_doc["source_root"] = str(src_root)
            plan_doc["source_repo_id"] = source_repo_id
            plan_doc["source_format"] = "v3.0"
            plan_path.write_text(json.dumps(plan_doc, indent=2, ensure_ascii=False) + "\n")
        write_result = {**write_result, "out_root": out_root}
    else:
        write_result = write_synthetic_dataset(
            src_root,
            plans,
            out_root,
            include_positives,
            progress_callback=progress_callback,
            source_repo_id=source_repo_id,
        )

    # Load the result through LeRobot metadata validation before registering it in the web app.
    if is_v3_dataset(out_root):
        V3DatasetMetadata(repo_id=f"local/{out_root.name}", root=out_root)
    else:
        LeRobotDatasetMetadata(repo_id=f"local/{out_root.name}", root=out_root)

    _emit(
        progress_callback,
        status="done",
        step="construction_done",
        current=len(plans),
        total=len(plans),
        message=f"Data construction complete: {out_root}",
    )
    return ConstructionResult(
        src_root=src_root,
        out_root=Path(write_result["out_root"]),
        repo_id=f"local/{Path(write_result['out_root']).name}",
        vocab=sorted(vocab),
        plans=plans,
        include_positives=include_positives,
        preview=preview,
    )
