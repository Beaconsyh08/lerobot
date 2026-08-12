from __future__ import annotations

import io
import json
import re
from pathlib import Path

from lerobot.data_platform.precompute.dataset_io import (
    read_episode_frame_image,
    update_episode_metadata,
)

_VARIANT_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def normalize_label_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    normalized = _VARIANT_RE.sub("_", str(variant).strip()).strip("._-")
    if normalized == "latest":
        return None
    return normalized or None


def labels_path(labeling_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_label_variant(variant)
    if variant is None:
        return Path(labeling_dir) / "labels.jsonl"
    return Path(labeling_dir) / f"labels_{variant}.jsonl"


def reviewed_path(labeling_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_label_variant(variant)
    if variant is None:
        return Path(labeling_dir) / "labels_reviewed.jsonl"
    return Path(labeling_dir) / f"labels_reviewed_{variant}.jsonl"


def source_path(labeling_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_label_variant(variant)
    if variant is None:
        return Path(labeling_dir) / "source.json"
    return Path(labeling_dir) / f"source_{variant}.json"


def _source_backend(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text()).get("backend")
    except (json.JSONDecodeError, OSError, AttributeError):
        return None
    return normalize_label_variant(value)


def latest_label_variant(labeling_dir: Path) -> str | None:
    return _source_backend(source_path(labeling_dir))


def resolved_labels_path(labeling_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_label_variant(variant)
    if variant is None:
        return labels_path(labeling_dir)
    path = labels_path(labeling_dir, variant)
    if path.is_file():
        return path
    if latest_label_variant(labeling_dir) == variant and labels_path(labeling_dir).is_file():
        return labels_path(labeling_dir)
    return path


def resolved_reviewed_path(labeling_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_label_variant(variant)
    if variant is None:
        return reviewed_path(labeling_dir)
    path = reviewed_path(labeling_dir, variant)
    if path.is_file():
        return path
    if latest_label_variant(labeling_dir) == variant and reviewed_path(labeling_dir).is_file():
        return reviewed_path(labeling_dir)
    return path


def migrate_latest_labels_to_variant(labeling_dir: Path) -> None:
    labeling_dir = Path(labeling_dir)
    variant = latest_label_variant(labeling_dir)
    if variant is None:
        return
    for generic, specific in (
        (labels_path(labeling_dir), labels_path(labeling_dir, variant)),
        (reviewed_path(labeling_dir), reviewed_path(labeling_dir, variant)),
        (source_path(labeling_dir), source_path(labeling_dir, variant)),
    ):
        if generic.is_file() and not specific.is_file():
            specific.write_bytes(generic.read_bytes())


def available_label_variants(labeling_dir: Path) -> list[dict]:
    labeling_dir = Path(labeling_dir)
    latest_variant = latest_label_variant(labeling_dir)
    variants: dict[str, dict] = {}

    if labels_path(labeling_dir).is_file() or reviewed_path(labeling_dir).is_file():
        variant = latest_variant or ""
        variants[variant] = {
            "id": variant,
            "label": variant or "latest",
            "is_latest": True,
        }

    for path in labeling_dir.glob("labels_*.jsonl"):
        if path.name.startswith("labels_reviewed"):
            continue
        variant = normalize_label_variant(path.stem.removeprefix("labels_"))
        if variant is None:
            continue
        variants.setdefault(
            variant,
            {
                "id": variant,
                "label": variant,
                "is_latest": variant == latest_variant,
            },
        )
        variants[variant]["is_latest"] = variants[variant].get("is_latest") or variant == latest_variant

    for path in labeling_dir.glob("labels_reviewed_*.jsonl"):
        variant = normalize_label_variant(path.stem.removeprefix("labels_reviewed_"))
        if variant is None:
            continue
        variants.setdefault(
            variant,
            {
                "id": variant,
                "label": variant,
                "is_latest": variant == latest_variant,
            },
        )
        variants[variant]["is_latest"] = variants[variant].get("is_latest") or variant == latest_variant

    out = []
    for variant, info in variants.items():
        labels_file = resolved_labels_path(labeling_dir, variant)
        reviewed_file = resolved_reviewed_path(labeling_dir, variant)
        source_file = source_path(labeling_dir, variant)
        labels_count = len(load_labels_jsonl(labels_file))
        reviewed_count = len(load_labels_jsonl(reviewed_file))
        if not source_file.is_file() and info.get("is_latest"):
            source_file = source_path(labeling_dir)
        out.append(
            {
                **info,
                "labels_path": str(labels_file),
                "reviewed_path": str(reviewed_file),
                "labels_count": max(labels_count, reviewed_count),
                "reviewed_count": reviewed_count,
            }
        )

    preferred = {"grounding_dino": 0, "qwen_remote": 1, "": 2, "latest": 2}
    out.sort(key=lambda item: (0 if item.get("is_latest") else 1, preferred.get(item["id"], 99), item["id"]))
    return out


def load_labels_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    path = Path(path)
    if path.is_dir():
        path = labels_path(path)
    if not path.is_file():
        return records
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[int(record["episode_index"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return records


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def save_reviewed_record(labeling_dir: Path, episode_index: int, record: dict) -> None:
    save_reviewed_record_for_variant(labeling_dir, episode_index, record, variant=None)


def save_reviewed_record_for_variant(
    labeling_dir: Path,
    episode_index: int,
    record: dict,
    variant: str | None = None,
) -> None:
    labeling_dir = Path(labeling_dir)
    variant = normalize_label_variant(variant)
    path = reviewed_path(labeling_dir, variant)
    reviewed = load_labels_jsonl(resolved_reviewed_path(labeling_dir, variant))
    reviewed[int(episode_index)] = record
    _write_jsonl_atomic(path, [reviewed[idx] for idx in sorted(reviewed)])
    if variant is not None and latest_label_variant(labeling_dir) == variant:
        _write_jsonl_atomic(reviewed_path(labeling_dir), [reviewed[idx] for idx in sorted(reviewed)])


def remove_reviewed_record(labeling_dir: Path, episode_index: int) -> None:
    remove_reviewed_record_for_variant(labeling_dir, episode_index, variant=None)


def remove_reviewed_record_for_variant(
    labeling_dir: Path, episode_index: int, variant: str | None = None
) -> None:
    labeling_dir = Path(labeling_dir)
    variant = normalize_label_variant(variant)
    path = reviewed_path(labeling_dir, variant)
    reviewed = load_labels_jsonl(resolved_reviewed_path(labeling_dir, variant))
    reviewed.pop(int(episode_index), None)
    _write_jsonl_atomic(path, [reviewed[idx] for idx in sorted(reviewed)])
    if variant is not None and latest_label_variant(labeling_dir) == variant:
        _write_jsonl_atomic(reviewed_path(labeling_dir), [reviewed[idx] for idx in sorted(reviewed)])


def uncertainty(record: dict) -> int:
    if record.get("skip_reason") == "exist_label_zero":
        return -1
    parsed = record.get("parsed")
    if parsed is None:
        return -1
    if record.get("selected") is None:
        return 100
    if record.get("relation_satisfied") is False:
        return 90
    detections = record.get("detections_target", [])
    if not detections:
        return 95
    confidence_source = (detections or [{}])[0].get("confidence_source")
    if confidence_source == "vlm_implicit":
        return 30
    top1 = detections[0]["confidence"]
    if confidence_source == "vlm_self_reported":
        if top1 < 0.45:
            return 85
        if top1 < 0.65:
            return 70
        if len(detections) >= 2 and (top1 - detections[1]["confidence"]) < 0.10:
            return 60
        return 30
    if top1 < 0.25:
        return 80
    if top1 < 0.30:
        return 70
    if len(detections) >= 2:
        gap = top1 - detections[1]["confidence"]
        if gap < 0.03:
            return 65
        if gap < 0.05:
            return 55
    if parsed.get("direction") and parsed.get("reference") is None:
        xs = sorted((d["bbox"]["left"] + d["bbox"]["right"]) / 2 for d in detections[:3])
        if len(xs) >= 2 and xs[-1] - xs[0] < 30:
            return 50
    return 10


def reason(record: dict) -> str:
    if record.get("skip_reason") == "exist_label_zero":
        return "exist_label=0"
    parsed = record.get("parsed")
    if parsed is None:
        return "not Pick-up"
    if record.get("selected") is None:
        if record.get("relation_satisfied") is False:
            return "no candidate on correct side"
        return "no detection"
    if record.get("relation_satisfied") is False:
        return "relation unsatisfied (fallback)"
    detections = record.get("detections_target", [])
    if not detections:
        return "no target candidates"
    top1 = detections[0]["confidence"]
    confidence_source = detections[0].get("confidence_source")
    if confidence_source == "vlm_implicit":
        return "vlm implicit conf"
    if confidence_source == "vlm_self_reported":
        if top1 < 0.65:
            return f"low vlm self-conf {top1:.2f}"
        if len(detections) >= 2 and (top1 - detections[1]["confidence"]) < 0.10:
            return f"close vlm self-conf (d={top1 - detections[1]['confidence']:.2f})"
        return f"vlm self-conf {top1:.2f}"
    if top1 < 0.30:
        return f"low conf {top1:.2f}"
    if len(detections) >= 2 and (top1 - detections[1]["confidence"]) < 0.05:
        return f"close top-2 (d={top1 - detections[1]['confidence']:.2f})"
    if parsed.get("direction") and parsed.get("reference") is None:
        xs = sorted((d["bbox"]["left"] + d["bbox"]["right"]) / 2 for d in detections[:3])
        if len(xs) >= 2 and xs[-1] - xs[0] < 30:
            return "absolute: x_centers close"
    return "ok"


def load_episode_record(labeling_dir: Path, episode_index: int, variant: str | None = None) -> dict | None:
    labeling_dir = Path(labeling_dir)
    variant = normalize_label_variant(variant)
    originals = load_labels_jsonl(resolved_labels_path(labeling_dir, variant))
    reviewed = load_labels_jsonl(resolved_reviewed_path(labeling_dir, variant))
    original = originals.get(int(episode_index)) or reviewed.get(int(episode_index))
    if original is None:
        return None
    return {
        "original": original,
        "current": reviewed.get(int(episode_index), original),
        "reason": reason(original),
        "reviewed": int(episode_index) in reviewed,
    }


def image_keys_from_meta(meta) -> list[str]:
    return [
        key
        for key, feature in getattr(meta, "features", {}).items()
        if feature.get("dtype") in {"image", "video"}
    ]


def read_first_frame_image(root: Path, meta, episode_index: int, image_key: str | None = None):
    return read_frame_image(root, meta, episode_index, 0, image_key=image_key)


def read_frame_image(root: Path, meta, episode_index: int, frame_index: int, image_key: str | None = None):
    image_keys = image_keys_from_meta(meta)
    if image_key is None:
        if not image_keys:
            raise ValueError("Object labeling requires at least one image feature in dataset metadata.")
        image_key = image_keys[0]
    elif image_key not in image_keys:
        raise ValueError(f"Image key is not an image feature: {image_key}")

    return read_episode_frame_image(
        Path(root),
        meta,
        int(episode_index),
        int(frame_index),
        image_key=image_key,
    )


def read_first_frame_jpeg(root: Path, meta, episode_index: int, image_key: str | None = None) -> bytes:
    image, _ = read_first_frame_image(root, meta, episode_index, image_key=image_key)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92)
    out.seek(0)
    return out.getvalue()


def _first_or_none(items: list[dict]) -> dict | None:
    return items[0] if items else None


def first_frame_bbox_from_record(record: dict, source: str = "reviewed") -> dict:
    return {
        "all_target": record.get("detections_all_target", record.get("detections_target", [])),
        "target": _first_or_none(record.get("detections_target", [])),
        "reference": _first_or_none(record.get("detections_ref", [])),
        "selected": record.get("selected"),
        "selected_target": record.get("selected_target", record.get("selected")),
        "target_selection_method": record.get("target_selection_method"),
        "active_arm": record.get("active_arm"),
        "relation_satisfied": record.get("relation_satisfied"),
        "source": source,
    }


def merge_reviewed_labels_to_metadata(root: Path, labeling_dir: Path, variant: str | None = None) -> dict:
    root = Path(root)
    labeling_dir = Path(labeling_dir)
    reviewed_file = resolved_reviewed_path(labeling_dir, variant)
    if not reviewed_file.is_file():
        raise FileNotFoundError(f"Reviewed labels not found: {reviewed_file}")

    reviewed = load_labels_jsonl(reviewed_file)
    updates = {
        int(episode_index): {"first_frame_bbox": first_frame_bbox_from_record(record, source="reviewed")}
        for episode_index, record in reviewed.items()
    }
    merged_episodes = update_episode_metadata(root, updates)
    return {
        "merged": len(merged_episodes),
        "episodes": merged_episodes,
        "episodes_path": str(
            root / "meta" / ("episodes" if (root / "meta" / "episodes").is_dir() else "episodes.jsonl")
        ),
    }
