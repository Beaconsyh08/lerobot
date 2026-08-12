from __future__ import annotations

import json
import re
from pathlib import Path

from lerobot.data_platform.precompute.dataset_io import update_episode_metadata
from lerobot.data_platform.precompute.tagging.schema import normalize_tag_values

_VARIANT_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def normalize_tag_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    normalized = _VARIANT_RE.sub("_", str(variant).strip()).strip("._-")
    return normalized or None


def tags_path(tagging_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_tag_variant(variant)
    if variant is None:
        return Path(tagging_dir) / "tags.jsonl"
    return Path(tagging_dir) / f"tags_{variant}.jsonl"


def reviewed_path(tagging_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_tag_variant(variant)
    if variant is None:
        return Path(tagging_dir) / "tags_reviewed.jsonl"
    return Path(tagging_dir) / f"tags_reviewed_{variant}.jsonl"


def source_path(tagging_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_tag_variant(variant)
    if variant is None:
        return Path(tagging_dir) / "source.json"
    return Path(tagging_dir) / f"source_{variant}.json"


def _source_variant(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text()).get("output_variant")
    except (json.JSONDecodeError, OSError, AttributeError):
        return None
    return normalize_tag_variant(value)


def latest_tag_variant(tagging_dir: Path) -> str | None:
    return _source_variant(source_path(tagging_dir))


def resolved_tags_path(tagging_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_tag_variant(variant)
    if variant is None:
        return tags_path(tagging_dir)
    path = tags_path(tagging_dir, variant)
    if path.is_file():
        return path
    if latest_tag_variant(tagging_dir) == variant and tags_path(tagging_dir).is_file():
        return tags_path(tagging_dir)
    return path


def resolved_reviewed_path(tagging_dir: Path, variant: str | None = None) -> Path:
    variant = normalize_tag_variant(variant)
    if variant is None:
        return reviewed_path(tagging_dir)
    path = reviewed_path(tagging_dir, variant)
    if path.is_file():
        return path
    if latest_tag_variant(tagging_dir) == variant and reviewed_path(tagging_dir).is_file():
        return reviewed_path(tagging_dir)
    return path


def available_tag_variants(tagging_dir: Path) -> list[dict]:
    tagging_dir = Path(tagging_dir)
    variants: dict[str, dict] = {}

    if tags_path(tagging_dir).is_file():
        variants["latest"] = {"id": "latest", "label": "latest"}

    for path in tagging_dir.glob("tags_*.jsonl"):
        if path.name.startswith("tags_reviewed"):
            continue
        variant = normalize_tag_variant(path.stem.removeprefix("tags_"))
        if variant is None:
            continue
        variants.setdefault(variant, {"id": variant, "label": variant})

    out = []
    for variant, info in variants.items():
        if variant == "latest":
            tags_file = tags_path(tagging_dir)
            reviewed_file = reviewed_path(tagging_dir)
        else:
            tags_file = resolved_tags_path(tagging_dir, variant)
            reviewed_file = resolved_reviewed_path(tagging_dir, variant)
        try:
            modified_at = tags_file.stat().st_mtime
        except OSError:
            modified_at = 0.0
        out.append(
            {
                **info,
                "tags_path": str(tags_file),
                "reviewed_path": str(reviewed_file),
                "tags_count": len(load_tags_jsonl(tags_file)),
                "reviewed_count": len(load_tags_jsonl(reviewed_file)),
                "modified_at": modified_at,
            }
        )

    latest_modified_at = max((item["modified_at"] for item in out), default=0.0)
    for item in out:
        item["is_latest"] = item["modified_at"] == latest_modified_at

    preferred = {"latest": 0, "trial": 1}
    out.sort(key=lambda item: (-item["modified_at"], preferred.get(item["id"], 99), item["id"]))
    return out


def load_tags_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    path = Path(path)
    if path.is_dir():
        path = tags_path(path)
    if not path.is_file():
        return records
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record["tags"] = normalize_tag_values(record.get("tags"))
            records[int(record["episode_index"])] = record
    return records


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def merge_tag_record(base: dict | None, update: dict | None, selected_tags: list[str] | None = None) -> dict:
    """Merge tag records field-by-field.

    `selected_tags` limits which tag keys are replaced. This keeps old tags when
    a later run computes only a subset of the schema.
    """

    base = dict(base or {})
    update = dict(update or {})
    record = dict(base)
    record.update({key: value for key, value in update.items() if key not in {"tags", "tag_details"}})
    record["episode_index"] = int(update.get("episode_index", base.get("episode_index", 0)))
    if update.get("task") or not record.get("task"):
        record["task"] = update.get("task", record.get("task"))

    tag_keys = set(selected_tags or (update.get("tags") or {}).keys())
    tags = normalize_tag_values(dict(base.get("tags") or {}))
    update_tags = normalize_tag_values(update.get("tags") or {})
    for key in tag_keys:
        if key in update_tags:
            tags[key] = update_tags[key]
        elif selected_tags is None and key not in tags:
            tags[key] = None
    record["tags"] = tags

    details = dict(base.get("tag_details") or {})
    update_details = dict(update.get("tag_details") or {})
    for key in tag_keys:
        if key in update_details:
            details[key] = update_details[key]
        elif key in details and key in update_tags:
            details.pop(key, None)
    if details:
        record["tag_details"] = details
    else:
        record.pop("tag_details", None)
    return record


def save_reviewed_tag(
    tagging_dir: Path,
    episode_index: int,
    tags: dict,
    manual: bool = True,
    variant: str | None = None,
) -> dict:
    tagging_dir = Path(tagging_dir)
    originals = load_tags_jsonl(resolved_tags_path(tagging_dir, variant))
    original = originals.get(int(episode_index), {"episode_index": int(episode_index), "tags": {}})
    reviewed = load_tags_jsonl(resolved_reviewed_path(tagging_dir, variant))
    record = merge_tag_record(
        original, {"episode_index": int(episode_index), "tags": normalize_tag_values(tags)}
    )
    record["manual"] = bool(manual)
    record["reviewed"] = True
    reviewed[int(episode_index)] = record
    _write_jsonl_atomic(reviewed_path(tagging_dir, variant), [reviewed[idx] for idx in sorted(reviewed)])
    return record


def remove_reviewed_tag(tagging_dir: Path, episode_index: int, variant: str | None = None) -> None:
    tagging_dir = Path(tagging_dir)
    reviewed = load_tags_jsonl(resolved_reviewed_path(tagging_dir, variant))
    reviewed.pop(int(episode_index), None)
    _write_jsonl_atomic(reviewed_path(tagging_dir, variant), [reviewed[idx] for idx in sorted(reviewed)])


def current_tags(tagging_dir: Path, variant: str | None = None) -> dict[int, dict]:
    originals = load_tags_jsonl(resolved_tags_path(tagging_dir, variant))
    reviewed = load_tags_jsonl(resolved_reviewed_path(tagging_dir, variant))
    current = dict(originals)
    for episode_index, reviewed_record in reviewed.items():
        current[episode_index] = merge_tag_record(originals.get(episode_index), reviewed_record)
    return current


def load_episode_record(tagging_dir: Path, episode_index: int, variant: str | None = None) -> dict | None:
    tagging_dir = Path(tagging_dir)
    originals = load_tags_jsonl(resolved_tags_path(tagging_dir, variant))
    original = originals.get(int(episode_index))
    if original is None:
        return None
    reviewed = load_tags_jsonl(resolved_reviewed_path(tagging_dir, variant))
    current = (
        merge_tag_record(original, reviewed.get(int(episode_index)))
        if int(episode_index) in reviewed
        else original
    )
    return {"original": original, "current": current, "reviewed": int(episode_index) in reviewed}


def merge_tags_to_metadata(root: Path, tagging_dir: Path, variant: str | None = None) -> dict:
    root = Path(root)
    tagging_dir = Path(tagging_dir)
    records = current_tags(tagging_dir, variant)
    if not records:
        raise FileNotFoundError(f"Tags not found: {resolved_tags_path(tagging_dir, variant)}")
    updates = {
        int(episode_index): {"tags": dict(record.get("tags") or {})}
        for episode_index, record in records.items()
    }
    merged_episodes = update_episode_metadata(root, updates)
    episodes_path = (
        root / "meta" / ("episodes" if (root / "meta" / "episodes").is_dir() else "episodes.jsonl")
    )
    return {
        "merged": len(merged_episodes),
        "episodes": merged_episodes,
        "episodes_path": str(episodes_path),
    }
