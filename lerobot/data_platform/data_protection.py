"""Dataset stage classification and source-protection policy helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

DATASET_STAGE_RAW = "raw"
DATASET_STAGE_STANDARD = "standard"
DATASET_STAGE_CURATED = "curated"
DATASET_STAGES = {
    DATASET_STAGE_RAW,
    DATASET_STAGE_STANDARD,
    DATASET_STAGE_CURATED,
}

RETENTION_PROTECTED_SOURCE = "protected_source"
RETENTION_MANAGED = "managed"


@dataclass(frozen=True)
class DatasetProtection:
    stage: str
    retention_class: str
    protected: bool
    source_root: str | None
    manual_source: bool
    manual_reason: str
    reasons: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def path_is_within(path: Path | str, parent: Path | str) -> bool:
    resolved = normalized_path(path)
    resolved_parent = normalized_path(parent)
    return resolved == resolved_parent or resolved_parent in resolved.parents


def infer_dataset_stage(root: Path | str, declared_stage: str | None = None) -> str:
    declared = str(declared_stage or "").strip().lower()
    if declared in DATASET_STAGES:
        return declared

    meta_dir = normalized_path(root) / "meta"
    if (meta_dir / "construction_plan.json").is_file():
        return DATASET_STAGE_CURATED
    try:
        if any(meta_dir.glob("preprocess_*.json")):
            return DATASET_STAGE_STANDARD
    except OSError:
        pass
    return DATASET_STAGE_RAW


def evaluate_dataset_protection(
    root: Path | str,
    *,
    source_roots: list[Path | str] | tuple[Path | str, ...] = (),
    manual_source: bool = False,
    manual_reason: str = "",
    stage: str | None = None,
) -> DatasetProtection:
    resolved_root = normalized_path(root)
    matching_roots = sorted(
        {
            str(normalized_path(source_root))
            for source_root in source_roots
            if path_is_within(resolved_root, source_root)
        },
        key=lambda value: (len(Path(value).parts), value),
        reverse=True,
    )
    source_root = matching_roots[0] if matching_roots else None
    reasons = []
    if source_root:
        reasons.append({"type": "source_root", "value": source_root})
    if manual_source:
        reasons.append({"type": "manual", "value": str(manual_reason or "Manual source protection")})
    protected = bool(reasons)
    return DatasetProtection(
        stage=infer_dataset_stage(resolved_root, stage),
        retention_class=RETENTION_PROTECTED_SOURCE if protected else RETENTION_MANAGED,
        protected=protected,
        source_root=source_root,
        manual_source=bool(manual_source),
        manual_reason=str(manual_reason or ""),
        reasons=reasons,
    )
