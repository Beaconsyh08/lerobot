"""Versioned Data Platform and Data Curation lifecycle contracts.

The lifecycle ledger lives outside LeRobot dataset roots.  Raw and published
dataset versions therefore remain immutable while draft curation decisions can
still evolve in the console workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from lerobot.data_platform.lifecycle_repository import SQLiteLifecycleRepository
from lerobot.data_platform.precompute.dataset_io import (
    V3DatasetMetadata,
    is_v3_dataset,
    load_episode_records,
    load_task_records,
    read_episode_table,
    replace_episode_column,
    update_episode_metadata,
    upsert_episode_column,
    write_episode_records,
    write_task_records,
)
from lerobot.data_platform.precompute.data_profile import resolve_data_profile
from lerobot.data_platform.precompute.mutations import (
    update_episode_stats_for_subtask_state,
    update_info_features,
)
from lerobot.data_platform.precompute.preprocess.action_dim import run_convert_action
from lerobot.data_platform.precompute.preprocess.common import (
    format_data_path,
    load_json,
    validate_dataset_root,
)
from lerobot.data_platform.precompute.preprocess.dataset_merge import run_merge
from lerobot.data_platform.precompute.preprocess.dataset_split import run_split
from lerobot.data_platform.precompute.preprocess.dataset_subtract import _fingerprints_for_root
from lerobot.data_platform.precompute.preprocess.field_ops import run_drop_field
from lerobot.data_platform.precompute.preprocess.flag_fixes import trim_episode_inplace
from lerobot.data_platform.precompute.preprocess.standardize import run_standardize_dataset
from lerobot.data_platform.precompute.preprocess.value_edit import run_value_edits

LIFECYCLE_SCHEMA_VERSION = 3
MANIFEST_DRAFT = "draft"
MANIFEST_IN_REVIEW = "in_review"
MANIFEST_APPROVED = "approved"
MANIFEST_PUBLISHED = "published"
MANIFEST_MATERIALIZED = "materialized"
MANIFEST_STATES = {
    MANIFEST_DRAFT,
    MANIFEST_IN_REVIEW,
    MANIFEST_APPROVED,
    MANIFEST_PUBLISHED,
    MANIFEST_MATERIALIZED,
}
_MANIFEST_TRANSITIONS = {
    MANIFEST_DRAFT: MANIFEST_IN_REVIEW,
    MANIFEST_IN_REVIEW: MANIFEST_APPROVED,
    MANIFEST_APPROVED: MANIFEST_PUBLISHED,
}
_SUPPORTED_REPAIR_OPS = {"trim", "value_edit"}
_ALLOWED_ANNOTATION_FIELDS = {
    "first_frame_bbox",
    "prompt",
    "subtask_state",
    "subtask_transitions",
    "tags",
    "task",
    "tasks",
}
PROFILE_PREPROCESSING = "preprocessing"
PROFILE_MATERIALIZATION = "materialization"
PROFILE_KINDS = {PROFILE_PREPROCESSING, PROFILE_MATERIALIZATION}
WORKSPACE_OPEN = "open"
WORKSPACE_PUBLISHED = "published"
MATERIALIZATION_PLANNED = "planned"
MATERIALIZATION_RUNNING = "running"
MATERIALIZATION_VALIDATING = "validating"
MATERIALIZATION_COMMITTED = "committed"
MATERIALIZATION_FAILED = "failed"
QUARANTINE_CATEGORIES = {
    "checksum",
    "missing_file",
    "schema",
    "parquet",
    "video",
    "timestamp",
    "index",
    "metadata",
    "unsupported_format",
}
QUARANTINE_STATES = {"open", "retrying", "resolved", "waived"}
SOURCE_KINDS = {"robot", "umi", "simulation", "generated", "imported"}
RETENTION_CLASSES = {"protected_source", "managed", "disposable"}
DATASET_STAGES = {"raw", "standard", "curated"}
DISPOSITION_OUTCOMES = {
    "received",
    "accepted",
    "quarantined",
    "excluded",
    "deduplicated",
    "repaired",
    "generated",
}
COHORT_DIMENSIONS = {
    "task",
    "scene",
    "object",
    "background",
    "stage",
    "source_kind",
    "robot_profile",
    "signal_schema",
    "dataset_format",
    "generated",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_content_manifest(root: Path) -> dict:
    """Hash every published dataset asset, including complete video bytes."""
    root = validate_dataset_root(root).resolve()
    files = []
    for directory_name in ("meta", "data", "videos"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    payload = {"schema_version": 1, "files": files}
    payload["dataset_fingerprint"] = _sha256(_canonical_json(payload))
    return payload


@dataclass(frozen=True)
class EpisodeRef:
    dataset_version_id: str
    episode_uid: str

    @classmethod
    def from_dict(cls, value: dict) -> EpisodeRef:
        if not isinstance(value, dict):
            raise ValueError("episode reference must be an object")
        version_id = str(value.get("dataset_version_id") or "").strip()
        episode_uid = str(value.get("episode_uid") or "").strip()
        if not version_id or not episode_uid:
            raise ValueError("episode reference requires dataset_version_id and episode_uid")
        return cls(version_id, episode_uid)


@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    version_id: str
    dataset_key: str
    root: str
    stage: str
    fingerprint: str
    dataset_format_version: str
    parent_version_ids: list[str]
    operation: str
    profile: dict
    episode_refs: list[dict]
    created_at: str
    schema: dict = field(default_factory=dict)
    identity_artifact_uri: str | None = None
    identity_artifact_digest: str | None = None
    identity_confidence: str = "confirmed"
    profile_id: str | None = None
    profile_digest: str | None = None
    executor_digest: str | None = None
    data_dimensions: dict = field(default_factory=dict)
    source_batch_ids: list[str] = field(default_factory=list)
    logical_snapshot_id: str | None = None
    semantic_episode_set_digest: str | None = None
    format_variant_of: str | None = None
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> DatasetVersion:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})

    def uid_by_index(self) -> dict[int, str]:
        return {int(item["episode_index"]): str(item["episode_uid"]) for item in self.episode_refs}

    def index_by_uid(self) -> dict[str, int]:
        return {uid: index for index, uid in self.uid_by_index().items()}

    def episode_uids(self) -> set[str]:
        return set(self.index_by_uid())


@dataclass(frozen=True)
class CurationManifestVersion:
    manifest_id: str
    workspace_id: str
    manifest_version: int
    supersedes_manifest_id: str | None
    base_dataset_version_id: str
    base_fingerprint: str
    status: str
    include: list[dict]
    exclude: list[dict]
    annotation_patches: list[dict]
    repair_recipes: list[dict]
    evidence: list[dict]
    rule_versions: dict
    model_versions: dict
    reviewer: str | None
    reason: str | None
    created_by: str
    created_at: str
    updated_at: str
    manifest_digest: str | None = None
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> CurationManifestVersion:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class DatasetReplica:
    replica_id: str
    dataset_version_id: str
    root: str
    fingerprint: str
    status: str
    created_at: str
    dataset_key: str | None = None
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> DatasetReplica:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class DatasetIdentityArtifact:
    dataset_id: str
    dataset_version_id: str
    dataset_fingerprint: str
    episodes: list[dict]
    artifact_digest: str
    generator_version: str
    created_at: str
    identity_confidence: str = "confirmed"
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> DatasetIdentityArtifact:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class ProfileVersion:
    profile_id: str
    kind: str
    name: str
    version: int
    steps: list[dict]
    content_options: dict
    digest: str
    executor_digest: str
    created_by: str
    created_at: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ProfileVersion:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class CurationWorkspace:
    workspace_id: str
    base_dataset_version_id: str
    base_fingerprint: str
    owner: str
    revision: int
    status: str
    decisions: list[dict]
    annotation_patches: list[dict]
    repair_recipes: list[dict]
    cohort_query_snapshot: dict
    evidence: list[dict]
    rule_versions: dict
    model_versions: dict
    created_at: str
    updated_at: str
    published_manifest_id: str | None = None
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> CurationWorkspace:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class MaterializationRun:
    materialization_id: str
    idempotency_key: str
    manifest_id: str
    base_dataset_version_id: str
    profile_id: str
    profile_digest: str
    output_root: str
    staging_root: str
    status: str
    created_at: str
    updated_at: str
    output_dataset_version_id: str | None = None
    error: str | None = None
    details: dict = field(default_factory=dict)
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> MaterializationRun:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class SourceBatchVersion:
    source_batch_id: str
    name: str
    version: int
    supersedes_source_batch_id: str | None
    source_kind: str
    source_uri: str
    robot_profile: str
    signal_schema: str
    dataset_format: str
    retention_class: str
    expected_episode_count: int | None
    metadata: dict
    digest: str
    created_by: str
    created_at: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SourceBatchVersion:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class EpisodeDisposition:
    outcome: str
    reason_code: str
    input_episode_ref: dict | None
    output_episode_refs: list[dict]
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProcessingReconciliationReport:
    reconciliation_id: str
    source_batch_ids: list[str]
    input_dataset_version_ids: list[str]
    output_dataset_version_id: str
    operation: str
    dispositions: list[dict]
    counts: dict
    digest: str
    created_at: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ProcessingReconciliationReport:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class DatasetProfileVersion:
    dataset_profile_id: str
    dataset_version_id: str
    dataset_fingerprint: str
    metrics: dict
    distributions: dict
    data_dimensions: dict
    digest: str
    created_by: str
    created_at: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> DatasetProfileVersion:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class DatasetRequirementVersion:
    requirement_id: str
    name: str
    version: int
    supersedes_requirement_id: str | None
    target_episode_count: int | None
    dimensions: dict
    quality_constraints: dict
    composition_constraints: dict
    reason: str | None
    digest: str
    created_by: str
    created_at: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> DatasetRequirementVersion:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class DataRecipeVersion:
    recipe_id: str
    name: str
    version: int
    supersedes_recipe_id: str | None
    requirement_id: str | None
    base_dataset_version_id: str
    base_fingerprint: str
    cohort_query_snapshot: dict
    include: list[dict]
    exclude: list[dict]
    composition: dict
    deduplication: dict
    random_seed: int
    digest: str
    created_by: str
    created_at: str
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> DataRecipeVersion:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


@dataclass(frozen=True)
class FeedbackReport:
    feedback_id: str
    training_run: dict
    dataset_version_id: str
    manifest_id: str | None
    failures: list[dict]
    coverage_gaps: list[dict]
    collection_brief: dict
    created_by: str
    created_at: str
    dataset_recommendations: list[dict] = field(default_factory=list)
    schema_version: int = LIFECYCLE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> FeedbackReport:
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})


def dataset_snapshot(root: Path) -> tuple[str, dict[int, str]]:
    """Fingerprint episode semantics and every byte of published dataset assets."""
    root = validate_dataset_root(root)
    episode_fingerprints, _counts = _fingerprints_for_root(root)
    return str(build_content_manifest(root)["dataset_fingerprint"]), episode_fingerprints


def _feedback_recommendations(failures: list[dict], coverage_gaps: list[dict]) -> list[dict]:
    recommendations = []
    failure_counts = Counter(
        str(item.get("failure_type") or "unknown") for item in failures if isinstance(item, dict)
    )
    for failure_type, count in sorted(failure_counts.items()):
        recommendations.append(
            {
                "kind": "failure_mode",
                "failure_type": failure_type,
                "episode_count": count,
                "action": "review referenced episodes and adjust the next curation cohort",
            }
        )
    for gap in coverage_gaps:
        if isinstance(gap, dict):
            recommendations.append(
                {
                    "kind": "coverage_gap",
                    "dimensions": {
                        key: gap[key]
                        for key in ("task", "scene", "tag", "stage", "failure_type")
                        if gap.get(key) not in (None, "")
                    },
                    "action": "increase representation in the next dataset version or collection round",
                }
            )
    return recommendations


def _generate_collection_brief(failures: list[dict], coverage_gaps: list[dict]) -> dict:
    priorities = []
    target_counts = {}
    for gap in coverage_gaps:
        if not isinstance(gap, dict):
            continue
        dimensions = [
            f"{key}={gap[key]}"
            for key in ("task", "scene", "tag", "stage", "failure_type")
            if gap.get(key) not in (None, "")
        ]
        label = ", ".join(dimensions) or str(gap.get("name") or "coverage gap")
        if label not in priorities:
            priorities.append(label)
        raw_target = gap.get("target_count", gap.get("gap"))
        if isinstance(raw_target, (int, float)) and raw_target > 0:
            target_counts[label] = int(raw_target)
    for failure_type in sorted(
        {str(item.get("failure_type")) for item in failures if item.get("failure_type")}
    ):
        label = f"failure_type={failure_type}"
        if label not in priorities:
            priorities.append(label)
    return {
        "priorities": priorities,
        "target_counts": target_counts,
        "acceptance_criteria": [
            "new episodes pass Data Platform integrity validation",
            "new coverage is reviewed and published through a Curation Manifest",
        ],
    }


class LifecycleStore:
    """SQLite-backed lifecycle service for one local Data Platform console."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.repository = SQLiteLifecycleRepository(self.root)
        self._upgrade_legacy_versions()

    def _directory(self, name: str) -> Path:
        return self.root / name

    def _list_records(self, name: str, *, connection=None) -> list[dict]:
        return self.repository.list(name, connection=connection)

    def _get_record(self, name: str, record_id: str, *, connection=None) -> dict | None:
        return self.repository.get(name, record_id, connection=connection)

    def _put_record(
        self,
        name: str,
        record_id: str,
        payload: dict,
        *,
        immutable: bool = False,
        connection=None,
    ) -> None:
        self.repository.put(
            name,
            record_id,
            payload,
            immutable=immutable,
            connection=connection,
        )

    def _identity_path(self, artifact_digest: str) -> Path:
        return self.root / "artifacts" / "identities" / f"{artifact_digest}.json"

    def _content_manifest_path(self, fingerprint: str) -> Path:
        return self.root / "artifacts" / "content" / f"{fingerprint}.json"

    def create_source_batch(
        self,
        name: str,
        *,
        source_kind: str,
        source_uri: str,
        robot_profile: str = "unknown",
        signal_schema: str = "unknown",
        dataset_format: str = "unknown",
        retention_class: str = "protected_source",
        expected_episode_count: int | None = None,
        metadata: dict | None = None,
        created_by: str = "unknown",
    ) -> SourceBatchVersion:
        name = str(name or "").strip()
        source_kind = str(source_kind or "").strip().lower()
        source_uri = str(source_uri or "").strip()
        retention_class = str(retention_class or "").strip().lower()
        if not name:
            raise ValueError("source batch name is required")
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"unsupported source kind: {source_kind}")
        if not source_uri:
            raise ValueError("source_uri is required")
        if retention_class not in RETENTION_CLASSES:
            raise ValueError(f"unsupported retention class: {retention_class}")
        if expected_episode_count is not None:
            expected_episode_count = int(expected_episode_count)
            if expected_episode_count < 0:
                raise ValueError("expected_episode_count must be non-negative")
        content = {
            "name": name,
            "source_kind": source_kind,
            "source_uri": source_uri,
            "robot_profile": str(robot_profile or "unknown"),
            "signal_schema": str(signal_schema or "unknown"),
            "dataset_format": str(dataset_format or "unknown"),
            "retention_class": retention_class,
            "expected_episode_count": expected_episode_count,
            "metadata": dict(metadata or {}),
        }
        digest = _sha256(_canonical_json(content))
        existing = next(
            (
                SourceBatchVersion.from_dict(item)
                for item in self._list_records("source_batches")
                if item.get("digest") == digest
            ),
            None,
        )
        if existing is not None:
            return existing
        previous = [
            SourceBatchVersion.from_dict(item)
            for item in self._list_records("source_batches")
            if item.get("name") == name
        ]
        latest = max(previous, key=lambda item: item.version) if previous else None
        batch = SourceBatchVersion(
            source_batch_id=f"sb_{digest[:24]}",
            version=(latest.version + 1 if latest else 1),
            supersedes_source_batch_id=(latest.source_batch_id if latest else None),
            digest=digest,
            created_by=str(created_by or "unknown"),
            created_at=_now(),
            **content,
        )
        self._put_record(
            "source_batches",
            batch.source_batch_id,
            batch.to_dict(),
            immutable=True,
        )
        return batch

    def get_source_batch(self, source_batch_id: str) -> SourceBatchVersion:
        payload = self._get_record("source_batches", str(source_batch_id))
        if payload is None:
            raise KeyError(f"source batch not found: {source_batch_id}")
        return SourceBatchVersion.from_dict(payload)

    def list_source_batches(self) -> list[SourceBatchVersion]:
        records = [SourceBatchVersion.from_dict(item) for item in self._list_records("source_batches")]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def _resolve_data_dimensions(
        self,
        root: Path,
        *,
        stage: str,
        info: dict,
        parents: list[DatasetVersion],
        source_batches: list[SourceBatchVersion],
        overrides: dict | None,
    ) -> dict:
        stage = str(stage or "").strip().lower()
        if stage not in DATASET_STAGES:
            raise ValueError(f"unsupported lifecycle stage: {stage}")
        inherited = dict(parents[0].data_dimensions) if len(parents) == 1 else {}
        portable = resolve_data_profile(root, info.get("features") or {})
        batch = source_batches[0] if len(source_batches) == 1 and not parents else None
        physical_format = str(info.get("codebase_version") or "unknown")
        if batch is not None:
            if batch.signal_schema not in {"", "unknown", portable.signal_schema}:
                raise ValueError(
                    "source batch signal_schema does not match the dataset: "
                    f"declared {batch.signal_schema}, detected {portable.signal_schema}"
                )
            if batch.dataset_format not in {"", "unknown", physical_format}:
                raise ValueError(
                    "source batch dataset_format does not match the dataset: "
                    f"declared {batch.dataset_format}, detected {physical_format}"
                )
            if (
                portable.confirmed
                and batch.robot_profile not in {"", "unknown", portable.robot_profile}
            ):
                raise ValueError(
                    "source batch robot_profile does not match the portable dataset profile"
                )
        if batch is not None:
            stage_profile = str(batch.metadata.get("stage_profile") or portable.stage_profile)
            gripper_encoding = str(
                batch.metadata.get("gripper_encoding") or portable.gripper_encoding
            )
        else:
            stage_profile = (
                portable.stage_profile
                if portable.confirmed or not inherited.get("stage_profile")
                else inherited["stage_profile"]
            )
            gripper_encoding = (
                portable.gripper_encoding
                if portable.confirmed or not inherited.get("gripper_encoding")
                else inherited["gripper_encoding"]
            )
        inferred = {
            **inherited,
            "robot_profile": (
                batch.robot_profile
                if batch
                else inherited.get("robot_profile") or portable.robot_profile
            ),
            "signal_schema": batch.signal_schema if batch else portable.signal_schema,
            "dataset_format": physical_format,
            "lifecycle_stage": stage,
            "source_kind": batch.source_kind if batch else inherited.get("source_kind", "imported"),
            "retention_class": (
                batch.retention_class
                if batch
                else "protected_source" if stage == "raw" else "managed"
            ),
            "stage_profile": stage_profile,
            "gripper_encoding": gripper_encoding,
        }
        inferred.update(dict(overrides or {}))
        if str(inferred.get("source_kind") or "").lower() not in SOURCE_KINDS:
            raise ValueError(f"unsupported source kind: {inferred.get('source_kind')}")
        if str(inferred.get("retention_class") or "").lower() not in RETENTION_CLASSES:
            raise ValueError(f"unsupported retention class: {inferred.get('retention_class')}")
        inferred["source_kind"] = str(inferred["source_kind"]).lower()
        inferred["retention_class"] = str(inferred["retention_class"]).lower()
        inferred["lifecycle_stage"] = stage
        inferred["generated"] = inferred["source_kind"] == "generated"
        return inferred

    def _dataset_id_for_source(self, dataset_key: str, root: Path) -> str:
        source_id = f"src_{_sha256(f'{dataset_key}:{root}')[:24]}"
        with self.repository.transaction() as connection:
            existing = self._get_record("sources", source_id, connection=connection)
            if existing is not None:
                return str(existing["dataset_id"])
            prior_version = next(
                (
                    item
                    for item in self._list_records("versions", connection=connection)
                    if item.get("dataset_key") == dataset_key
                    and Path(str(item.get("root") or "")).expanduser().resolve() == root
                ),
                None,
            )
            dataset_id = (
                str(prior_version["dataset_id"])
                if prior_version is not None
                else f"ds_{uuid.uuid4().hex[:20]}"
            )
            self._put_record(
                "sources",
                source_id,
                {
                    "source_id": source_id,
                    "dataset_key": dataset_key,
                    "root": str(root),
                    "dataset_id": dataset_id,
                    "created_at": _now(),
                },
                immutable=True,
                connection=connection,
            )
            return dataset_id

    def _initial_episode_uids(
        self,
        dataset_id: str,
        fingerprint: str,
        episode_indices: list[int],
    ) -> tuple[dict[int, str], str]:
        seed_id = f"uid_{_sha256(f'{dataset_id}:{fingerprint}')[:24]}"
        with self.repository.transaction() as connection:
            existing = self._get_record("identity_seeds", seed_id, connection=connection)
            if existing is not None:
                mapping = {int(index): str(uid) for index, uid in existing["episode_uids"].items()}
                if set(mapping) != set(episode_indices):
                    raise ValueError("persisted identity seed does not match dataset episodes")
                return mapping, str(existing["created_at"])
            created_at = _now()
            mapping = {index: f"ep_{uuid.uuid4().hex}" for index in episode_indices}
            self._put_record(
                "identity_seeds",
                seed_id,
                {
                    "seed_id": seed_id,
                    "dataset_id": dataset_id,
                    "dataset_fingerprint": fingerprint,
                    "episode_uids": {str(index): uid for index, uid in mapping.items()},
                    "created_at": created_at,
                },
                immutable=True,
                connection=connection,
            )
            return mapping, created_at

    @staticmethod
    def _artifact_payload(
        dataset_id: str,
        version_id: str,
        fingerprint: str,
        episode_refs: list[dict],
        *,
        created_at: str,
        identity_confidence: str = "confirmed",
    ) -> dict:
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "dataset_version_id": version_id,
            "dataset_fingerprint": fingerprint,
            "episodes": [
                {
                    "episode_index": int(item["episode_index"]),
                    "episode_uid": str(item["episode_uid"]),
                    "content_fingerprint": str(item["fingerprint"]),
                    **(
                        {"source_episode_ref": dict(item["source_episode_ref"])}
                        if item.get("source_episode_ref")
                        else {}
                    ),
                }
                for item in episode_refs
            ],
            "generator_version": f"data-platform-lifecycle/{LIFECYCLE_SCHEMA_VERSION}",
            "created_at": created_at,
            "identity_confidence": str(identity_confidence),
        }

    def _persist_identity_artifact(
        self,
        payload: dict,
        *,
        connection=None,
    ) -> DatasetIdentityArtifact:
        digest_payload = dict(payload)
        digest_payload.pop("artifact_digest", None)
        digest = _sha256(_canonical_json(digest_payload))
        artifact = DatasetIdentityArtifact.from_dict({**digest_payload, "artifact_digest": digest})
        path = self._identity_path(digest)
        if not path.is_file():
            _atomic_write_json(path, artifact.to_dict())
        self._put_record(
            "identities",
            digest,
            artifact.to_dict(),
            immutable=True,
            connection=connection,
        )
        return artifact

    def _load_identity_artifact(self, value: Path | dict) -> DatasetIdentityArtifact:
        payload = _read_json(Path(value).expanduser()) if not isinstance(value, dict) else dict(value)
        artifact = DatasetIdentityArtifact.from_dict(payload)
        expected = _sha256(
            _canonical_json({key: item for key, item in payload.items() if key != "artifact_digest"})
        )
        if artifact.artifact_digest != expected:
            raise ValueError("identity artifact digest does not match its content")
        return artifact

    def _upgrade_legacy_versions(self) -> None:
        for payload in self._list_records("versions"):
            refs = list(payload.get("episode_refs") or [])
            if not refs:
                continue
            upgraded = dict(payload)
            if not payload.get("identity_artifact_digest"):
                artifact = self._persist_identity_artifact(
                    self._artifact_payload(
                        str(payload["dataset_id"]),
                        str(payload["version_id"]),
                        str(payload["fingerprint"]),
                        refs,
                        created_at=str(payload.get("created_at") or _now()),
                        identity_confidence="legacy_inferred",
                    )
                )
                upgraded.update(
                    identity_artifact_uri=str(self._identity_path(artifact.artifact_digest)),
                    identity_artifact_digest=artifact.artifact_digest,
                    identity_confidence="legacy_inferred",
                )
            semantic_digest = str(payload.get("semantic_episode_set_digest") or "")
            if not semantic_digest:
                semantic_digest = _sha256(
                    _canonical_json(
                        sorted(
                            (str(item["episode_uid"]), str(item["fingerprint"]))
                            for item in refs
                        )
                    )
                )
                upgraded.update(
                    semantic_episode_set_digest=semantic_digest,
                    logical_snapshot_id=f"lds_{semantic_digest[:24]}",
                    format_variant_of=None,
                )
            if not isinstance(payload.get("source_batch_ids"), list):
                upgraded["source_batch_ids"] = []
            if not isinstance(payload.get("data_dimensions"), dict):
                stage = str(payload.get("stage") or "raw")
                root = Path(str(payload.get("root") or ""))
                if root.is_dir():
                    info = load_json(root / "meta" / "info.json")
                    upgraded["data_dimensions"] = self._resolve_data_dimensions(
                        root,
                        stage=stage,
                        info=info,
                        parents=[],
                        source_batches=[],
                        overrides=None,
                    )
                else:
                    upgraded["data_dimensions"] = {
                        "robot_profile": "unknown",
                        "signal_schema": "unknown",
                        "dataset_format": str(payload.get("dataset_format_version") or "unknown"),
                        "lifecycle_stage": stage,
                        "source_kind": "imported",
                        "retention_class": "protected_source" if stage == "raw" else "managed",
                        "generated": False,
                    }
            upgraded["schema_version"] = LIFECYCLE_SCHEMA_VERSION
            if upgraded != payload:
                self._put_record("versions", str(payload["version_id"]), upgraded)

    def export_identity_artifact(self, version_id: str, out_path: Path) -> Path:
        version = self.get_version(version_id)
        if not version.identity_artifact_uri:
            raise ValueError(f"dataset version has no identity artifact: {version_id}")
        out_path = Path(out_path).expanduser()
        if out_path.exists():
            raise FileExistsError(f"identity artifact output already exists: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(version.identity_artifact_uri, out_path)
        return out_path

    def list_replicas(self, *, dataset_version_id: str | None = None) -> list[DatasetReplica]:
        replicas = [DatasetReplica.from_dict(item) for item in self._list_records("replicas")]
        if dataset_version_id:
            replicas = [item for item in replicas if item.dataset_version_id == dataset_version_id]
        return sorted(replicas, key=lambda item: item.created_at, reverse=True)

    def _register_replica(
        self,
        version_id: str,
        root: Path,
        fingerprint: str,
        *,
        dataset_key: str | None = None,
        connection=None,
    ) -> DatasetReplica:
        resolved = Path(root).expanduser().resolve()
        replica = DatasetReplica(
            replica_id=f"rep_{_sha256(f'{version_id}:{resolved}')[:24]}",
            dataset_version_id=version_id,
            root=str(resolved),
            fingerprint=fingerprint,
            status="available",
            created_at=_now(),
            dataset_key=str(dataset_key) if dataset_key else None,
        )
        existing = self._get_record("replicas", replica.replica_id, connection=connection)
        if existing is not None:
            return DatasetReplica.from_dict(existing)
        try:
            self._put_record(
                "replicas",
                replica.replica_id,
                replica.to_dict(),
                immutable=True,
                connection=connection,
            )
            return replica
        except ValueError:
            payload = self._get_record("replicas", replica.replica_id, connection=connection)
            if payload is None:
                raise
            return DatasetReplica.from_dict(payload)

    def ingest(
        self,
        root: Path,
        dataset_key: str,
        *,
        stage: str = "raw",
        dataset_id: str | None = None,
        parent_version_ids: list[str] | None = None,
        operation: str = "ingest",
        profile: dict | None = None,
        episode_uid_by_index: dict[int, str] | None = None,
        episode_lineage_by_index: dict[int, dict] | None = None,
        identity_artifact: Path | dict | None = None,
        identity_confidence: str = "confirmed",
        profile_id: str | None = None,
        profile_digest: str | None = None,
        executor_digest: str | None = None,
        source_batch_ids: list[str] | None = None,
        data_dimensions: dict | None = None,
        connection=None,
    ) -> DatasetVersion:
        root = validate_dataset_root(root).resolve()
        dataset_key = str(dataset_key).strip()
        if not dataset_key:
            raise ValueError("dataset_key is required")
        parent_version_ids = [str(item) for item in (parent_version_ids or [])]
        parents = [self.get_version(item) for item in parent_version_ids]
        inherited_source_batch_ids = sorted(
            {
                source_batch_id
                for parent in parents
                for source_batch_id in parent.source_batch_ids
            }
        )
        normalized_source_batch_ids = sorted(
            {str(item) for item in (source_batch_ids or inherited_source_batch_ids) if str(item)}
        )
        source_batches = [self.get_source_batch(item) for item in normalized_source_batch_ids]
        imported_artifact = (
            self._load_identity_artifact(identity_artifact) if identity_artifact is not None else None
        )
        if imported_artifact is not None:
            dataset_id = imported_artifact.dataset_id
            identity_confidence = imported_artifact.identity_confidence
        elif dataset_id is None:
            parent_dataset_ids = {parent.dataset_id for parent in parents}
            if len(parent_dataset_ids) == 1:
                dataset_id = parents[0].dataset_id
            elif parents:
                merged_identity = _canonical_json(sorted(parent.version_id for parent in parents))
                dataset_id = f"ds_{_sha256(f'{dataset_key}:{merged_identity}')[:20]}"
            else:
                dataset_id = self._dataset_id_for_source(dataset_key, root)

        validation = validate_materialized_dataset(
            root,
            expected_episode_count=len(load_episode_records(root)),
        )
        content_manifest = validation["content_manifest"]
        content_path = self._content_manifest_path(str(content_manifest["dataset_fingerprint"]))
        if not content_path.is_file():
            _atomic_write_json(content_path, content_manifest)
        fingerprint = str(content_manifest["dataset_fingerprint"])
        episode_fingerprints, _episode_counts = _fingerprints_for_root(root)
        episode_indices = sorted(episode_fingerprints)
        identity_created_at = None
        imported_lineage_by_index = {}
        if imported_artifact is not None:
            if imported_artifact.dataset_fingerprint != fingerprint:
                raise ValueError("identity artifact fingerprint does not match the dataset content")
            imported_episodes = {int(item["episode_index"]): item for item in imported_artifact.episodes}
            if set(imported_episodes) != set(episode_indices):
                raise ValueError("identity artifact episodes do not match the dataset")
            for episode_index, item in imported_episodes.items():
                if str(item["content_fingerprint"]) != episode_fingerprints[episode_index]:
                    raise ValueError(f"identity artifact episode fingerprint mismatch: {episode_index}")
            episode_uid_by_index = {
                index: str(item["episode_uid"]) for index, item in imported_episodes.items()
            }
            imported_lineage_by_index = {
                index: dict(item["source_episode_ref"])
                for index, item in imported_episodes.items()
                if item.get("source_episode_ref")
            }
        elif episode_uid_by_index is None:
            episode_uid_by_index, identity_created_at = self._initial_episode_uids(
                str(dataset_id),
                fingerprint,
                episode_indices,
            )
        normalized_uids = {int(index): str(uid) for index, uid in episode_uid_by_index.items()}
        if set(normalized_uids) != set(episode_indices):
            raise ValueError("episode_uid_by_index must cover every output episode exactly once")
        if len(set(normalized_uids.values())) != len(normalized_uids):
            raise ValueError("episode_uid values must be unique within a dataset version")

        info = load_json(root / "meta" / "info.json")
        resolved_dimensions = self._resolve_data_dimensions(
            root,
            stage=stage,
            info=info,
            parents=parents,
            source_batches=source_batches,
            overrides=data_dimensions,
        )
        identity = {
            "dataset_id": dataset_id,
            "fingerprint": fingerprint,
            "parents": parent_version_ids,
            "operation": operation,
            "profile_digest": profile_digest or _sha256(_canonical_json(profile or {})),
            "source_batch_ids": normalized_source_batch_ids,
            "data_dimensions": resolved_dimensions,
        }
        version_id = (
            imported_artifact.dataset_version_id
            if imported_artifact is not None
            else f"dv_{_sha256(_canonical_json(identity))[:24]}"
        )
        existing_payload = self._get_record("versions", version_id, connection=connection)
        if existing_payload is not None:
            existing = DatasetVersion.from_dict(existing_payload)
            if existing.fingerprint != fingerprint:
                raise ValueError("dataset version identity collides with different content")
            self._register_replica(
                version_id,
                root,
                fingerprint,
                dataset_key=dataset_key,
                connection=connection,
            )
            self._record_automatic_reconciliation(
                parents,
                existing,
                operation=operation,
                connection=connection,
            )
            return existing

        lineage = {
            **imported_lineage_by_index,
            **{
                int(index): dict(value)
                for index, value in (episode_lineage_by_index or {}).items()
            },
        }
        episode_refs = [
            {
                "episode_index": episode_index,
                "episode_uid": normalized_uids[episode_index],
                "fingerprint": episode_fingerprints[episode_index],
                **(
                    {"source_episode_ref": lineage[episode_index]}
                    if episode_index in lineage
                    else {}
                ),
            }
            for episode_index in episode_indices
        ]
        semantic_episode_set_digest = _sha256(
            _canonical_json(
                sorted(
                    (item["episode_uid"], item["fingerprint"])
                    for item in episode_refs
                )
            )
        )
        format_variant_of = next(
            (
                parent.version_id
                for parent in parents
                if parent.semantic_episode_set_digest == semantic_episode_set_digest
                and parent.dataset_format_version != str(info.get("codebase_version") or "unknown")
            ),
            None,
        )
        created_at = identity_created_at or _now()
        artifact = imported_artifact or self._persist_identity_artifact(
            self._artifact_payload(
                str(dataset_id),
                version_id,
                fingerprint,
                episode_refs,
                created_at=created_at,
                identity_confidence=identity_confidence,
            ),
            connection=connection,
        )
        if imported_artifact is not None:
            artifact = self._persist_identity_artifact(
                imported_artifact.to_dict(),
                connection=connection,
            )
        record = DatasetVersion(
            dataset_id=dataset_id,
            version_id=version_id,
            dataset_key=dataset_key,
            root=str(root),
            stage=str(stage),
            fingerprint=fingerprint,
            dataset_format_version=str(info.get("codebase_version") or "unknown"),
            parent_version_ids=parent_version_ids,
            operation=str(operation),
            profile=dict(profile or {}),
            episode_refs=episode_refs,
            created_at=created_at,
            schema=dict(info.get("features") or {}),
            identity_artifact_uri=str(self._identity_path(artifact.artifact_digest)),
            identity_artifact_digest=artifact.artifact_digest,
            identity_confidence=str(identity_confidence),
            profile_id=profile_id,
            profile_digest=profile_digest or identity["profile_digest"],
            executor_digest=executor_digest,
            data_dimensions=resolved_dimensions,
            source_batch_ids=normalized_source_batch_ids,
            logical_snapshot_id=f"lds_{semantic_episode_set_digest[:24]}",
            semantic_episode_set_digest=semantic_episode_set_digest,
            format_variant_of=format_variant_of,
        )
        self._put_record("versions", version_id, record.to_dict(), immutable=True, connection=connection)
        self._register_replica(
            version_id,
            root,
            fingerprint,
            dataset_key=dataset_key,
            connection=connection,
        )
        self._record_automatic_reconciliation(
            parents,
            record,
            operation=operation,
            connection=connection,
        )
        return record

    def register_derived(
        self,
        root: Path,
        dataset_key: str,
        *,
        parent_version_ids: list[str],
        operation: str,
        profile: dict | None = None,
        stage: str = "standard",
        excluded_episode_indices: dict[str, list[int]] | None = None,
        episode_lineage: list[dict] | None = None,
        profile_id: str | None = None,
        profile_digest: str | None = None,
        executor_digest: str | None = None,
    ) -> DatasetVersion:
        """Register an output while carrying parent episode UIDs across reindexing."""
        if not parent_version_ids:
            raise ValueError("derived dataset requires at least one parent version")
        parents = [self.get_version(item) for item in parent_version_ids]
        for parent in parents:
            self.assert_version_current(parent)

        _fingerprint, output_fingerprints = dataset_snapshot(root)
        parent_by_id = {parent.version_id: parent for parent in parents}
        if episode_lineage is not None:
            uid_by_index = {}
            lineage_by_index = {}
            for item in episode_lineage:
                output_index = int(item["output_episode_index"])
                source_version_id = str(item["source_dataset_version_id"])
                source_index = int(item["source_episode_index"])
                parent = parent_by_id.get(source_version_id)
                if parent is None:
                    raise ValueError(f"lineage references undeclared parent version: {source_version_id}")
                source_uid = parent.uid_by_index().get(source_index)
                if source_uid is None:
                    raise ValueError(
                        f"lineage references missing source episode: {source_version_id}/{source_index}"
                    )
                if output_index in uid_by_index:
                    raise ValueError(f"duplicate lineage output episode: {output_index}")
                uid_by_index[output_index] = source_uid
                lineage_by_index[output_index] = asdict(EpisodeRef(source_version_id, source_uid))
            if set(uid_by_index) != set(output_fingerprints):
                raise ValueError("episode lineage must cover every output episode exactly once")
            return self.ingest(
                root,
                dataset_key,
                stage=stage,
                parent_version_ids=[parent.version_id for parent in parents],
                operation=operation,
                profile=profile,
                episode_uid_by_index=uid_by_index,
                episode_lineage_by_index=lineage_by_index,
                identity_confidence="confirmed",
                profile_id=profile_id,
                profile_digest=profile_digest,
                executor_digest=executor_digest,
            )

        exclusions = {
            str(version_id): {int(index) for index in indices}
            for version_id, indices in (excluded_episode_indices or {}).items()
        }
        candidates = []
        for parent in parents:
            excluded = exclusions.get(parent.version_id, set())
            candidates.extend(
                {
                    "episode_index": int(item["episode_index"]),
                    "episode_uid": str(item["episode_uid"]),
                    "fingerprint": str(item["fingerprint"]),
                }
                for item in sorted(parent.episode_refs, key=lambda value: int(value["episode_index"]))
                if int(item["episode_index"]) not in excluded
            )

        by_fingerprint: dict[str, deque[str]] = defaultdict(deque)
        for item in candidates:
            by_fingerprint[item["fingerprint"]].append(item["episode_uid"])
        matched: dict[int, str] = {}
        exact_match = True
        for episode_index, episode_fingerprint in sorted(output_fingerprints.items()):
            bucket = by_fingerprint.get(episode_fingerprint)
            if not bucket:
                exact_match = False
                break
            matched[episode_index] = bucket.popleft()

        if exact_match:
            uid_by_index = matched
        elif len(parents) == 1 and len(candidates) == len(output_fingerprints):
            uid_by_index = {
                output_index: candidate["episode_uid"]
                for output_index, candidate in zip(
                    sorted(output_fingerprints),
                    candidates,
                    strict=True,
                )
            }
        else:
            raise ValueError(
                "cannot preserve episode_uid for derived dataset: output episodes do not match "
                "the declared parent lineage"
            )

        return self.ingest(
            root,
            dataset_key,
            stage=stage,
            parent_version_ids=[parent.version_id for parent in parents],
            operation=operation,
            profile=profile,
            episode_uid_by_index=uid_by_index,
            identity_confidence="legacy_inferred",
            profile_id=profile_id,
            profile_digest=profile_digest,
            executor_digest=executor_digest,
        )

    def get_version(self, version_id: str) -> DatasetVersion:
        payload = self._get_record("versions", str(version_id))
        if payload is None:
            raise KeyError(f"dataset version not found: {version_id}")
        version = DatasetVersion.from_dict(payload)
        if Path(version.root).is_dir():
            return version
        replica = next(
            (
                item
                for item in self.list_replicas(dataset_version_id=version.version_id)
                if Path(item.root).is_dir()
            ),
            None,
        )
        return (
            DatasetVersion.from_dict({**version.to_dict(), "root": replica.root})
            if replica is not None
            else version
        )

    def version_at_root(self, root: Path) -> DatasetVersion | None:
        resolved = Path(root).expanduser().resolve()
        for version in self.list_versions():
            if Path(version.root).expanduser().resolve() == resolved:
                return DatasetVersion.from_dict({**version.to_dict(), "root": str(resolved)})
            if any(
                Path(replica.root).expanduser().resolve() == resolved
                for replica in self.list_replicas(dataset_version_id=version.version_id)
            ):
                return DatasetVersion.from_dict({**version.to_dict(), "root": str(resolved)})
        return None

    def list_versions(self, *, dataset_key: str | None = None) -> list[DatasetVersion]:
        records = [DatasetVersion.from_dict(item) for item in self._list_records("versions")]
        if dataset_key:
            matching_replicas = {
                item.dataset_version_id: item
                for item in self.list_replicas()
                if item.dataset_key == dataset_key
            }
            records = [
                (
                    DatasetVersion.from_dict(
                        {**item.to_dict(), "root": matching_replicas[item.version_id].root}
                    )
                    if item.version_id in matching_replicas
                    else item
                )
                for item in records
                if item.dataset_key == dataset_key or item.version_id in matching_replicas
            ]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def version_has_dataset_key(self, version_id: str, dataset_key: str) -> bool:
        version = self.get_version(version_id)
        return version.dataset_key == dataset_key or any(
            replica.dataset_key == dataset_key
            for replica in self.list_replicas(dataset_version_id=version_id)
        )

    def _persist_reconciliation(
        self,
        *,
        source_batch_ids: list[str],
        input_dataset_version_ids: list[str],
        output_dataset_version_id: str,
        operation: str,
        dispositions: list[dict],
        connection=None,
    ) -> ProcessingReconciliationReport:
        normalized = []
        outcome_counts = Counter()
        for raw_item in dispositions:
            item = dict(raw_item)
            outcome = str(item.get("outcome") or "").strip().lower()
            reason_code = str(item.get("reason_code") or "").strip()
            if outcome not in DISPOSITION_OUTCOMES:
                raise ValueError(f"unsupported episode disposition: {outcome}")
            if not reason_code:
                raise ValueError("episode disposition reason_code is required")
            input_ref = item.get("input_episode_ref")
            output_refs = list(item.get("output_episode_refs") or [])
            if input_ref is not None:
                input_ref = asdict(EpisodeRef.from_dict(input_ref))
            output_refs = [asdict(EpisodeRef.from_dict(ref)) for ref in output_refs]
            normalized_item = EpisodeDisposition(
                outcome=outcome,
                reason_code=reason_code,
                input_episode_ref=input_ref,
                output_episode_refs=output_refs,
                evidence=dict(item.get("evidence") or {}),
            ).to_dict()
            normalized.append(normalized_item)
            outcome_counts[outcome] += 1
        content = {
            "source_batch_ids": sorted({str(item) for item in source_batch_ids}),
            "input_dataset_version_ids": [str(item) for item in input_dataset_version_ids],
            "output_dataset_version_id": str(output_dataset_version_id),
            "operation": str(operation),
            "dispositions": normalized,
        }
        digest = _sha256(_canonical_json(content))
        expected_counts = [
            self.get_source_batch(item).expected_episode_count
            for item in content["source_batch_ids"]
        ]
        expected_episode_count = (
            sum(int(item) for item in expected_counts)
            if expected_counts and all(item is not None for item in expected_counts)
            else None
        )
        output_episode_count = len(
            {
                ref["episode_uid"]
                for item in normalized
                for ref in item["output_episode_refs"]
            }
        )
        report = ProcessingReconciliationReport(
            reconciliation_id=f"rec_{digest[:24]}",
            counts={
                "input_episode_count": sum(
                    item["input_episode_ref"] is not None for item in normalized
                ),
                "output_episode_count": output_episode_count,
                "expected_episode_count": expected_episode_count,
                "expected_output_delta": (
                    output_episode_count - expected_episode_count
                    if expected_episode_count is not None
                    else None
                ),
                "outcomes": dict(sorted(outcome_counts.items())),
            },
            digest=digest,
            created_at=_now(),
            **content,
        )
        existing = self._get_record(
            "reconciliations",
            report.reconciliation_id,
            connection=connection,
        )
        if existing is not None:
            return ProcessingReconciliationReport.from_dict(existing)
        self._put_record(
            "reconciliations",
            report.reconciliation_id,
            report.to_dict(),
            immutable=True,
            connection=connection,
        )
        return report

    def _record_automatic_reconciliation(
        self,
        parents: list[DatasetVersion],
        output: DatasetVersion,
        *,
        operation: str,
        connection=None,
    ) -> ProcessingReconciliationReport | None:
        if not parents and not output.source_batch_ids:
            return None
        output_by_uid = {str(item["episode_uid"]): item for item in output.episode_refs}
        parent_uids = {
            str(item["episode_uid"])
            for parent in parents
            for item in parent.episode_refs
        }
        dispositions = []
        if not parents:
            dispositions.extend(
                EpisodeDisposition(
                    outcome="received",
                    reason_code="source_batch_ingest",
                    input_episode_ref=None,
                    output_episode_refs=[asdict(EpisodeRef(output.version_id, uid))],
                    evidence={"automatic": True},
                ).to_dict()
                for uid in sorted(output_by_uid)
            )
        for parent in parents:
            for item in sorted(parent.episode_refs, key=lambda value: int(value["episode_index"])):
                uid = str(item["episode_uid"])
                output_item = output_by_uid.get(uid)
                if output_item is None:
                    outcome = "excluded"
                    output_refs = []
                    reason_code = f"{operation}_not_in_output"
                elif output_item.get("fingerprint") != item.get("fingerprint"):
                    outcome = "repaired"
                    output_refs = [asdict(EpisodeRef(output.version_id, uid))]
                    reason_code = f"{operation}_content_changed"
                else:
                    outcome = "accepted"
                    output_refs = [asdict(EpisodeRef(output.version_id, uid))]
                    reason_code = "preserved"
                dispositions.append(
                    EpisodeDisposition(
                        outcome=outcome,
                        reason_code=reason_code,
                        input_episode_ref=asdict(EpisodeRef(parent.version_id, uid)),
                        output_episode_refs=output_refs,
                        evidence={"automatic": True, "operation": operation},
                    ).to_dict()
                )
        if parents:
            dispositions.extend(
                EpisodeDisposition(
                    outcome="generated",
                    reason_code=f"{operation}_new_episode",
                    input_episode_ref=None,
                    output_episode_refs=[asdict(EpisodeRef(output.version_id, uid))],
                    evidence={"automatic": True, "operation": operation},
                ).to_dict()
                for uid in sorted(set(output_by_uid) - parent_uids)
            )
        return self._persist_reconciliation(
            source_batch_ids=output.source_batch_ids,
            input_dataset_version_ids=[item.version_id for item in parents],
            output_dataset_version_id=output.version_id,
            operation=operation,
            dispositions=dispositions,
            connection=connection,
        )

    def list_reconciliations(
        self,
        *,
        dataset_version_id: str | None = None,
        source_batch_id: str | None = None,
    ) -> list[ProcessingReconciliationReport]:
        records = [
            ProcessingReconciliationReport.from_dict(item)
            for item in self._list_records("reconciliations")
        ]
        if dataset_version_id:
            records = [
                item
                for item in records
                if item.output_dataset_version_id == dataset_version_id
                or dataset_version_id in item.input_dataset_version_ids
            ]
        if source_batch_id:
            records = [item for item in records if source_batch_id in item.source_batch_ids]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    @staticmethod
    def _episode_tasks(record: dict, tasks_by_index: dict[int, str]) -> list[str]:
        values = record.get("tasks")
        if isinstance(values, str):
            return [values]
        if isinstance(values, list):
            return [str(item) for item in values if str(item).strip()]
        if record.get("task") not in (None, ""):
            return [str(record["task"])]
        try:
            task = tasks_by_index.get(int(record["task_index"]))
        except (KeyError, TypeError, ValueError):
            task = None
        return [task] if task else []

    def _episode_metadata_by_uid(self, version: DatasetVersion) -> dict[str, dict]:
        records = {
            int(item["episode_index"]): dict(item)
            for item in load_episode_records(Path(version.root))
        }
        tasks_by_index = {
            int(item["task_index"]): str(item.get("task") or "")
            for item in load_task_records(Path(version.root))
            if item.get("task_index") is not None
        }
        metadata = {}
        for item in version.episode_refs:
            record = records.get(int(item["episode_index"]), {})
            metadata[str(item["episode_uid"])] = {
                **record,
                "tasks": self._episode_tasks(record, tasks_by_index),
            }
        return metadata

    def create_dataset_profile(
        self,
        dataset_version_id: str,
        *,
        distributions: dict | None = None,
        created_by: str = "unknown",
    ) -> DatasetProfileVersion:
        version = self.get_version(dataset_version_id)
        self.assert_version_current(version)
        info = load_json(Path(version.root) / "meta" / "info.json")
        episode_metadata = self._episode_metadata_by_uid(version)
        task_counts = Counter(
            task
            for item in episode_metadata.values()
            for task in item.get("tasks") or ["unassigned"]
        )
        episode_count = len(version.episode_refs)
        frame_count = int(info.get("total_frames") or 0)
        fps = float(info.get("fps") or 0)
        normalized_distributions = {"task": dict(sorted(task_counts.items()))}
        for key, value in dict(distributions or {}).items():
            if not isinstance(value, dict):
                raise ValueError(f"dataset profile distribution must be an object: {key}")
            normalized_distributions[str(key)] = dict(value)
        content = {
            "dataset_version_id": version.version_id,
            "dataset_fingerprint": version.fingerprint,
            "metrics": {
                "episode_count": episode_count,
                "frame_count": frame_count,
                "fps": fps,
                "duration_seconds": frame_count / fps if fps > 0 else None,
            },
            "distributions": normalized_distributions,
            "data_dimensions": dict(version.data_dimensions),
        }
        digest = _sha256(_canonical_json(content))
        profile = DatasetProfileVersion(
            dataset_profile_id=f"dsp_{digest[:24]}",
            digest=digest,
            created_by=str(created_by or "unknown"),
            created_at=_now(),
            **content,
        )
        existing = self._get_record("dataset_profiles", profile.dataset_profile_id)
        if existing is not None:
            return DatasetProfileVersion.from_dict(existing)
        self._put_record(
            "dataset_profiles",
            profile.dataset_profile_id,
            profile.to_dict(),
            immutable=True,
        )
        return profile

    def list_dataset_profiles(
        self,
        *,
        dataset_version_id: str | None = None,
    ) -> list[DatasetProfileVersion]:
        records = [
            DatasetProfileVersion.from_dict(item)
            for item in self._list_records("dataset_profiles")
        ]
        if dataset_version_id:
            records = [item for item in records if item.dataset_version_id == dataset_version_id]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def create_requirement(
        self,
        name: str,
        *,
        target_episode_count: int | None = None,
        dimensions: dict | None = None,
        quality_constraints: dict | None = None,
        composition_constraints: dict | None = None,
        reason: str | None = None,
        created_by: str = "unknown",
    ) -> DatasetRequirementVersion:
        name = str(name or "").strip()
        if not name:
            raise ValueError("dataset requirement name is required")
        if target_episode_count is not None:
            target_episode_count = int(target_episode_count)
            if target_episode_count <= 0:
                raise ValueError("target_episode_count must be positive")
        dimensions = dict(dimensions or {})
        unsupported = sorted(set(dimensions) - COHORT_DIMENSIONS)
        if unsupported:
            raise ValueError(f"unsupported requirement dimensions: {unsupported}")
        content = {
            "name": name,
            "target_episode_count": target_episode_count,
            "dimensions": dimensions,
            "quality_constraints": dict(quality_constraints or {}),
            "composition_constraints": dict(composition_constraints or {}),
            "reason": str(reason) if reason else None,
        }
        digest = _sha256(_canonical_json(content))
        existing = next(
            (
                DatasetRequirementVersion.from_dict(item)
                for item in self._list_records("requirements")
                if item.get("digest") == digest
            ),
            None,
        )
        if existing is not None:
            return existing
        previous = [
            DatasetRequirementVersion.from_dict(item)
            for item in self._list_records("requirements")
            if item.get("name") == name
        ]
        latest = max(previous, key=lambda item: item.version) if previous else None
        requirement = DatasetRequirementVersion(
            requirement_id=f"dreq_{digest[:24]}",
            version=(latest.version + 1 if latest else 1),
            supersedes_requirement_id=(latest.requirement_id if latest else None),
            digest=digest,
            created_by=str(created_by or "unknown"),
            created_at=_now(),
            **content,
        )
        self._put_record(
            "requirements",
            requirement.requirement_id,
            requirement.to_dict(),
            immutable=True,
        )
        return requirement

    def get_requirement(self, requirement_id: str) -> DatasetRequirementVersion:
        payload = self._get_record("requirements", str(requirement_id))
        if payload is None:
            raise KeyError(f"dataset requirement not found: {requirement_id}")
        return DatasetRequirementVersion.from_dict(payload)

    def list_requirements(self) -> list[DatasetRequirementVersion]:
        records = [
            DatasetRequirementVersion.from_dict(item)
            for item in self._list_records("requirements")
        ]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    @staticmethod
    def _cohort_value(version: DatasetVersion, metadata: dict, key: str):
        if key in version.data_dimensions:
            return version.data_dimensions[key]
        if key == "task":
            return metadata.get("tasks") or []
        if key in metadata:
            return metadata[key]
        tags = metadata.get("tags")
        return tags.get(key) if isinstance(tags, dict) else None

    def resolve_cohort(self, dataset_version_id: str, query: dict | None = None) -> dict:
        version = self.get_version(dataset_version_id)
        query = dict(query or {})
        allowed = {"episode_indices", "episode_uids", "task_contains", "metadata"}
        unsupported = sorted(set(query) - allowed)
        if unsupported:
            raise ValueError(f"unsupported cohort query fields: {unsupported}")
        metadata_filters = dict(query.get("metadata") or {})
        unsupported_dimensions = sorted(set(metadata_filters) - COHORT_DIMENSIONS)
        if unsupported_dimensions:
            raise ValueError(f"unsupported cohort metadata dimensions: {unsupported_dimensions}")
        requested_indices = (
            {int(item) for item in query.get("episode_indices") or []}
            if query.get("episode_indices") is not None
            else None
        )
        requested_uids = (
            {str(item) for item in query.get("episode_uids") or []}
            if query.get("episode_uids") is not None
            else None
        )
        task_terms = query.get("task_contains") or []
        if isinstance(task_terms, str):
            task_terms = [task_terms]
        task_terms = [str(item).lower() for item in task_terms if str(item).strip()]
        metadata_by_uid = self._episode_metadata_by_uid(version)
        resolved = []
        for item in sorted(version.episode_refs, key=lambda value: int(value["episode_index"])):
            index = int(item["episode_index"])
            uid = str(item["episode_uid"])
            if requested_indices is not None and index not in requested_indices:
                continue
            if requested_uids is not None and uid not in requested_uids:
                continue
            metadata = metadata_by_uid.get(uid, {})
            tasks = [str(task).lower() for task in metadata.get("tasks") or []]
            if task_terms and not any(term in task for term in task_terms for task in tasks):
                continue
            matched = True
            for key, expected in metadata_filters.items():
                actual = self._cohort_value(version, metadata, key)
                expected_values = expected if isinstance(expected, list) else [expected]
                actual_values = actual if isinstance(actual, list) else [actual]
                if not set(map(str, actual_values)) & set(map(str, expected_values)):
                    matched = False
                    break
            if matched:
                resolved.append(asdict(EpisodeRef(version.version_id, uid)))
        snapshot = {
            "base_dataset_version_id": version.version_id,
            "base_fingerprint": version.fingerprint,
            "query": query,
            "episode_refs": resolved,
        }
        snapshot["digest"] = _sha256(_canonical_json(snapshot))
        snapshot["episode_count"] = len(resolved)
        return snapshot

    @staticmethod
    def _stable_recipe_order(refs: list[dict], seed: int) -> list[dict]:
        return sorted(
            refs,
            key=lambda item: _sha256(f"{seed}:{item['episode_uid']}"),
        )

    def _apply_recipe_composition(
        self,
        base: DatasetVersion,
        refs: list[dict],
        composition: dict,
        *,
        random_seed: int,
    ) -> list[dict]:
        if not composition:
            return refs
        allowed = {"max_episodes", "group_by", "target_counts", "ratios", "target_episode_count"}
        unsupported = sorted(set(composition) - allowed)
        if unsupported:
            raise ValueError(f"unsupported recipe composition fields: {unsupported}")
        ordered = self._stable_recipe_order(refs, random_seed)
        group_by = str(composition.get("group_by") or "").strip()
        if not group_by:
            if composition.get("target_counts") or composition.get("ratios"):
                raise ValueError("recipe composition group_by is required for grouped targets")
            if composition.get("max_episodes") is not None:
                limit = int(composition["max_episodes"])
                if limit <= 0:
                    raise ValueError("recipe max_episodes must be positive")
                ordered = ordered[:limit]
            return sorted(ordered, key=lambda item: base.index_by_uid()[item["episode_uid"]])
        if group_by not in COHORT_DIMENSIONS:
            raise ValueError(f"unsupported recipe group_by dimension: {group_by}")
        metadata_by_uid = self._episode_metadata_by_uid(base)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for ref in ordered:
            value = self._cohort_value(base, metadata_by_uid.get(ref["episode_uid"], {}), group_by)
            values = value if isinstance(value, list) else [value]
            labels = {str(item) for item in values if item not in (None, "")}
            if len(labels) != 1:
                raise ValueError(
                    f"recipe grouped composition requires exactly one {group_by} value per episode"
                )
            grouped[next(iter(labels))].append(ref)
        raw_targets = dict(composition.get("target_counts") or {})
        if composition.get("ratios"):
            total = int(composition.get("target_episode_count") or 0)
            ratios = {str(key): float(value) for key, value in dict(composition["ratios"]).items()}
            if total <= 0 or not ratios or any(value < 0 for value in ratios.values()):
                raise ValueError(
                    "recipe ratios require positive target_episode_count and non-negative ratios"
                )
            ratio_sum = sum(ratios.values())
            if ratio_sum <= 0:
                raise ValueError("recipe ratios must contain a positive value")
            raw_targets = {
                key: int(total * value / ratio_sum)
                for key, value in ratios.items()
            }
            remainder = total - sum(raw_targets.values())
            for key in sorted(ratios, key=lambda item: (-ratios[item], item))[:remainder]:
                raw_targets[key] += 1
        if not raw_targets:
            raise ValueError("grouped recipe composition requires target_counts or ratios")
        selected = []
        for label, raw_count in raw_targets.items():
            count = int(raw_count)
            if count < 0:
                raise ValueError("recipe target counts must be non-negative")
            available = grouped.get(str(label), [])
            if len(available) < count:
                raise ValueError(
                    f"recipe target exceeds available episodes for {group_by}={label}: "
                    f"requested {count}, available {len(available)}"
                )
            selected.extend(available[:count])
        unique = {item["episode_uid"]: item for item in selected}
        return sorted(unique.values(), key=lambda item: base.index_by_uid()[item["episode_uid"]])

    def create_recipe(
        self,
        name: str,
        base_dataset_version_id: str,
        *,
        requirement_id: str | None = None,
        cohort_query_snapshot: dict | None = None,
        include: list[dict] | None = None,
        exclude: list[dict] | None = None,
        composition: dict | None = None,
        deduplication: dict | None = None,
        random_seed: int = 0,
        created_by: str = "unknown",
    ) -> DataRecipeVersion:
        name = str(name or "").strip()
        if not name:
            raise ValueError("data recipe name is required")
        base = self.get_version(base_dataset_version_id)
        self.assert_version_current(base)
        if requirement_id:
            self.get_requirement(requirement_id)
        include_refs = self._validate_refs(base, list(include or []), "recipe include")
        exclude_refs = self._validate_refs(base, list(exclude or []), "recipe exclude")
        if {item["episode_uid"] for item in include_refs} & {
            item["episode_uid"] for item in exclude_refs
        }:
            raise ValueError("recipe cannot include and exclude the same episode")
        snapshot = dict(cohort_query_snapshot or {})
        if snapshot:
            if snapshot.get("base_dataset_version_id") != base.version_id:
                raise ValueError("cohort snapshot base version does not match recipe base")
            if snapshot.get("base_fingerprint") != base.fingerprint:
                raise ValueError("cohort snapshot fingerprint does not match recipe base")
            snapshot_refs = self._validate_refs(
                base,
                list(snapshot.get("episode_refs") or []),
                "cohort snapshot",
            )
            if not include_refs:
                include_refs = snapshot_refs
            snapshot = {**snapshot, "episode_refs": snapshot_refs}
        excluded_uids = {item["episode_uid"] for item in exclude_refs}
        candidates = include_refs or [
            asdict(EpisodeRef(base.version_id, item["episode_uid"]))
            for item in sorted(base.episode_refs, key=lambda value: int(value["episode_index"]))
        ]
        candidates = [item for item in candidates if item["episode_uid"] not in excluded_uids]
        deduplication = dict(deduplication or {})
        dedup_mode = str(deduplication.get("mode") or "none")
        if dedup_mode not in {"none", "exact"}:
            raise ValueError("recipe deduplication mode must be none or exact")
        if dedup_mode == "exact":
            refs_by_uid = {item["episode_uid"]: item for item in base.episode_refs}
            seen_fingerprints = set()
            deduplicated = []
            for ref in candidates:
                fingerprint = refs_by_uid[ref["episode_uid"]]["fingerprint"]
                if fingerprint in seen_fingerprints:
                    exclude_refs.append(ref)
                    continue
                seen_fingerprints.add(fingerprint)
                deduplicated.append(ref)
            candidates = deduplicated
        composition = dict(composition or {})
        selected = self._apply_recipe_composition(
            base,
            candidates,
            composition,
            random_seed=int(random_seed),
        )
        selection_is_explicit = bool(include_refs or snapshot or composition or dedup_mode == "exact")
        resolved_include = selected if selection_is_explicit else []
        content = {
            "name": name,
            "requirement_id": str(requirement_id) if requirement_id else None,
            "base_dataset_version_id": base.version_id,
            "base_fingerprint": base.fingerprint,
            "cohort_query_snapshot": snapshot,
            "include": resolved_include,
            "exclude": sorted(
                {item["episode_uid"]: item for item in exclude_refs}.values(),
                key=lambda item: base.index_by_uid()[item["episode_uid"]],
            ),
            "composition": composition,
            "deduplication": {**deduplication, "mode": dedup_mode},
            "random_seed": int(random_seed),
        }
        digest = _sha256(_canonical_json(content))
        existing = next(
            (
                DataRecipeVersion.from_dict(item)
                for item in self._list_records("recipes")
                if item.get("digest") == digest
            ),
            None,
        )
        if existing is not None:
            return existing
        previous = [
            DataRecipeVersion.from_dict(item)
            for item in self._list_records("recipes")
            if item.get("name") == name
        ]
        latest = max(previous, key=lambda item: item.version) if previous else None
        recipe = DataRecipeVersion(
            recipe_id=f"drcp_{digest[:24]}",
            version=(latest.version + 1 if latest else 1),
            supersedes_recipe_id=(latest.recipe_id if latest else None),
            digest=digest,
            created_by=str(created_by or "unknown"),
            created_at=_now(),
            **content,
        )
        self._put_record("recipes", recipe.recipe_id, recipe.to_dict(), immutable=True)
        return recipe

    def get_recipe(self, recipe_id: str) -> DataRecipeVersion:
        payload = self._get_record("recipes", str(recipe_id))
        if payload is None:
            raise KeyError(f"data recipe not found: {recipe_id}")
        return DataRecipeVersion.from_dict(payload)

    def list_recipes(
        self,
        *,
        base_dataset_version_id: str | None = None,
    ) -> list[DataRecipeVersion]:
        records = [DataRecipeVersion.from_dict(item) for item in self._list_records("recipes")]
        if base_dataset_version_id:
            records = [
                item for item in records if item.base_dataset_version_id == base_dataset_version_id
            ]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def validate_recipe(self, recipe_id: str) -> dict:
        recipe = self.get_recipe(recipe_id)
        base = self.get_version(recipe.base_dataset_version_id)
        selected_uids = {item["episode_uid"] for item in recipe.include} or base.episode_uids()
        selected_uids -= {item["episode_uid"] for item in recipe.exclude}
        errors = []
        warnings = []
        requirement = self.get_requirement(recipe.requirement_id) if recipe.requirement_id else None
        if requirement is not None:
            if (
                requirement.target_episode_count is not None
                and len(selected_uids) != requirement.target_episode_count
            ):
                errors.append(
                    {
                        "field": "target_episode_count",
                        "expected": requirement.target_episode_count,
                        "actual": len(selected_uids),
                    }
                )
            metadata_by_uid = self._episode_metadata_by_uid(base)
            for dimension, raw_expected in requirement.dimensions.items():
                expected_values = raw_expected if isinstance(raw_expected, list) else [raw_expected]
                actual_values = set()
                for uid in selected_uids:
                    value = self._cohort_value(base, metadata_by_uid.get(uid, {}), dimension)
                    values = value if isinstance(value, list) else [value]
                    actual_values.update(str(item) for item in values if item not in (None, ""))
                missing = sorted(set(map(str, expected_values)) - actual_values)
                if missing:
                    errors.append(
                        {
                            "field": f"dimensions.{dimension}",
                            "missing": missing,
                            "actual": sorted(actual_values),
                        }
                    )
            for key, expected in requirement.composition_constraints.items():
                if recipe.composition.get(key) != expected:
                    errors.append(
                        {
                            "field": f"composition.{key}",
                            "expected": expected,
                            "actual": recipe.composition.get(key),
                        }
                    )
            if requirement.quality_constraints:
                warnings.append(
                    {
                        "field": "quality_constraints",
                        "message": "quality constraints require reviewed quality distributions",
                    }
                )
        return {
            "recipe_id": recipe.recipe_id,
            "requirement_id": recipe.requirement_id,
            "selected_episode_count": len(selected_uids),
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def compile_recipe(self, recipe_id: str, *, owner: str = "unknown") -> CurationWorkspace:
        recipe = self.get_recipe(recipe_id)
        validation = self.validate_recipe(recipe.recipe_id)
        if not validation["valid"]:
            raise ValueError(f"data recipe does not satisfy its requirement: {validation['errors']}")
        base = self.get_version(recipe.base_dataset_version_id)
        self.assert_version_current(base)
        decisions = [
            {"episode_ref": item, "decision": "keep", "reason": f"recipe:{recipe.recipe_id}"}
            for item in recipe.include
        ]
        decisions.extend(
            {
                "episode_ref": item,
                "decision": "exclude",
                "reason": f"recipe:{recipe.recipe_id}",
            }
            for item in recipe.exclude
        )
        return self.create_workspace(
            base.version_id,
            owner=owner,
            decisions=decisions,
            cohort_query_snapshot={
                **recipe.cohort_query_snapshot,
                "recipe_id": recipe.recipe_id,
                "recipe_digest": recipe.digest,
                "resolved_include": recipe.include,
                "resolved_exclude": recipe.exclude,
            },
            evidence=[
                {
                    "kind": "data_recipe",
                    "recipe_id": recipe.recipe_id,
                    "recipe_digest": recipe.digest,
                    "requirement_id": recipe.requirement_id,
                }
            ],
        )

    def record_quarantine(
        self,
        *,
        dataset_key: str,
        root: Path,
        error: str | Exception,
        operation: str = "ingest",
        category: str | None = None,
        detector_version: str = "dataset-root-validator/v1",
        evidence: dict | None = None,
        affected_files: list[str] | None = None,
        affected_episode_uids: list[str] | None = None,
    ) -> dict:
        if category is None:
            message = str(error).lower()
            if isinstance(error, FileNotFoundError) or "missing" in message or "not found" in message:
                category = "missing_file"
            elif "parquet" in message:
                category = "parquet"
            elif "video" in message or "mp4" in message:
                category = "video"
            elif "timestamp" in message or " pts" in message:
                category = "timestamp"
            elif "schema" in message or "feature" in message:
                category = "schema"
            elif "format" in message or "version" in message:
                category = "unsupported_format"
            else:
                category = "metadata"
        category = str(category).strip().lower()
        if category not in QUARANTINE_CATEGORIES:
            raise ValueError(f"unsupported quarantine category: {category}")
        record = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "quarantine_id": f"iq_{uuid.uuid4().hex}",
            "quality_domain": "infra_quality",
            "category": category,
            "status": "open",
            "dataset_key": str(dataset_key),
            "root": str(Path(root).expanduser().resolve()),
            "operation": str(operation),
            "error": str(error),
            "detector_version": str(detector_version),
            "evidence": dict(evidence or {}),
            "affected_files": [str(item) for item in (affected_files or [])],
            "affected_episode_uids": [str(item) for item in (affected_episode_uids or [])],
            "retry_job_id": None,
            "resolved_dataset_version_id": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._put_record("quarantine", record["quarantine_id"], record, immutable=True)
        return record

    def list_quarantine(self, *, dataset_key: str | None = None) -> list[dict]:
        records = self._list_records("quarantine")
        if dataset_key:
            records = [item for item in records if item.get("dataset_key") == dataset_key]
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)

    def transition_quarantine(
        self,
        quarantine_id: str,
        status: str,
        *,
        retry_job_id: str | None = None,
        resolved_dataset_version_id: str | None = None,
    ) -> dict:
        status = str(status).strip().lower()
        if status not in QUARANTINE_STATES:
            raise ValueError(f"unsupported quarantine status: {status}")
        with self.repository.transaction() as connection:
            payload = self._get_record("quarantine", quarantine_id, connection=connection)
            if payload is None:
                raise KeyError(f"quarantine record not found: {quarantine_id}")
            current = str(payload.get("status") or "open")
            allowed = {
                "open": {"retrying", "waived"},
                "retrying": {"open", "resolved", "waived"},
                "resolved": set(),
                "waived": set(),
            }
            if status not in allowed.get(current, set()):
                raise ValueError(f"invalid quarantine transition: {current} -> {status}")
            if status == "retrying" and not retry_job_id:
                raise ValueError("retry_job_id is required when retrying quarantine")
            if status == "resolved":
                if not resolved_dataset_version_id:
                    raise ValueError("resolved_dataset_version_id is required")
                self.get_version(resolved_dataset_version_id)
            updated = {
                **payload,
                "status": status,
                "retry_job_id": retry_job_id or payload.get("retry_job_id"),
                "resolved_dataset_version_id": (
                    resolved_dataset_version_id or payload.get("resolved_dataset_version_id")
                ),
                "updated_at": _now(),
            }
            self._put_record(
                "quarantine",
                quarantine_id,
                updated,
                connection=connection,
            )
        return updated

    def assert_version_current(self, version: DatasetVersion) -> None:
        root = Path(version.root)
        if not root.is_dir():
            raise ValueError(f"dataset version root is unavailable: {root}")
        fingerprint, _episodes = dataset_snapshot(root)
        if fingerprint != version.fingerprint:
            raise ValueError(
                f"dataset version is stale: expected {version.fingerprint}, found {fingerprint}; "
                "ingest the changed dataset as a new version"
            )

    def create_profile(
        self,
        kind: str,
        name: str,
        *,
        steps: list[dict] | None = None,
        content_options: dict | None = None,
        created_by: str = "unknown",
    ) -> ProfileVersion:
        kind = str(kind).strip().lower()
        name = str(name).strip()
        if kind not in PROFILE_KINDS:
            raise ValueError(f"unsupported profile kind: {kind}")
        if not name:
            raise ValueError("profile name is required")
        steps = [dict(item) for item in (steps or [])]
        allowed_ops = (
            {"copy", "split", "standardize", "convert_action", "drop_field"}
            if kind == PROFILE_PREPROCESSING
            else {"validate"}
        )
        allowed_params = {
            "copy": set(),
            "split": {"episode_range", "task_filter"},
            "standardize": {"data_version"},
            "convert_action": {"target_dim"},
            "drop_field": {"field_name"},
            "validate": {"level"},
        }
        for step in steps:
            operation = str(step.get("op") or "").strip()
            if operation not in allowed_ops:
                raise ValueError(f"unsupported {kind} profile step: {operation}")
            params = step.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("profile step params must be an object")
            unsupported_params = sorted(set(params) - allowed_params[operation])
            if unsupported_params:
                raise ValueError(
                    f"unsupported params for profile step {operation}: {unsupported_params}"
                )
            step["op"] = operation
            step["params"] = dict(params)
        if kind == PROFILE_PREPROCESSING and not steps:
            raise ValueError("preprocessing profile requires at least one executable step")
        if kind == PROFILE_MATERIALIZATION and not steps:
            steps = [{"op": "validate", "params": {"level": "full"}}]
        options = dict(content_options or {})
        if kind == PROFILE_MATERIALIZATION:
            unsupported_options = sorted(set(options) - {"validation_level", "video_mode"})
            if unsupported_options:
                raise ValueError(f"unsupported materialization content options: {unsupported_options}")
            validation_level = str(options.get("validation_level") or "full")
            if validation_level != "full":
                raise ValueError("published materialization profiles require full validation")
            options["validation_level"] = "full"
            video_mode = str(options.get("video_mode") or "copy")
            if video_mode != "copy":
                raise ValueError("current materializer only supports deterministic video_mode=copy")
            options["video_mode"] = "copy"
        executor_digest = _sha256(
            _canonical_json(
                {
                    "schema_version": LIFECYCLE_SCHEMA_VERSION,
                    "kind": kind,
                    "operations": sorted(allowed_ops),
                }
            )
        )
        with self.repository.transaction() as connection:
            existing = [
                ProfileVersion.from_dict(item)
                for item in self._list_records("profiles", connection=connection)
                if item.get("kind") == kind and item.get("name") == name
            ]
            matching = next(
                (
                    item
                    for item in existing
                    if item.steps == steps and item.content_options == options
                ),
                None,
            )
            if matching is not None:
                return matching
            version = max((item.version for item in existing), default=0) + 1
            digest_payload = {
                "kind": kind,
                "name": name,
                "version": version,
                "steps": steps,
                "content_options": options,
            }
            digest = _sha256(_canonical_json(digest_payload))
            profile = ProfileVersion(
                profile_id=f"pf_{digest[:24]}",
                kind=kind,
                name=name,
                version=version,
                steps=steps,
                content_options=options,
                digest=digest,
                executor_digest=executor_digest,
                created_by=str(created_by or "unknown"),
                created_at=_now(),
            )
            self._put_record(
                "profiles",
                profile.profile_id,
                profile.to_dict(),
                immutable=True,
                connection=connection,
            )
            return profile

    def get_profile(self, profile_id: str, *, kind: str | None = None) -> ProfileVersion:
        payload = self._get_record("profiles", str(profile_id))
        if payload is None:
            raise KeyError(f"profile not found: {profile_id}")
        profile = ProfileVersion.from_dict(payload)
        if kind and profile.kind != kind:
            raise ValueError(f"profile {profile_id} is not a {kind} profile")
        return profile

    def list_profiles(self, *, kind: str | None = None) -> list[ProfileVersion]:
        profiles = [ProfileVersion.from_dict(item) for item in self._list_records("profiles")]
        if kind:
            profiles = [item for item in profiles if item.kind == kind]
        return sorted(profiles, key=lambda item: (item.name, item.version), reverse=True)

    def default_materialization_profile(self) -> ProfileVersion:
        existing = [
            item
            for item in self.list_profiles(kind=PROFILE_MATERIALIZATION)
            if item.name == "default-full-validation"
        ]
        return existing[0] if existing else self.create_profile(
            PROFILE_MATERIALIZATION,
            "default-full-validation",
            content_options={"validation_level": "full"},
            created_by="system",
        )

    def create_workspace(
        self,
        base_dataset_version_id: str,
        *,
        owner: str = "unknown",
        decisions: list[dict] | None = None,
        annotation_patches: list[dict] | None = None,
        repair_recipes: list[dict] | None = None,
        cohort_query_snapshot: dict | None = None,
        evidence: list[dict] | None = None,
        rule_versions: dict | None = None,
        model_versions: dict | None = None,
    ) -> CurationWorkspace:
        base = self.get_version(base_dataset_version_id)
        workspace = CurationWorkspace(
            workspace_id=f"cw_{uuid.uuid4().hex}",
            base_dataset_version_id=base.version_id,
            base_fingerprint=base.fingerprint,
            owner=str(owner or "unknown"),
            revision=1,
            status=WORKSPACE_OPEN,
            decisions=list(decisions or []),
            annotation_patches=list(annotation_patches or []),
            repair_recipes=list(repair_recipes or []),
            cohort_query_snapshot=dict(cohort_query_snapshot or {}),
            evidence=list(evidence or []),
            rule_versions=dict(rule_versions or {}),
            model_versions=dict(model_versions or {}),
            created_at=_now(),
            updated_at=_now(),
        )
        self._validate_workspace_payload(workspace)
        self._put_record("workspaces", workspace.workspace_id, workspace.to_dict(), immutable=True)
        return workspace

    def get_workspace(self, workspace_id: str) -> CurationWorkspace:
        payload = self._get_record("workspaces", str(workspace_id))
        if payload is None:
            raise KeyError(f"curation workspace not found: {workspace_id}")
        return CurationWorkspace.from_dict(payload)

    def list_workspaces(
        self,
        *,
        base_dataset_version_id: str | None = None,
    ) -> list[CurationWorkspace]:
        workspaces = [CurationWorkspace.from_dict(item) for item in self._list_records("workspaces")]
        if base_dataset_version_id:
            workspaces = [
                item for item in workspaces if item.base_dataset_version_id == base_dataset_version_id
            ]
        return sorted(workspaces, key=lambda item: item.updated_at, reverse=True)

    def _normalize_workspace_decisions(
        self,
        base: DatasetVersion,
        decisions: list[dict],
    ) -> list[dict]:
        normalized = []
        seen = set()
        for item in decisions:
            if not isinstance(item, dict):
                raise ValueError("workspace decisions must be objects")
            ref = EpisodeRef.from_dict(item.get("episode_ref") or {})
            self._validate_refs(base, [asdict(ref)], "workspace decision")
            decision = str(item.get("decision") or "").strip().lower()
            if decision not in {"keep", "exclude", "review", "repair"}:
                raise ValueError(f"unsupported workspace decision: {decision}")
            if ref.episode_uid in seen:
                raise ValueError(f"duplicate workspace decision: {ref.episode_uid}")
            seen.add(ref.episode_uid)
            normalized.append(
                {
                    "episode_ref": asdict(ref),
                    "decision": decision,
                    "reason": str(item.get("reason") or ""),
                }
            )
        return normalized

    def _validate_workspace_payload(self, workspace: CurationWorkspace) -> dict:
        base = self.get_version(workspace.base_dataset_version_id)
        if base.fingerprint != workspace.base_fingerprint:
            raise ValueError("workspace base fingerprint does not match its dataset version")
        decisions = self._normalize_workspace_decisions(base, workspace.decisions)
        excluded_uids = {
            item["episode_ref"]["episode_uid"]
            for item in decisions
            if item["decision"] == "exclude"
        }
        selected_uids = base.episode_uids() - excluded_uids
        if not selected_uids:
            raise ValueError("workspace would exclude every episode")
        patches = self._validate_annotation_patches(base, workspace.annotation_patches, selected_uids)
        repairs = self._validate_repair_recipes(base, workspace.repair_recipes, selected_uids)
        return {
            "valid": True,
            "revision": workspace.revision,
            "normalized_decisions": decisions,
            "annotation_patch_count": len(patches),
            "repair_recipe_count": len(repairs),
            "unresolved_review_count": sum(item["decision"] == "review" for item in decisions),
        }

    def validate_workspace(self, workspace_id: str) -> dict:
        workspace = self.get_workspace(workspace_id)
        base = self.get_version(workspace.base_dataset_version_id)
        self.assert_version_current(base)
        return self._validate_workspace_payload(workspace)

    def update_workspace(
        self,
        workspace_id: str,
        *,
        expected_revision: int,
        changes: dict,
    ) -> CurationWorkspace:
        with self.repository.transaction() as connection:
            payload = self._get_record("workspaces", workspace_id, connection=connection)
            if payload is None:
                raise KeyError(f"curation workspace not found: {workspace_id}")
            current = CurationWorkspace.from_dict(payload)
            if current.status != WORKSPACE_OPEN:
                raise ValueError("published workspaces are immutable")
            if int(expected_revision) != current.revision:
                raise ValueError(
                    f"workspace revision conflict: expected {expected_revision}, current {current.revision}"
                )
            mutable_fields = {
                "decisions",
                "annotation_patches",
                "repair_recipes",
                "cohort_query_snapshot",
                "evidence",
                "rule_versions",
                "model_versions",
            }
            unsupported = sorted(set(changes) - mutable_fields)
            if unsupported:
                raise ValueError(f"unsupported workspace changes: {unsupported}")
            updated_payload = current.to_dict()
            updated_payload.update({key: changes[key] for key in changes if key in mutable_fields})
            updated_payload.update(revision=current.revision + 1, updated_at=_now())
            updated = CurationWorkspace.from_dict(updated_payload)
            self._validate_workspace_payload(updated)
            self._put_record("workspaces", workspace_id, updated.to_dict(), connection=connection)
        return updated

    def rebase_workspace(
        self,
        workspace_id: str,
        new_base_dataset_version_id: str,
        *,
        expected_revision: int,
    ) -> dict:
        lease_owner = f"pid:{os.getpid()}:{threading.get_ident()}"
        lease_id = f"workspace-rebase:{workspace_id}"
        if not self.repository.claim_job(
            lease_id,
            "workspace_rebase",
            owner=lease_owner,
            lease_seconds=120,
        ):
            raise ValueError("workspace rebase is already running in another process")
        try:
            return self._rebase_workspace_locked(
                workspace_id,
                new_base_dataset_version_id,
                expected_revision=expected_revision,
            )
        finally:
            self.repository.release_job(lease_id, owner=lease_owner)

    def _rebase_workspace_locked(
        self,
        workspace_id: str,
        new_base_dataset_version_id: str,
        *,
        expected_revision: int,
    ) -> dict:
        workspace = self.get_workspace(workspace_id)
        if workspace.revision != int(expected_revision):
            raise ValueError(
                f"workspace revision conflict: expected {expected_revision}, current {workspace.revision}"
            )
        old_base = self.get_version(workspace.base_dataset_version_id)
        new_base = self.get_version(new_base_dataset_version_id)
        if old_base.dataset_id != new_base.dataset_id:
            raise ValueError("workspace can only rebase within the same logical dataset")
        new_uids = new_base.episode_uids()
        referenced_uids = {
            str(item["episode_ref"]["episode_uid"])
            for item in [*workspace.decisions, *workspace.annotation_patches]
        }
        for recipe in workspace.repair_recipes:
            referenced_uids.update(str(item["episode_uid"]) for item in recipe.get("episode_refs") or [])
        missing = sorted(referenced_uids - new_uids)
        if missing:
            return {"rebased": False, "conflicts": [{"episode_uid": uid, "reason": "missing"} for uid in missing]}

        def _new_ref(value: dict) -> dict:
            return asdict(EpisodeRef(new_base.version_id, str(value["episode_uid"])))

        payload = workspace.to_dict()
        payload.update(
            base_dataset_version_id=new_base.version_id,
            base_fingerprint=new_base.fingerprint,
            revision=workspace.revision + 1,
            updated_at=_now(),
        )
        for item in payload["decisions"]:
            item["episode_ref"] = _new_ref(item["episode_ref"])
        for item in payload["annotation_patches"]:
            item["episode_ref"] = _new_ref(item["episode_ref"])
        for recipe in payload["repair_recipes"]:
            recipe["episode_refs"] = [_new_ref(item) for item in recipe.get("episode_refs") or []]
        rebased = CurationWorkspace.from_dict(payload)
        self._validate_workspace_payload(rebased)
        self._put_record("workspaces", workspace_id, rebased.to_dict())
        return {"rebased": True, "workspace": rebased.to_dict(), "conflicts": []}

    def publish_workspace(
        self,
        workspace_id: str,
        *,
        expected_revision: int,
        reviewer: str,
        reason: str | None = None,
    ) -> CurationManifestVersion:
        lease_owner = f"pid:{os.getpid()}:{threading.get_ident()}"
        lease_id = f"workspace-publish:{workspace_id}"
        if not self.repository.claim_job(
            lease_id,
            "workspace_publish",
            owner=lease_owner,
            lease_seconds=120,
        ):
            raise ValueError("workspace publish is already running in another process")
        try:
            workspace = self.get_workspace(workspace_id)
            if workspace.status != WORKSPACE_OPEN:
                raise ValueError("workspace has already been published")
            if workspace.revision != int(expected_revision):
                raise ValueError(
                    f"workspace revision conflict: expected {expected_revision}, current {workspace.revision}"
                )
            existing = [
                CurationManifestVersion.from_dict(item)
                for item in self._list_records("manifests")
                if item.get("workspace_id") == workspace.workspace_id
            ]
            manifest = next((item for item in existing if item.status == MANIFEST_PUBLISHED), None)
            if manifest is None:
                validation = self.validate_workspace(workspace_id)
                if validation["unresolved_review_count"]:
                    raise ValueError("workspace contains unresolved review decisions")
                base = self.get_version(workspace.base_dataset_version_id)
                normalized = self._normalize_workspace_decisions(base, workspace.decisions)
                include = [
                    item["episode_ref"]
                    for item in normalized
                    if item["decision"] in {"keep", "repair"}
                ]
                exclude = [
                    item["episode_ref"] for item in normalized if item["decision"] == "exclude"
                ]
                manifest = max(existing, key=lambda item: item.created_at) if existing else None
                if manifest is None:
                    manifest = self.create_manifest(
                        base.version_id,
                        include=include,
                        exclude=exclude,
                        annotation_patches=workspace.annotation_patches,
                        repair_recipes=workspace.repair_recipes,
                        evidence=workspace.evidence,
                        rule_versions=workspace.rule_versions,
                        model_versions=workspace.model_versions,
                        created_by=workspace.owner,
                        reason=reason,
                        workspace_id=workspace.workspace_id,
                    )
                if manifest.status == MANIFEST_DRAFT:
                    manifest = self.transition_manifest(manifest.manifest_id, MANIFEST_IN_REVIEW)
                if manifest.status == MANIFEST_IN_REVIEW:
                    manifest = self.transition_manifest(
                        manifest.manifest_id,
                        MANIFEST_APPROVED,
                        reviewer=reviewer,
                    )
                if manifest.status == MANIFEST_APPROVED:
                    manifest = self.transition_manifest(
                        manifest.manifest_id,
                        MANIFEST_PUBLISHED,
                        reviewer=reviewer,
                    )
            payload = workspace.to_dict()
            payload.update(
                status=WORKSPACE_PUBLISHED,
                published_manifest_id=manifest.manifest_id,
                updated_at=_now(),
            )
            self._put_record("workspaces", workspace.workspace_id, payload)
            return manifest
        finally:
            self.repository.release_job(lease_id, owner=lease_owner)

    def create_manifest(
        self,
        base_dataset_version_id: str,
        *,
        include: list[dict] | None = None,
        exclude: list[dict] | None = None,
        annotation_patches: list[dict] | None = None,
        repair_recipes: list[dict] | None = None,
        evidence: list[dict] | None = None,
        rule_versions: dict | None = None,
        model_versions: dict | None = None,
        created_by: str = "unknown",
        reason: str | None = None,
        supersedes_manifest_id: str | None = None,
        workspace_id: str | None = None,
    ) -> CurationManifestVersion:
        base = self.get_version(base_dataset_version_id)
        supersedes = self.get_manifest(supersedes_manifest_id) if supersedes_manifest_id else None
        if supersedes and supersedes.base_dataset_version_id != base.version_id:
            raise ValueError("superseded manifest must use the same base dataset version")
        manifest_version = (supersedes.manifest_version + 1) if supersedes else 1
        workspace_id = str(workspace_id or (supersedes.workspace_id if supersedes else base.dataset_id))
        normalized_include = self._validate_refs(base, include or [], "include")
        normalized_exclude = self._validate_refs(base, exclude or [], "exclude")
        include_uids = {item["episode_uid"] for item in normalized_include}
        exclude_uids = {item["episode_uid"] for item in normalized_exclude}
        overlap = sorted(include_uids & exclude_uids)
        if overlap:
            raise ValueError(f"episode cannot be both included and excluded: {overlap}")
        selected_uids = include_uids or base.episode_uids()
        selected_uids -= exclude_uids
        if not selected_uids:
            raise ValueError("manifest would select no episodes")

        normalized_patches = self._validate_annotation_patches(base, annotation_patches or [], selected_uids)
        normalized_repairs = self._validate_repair_recipes(base, repair_recipes or [], selected_uids)
        created_at = _now()
        manifest_id = f"cm_{uuid.uuid4().hex}"
        manifest_digest = _sha256(
            _canonical_json(
                {
                    "base_dataset_version_id": base.version_id,
                    "base_fingerprint": base.fingerprint,
                    "include": normalized_include,
                    "exclude": normalized_exclude,
                    "annotation_patches": normalized_patches,
                    "repair_recipes": normalized_repairs,
                    "evidence": list(evidence or []),
                    "rule_versions": dict(rule_versions or {}),
                    "model_versions": dict(model_versions or {}),
                }
            )
        )
        record = CurationManifestVersion(
            manifest_id=manifest_id,
            workspace_id=workspace_id,
            manifest_version=manifest_version,
            supersedes_manifest_id=supersedes_manifest_id,
            base_dataset_version_id=base.version_id,
            base_fingerprint=base.fingerprint,
            status=MANIFEST_DRAFT,
            include=normalized_include,
            exclude=normalized_exclude,
            annotation_patches=normalized_patches,
            repair_recipes=normalized_repairs,
            evidence=list(evidence or []),
            rule_versions=dict(rule_versions or {}),
            model_versions=dict(model_versions or {}),
            reviewer=None,
            reason=str(reason) if reason else None,
            created_by=str(created_by or "unknown"),
            created_at=created_at,
            updated_at=created_at,
            manifest_digest=manifest_digest,
        )
        self._put_record("manifests", manifest_id, record.to_dict(), immutable=True)
        return record

    def _validate_refs(self, base: DatasetVersion, refs: list[dict], field_name: str) -> list[dict]:
        normalized = []
        seen = set()
        valid_uids = base.episode_uids()
        for value in refs:
            ref = EpisodeRef.from_dict(value)
            if ref.dataset_version_id != base.version_id:
                raise ValueError(f"{field_name} reference must target base dataset version")
            if ref.episode_uid not in valid_uids:
                raise ValueError(f"{field_name} references unknown episode_uid: {ref.episode_uid}")
            if ref.episode_uid in seen:
                raise ValueError(f"duplicate {field_name} reference: {ref.episode_uid}")
            seen.add(ref.episode_uid)
            normalized.append(asdict(ref))
        return normalized

    def _validate_annotation_patches(
        self,
        base: DatasetVersion,
        patches: list[dict],
        selected_uids: set[str],
    ) -> list[dict]:
        normalized = []
        seen_fields = set()
        for patch in patches:
            if not isinstance(patch, dict):
                raise ValueError("annotation patch must be an object")
            ref = EpisodeRef.from_dict(patch.get("episode_ref") or {})
            self._validate_refs(base, [asdict(ref)], "annotation patch")
            if ref.episode_uid not in selected_uids:
                raise ValueError(f"annotation patch targets excluded episode: {ref.episode_uid}")
            raw_fields = patch.get("fields")
            if not isinstance(raw_fields, dict) or not raw_fields:
                raise ValueError("annotation patch requires non-empty fields")
            fields = dict(raw_fields)
            unsupported = sorted(set(fields) - _ALLOWED_ANNOTATION_FIELDS)
            if unsupported:
                raise ValueError(f"unsupported annotation patch fields: {unsupported}")
            if "prompt" in fields:
                if "task" in fields or "tasks" in fields:
                    raise ValueError("annotation patch cannot combine prompt with task/tasks")
                fields["task"] = fields.pop("prompt")
            for name in fields:
                key = (ref.episode_uid, str(name))
                if key in seen_fields:
                    raise ValueError(f"conflicting annotation patch: {ref.episode_uid}.{name}")
                seen_fields.add(key)
            normalized.append({"episode_ref": asdict(ref), "fields": fields})
        return normalized

    def _validate_repair_recipes(
        self,
        base: DatasetVersion,
        recipes: list[dict],
        selected_uids: set[str],
    ) -> list[dict]:
        normalized = []
        edit_targets = set()
        trim_targets = set()
        features = load_json(Path(base.root) / "meta" / "info.json").get("features") or {}
        for recipe in recipes:
            if not isinstance(recipe, dict):
                raise ValueError("repair recipe must be an object")
            operation = str(recipe.get("op") or "").strip()
            if operation not in _SUPPORTED_REPAIR_OPS:
                raise ValueError(f"unsupported repair op: {operation}")
            refs = self._validate_refs(base, recipe.get("episode_refs") or [], "repair recipe")
            target_uids = {item["episode_uid"] for item in refs} or set(selected_uids)
            excluded_targets = sorted(target_uids - selected_uids)
            if excluded_targets:
                raise ValueError(f"repair recipe targets excluded episodes: {excluded_targets}")
            params = dict(recipe.get("params") or {})
            if operation == "trim":
                if not refs:
                    raise ValueError("trim repair requires explicit episode_refs")
                try:
                    start_frame = int(params["start_frame"])
                    end_frame = int(params["end_frame"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("trim repair requires integer start_frame and end_frame") from exc
                if start_frame < 0 or end_frame < start_frame:
                    raise ValueError("trim repair requires 0 <= start_frame <= end_frame")
                duplicate = sorted(target_uids & trim_targets)
                if duplicate:
                    raise ValueError(f"duplicate trim repair for episode(s): {duplicate}")
                trim_targets.update(target_uids)
                params = {**params, "start_frame": start_frame, "end_frame": end_frame}
            if operation == "value_edit":
                edits = params.get("edits")
                if not isinstance(edits, list) or not edits:
                    raise ValueError("value_edit repair requires non-empty params.edits")
                normalized_edits = []
                for edit in edits:
                    if not isinstance(edit, dict):
                        raise ValueError("value_edit entries must be objects")
                    field_name = str(edit.get("field") or edit.get("field_name") or "").strip()
                    if field_name not in {"action", "state"}:
                        raise ValueError("value_edit field must be action or state")
                    try:
                        dimension = int(edit["dimension"])
                        value = float(edit["value"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("value_edit requires field, dimension, and numeric value") from exc
                    shape = (features.get(field_name) or {}).get("shape") or []
                    if not shape or dimension < 0 or dimension >= int(shape[0]):
                        raise ValueError(f"value_edit dimension is out of range for {field_name}")
                    if not np.isfinite(value):
                        raise ValueError("value_edit value must be finite")
                    for uid in target_uids:
                        key = (uid, field_name, dimension)
                        if key in edit_targets:
                            raise ValueError(
                                f"conflicting value_edit repair: {uid}.{field_name}[{dimension}]"
                            )
                        edit_targets.add(key)
                    normalized_edits.append({"field": field_name, "dimension": dimension, "value": value})
                params["edits"] = normalized_edits
            normalized.append({"op": operation, "episode_refs": refs, "params": params})
        return normalized

    def get_manifest(self, manifest_id: str | None) -> CurationManifestVersion:
        if not manifest_id:
            raise KeyError("manifest_id is required")
        payload = self._get_record("manifests", str(manifest_id))
        if payload is None:
            raise KeyError(f"curation manifest not found: {manifest_id}")
        return CurationManifestVersion.from_dict(payload)

    def list_manifests(
        self,
        *,
        base_dataset_version_id: str | None = None,
    ) -> list[dict]:
        materialized_ids = {
            str(item.get("manifest_id"))
            for item in self._list_records("materializations")
            if item.get("status") == MATERIALIZATION_COMMITTED
        }
        records = []
        for item in self._list_records("manifests"):
            manifest = CurationManifestVersion.from_dict(item)
            if base_dataset_version_id and manifest.base_dataset_version_id != base_dataset_version_id:
                continue
            payload = manifest.to_dict()
            payload["lifecycle_status"] = (
                MANIFEST_MATERIALIZED if manifest.manifest_id in materialized_ids else manifest.status
            )
            records.append(payload)
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def transition_manifest(
        self,
        manifest_id: str,
        target_status: str,
        *,
        reviewer: str | None = None,
    ) -> CurationManifestVersion:
        target_status = str(target_status).strip().lower()
        if target_status not in MANIFEST_STATES - {MANIFEST_MATERIALIZED}:
            raise ValueError(f"unsupported manifest status: {target_status}")
        with self.repository.transaction() as connection:
            payload = self._get_record("manifests", manifest_id, connection=connection)
            if payload is None:
                raise KeyError(f"curation manifest not found: {manifest_id}")
            manifest = CurationManifestVersion.from_dict(payload)
            if manifest.status == MANIFEST_PUBLISHED:
                raise ValueError("published manifests are immutable")
            expected = _MANIFEST_TRANSITIONS.get(manifest.status)
            if target_status != expected:
                raise ValueError(f"manifest transition must be {manifest.status} -> {expected}")
            if target_status in {MANIFEST_APPROVED, MANIFEST_PUBLISHED} and not reviewer:
                raise ValueError("reviewer is required to approve or publish a manifest")
            if target_status == MANIFEST_PUBLISHED:
                base = self.get_version(manifest.base_dataset_version_id)
                self.assert_version_current(base)
            updated = manifest.to_dict()
            updated.update(
                status=target_status,
                reviewer=str(reviewer) if reviewer else manifest.reviewer,
                updated_at=_now(),
            )
            self._put_record(
                "manifests",
                manifest.manifest_id,
                updated,
                connection=connection,
            )
        return self.get_manifest(manifest_id)

    def plan_materialization(
        self,
        manifest: CurationManifestVersion,
        profile: ProfileVersion,
        out_root: Path,
    ) -> MaterializationRun:
        if not manifest.manifest_digest:
            raise ValueError("manifest has no content digest")
        idempotency_key = _sha256(
            f"{manifest.base_dataset_version_id}:{manifest.manifest_digest}:{profile.digest}"
        )
        resolved_output = Path(out_root).expanduser().resolve()
        materialization_id = f"mat_{_sha256(f'{idempotency_key}:{resolved_output}')[:24]}"
        staging_root = resolved_output.with_name(f".{resolved_output.name}.materializing-{materialization_id}")
        with self.repository.transaction() as connection:
            existing = self._get_record(
                "materializations",
                materialization_id,
                connection=connection,
            )
            if existing is not None:
                return MaterializationRun.from_dict(existing)
            now = _now()
            run = MaterializationRun(
                materialization_id=materialization_id,
                idempotency_key=idempotency_key,
                manifest_id=manifest.manifest_id,
                base_dataset_version_id=manifest.base_dataset_version_id,
                profile_id=profile.profile_id,
                profile_digest=profile.digest,
                output_root=str(resolved_output),
                staging_root=str(staging_root),
                status=MATERIALIZATION_PLANNED,
                created_at=now,
                updated_at=now,
            )
            self._put_record(
                "materializations",
                materialization_id,
                run.to_dict(),
                immutable=True,
                connection=connection,
            )
        return run

    def get_materialization(self, materialization_id: str) -> MaterializationRun:
        payload = self._get_record("materializations", str(materialization_id))
        if payload is None:
            raise KeyError(f"materialization run not found: {materialization_id}")
        return MaterializationRun.from_dict(payload)

    def update_materialization(
        self,
        materialization_id: str,
        status: str,
        *,
        output_dataset_version_id: str | None = None,
        error: str | None = None,
        details: dict | None = None,
        connection=None,
    ) -> MaterializationRun:
        allowed = {
            MATERIALIZATION_PLANNED: {MATERIALIZATION_RUNNING, MATERIALIZATION_FAILED},
            MATERIALIZATION_RUNNING: {MATERIALIZATION_VALIDATING, MATERIALIZATION_FAILED},
            MATERIALIZATION_VALIDATING: {MATERIALIZATION_COMMITTED, MATERIALIZATION_FAILED},
            MATERIALIZATION_FAILED: {MATERIALIZATION_RUNNING},
            MATERIALIZATION_COMMITTED: set(),
        }

        def _update(active) -> MaterializationRun:
            payload = self._get_record(
                "materializations",
                materialization_id,
                connection=active,
            )
            if payload is None:
                raise KeyError(f"materialization run not found: {materialization_id}")
            current = MaterializationRun.from_dict(payload)
            if status == current.status:
                return current
            if status not in allowed.get(current.status, set()):
                raise ValueError(f"invalid materialization transition: {current.status} -> {status}")
            updated_payload = current.to_dict()
            updated_payload.update(
                status=status,
                output_dataset_version_id=(
                    output_dataset_version_id or current.output_dataset_version_id
                ),
                error=error,
                details={**current.details, **dict(details or {})},
                updated_at=_now(),
            )
            self._put_record(
                "materializations",
                materialization_id,
                updated_payload,
                connection=active,
            )
            return MaterializationRun.from_dict(updated_payload)

        if connection is not None:
            return _update(connection)
        with self.repository.transaction() as active:
            return _update(active)

    def list_materializations(self, *, manifest_id: str | None = None) -> list[dict]:
        records = self._list_records("materializations")
        if manifest_id:
            records = [item for item in records if item.get("manifest_id") == manifest_id]
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)

    def publish_feedback(
        self,
        dataset_version_id: str,
        *,
        training_run: dict,
        failures: list[dict] | None = None,
        coverage_gaps: list[dict] | None = None,
        collection_brief: dict | None = None,
        created_by: str = "unknown",
        manifest_id: str | None = None,
    ) -> FeedbackReport:
        version = self.get_version(dataset_version_id)
        expected_manifest_id = (
            str(version.profile.get("manifest_id") or "") if version.operation == "materialize" else ""
        )
        if expected_manifest_id:
            if manifest_id and manifest_id != expected_manifest_id:
                raise ValueError("feedback manifest does not match the curated dataset version")
            manifest_id = expected_manifest_id
        if manifest_id:
            manifest = self.get_manifest(manifest_id)
            if manifest.status != MANIFEST_PUBLISHED:
                raise ValueError("feedback manifest must be published")
            materializations = self.list_materializations(manifest_id=manifest.manifest_id)
            if not any(
                item.get("output_dataset_version_id") == version.version_id for item in materializations
            ):
                raise ValueError("feedback manifest did not materialize the requested dataset version")
        if not isinstance(training_run, dict) or not training_run:
            raise ValueError("training_run is required")
        normalized_failures = []
        for failure in failures or []:
            if not isinstance(failure, dict):
                raise ValueError("failure entries must be objects")
            ref = EpisodeRef.from_dict(failure.get("episode_ref") or {})
            if ref.dataset_version_id != version.version_id or ref.episode_uid not in version.episode_uids():
                raise ValueError("failure episode_ref must belong to the feedback dataset version")
            normalized_failures.append({**failure, "episode_ref": asdict(ref)})
        normalized_gaps = list(coverage_gaps or [])
        if not all(isinstance(item, dict) for item in normalized_gaps):
            raise ValueError("coverage gap entries must be objects")
        normalized_brief = dict(collection_brief or {})
        if not normalized_brief:
            normalized_brief = _generate_collection_brief(normalized_failures, normalized_gaps)
        report = FeedbackReport(
            feedback_id=f"fb_{uuid.uuid4().hex}",
            training_run=dict(training_run),
            dataset_version_id=version.version_id,
            manifest_id=manifest_id,
            failures=normalized_failures,
            coverage_gaps=normalized_gaps,
            collection_brief=normalized_brief,
            created_by=str(created_by or "unknown"),
            created_at=_now(),
            dataset_recommendations=_feedback_recommendations(normalized_failures, normalized_gaps),
        )
        self._put_record("feedback", report.feedback_id, report.to_dict(), immutable=True)
        return report

    def list_feedback(self, *, dataset_version_id: str | None = None) -> list[FeedbackReport]:
        records = [FeedbackReport.from_dict(item) for item in self._list_records("feedback")]
        if dataset_version_id:
            records = [item for item in records if item.dataset_version_id == dataset_version_id]
        return sorted(records, key=lambda item: item.created_at, reverse=True)


def execute_preprocessing_profile(
    store: LifecycleStore,
    base_dataset_version_id: str,
    profile_id: str,
    out_root: Path,
    *,
    execution_options: dict | None = None,
    progress_callback=None,
) -> tuple[DatasetVersion, dict]:
    """Execute one immutable Raw-to-Standard profile and register explicit lineage."""
    base = store.get_version(base_dataset_version_id)
    store.assert_version_current(base)
    profile = store.get_profile(profile_id, kind=PROFILE_PREPROCESSING)
    options = dict(execution_options or {})
    workers = max(1, int(options.get("workers") or 8))
    out_root = Path(out_root).expanduser().resolve()
    if out_root.exists():
        raise FileExistsError(f"Output dataset already exists: {out_root}")
    if out_root == Path(base.root).resolve() or Path(base.root).resolve() in out_root.parents:
        raise ValueError("preprocessing output must be a sibling outside the base dataset root")
    out_root.parent.mkdir(parents=True, exist_ok=True)
    staging = out_root.with_name(f".{out_root.name}.profile-{uuid.uuid4().hex}")
    lease_id = f"preprocess:{_sha256(f'{base.version_id}:{profile.digest}:{out_root}')[:24]}"
    lease_owner = f"pid:{os.getpid()}:{threading.get_ident()}"
    if not store.repository.claim_job(
        lease_id,
        "preprocess_profile",
        payload={"output_root": str(out_root), "profile_id": profile.profile_id},
        owner=lease_owner,
        lease_seconds=300,
    ):
        raise ValueError("preprocessing profile is already running in another process")

    def _progress(payload: dict) -> None:
        store.repository.heartbeat_job(lease_id, owner=lease_owner, lease_seconds=300)
        if progress_callback is not None:
            progress_callback(payload)

    current_root = Path(base.root)
    current_to_base = {index: index for index in base.uid_by_index()}
    step_summaries = []
    try:
        for position, step in enumerate(profile.steps):
            operation = step["op"]
            params = dict(step.get("params") or {})
            step_out = out_root if position == len(profile.steps) - 1 else staging / f"step-{position:03d}"
            if operation == "copy":
                result = run_merge(
                    [current_root],
                    out_root=step_out,
                    workers=workers,
                    _allow_single_source=True,
                    _op="profile_copy",
                    _default_op="profile_copy",
                    progress_callback=_progress,
                )
            elif operation == "split":
                result = run_split(
                    current_root,
                    out_root=step_out,
                    episode_range=params.get("episode_range"),
                    task_filter=params.get("task_filter"),
                    progress_callback=_progress,
                )
            elif operation == "standardize":
                result = run_standardize_dataset(
                    current_root,
                    out_root=step_out,
                    data_version=params.get("data_version"),
                    workers=workers,
                    progress_callback=_progress,
                )
                # Standardization is a semantic pipeline, not only a 16D projection.
                # Reuse the same Stage/index executor as the CLI and web flow.
                from lerobot.data_platform.cli import run_precompute

                run_precompute(
                    root=step_out,
                    repo_id=result.repo_id,
                    output_dir=staging / f"step-{position:03d}-cache",
                    prepare_videos=False,
                    prepare_csv=True,
                    prepare_workers=workers,
                    fix_episode_indices_enabled=True,
                    annotate=True,
                    write_parquet=True,
                    force_recompute_stage=True,
                    write_subtask=True,
                    overwrite_csv=True,
                    data_version=str(result.summary["data_version"]),
                    progress_callback=_progress,
                    show_progress=False,
                )
            elif operation == "convert_action":
                result = run_convert_action(
                    current_root,
                    out_root=step_out,
                    target_dim=int(params.get("target_dim") or 16),
                    progress_callback=_progress,
                )
            elif operation == "drop_field":
                result = run_drop_field(
                    current_root,
                    out_root=step_out,
                    field_name=str(params.get("field_name") or ""),
                    progress_callback=_progress,
                )
            else:  # create_profile rejects this; keep execution defensive.
                raise ValueError(f"unsupported preprocessing profile step: {operation}")

            lineage = list(result.episode_lineage or [])
            if not lineage:
                source_indices = sorted(int(item["episode_index"]) for item in load_episode_records(current_root))
                output_indices = sorted(int(item["episode_index"]) for item in load_episode_records(step_out))
                if source_indices != output_indices:
                    raise ValueError(
                        f"profile step {operation} changed episode identity without explicit lineage"
                    )
                lineage = [
                    {"source_episode_index": index, "output_episode_index": index}
                    for index in output_indices
                ]
            next_to_base = {}
            for item in lineage:
                source_index = int(item["source_episode_index"])
                output_index = int(item["output_episode_index"])
                if source_index not in current_to_base:
                    raise ValueError(
                        f"profile step {operation} references unknown source episode {source_index}"
                    )
                next_to_base[output_index] = current_to_base[source_index]
            current_to_base = next_to_base
            current_root = step_out
            step_summaries.append({"op": operation, "summary": dict(result.summary or {})})

        lineage_to_base = [
            {
                "source_dataset_version_id": base.version_id,
                "source_episode_index": source_index,
                "output_episode_index": output_index,
            }
            for output_index, source_index in sorted(current_to_base.items())
        ]
        output_stage = "curated" if base.stage == "curated" else "standard"
        version = store.register_derived(
            out_root,
            f"local/{out_root.name}",
            parent_version_ids=[base.version_id],
            operation="preprocess_profile",
            profile={"profile_id": profile.profile_id, "steps": step_summaries},
            stage=output_stage,
            episode_lineage=lineage_to_base,
            profile_id=profile.profile_id,
            profile_digest=profile.digest,
            executor_digest=profile.executor_digest,
        )
        return version, {
            "profile_id": profile.profile_id,
            "profile_digest": profile.digest,
            "steps": step_summaries,
            "output_root": str(out_root),
        }
    except Exception:
        if out_root.exists():
            shutil.rmtree(out_root)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        store.repository.release_job(lease_id, owner=lease_owner)


def _legacy_meta(root: Path):
    info = load_json(root / "meta" / "info.json")

    class _LegacyMeta:
        def get_data_file_path(self, episode_index: int) -> Path:
            return format_data_path(info, int(episode_index))

    return _LegacyMeta()


def _metadata(root: Path):
    return V3DatasetMetadata(f"local/{root.name}", root) if is_v3_dataset(root) else _legacy_meta(root)


def _set_episode_task(root: Path, episode_index: int, task: str) -> None:
    task = str(task).strip()
    if not task:
        raise ValueError("task annotation cannot be empty")
    tasks = load_task_records(root)
    task_to_index = {str(item["task"]): int(item["task_index"]) for item in tasks}
    if task not in task_to_index:
        task_to_index[task] = max(task_to_index.values(), default=-1) + 1
        tasks.append({"task_index": task_to_index[task], "task": task})
        write_task_records(root, tasks)
    episodes = load_episode_records(root)
    found = False
    for item in episodes:
        if int(item["episode_index"]) == int(episode_index):
            item["tasks"] = [task]
            found = True
            break
    if not found:
        raise ValueError(f"episode {episode_index} not found while applying task annotation")
    write_episode_records(root, episodes)

    meta = _metadata(root)
    table = read_episode_table(root, meta, episode_index, columns=["task_index"])
    if "task_index" in table.column_names:
        replace_episode_column(
            root,
            meta,
            episode_index,
            "task_index",
            [task_to_index[task]] * table.num_rows,
        )


def _states_from_transitions(timestamps: list[float], transitions: list[dict]) -> list[int]:
    normalized = sorted(
        (
            {"time": float(item["time"]), "state": int(item["state"])}
            for item in transitions
            if isinstance(item, dict) and "time" in item and "state" in item
        ),
        key=lambda item: item["time"],
    )
    if not normalized:
        raise ValueError("subtask_transitions must contain time/state entries")
    states = []
    for timestamp in timestamps:
        state = normalized[0]["state"]
        for transition in normalized:
            if transition["time"] <= float(timestamp):
                state = transition["state"]
            else:
                break
        states.append(state)
    return states


def _set_subtask_state(root: Path, episode_index: int, values: list[int]) -> None:
    meta = _metadata(root)
    table = read_episode_table(root, meta, episode_index)
    states = [int(value) for value in values]
    if len(states) != table.num_rows:
        raise ValueError(
            f"subtask_state length {len(states)} does not match episode {episode_index} rows {table.num_rows}"
        )
    upsert_episode_column(root, meta, episode_index, "subtask_state", states, pa.int32())
    values_array = np.asarray(states, dtype=np.float64)
    update_info_features(
        root,
        {"subtask_state": {"dtype": "int32", "shape": [1], "names": None}},
    )
    update_episode_stats_for_subtask_state(
        root,
        {
            int(episode_index): {
                "min": [int(values_array.min())],
                "max": [int(values_array.max())],
                "mean": [float(values_array.mean())],
                "std": [float(values_array.std())],
                "count": [len(states)],
            }
        },
    )


def _apply_annotation_patches(
    root: Path,
    manifest: CurationManifestVersion,
    output_index_by_uid: dict[str, int],
) -> list[int]:
    metadata_updates: dict[int, dict] = {}
    patched = []
    for patch in manifest.annotation_patches:
        uid = patch["episode_ref"]["episode_uid"]
        episode_index = output_index_by_uid[uid]
        fields = dict(patch["fields"])
        task = fields.pop("task", None)
        tasks = fields.pop("tasks", None)
        if task is None and isinstance(tasks, list) and tasks:
            task = tasks[0]
        if task is not None:
            _set_episode_task(root, episode_index, str(task))
        states = fields.pop("subtask_state", None)
        transitions = fields.pop("subtask_transitions", None)
        if transitions is not None:
            meta = _metadata(root)
            table = read_episode_table(root, meta, episode_index, columns=["timestamp"])
            states = _states_from_transitions(table["timestamp"].to_pylist(), transitions)
        if states is not None:
            _set_subtask_state(root, episode_index, list(states))
        if fields:
            metadata_updates.setdefault(episode_index, {}).update(fields)
        patched.append(episode_index)
    if metadata_updates:
        update_episode_metadata(root, metadata_updates)
    return sorted(set(patched))


def _apply_repair_recipes(
    root: Path,
    manifest: CurationManifestVersion,
    output_index_by_uid: dict[str, int],
    *,
    workers: int,
) -> list[dict]:
    results = []
    for recipe in manifest.repair_recipes:
        operation = recipe["op"]
        target_uids = [item["episode_uid"] for item in recipe["episode_refs"]]
        if not target_uids:
            target_uids = sorted(output_index_by_uid)
        episode_ids = [output_index_by_uid[uid] for uid in target_uids]
        if operation == "trim":
            params = recipe["params"]
            per_episode = []
            for episode_index in episode_ids:
                per_episode.append(
                    trim_episode_inplace(
                        root,
                        episode_index,
                        int(params["start_frame"]),
                        int(params["end_frame"]),
                        workers=workers,
                    )
                )
            results.append({"op": operation, "episodes": episode_ids, "results": per_episode})
        elif operation == "value_edit":
            result = run_value_edits(
                root,
                edits=list(recipe["params"]["edits"]),
                episode_ids=episode_ids,
                in_place=True,
            )
            results.append({"op": operation, "episodes": episode_ids, "summary": result.summary})
    return results


def validate_materialized_dataset(root: Path, *, expected_episode_count: int) -> dict:
    """Perform structural, temporal, row-count, and full-file validation."""
    root = validate_dataset_root(root)
    info = load_json(root / "meta" / "info.json")
    episodes = load_episode_records(root)
    if len(episodes) != int(expected_episode_count):
        raise ValueError(
            f"materialized episode count mismatch: expected {expected_episode_count}, found {len(episodes)}"
        )
    indices = [int(item["episode_index"]) for item in episodes]
    if indices != list(range(len(episodes))):
        raise ValueError("materialized episode indices must be contiguous and zero-based")
    if int(info.get("total_episodes") or 0) != len(episodes):
        raise ValueError("meta/info total_episodes does not match episode metadata")
    meta = _metadata(root)
    declared_features = dict(info.get("features") or {})
    required_table_features = {
        name
        for name, feature in declared_features.items()
        if str((feature or {}).get("dtype") or "") != "video"
    }
    total_frames = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        expected_rows = int(episode.get("length") or 0)
        parquet_path = root / meta.get_data_file_path(episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing source parquet: {parquet_path}")
        table = read_episode_table(root, meta, episode_index)
        missing_features = sorted(required_table_features - set(table.column_names))
        if missing_features:
            raise ValueError(
                f"episode {episode_index} parquet is missing declared features: {missing_features}"
            )
        if table.num_rows != expected_rows:
            raise ValueError(
                f"episode {episode_index} row count mismatch: expected {expected_rows}, found {table.num_rows}"
            )
        if "episode_index" in table.column_names:
            values = {int(value) for value in table["episode_index"].to_pylist()}
            if values != {episode_index}:
                raise ValueError(f"episode {episode_index} parquet contains invalid episode_index values")
        if "timestamp" in table.column_names:
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            if not np.isfinite(timestamps).all():
                raise ValueError(f"episode {episode_index} contains non-finite timestamps")
            if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
                raise ValueError(f"episode {episode_index} timestamps are not monotonic")
        for name in ("action", "state"):
            if name not in table.column_names or name not in declared_features:
                continue
            shape = list((declared_features[name] or {}).get("shape") or [])
            if not shape or not table.num_rows:
                continue
            first_value = table[name][0].as_py()
            if isinstance(first_value, list) and len(first_value) != int(shape[0]):
                raise ValueError(
                    f"episode {episode_index} {name} width does not match declared schema"
                )
        total_frames += table.num_rows
    if int(info.get("total_frames") or 0) != total_frames:
        raise ValueError("meta/info total_frames does not match parquet row counts")
    content_manifest = build_content_manifest(root)
    return {
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "content_manifest": content_manifest,
    }


def materialize_manifest(
    store: LifecycleStore,
    manifest_id: str,
    out_root: Path,
    *,
    profile_id: str | None = None,
    profile: dict | None = None,
    workers: int = 8,
    progress_callback=None,
) -> tuple[DatasetVersion, dict]:
    """Idempotently materialize one published manifest into an immutable replica."""
    manifest = store.get_manifest(manifest_id)
    if manifest.status != MANIFEST_PUBLISHED:
        raise ValueError("only published manifests can be materialized")
    requested_profile_id = profile_id or str((profile or {}).get("profile_id") or "")
    materialization_profile = (
        store.get_profile(requested_profile_id, kind=PROFILE_MATERIALIZATION)
        if requested_profile_id
        else store.default_materialization_profile()
    )
    base = store.get_version(manifest.base_dataset_version_id)
    if base.fingerprint != manifest.base_fingerprint:
        raise ValueError("manifest base fingerprint does not match its dataset version")
    store.assert_version_current(base)

    out_root = Path(out_root).expanduser().resolve()
    resolved_output = out_root
    resolved_base = Path(base.root).resolve()
    if resolved_output == resolved_base or resolved_base in resolved_output.parents:
        raise ValueError("materialized output must be a sibling path outside the base dataset root")
    out_root.parent.mkdir(parents=True, exist_ok=True)
    include_uids = {item["episode_uid"] for item in manifest.include}
    exclude_uids = {item["episode_uid"] for item in manifest.exclude}
    selected_uids = include_uids or base.episode_uids()
    selected_uids -= exclude_uids
    base_index_by_uid = base.index_by_uid()
    selected_indices = sorted(base_index_by_uid[uid] for uid in selected_uids)
    all_indices = set(base.uid_by_index())
    excluded_indices = sorted(all_indices - set(selected_indices))
    uid_by_output_index = {
        new_index: base.uid_by_index()[old_index] for new_index, old_index in enumerate(selected_indices)
    }
    output_index_by_uid = {uid: index for index, uid in uid_by_output_index.items()}

    run = store.plan_materialization(manifest, materialization_profile, out_root)
    if run.status == MATERIALIZATION_COMMITTED:
        version = store.get_version(str(run.output_dataset_version_id))
        if not out_root.is_dir():
            raise ValueError(
                "materialization is committed but the requested replica is unavailable; "
                "choose a new output path"
            )
        store.assert_version_current(
            DatasetVersion.from_dict({**version.to_dict(), "root": str(out_root)})
        )
        return version, {**run.to_dict(), **run.details}

    lease_owner = f"pid:{os.getpid()}:{threading.get_ident()}"
    if not store.repository.claim_job(
        run.materialization_id,
        "materialization",
        payload={"output_root": str(out_root)},
        owner=lease_owner,
        lease_seconds=300,
    ):
        raise ValueError("materialization is already running in another process")

    def _progress(payload: dict) -> None:
        store.repository.heartbeat_job(
            run.materialization_id,
            owner=lease_owner,
            lease_seconds=300,
        )
        if progress_callback is not None:
            progress_callback(payload)

    if run.status == MATERIALIZATION_PLANNED and out_root.exists():
        store.repository.release_job(run.materialization_id, owner=lease_owner)
        raise FileExistsError(f"Output dataset already exists: {out_root}")
    building_root = Path(run.staging_root)
    try:
        if run.status in {MATERIALIZATION_PLANNED, MATERIALIZATION_FAILED}:
            run = store.update_materialization(
                run.materialization_id,
                MATERIALIZATION_RUNNING,
            )
        if out_root.exists():
            validation = validate_materialized_dataset(
                out_root,
                expected_episode_count=len(selected_indices),
            )
            if run.status == MATERIALIZATION_RUNNING:
                run = store.update_materialization(
                    run.materialization_id,
                    MATERIALIZATION_VALIDATING,
                )
            result_summary = dict(run.details.get("preprocess_summary") or {})
            patched_episodes = list(run.details.get("patched_episodes") or [])
            repair_results = list(run.details.get("repair_results") or [])
        else:
            if building_root.exists():
                shutil.rmtree(building_root)
            result = run_merge(
                [Path(base.root)],
                out_root=building_root,
                workers=max(1, int(workers or 1)),
                exclude_episodes=[excluded_indices],
                _allow_single_source=True,
                _op="materialize",
                _default_op="materialize",
                _summary_extra={"manifest_id": manifest.manifest_id},
                progress_callback=_progress,
            )
            patched_episodes = _apply_annotation_patches(
                building_root,
                manifest,
                output_index_by_uid,
            )
            repair_results = _apply_repair_recipes(
                building_root,
                manifest,
                output_index_by_uid,
                workers=max(1, int(workers or 1)),
            )
            run = store.update_materialization(
                run.materialization_id,
                MATERIALIZATION_VALIDATING,
                details={
                    "selected_episode_count": len(selected_indices),
                    "excluded_episode_count": len(excluded_indices),
                    "patched_episodes": patched_episodes,
                    "repair_results": repair_results,
                    "preprocess_summary": result.summary,
                },
            )
            validation = validate_materialized_dataset(
                building_root,
                expected_episode_count=len(selected_indices),
            )
            result_summary = dict(result.summary)
            building_root.replace(out_root)

        content_manifest = validation["content_manifest"]
        content_path = store._content_manifest_path(content_manifest["dataset_fingerprint"])
        if not content_path.is_file():
            _atomic_write_json(content_path, content_manifest)
        lineage_by_index = {
            output_index: asdict(EpisodeRef(base.version_id, uid))
            for output_index, uid in uid_by_output_index.items()
        }
        with store.repository.transaction() as connection:
            version = store.ingest(
                out_root,
                f"local/{out_root.name}",
                stage="curated",
                dataset_id=base.dataset_id,
                parent_version_ids=[base.version_id],
                operation="materialize",
                profile={
                    "manifest_id": manifest.manifest_id,
                    "profile_id": materialization_profile.profile_id,
                },
                episode_uid_by_index=uid_by_output_index,
                episode_lineage_by_index=lineage_by_index,
                profile_id=materialization_profile.profile_id,
                profile_digest=materialization_profile.digest,
                executor_digest=materialization_profile.executor_digest,
                connection=connection,
            )
            details = {
                "selected_episode_count": len(selected_indices),
                "excluded_episode_count": len(excluded_indices),
                "patched_episodes": patched_episodes,
                "repair_results": repair_results,
                "preprocess_summary": result_summary,
                "content_manifest_uri": str(content_path),
                "content_fingerprint": content_manifest["dataset_fingerprint"],
            }
            run = store.update_materialization(
                run.materialization_id,
                MATERIALIZATION_COMMITTED,
                output_dataset_version_id=version.version_id,
                details=details,
                connection=connection,
            )
        return version, {**run.to_dict(), **run.details}
    except Exception as exc:
        if building_root.exists():
            shutil.rmtree(building_root)
        current = store.get_materialization(run.materialization_id)
        if current.status != MATERIALIZATION_COMMITTED:
            try:
                store.update_materialization(
                    run.materialization_id,
                    MATERIALIZATION_FAILED,
                    error=str(exc),
                )
            except ValueError:
                pass
        raise
    finally:
        store.repository.release_job(run.materialization_id, owner=lease_owner)


def _json_file(path: Path, default):
    try:
        return _read_json(path) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def collect_curation_sidecars(static_dir: Path, base: DatasetVersion) -> tuple[list[dict], list[dict]]:
    """Collect current sidecar decisions without mutating the source dataset."""
    static_dir = Path(static_dir)
    uid_by_index = base.uid_by_index()
    fields_by_uid: dict[str, dict] = {}

    def add_fields(episode_index: int, fields: dict) -> None:
        uid = uid_by_index.get(int(episode_index))
        if uid is not None and fields:
            fields_by_uid.setdefault(uid, {}).update(fields)

    subtask_annotations = _json_file(static_dir / "subtask_annotations.json", {})
    if isinstance(subtask_annotations, dict):
        for episode_index, transitions in subtask_annotations.items():
            add_fields(int(episode_index), {"subtask_transitions": transitions})

    pending = _json_file(static_dir / "prompt_assignments_pending.json", {})
    for item in pending.get("assignments", []) if isinstance(pending, dict) else []:
        if isinstance(item, dict) and item.get("selected_task"):
            add_fields(int(item["episode_index"]), {"task": str(item["selected_task"])})

    try:
        from lerobot.data_platform.precompute.labeling.review import (
            first_frame_bbox_from_record,
            load_labels_jsonl,
            resolved_reviewed_path,
        )

        labeling_dir = static_dir / "labeling"
        reviewed = load_labels_jsonl(resolved_reviewed_path(labeling_dir))
        for episode_index, record in reviewed.items():
            add_fields(
                int(episode_index),
                {"first_frame_bbox": first_frame_bbox_from_record(record, source="reviewed")},
            )
    except (FileNotFoundError, OSError, ValueError):
        pass

    try:
        from lerobot.data_platform.precompute.tagging.review import (
            load_tags_jsonl,
        )
        from lerobot.data_platform.precompute.tagging.review import (
            resolved_reviewed_path as tagging_reviewed_path,
        )

        tagging_dir = static_dir / "tagging"
        reviewed_tags = load_tags_jsonl(tagging_reviewed_path(tagging_dir))
        for episode_index, record in reviewed_tags.items():
            add_fields(int(episode_index), {"tags": dict(record.get("tags") or {})})
    except (FileNotFoundError, OSError, ValueError):
        pass

    evidence = []
    flagged = _json_file(static_dir / "flagged_episodes.json", {})
    flagged_ids = flagged.get("flagged_episodes", []) if isinstance(flagged, dict) else []
    reasons = flagged.get("flag_reasons", {}) if isinstance(flagged, dict) else {}
    for episode_index in flagged_ids:
        uid = uid_by_index.get(int(episode_index))
        if uid is not None:
            evidence.append(
                {
                    "episode_ref": asdict(EpisodeRef(base.version_id, uid)),
                    "kind": "quality_flag",
                    "reasons": reasons.get(str(episode_index), []),
                }
            )

    patches = [
        {
            "episode_ref": asdict(EpisodeRef(base.version_id, uid)),
            "fields": fields,
        }
        for uid, fields in sorted(fields_by_uid.items())
    ]
    return patches, evidence
