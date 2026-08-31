"""Portable robot semantics and signal-layout metadata for local datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from lerobot.data_platform.precompute.timeseries import (
    DATA_VERSION_DVT1,
    DATA_VERSION_DVT2,
    feature_vector_dim,
    infer_data_version_from_features,
)

DATA_PROFILE_FILENAME = "data_profile.json"
DATA_PROFILE_SCHEMA_VERSION = 1
ROBOT_PROFILE_DVT1 = "h10w_dvt1"
ROBOT_PROFILE_DVT2 = "h10w_dvt2"
STAGE_PROFILE_DVT1 = "h10w_dvt1_stage_v1"
STAGE_PROFILE_DVT2 = "h10w_dvt2_stage_v1"
SIGNAL_SCHEMA_STANDARD_16D = "dual_arm_standard_16d"
# Backward-compatible import name for callers written before the Data Platform/Curation scope split.
SIGNAL_SCHEMA_TRAIN_16D = SIGNAL_SCHEMA_STANDARD_16D
_LEGACY_SIGNAL_SCHEMA_TRAIN_16D = "dual_arm_train_16d"


@dataclass(frozen=True)
class DatasetDataProfile:
    schema_version: int
    robot_profile: str
    signal_schema: str
    gripper_encoding: str
    stage_profile: str
    legacy_data_version: str
    resolution_source: str
    confirmed: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def for_signal_schema(
        self,
        signal_schema: str,
        *,
        gripper_encoding: str | None = None,
        resolution_source: str | None = None,
    ) -> DatasetDataProfile:
        return replace(
            self,
            signal_schema=signal_schema,
            gripper_encoding=gripper_encoding or self.gripper_encoding,
            resolution_source=resolution_source or self.resolution_source,
        )


def signal_schema_from_features(features: dict | None) -> str:
    features = features or {}
    action_dim = feature_vector_dim(features.get("action"))
    state_dim = feature_vector_dim(features.get("state"))
    max_dim = max(action_dim, state_dim)
    if action_dim == 16 and state_dim == 16:
        return SIGNAL_SCHEMA_STANDARD_16D
    if max_dim:
        return f"action_{action_dim}d_state_{state_dim}d"
    return "unknown"


def has_body_joint_dimensions(features: dict | None) -> bool:
    """Return whether the stored signal layout contains DVT2 body joints 16..18."""
    features = features or {}
    return max(
        feature_vector_dim(features.get("action")),
        feature_vector_dim(features.get("state")),
    ) >= 18


def has_legacy_flag_dimension(features: dict | None) -> bool:
    features = features or {}
    max_dim = max(
        feature_vector_dim(features.get("action")),
        feature_vector_dim(features.get("state")),
    )
    return max_dim == 17


def profile_from_data_version(
    data_version: str,
    features: dict | None,
    *,
    resolution_source: str,
    confirmed: bool,
) -> DatasetDataProfile:
    normalized = str(data_version).upper()
    if normalized not in {DATA_VERSION_DVT1, DATA_VERSION_DVT2}:
        raise ValueError(f"Unsupported data_version: {data_version}")
    is_dvt2 = normalized == DATA_VERSION_DVT2
    return DatasetDataProfile(
        schema_version=DATA_PROFILE_SCHEMA_VERSION,
        robot_profile=ROBOT_PROFILE_DVT2 if is_dvt2 else ROBOT_PROFILE_DVT1,
        signal_schema=signal_schema_from_features(features),
        gripper_encoding="auto_detect" if is_dvt2 else "legacy",
        stage_profile=STAGE_PROFILE_DVT2 if is_dvt2 else STAGE_PROFILE_DVT1,
        legacy_data_version=normalized,
        resolution_source=resolution_source,
        confirmed=bool(confirmed),
    )


def _profile_from_dict(payload: dict, source: str) -> DatasetDataProfile:
    data_version = str(payload.get("legacy_data_version") or "").upper()
    if data_version not in {DATA_VERSION_DVT1, DATA_VERSION_DVT2}:
        raise ValueError(f"Invalid data profile legacy_data_version in {source}")
    signal_schema = str(payload.get("signal_schema") or "unknown")
    if signal_schema == _LEGACY_SIGNAL_SCHEMA_TRAIN_16D:
        signal_schema = SIGNAL_SCHEMA_STANDARD_16D
    return DatasetDataProfile(
        schema_version=int(payload.get("schema_version") or DATA_PROFILE_SCHEMA_VERSION),
        robot_profile=str(payload.get("robot_profile") or ""),
        signal_schema=signal_schema,
        gripper_encoding=str(payload.get("gripper_encoding") or "unknown"),
        stage_profile=str(payload.get("stage_profile") or ""),
        legacy_data_version=data_version,
        resolution_source=str(payload.get("resolution_source") or source),
        confirmed=bool(payload.get("confirmed", False)),
    )


def resolve_data_profile(
    root: Path,
    features: dict | None = None,
    *,
    data_version_override: str | None = None,
    default_data_version: str | None = None,
) -> DatasetDataProfile:
    """Resolve semantic profile without treating vector dimensions as authoritative identity."""
    root = Path(root).expanduser()
    info: dict = {}
    if features is None:
        try:
            info = json.loads((root / "meta" / "info.json").read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
        features = info.get("features") or {}

    if data_version_override:
        return profile_from_data_version(
            data_version_override,
            features,
            resolution_source="explicit_override",
            confirmed=True,
        )

    profile_path = root / "meta" / DATA_PROFILE_FILENAME
    if profile_path.is_file():
        try:
            payload = json.loads(profile_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid data profile: {profile_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid data profile: {profile_path}")
        return _profile_from_dict(payload, str(profile_path))

    if not info:
        try:
            info = json.loads((root / "meta" / "info.json").read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
    embedded = info.get("data_profile")
    if isinstance(embedded, dict):
        return _profile_from_dict(embedded, "meta/info.json")

    standardize_path = root / "meta" / "preprocess_standardize.json"
    if standardize_path.is_file():
        try:
            standardize = json.loads(standardize_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid standardize provenance: {standardize_path}") from exc
        source_version = str(standardize.get("source_data_version") or "").upper()
        if source_version in {DATA_VERSION_DVT1, DATA_VERSION_DVT2}:
            profile = profile_from_data_version(
                source_version,
                features,
                resolution_source="standardize_provenance",
                confirmed=True,
            )
            return profile.for_signal_schema(
                SIGNAL_SCHEMA_STANDARD_16D,
                gripper_encoding="normalized_0_1" if source_version == DATA_VERSION_DVT2 else "legacy",
            )

    if default_data_version:
        return profile_from_data_version(
            default_data_version,
            features,
            resolution_source="operation_default",
            confirmed=True,
        )

    inferred = infer_data_version_from_features(features)
    return profile_from_data_version(
        inferred,
        features,
        resolution_source="dimension_inference",
        confirmed=False,
    )


def write_data_profile(root: Path, profile: DatasetDataProfile, *, info: dict | None = None) -> Path:
    root = Path(root)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = profile.to_dict()
    path = meta_dir / DATA_PROFILE_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if info is not None:
        info["data_profile"] = payload
    return path
