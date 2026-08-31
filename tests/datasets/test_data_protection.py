import json
from pathlib import Path

import pytest

from lerobot.data_platform.data_protection import (
    RETENTION_MANAGED,
    RETENTION_PROTECTED_SOURCE,
    evaluate_dataset_protection,
    infer_dataset_stage,
)


def _make_light_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": 0,
                "total_frames": 0,
                "features": {},
            }
        )
    )


def test_stage_and_source_protection_are_independent(tmp_path: Path):
    source_zone = tmp_path / "source-zone"
    raw = source_zone / "raw"
    standard = tmp_path / "managed" / "standard"
    curated = tmp_path / "managed" / "curated"
    for root in (raw, standard, curated):
        _make_light_dataset(root)
    (standard / "meta" / "preprocess_standardize.json").write_text("{}")
    (curated / "meta" / "construction_plan.json").write_text("{}")

    raw_policy = evaluate_dataset_protection(raw, source_roots=[source_zone])
    standard_policy = evaluate_dataset_protection(standard, source_roots=[source_zone])

    assert raw_policy.stage == "raw"
    assert raw_policy.protected is True
    assert raw_policy.retention_class == RETENTION_PROTECTED_SOURCE
    assert infer_dataset_stage(standard) == "standard"
    assert standard_policy.protected is False
    assert standard_policy.retention_class == RETENTION_MANAGED
    assert infer_dataset_stage(curated) == "curated"

    manual_policy = evaluate_dataset_protection(
        standard,
        manual_source=True,
        manual_reason="authoritative import",
    )
    assert manual_policy.stage == "standard"
    assert manual_policy.protected is True
    assert manual_policy.manual_reason == "authoritative import"


def test_console_protects_source_datasets_and_hides_cache_only_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from flask import Flask

    from lerobot.data_platform import viewer

    source_zone = tmp_path / "source-zone"
    source = source_zone / "raw"
    managed = tmp_path / "managed"
    cache_only = tmp_path / "viewer-cache"
    duplicate_a = tmp_path / "a" / "shared"
    duplicate_b = tmp_path / "b" / "shared"
    _make_light_dataset(source)
    _make_light_dataset(managed)
    _make_light_dataset(duplicate_a)
    _make_light_dataset(duplicate_b)
    cache_only.mkdir()

    captured = {}
    monkeypatch.setattr(Flask, "run", lambda self, **_kwargs: captured.update(app=self))
    static_dir = tmp_path / "vis" / "_console" / "static"
    static_dir.mkdir(parents=True)
    viewer.run_server(
        dataset=None,
        episodes=None,
        max_frames=None,
        prepare_videos=False,
        downsample=None,
        precompute_csv=False,
        precomputed_only=True,
        host="127.0.0.1",
        port=0,
        static_folder=static_dir,
        template_folder=Path(viewer.__file__).parent / "templates",
        datasets_root=tmp_path,
        legacy_mutations_enabled=True,
        protected_source_roots=[source_zone],
    )
    client = captured["app"].test_client()
    admin_setup = client.post(
        "/api/admin/setup",
        json={"password": "source-admin-password", "confirm_password": "source-admin-password"},
    )
    assert admin_setup.status_code == 200

    source_payload = client.post("/api/datasets/register", json={"root": str(source)}).get_json()[
        "dataset"
    ]
    managed_payload = client.post("/api/datasets/register", json={"root": str(managed)}).get_json()[
        "dataset"
    ]
    assert source_payload["source_protected"] is True
    assert source_payload["retention_class"] == RETENTION_PROTECTED_SOURCE
    assert managed_payload["source_protected"] is False

    duplicate_a_payload = client.post(
        "/api/datasets/register", json={"root": str(duplicate_a)}
    ).get_json()["dataset"]
    duplicate_b_payload = client.post(
        "/api/datasets/register", json={"root": str(duplicate_b)}
    ).get_json()["dataset"]
    assert duplicate_a_payload["key"] != duplicate_b_payload["key"]

    blocked = client.post(
        "/api/preprocess/delete_episodes/start",
        json={"dataset_key": source_payload["key"], "options": {"episodes": "0"}},
    )
    assert blocked.status_code == 403
    assert "Protected source dataset" in blocked.get_json()["error"]
    direct_delete = client.post(
        f"/{source_payload['key']}/delete_episode",
        json={"episode_id": 0},
    )
    assert direct_delete.status_code == 403
    tagging_merge = client.post(f"/api/tagging/{source_payload['key']}/merge")
    assert tagging_merge.status_code == 403

    dynamic_root = client.patch(
        "/api/source_roots",
        json={"root": str(managed), "protected": True},
    )
    assert dynamic_root.status_code == 200
    dynamic_managed = next(
        item
        for item in dynamic_root.get_json()["datasets"]
        if item["key"] == managed_payload["key"]
    )
    assert dynamic_managed["source_protected"] is True
    removed_dynamic_root = client.patch(
        "/api/source_roots",
        json={"root": str(managed), "protected": False},
    )
    assert removed_dynamic_root.status_code == 200

    protected_manually = client.patch(
        f"/api/datasets/{managed_payload['key']}/protection",
        json={"protected": True, "reason": "authoritative import"},
    )
    assert protected_manually.status_code == 200
    assert protected_manually.get_json()["dataset"]["protection"]["manual_source"] is True

    unprotected = client.patch(
        f"/api/datasets/{managed_payload['key']}/protection",
        json={"protected": False},
    )
    assert unprotected.get_json()["dataset"]["source_protected"] is False

    rejected_cache = client.post("/api/datasets/register", json={"root": str(cache_only)})
    assert rejected_cache.status_code == 400
    assert "cache-only registration is temporarily unavailable" in rejected_cache.get_json()["error"]

    configured_remove = client.patch(
        "/api/source_roots",
        json={"root": str(source_zone), "protected": False},
    )
    assert configured_remove.status_code == 409

    unregistered = client.delete(f"/api/datasets/{source_payload['key']}")
    assert unregistered.status_code == 200
    assert (source / "meta" / "info.json").is_file()
