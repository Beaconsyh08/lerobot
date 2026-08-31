import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from lerobot.data_platform.lifecycle import (
    MANIFEST_APPROVED,
    MANIFEST_IN_REVIEW,
    MANIFEST_PUBLISHED,
    EpisodeRef,
    LifecycleStore,
    execute_preprocessing_profile,
    materialize_manifest,
)
from lerobot.data_platform.precompute.preprocess.dataset_merge import run_merge
from lerobot.data_platform.precompute.preprocess.dataset_split import run_split
from lerobot.data_platform.precompute.preprocess.dataset_version import run_convert_v3


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "robot_type": "test",
        "fps": 10,
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [2], "names": None},
            "state": {"dtype": "float32", "shape": [2], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "pick cube"}])
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick cube"], "length": 2},
            {"episode_index": 1, "tasks": ["pick cube"], "length": 3},
        ],
    )
    stats = []
    offset = 0
    for episode_index, length in ((0, 2), (1, 3)):
        stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "episode_index": {
                        "min": [episode_index],
                        "max": [episode_index],
                        "mean": [float(episode_index)],
                        "std": [0.0],
                        "count": [length],
                    },
                    "index": {
                        "min": [offset],
                        "max": [offset + length - 1],
                        "mean": [offset + (length - 1) / 2],
                        "std": [0.0],
                        "count": [length],
                    },
                    "action": {
                        "min": [0.0, 0.0],
                        "max": [float(length - 1), float(length - 1)],
                        "mean": [0.5, 0.5],
                        "std": [0.5, 0.5],
                        "count": [length],
                    },
                    "state": {
                        "min": [0.0, 0.0],
                        "max": [float(length - 1), float(length - 1)],
                        "mean": [0.5, 0.5],
                        "std": [0.5, 0.5],
                        "count": [length],
                    },
                },
            }
        )
        frame = pd.DataFrame(
            {
                "episode_index": [episode_index] * length,
                "frame_index": list(range(length)),
                "index": list(range(offset, offset + length)),
                "timestamp": [frame_index / 10 for frame_index in range(length)],
                "task_index": [0] * length,
                "action": [[float(frame_index), float(frame_index)] for frame_index in range(length)],
                "state": [[float(frame_index), float(frame_index)] for frame_index in range(length)],
            }
        )
        frame.to_parquet(root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet")
        offset += length
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)
    (root / "meta" / "stats.json").write_text(json.dumps({}))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def _publish(store: LifecycleStore, manifest_id: str):
    assert store.transition_manifest(manifest_id, MANIFEST_IN_REVIEW).status == MANIFEST_IN_REVIEW
    assert (
        store.transition_manifest(manifest_id, MANIFEST_APPROVED, reviewer="reviewer").status
        == MANIFEST_APPROVED
    )
    return store.transition_manifest(manifest_id, MANIFEST_PUBLISHED, reviewer="reviewer")


def test_manifest_materialization_preserves_source_and_episode_uid(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    source_before = _snapshot(source)
    store = LifecycleStore(tmp_path / "ledger")

    base = store.ingest(source, "local/source")
    assert _snapshot(source) == source_before
    assert store.ingest(source, "local/source").version_id == base.version_id
    assert base.schema["action"]["shape"] == [2]
    uid0 = base.uid_by_index()[0]
    uid1 = base.uid_by_index()[1]

    manifest = store.create_manifest(
        base.version_id,
        exclude=[asdict(EpisodeRef(base.version_id, uid0))],
        annotation_patches=[
            {
                "episode_ref": asdict(EpisodeRef(base.version_id, uid1)),
                "fields": {
                    "task": "pick red cube",
                    "tags": {"scene": "table"},
                    "subtask_state": [0, 1, 1],
                },
            }
        ],
        repair_recipes=[
            {
                "op": "trim",
                "episode_refs": [asdict(EpisodeRef(base.version_id, uid1))],
                "params": {"start_frame": 1, "end_frame": 2},
            },
            {
                "op": "value_edit",
                "episode_refs": [asdict(EpisodeRef(base.version_id, uid1))],
                "params": {"edits": [{"field": "state", "dimension": 0, "value": 9.0}]},
            },
        ],
        created_by="curator",
        reason="remove failed grasp and repair the accepted trajectory",
    )
    _publish(store, manifest.manifest_id)
    cloned_ledger = tmp_path / "ledger_clone"
    shutil.copytree(store.root, cloned_ledger)
    with pytest.raises(ValueError, match="sibling path"):
        materialize_manifest(store, manifest.manifest_id, source / "nested_output")

    output = tmp_path / "curated"
    curated, materialization = materialize_manifest(store, manifest.manifest_id, output)
    reproduced_output = tmp_path / "curated_reproduced"
    reproduced, _ = materialize_manifest(
        LifecycleStore(cloned_ledger),
        manifest.manifest_id,
        reproduced_output,
    )

    assert _snapshot(source) == source_before
    assert reproduced.fingerprint == curated.fingerprint
    assert _snapshot(reproduced_output) == _snapshot(output)
    linked_feedback = store.publish_feedback(
        curated.version_id,
        training_run={"run_id": "eval-curated"},
        collection_brief={"priorities": ["more accepted trajectories"]},
    )
    assert linked_feedback.manifest_id == manifest.manifest_id
    assert curated.parent_version_ids == [base.version_id]
    assert curated.uid_by_index() == {0: uid1}
    assert materialization["selected_episode_count"] == 1
    episodes = [json.loads(line) for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert episodes == [
        {
            "episode_index": 0,
            "tasks": ["pick red cube"],
            "length": 2,
            "tags": {"scene": "table"},
        }
    ]
    table = pd.read_parquet(output / "data" / "chunk-000" / "episode_000000.parquet")
    assert [value.tolist() for value in table["state"]] == [[9.0, 1.0], [9.0, 2.0]]
    assert table["subtask_state"].tolist() == [1, 1]
    assert table["task_index"].nunique() == 1
    tasks = [json.loads(line) for line in (output / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert {item["task"] for item in tasks} == {"pick cube", "pick red cube"}
    another_replica, repeated = materialize_manifest(
        store,
        manifest.manifest_id,
        tmp_path / "curated_again",
    )
    assert another_replica.version_id == curated.version_id
    assert repeated["idempotency_key"] == materialization["idempotency_key"]
    assert len(store.list_replicas(dataset_version_id=curated.version_id)) == 2
    assert store.list_versions(dataset_key="local/curated_again")[0].root == str(
        (tmp_path / "curated_again").resolve()
    )


def test_manifest_rejects_conflicts_and_stale_base(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    store = LifecycleStore(tmp_path / "ledger")
    base = store.ingest(source, "local/source")
    ref = asdict(EpisodeRef(base.version_id, base.uid_by_index()[0]))

    with pytest.raises(ValueError, match="both included and excluded"):
        store.create_manifest(base.version_id, include=[ref], exclude=[ref])
    with pytest.raises(ValueError, match="conflicting annotation patch"):
        store.create_manifest(
            base.version_id,
            annotation_patches=[
                {"episode_ref": ref, "fields": {"tags": {"a": 1}}},
                {"episode_ref": ref, "fields": {"tags": {"a": 2}}},
            ],
        )
    with pytest.raises(ValueError, match="unsupported annotation patch fields"):
        store.create_manifest(
            base.version_id,
            annotation_patches=[{"episode_ref": ref, "fields": {"episode_index": 99}}],
        )
    with pytest.raises(ValueError, match="must be finite"):
        store.create_manifest(
            base.version_id,
            repair_recipes=[
                {
                    "op": "value_edit",
                    "episode_refs": [ref],
                    "params": {"edits": [{"field": "action", "dimension": 0, "value": float("nan")}]},
                }
            ],
        )

    manifest = store.create_manifest(base.version_id)
    _publish(store, manifest.manifest_id)
    table_path = source / "data" / "chunk-000" / "episode_000000.parquet"
    table = pd.read_parquet(table_path)
    table.at[0, "action"] = [99.0, 99.0]
    table.to_parquet(table_path)
    with pytest.raises(ValueError, match="dataset version is stale"):
        materialize_manifest(store, manifest.manifest_id, tmp_path / "stale")
    assert not (tmp_path / "stale").exists()


def test_derived_versions_preserve_uids_for_transform_and_merge(tmp_path: Path):
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    _make_dataset(source_a)
    _make_dataset(source_b)
    store = LifecycleStore(tmp_path / "ledger")
    version_a = store.ingest(source_a, "local/source_a")
    version_b = store.ingest(source_b, "local/source_b")

    transformed_root = tmp_path / "transformed"
    shutil.copytree(source_a, transformed_root)
    table_path = transformed_root / "data" / "chunk-000" / "episode_000000.parquet"
    table = pd.read_parquet(table_path)
    table.at[0, "state"] = [7.0, 7.0]
    table.to_parquet(table_path)
    transformed = store.register_derived(
        transformed_root,
        "local/transformed",
        parent_version_ids=[version_a.version_id],
        operation="test_transform",
    )
    assert transformed.uid_by_index() == version_a.uid_by_index()

    merged_root = tmp_path / "merged"
    run_merge([source_a, source_b], out_root=merged_root, workers=1)
    merged = store.register_derived(
        merged_root,
        "local/merged",
        parent_version_ids=[version_a.version_id, version_b.version_id],
        operation="merge",
    )
    assert merged.parent_version_ids == [version_a.version_id, version_b.version_id]
    assert list(merged.uid_by_index().values()) == [
        *version_a.uid_by_index().values(),
        *version_b.uid_by_index().values(),
    ]
    assert len(merged.episode_uids()) == 4


def test_source_delivery_dimensions_and_reconciliation_follow_derived_versions(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    store = LifecycleStore(tmp_path / "ledger")
    batch = store.create_source_batch(
        "dvt2-delivery",
        source_kind="robot",
        source_uri=str(source),
        robot_profile="h10w_dvt2",
        signal_schema="action_2d_state_2d",
        dataset_format="v2.1",
        expected_episode_count=2,
        created_by="collector",
    )
    raw = store.ingest(
        source,
        "local/source",
        source_batch_ids=[batch.source_batch_id],
    )

    assert raw.source_batch_ids == [batch.source_batch_id]
    assert raw.data_dimensions == {
        "robot_profile": "h10w_dvt2",
        "signal_schema": "action_2d_state_2d",
        "dataset_format": "v2.1",
        "lifecycle_stage": "raw",
        "source_kind": "robot",
        "retention_class": "protected_source",
        "stage_profile": "h10w_dvt1_stage_v1",
        "gripper_encoding": "legacy",
        "generated": False,
    }
    received = store.list_reconciliations(dataset_version_id=raw.version_id)
    assert received[0].counts["outcomes"] == {"received": 2}
    assert received[0].counts["expected_output_delta"] == 0

    split_root = tmp_path / "standard"
    result = run_split(source, split_root, episode_range="0:1")
    standard = store.register_derived(
        split_root,
        "local/standard",
        parent_version_ids=[raw.version_id],
        operation="split",
        stage="standard",
        episode_lineage=[
            {**item, "source_dataset_version_id": raw.version_id}
            for item in result.episode_lineage
        ],
    )

    assert standard.source_batch_ids == [batch.source_batch_id]
    assert standard.data_dimensions["robot_profile"] == "h10w_dvt2"
    assert standard.data_dimensions["signal_schema"] == "action_2d_state_2d"
    assert standard.data_dimensions["lifecycle_stage"] == "standard"
    assert standard.data_dimensions["retention_class"] == "managed"
    reconciled = store.list_reconciliations(dataset_version_id=standard.version_id)[0]
    assert reconciled.counts["outcomes"] == {"accepted": 1, "excluded": 1}
    assert reconciled.counts["input_episode_count"] == 2
    assert reconciled.counts["output_episode_count"] == 1


def test_requirement_profile_recipe_and_workspace_are_deterministic(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    store = LifecycleStore(tmp_path / "ledger")
    base = store.ingest(source, "local/source")

    profile = store.create_dataset_profile(base.version_id, created_by="curator")
    assert profile.metrics["episode_count"] == 2
    assert profile.distributions["task"] == {"pick cube": 2}
    assert store.create_dataset_profile(base.version_id).dataset_profile_id == profile.dataset_profile_id

    requirement = store.create_requirement(
        "one-pick",
        target_episode_count=1,
        dimensions={"task": ["pick cube"]},
        quality_constraints={"max_flagged_ratio": 0.0},
        created_by="curator",
    )
    snapshot = store.resolve_cohort(base.version_id, {"task_contains": ["pick"]})
    assert snapshot["episode_count"] == 2
    recipe = store.create_recipe(
        "one-pick-seeded",
        base.version_id,
        requirement_id=requirement.requirement_id,
        cohort_query_snapshot=snapshot,
        composition={"max_episodes": 1},
        random_seed=7,
        created_by="curator",
    )
    repeated = store.create_recipe(
        "one-pick-seeded",
        base.version_id,
        requirement_id=requirement.requirement_id,
        cohort_query_snapshot=snapshot,
        composition={"max_episodes": 1},
        random_seed=7,
    )
    assert repeated.recipe_id == recipe.recipe_id
    assert len(recipe.include) == 1
    validation = store.validate_recipe(recipe.recipe_id)
    assert validation["valid"] is True
    assert validation["selected_episode_count"] == 1
    assert validation["warnings"][0]["field"] == "quality_constraints"
    workspace = store.compile_recipe(recipe.recipe_id, owner="curator")
    assert workspace.cohort_query_snapshot["recipe_id"] == recipe.recipe_id
    assert workspace.decisions == [
        {
            "episode_ref": recipe.include[0],
            "decision": "keep",
            "reason": f"recipe:{recipe.recipe_id}",
        }
    ]
    manifest = store.publish_workspace(
        workspace.workspace_id,
        expected_revision=workspace.revision,
        reviewer="reviewer",
    )
    assert manifest.include == recipe.include

    with pytest.raises(ValueError, match="target exceeds available episodes"):
        store.create_recipe(
            "too-many",
            base.version_id,
            composition={"group_by": "task", "target_counts": {"pick cube": 3}},
        )


def test_feedback_is_bound_to_published_dataset_version(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    store = LifecycleStore(tmp_path / "ledger")
    version = store.ingest(source, "local/source")
    ref = asdict(EpisodeRef(version.version_id, version.uid_by_index()[1]))
    manifest = store.create_manifest(version.version_id)
    _publish(store, manifest.manifest_id)

    with pytest.raises(ValueError, match="did not materialize"):
        store.publish_feedback(
            version.version_id,
            training_run={"run_id": "eval-unrelated"},
            collection_brief={"priorities": ["retry"]},
            manifest_id=manifest.manifest_id,
        )

    report = store.publish_feedback(
        version.version_id,
        training_run={"run_id": "eval-42", "checkpoint": "ckpt/1000"},
        failures=[{"episode_ref": ref, "failure_type": "grasp_miss"}],
        coverage_gaps=[{"task": "pick", "scene": "dark", "gap": 20}],
        collection_brief={
            "priorities": ["dark scenes"],
            "target_counts": {"pick": 100},
            "acceptance_criteria": ["success >= 90%"],
        },
        created_by="trainer",
    )

    assert report.dataset_version_id == version.version_id
    assert report.failures[0]["episode_ref"] == ref
    assert {item["kind"] for item in report.dataset_recommendations} == {
        "coverage_gap",
        "failure_mode",
    }

    generated = store.publish_feedback(
        version.version_id,
        training_run={"run_id": "eval-generated-brief"},
        coverage_gaps=[{"task": "pick", "scene": "rain", "target_count": 25}],
    )
    assert generated.collection_brief["target_counts"]["task=pick, scene=rain"] == 25
    assert generated in store.list_feedback(dataset_version_id=version.version_id)


def test_console_exposes_two_workspaces_and_guards_legacy_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from flask import Flask

    from lerobot.data_platform import viewer

    source = tmp_path / "source"
    _make_dataset(source)
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
        legacy_mutations_enabled=False,
    )
    client = captured["app"].test_client()
    registered = client.post("/api/datasets/register", json={"root": str(source)})
    assert registered.status_code == 200

    homepage = client.get("/").get_data(as_text=True)
    assert "Data Platform" in homepage
    assert "Data Curation" in homepage
    assert "Versions" in homepage and "Lineage" in homepage
    assert "Legacy Admin" not in homepage

    batch_response = client.post(
        "/api/lifecycle/source-batches",
        json={
            "name": "test-delivery",
            "source_kind": "robot",
            "source_uri": str(source),
            "robot_profile": "h10w_dvt2",
            "signal_schema": "action_2d_state_2d",
            "dataset_format": "v2.1",
            "expected_episode_count": 2,
        },
    )
    assert batch_response.status_code == 200
    source_batch_id = batch_response.get_json()["source_batch"]["source_batch_id"]
    ingested = client.post(
        "/api/lifecycle/ingest",
        json={"dataset_key": "local/source", "source_batch_id": source_batch_id},
    )
    assert ingested.status_code == 200
    version_id = ingested.get_json()["dataset_version"]["version_id"]
    reconciliation = client.get(
        f"/api/lifecycle/reconciliations?dataset_version_id={version_id}"
    ).get_json()["reconciliations"]
    assert reconciliation[0]["counts"]["outcomes"] == {"received": 2}
    dataset_profile = client.post(
        "/api/curation/dataset-profiles",
        json={"dataset_version_id": version_id},
    )
    assert dataset_profile.status_code == 200
    requirement_response = client.post(
        "/api/curation/requirements",
        json={
            "name": "one-pick",
            "target_episode_count": 1,
            "dimensions": {"task": ["pick cube"]},
        },
    )
    assert requirement_response.status_code == 200
    recipe_response = client.post(
        "/api/curation/recipes",
        json={
            "name": "first-pick",
            "base_dataset_version_id": version_id,
            "requirement_id": requirement_response.get_json()["requirement"]["requirement_id"],
            "cohort_query": {"task_contains": ["pick"]},
            "composition": {"max_episodes": 1},
            "random_seed": 3,
        },
    )
    assert recipe_response.status_code == 200
    compiled = client.post(
        f"/api/curation/recipes/{recipe_response.get_json()['recipe']['recipe_id']}/compile",
        json={"owner": "curator"},
    )
    assert compiled.status_code == 200
    assert compiled.get_json()["workspace"]["cohort_query_snapshot"]["recipe_id"]
    immutable_overwrite = client.post(
        "/api/preprocess/standardize/start",
        json={
            "dataset_key": "local/source",
            "options": {"out_root": str(source), "overwrite_output": True},
        },
    )
    assert immutable_overwrite.status_code == 409
    assert "published dataset version" in immutable_overwrite.get_json()["error"]
    draft = client.post(
        "/api/curation/manifests",
        json={
            "dataset_key": "local/source",
            "base_dataset_version_id": version_id,
            "exclude_episode_ids": "0",
            "collect_current_artifacts": False,
        },
    )
    assert draft.status_code == 200
    assert draft.get_json()["manifest"]["exclude"][0]["episode_uid"]
    identity = client.get(f"/api/lifecycle/versions/{version_id}/identity")
    assert identity.status_code == 200
    assert identity.get_json()["identity_artifact"]["dataset_version_id"] == version_id
    workspace_response = client.post(
        "/api/curation/workspaces",
        json={
            "dataset_key": "local/source",
            "base_dataset_version_id": version_id,
            "exclude_episode_ids": "1",
            "collect_current_artifacts": False,
        },
    )
    assert workspace_response.status_code == 200
    workspace = workspace_response.get_json()["workspace"]
    published = client.post(
        f"/api/curation/workspaces/{workspace['workspace_id']}/publish",
        json={"expected_revision": workspace["revision"], "reviewer": "reviewer"},
    )
    assert published.status_code == 200
    assert published.get_json()["manifest"]["status"] == MANIFEST_PUBLISHED
    profile_response = client.post(
        "/api/lifecycle/profiles",
        json={
            "kind": "preprocessing",
            "name": "copy-profile",
            "steps": [{"op": "copy", "params": {}}],
        },
    )
    assert profile_response.status_code == 200
    assert profile_response.get_json()["profile"]["kind"] == "preprocessing"

    denied = client.post("/local/source/delete_episode", json={"episode_id": 0})
    assert denied.status_code == 403
    assert "Admin Mode is locked" in denied.get_json()["error"]

    finalize_denied = client.post("/api/construction/local/source/finalize")
    assert finalize_denied.status_code == 403

    status = client.get("/api/admin/status").get_json()
    assert status == {"configured": False, "authenticated": False}
    setup = client.post(
        "/api/admin/setup",
        json={"password": "local-admin-password", "confirm_password": "local-admin-password"},
    )
    assert setup.status_code == 200
    assert client.get("/api/admin/status").get_json() == {
        "configured": True,
        "authenticated": True,
    }
    auth_path = tmp_path / "vis" / "_console" / "admin_auth.json"
    assert auth_path.is_file()
    assert not (static_dir / "admin_auth.json").exists()
    assert "local-admin-password" not in auth_path.read_text()
    assert client.post("/api/admin/logout", json={}).status_code == 200
    assert client.get("/api/admin/status").get_json()["authenticated"] is False

    broken = tmp_path / "broken"
    _make_dataset(broken)
    (broken / "data" / "chunk-000" / "episode_000001.parquet").unlink()
    assert client.post("/api/datasets/register", json={"root": str(broken)}).status_code == 200
    failed_ingest = client.post("/api/lifecycle/ingest", json={"dataset_key": "local/broken"})
    assert failed_ingest.status_code == 400
    quarantine = client.get("/api/lifecycle/quarantine?dataset_key=local/broken").get_json()
    assert quarantine["quarantine"][0]["quality_domain"] == "infra_quality"
    assert quarantine["quarantine"][0]["category"] == "missing_file"
    assert quarantine["quarantine"][0]["status"] == "open"
    assert "Missing source parquet" in quarantine["quarantine"][0]["error"]
    waived = client.post(
        f"/api/lifecycle/quarantine/{quarantine['quarantine'][0]['quarantine_id']}/transition",
        json={"status": "waived"},
    )
    assert waived.status_code == 200
    assert waived.get_json()["quarantine"]["status"] == "waived"


def test_identity_artifact_is_portable_and_duplicate_content_gets_unique_uid(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 4
    info_path.write_text(json.dumps(info))
    episodes_path = source / "meta" / "episodes.jsonl"
    _write_jsonl(
        episodes_path,
        [
            {"episode_index": 0, "tasks": ["pick cube"], "length": 2},
            {"episode_index": 1, "tasks": ["pick cube"], "length": 2},
        ],
    )
    first = pd.read_parquet(source / "data" / "chunk-000" / "episode_000000.parquet")
    duplicate = first.copy()
    duplicate["episode_index"] = 1
    duplicate["index"] = [2, 3]
    duplicate.to_parquet(source / "data" / "chunk-000" / "episode_000001.parquet", index=False)

    first_store = LifecycleStore(tmp_path / "ledger-a")
    original = first_store.ingest(source, "local/source")
    assert len(set(original.uid_by_index().values())) == 2
    assert original.episode_refs[0]["fingerprint"] == original.episode_refs[1]["fingerprint"]
    exported = first_store.export_identity_artifact(
        original.version_id,
        tmp_path / "source.identity.json",
    )

    copied = tmp_path / "copied-source"
    shutil.copytree(source, copied)
    second_store = LifecycleStore(tmp_path / "ledger-b")
    imported = second_store.ingest(
        copied,
        "local/copied-source",
        identity_artifact=exported,
    )
    assert imported.version_id == original.version_id
    assert imported.uid_by_index() == original.uid_by_index()
    assert second_store.list_replicas(dataset_version_id=imported.version_id)[0].root == str(
        copied.resolve()
    )


def test_workspace_revision_profile_execution_and_publish(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    store = LifecycleStore(tmp_path / "ledger")
    base = store.ingest(source, "local/source")
    uid0 = base.uid_by_index()[0]
    workspace = store.create_workspace(
        base.version_id,
        owner="curator",
        decisions=[
            {
                "episode_ref": asdict(EpisodeRef(base.version_id, uid0)),
                "decision": "exclude",
                "reason": "failed grasp",
            }
        ],
    )
    updated = store.update_workspace(
        workspace.workspace_id,
        expected_revision=workspace.revision,
        changes={"cohort_query_snapshot": {"task": "pick cube"}},
    )
    with pytest.raises(ValueError, match="revision conflict"):
        store.update_workspace(
            workspace.workspace_id,
            expected_revision=workspace.revision,
            changes={"evidence": []},
        )
    manifest = store.publish_workspace(
        workspace.workspace_id,
        expected_revision=updated.revision,
        reviewer="reviewer",
    )
    assert manifest.status == MANIFEST_PUBLISHED
    assert store.get_workspace(workspace.workspace_id).published_manifest_id == manifest.manifest_id

    profile = store.create_profile(
        "preprocessing",
        "first-episode-only",
        steps=[{"op": "split", "params": {"episode_range": "0:1"}}],
        created_by="infra",
    )
    standard, summary = execute_preprocessing_profile(
        store,
        base.version_id,
        profile.profile_id,
        tmp_path / "standard",
    )
    assert standard.stage == "standard"
    assert standard.profile_id == profile.profile_id
    assert standard.identity_confidence == "confirmed"
    assert standard.uid_by_index() == {0: uid0}
    assert summary["steps"][0]["op"] == "split"


def test_workspace_optimistic_lock_is_safe_across_store_instances(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    ledger = tmp_path / "ledger"
    store = LifecycleStore(ledger)
    base = store.ingest(source, "local/source")
    workspace = store.create_workspace(base.version_id, owner="curator")

    def update(value: str):
        return LifecycleStore(ledger).update_workspace(
            workspace.workspace_id,
            expected_revision=workspace.revision,
            changes={"cohort_query_snapshot": {"worker": value}},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(update, value) for value in ("a", "b")]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("ok", future.result().revision))
            except ValueError as exc:
                outcomes.append(("error", str(exc)))
    assert sum(kind == "ok" for kind, _ in outcomes) == 1
    assert sum(kind == "error" and "revision conflict" in value for kind, value in outcomes) == 1


def test_first_ingest_is_idempotent_across_store_instances(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    ledger = tmp_path / "ledger"

    def ingest():
        return LifecycleStore(ledger).ingest(source, "local/source")

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = [future.result() for future in [executor.submit(ingest), executor.submit(ingest)]]
    assert versions[0].version_id == versions[1].version_id
    assert versions[0].uid_by_index() == versions[1].uid_by_index()
    assert len(LifecycleStore(ledger).list_versions()) == 1


def test_materialization_plan_is_idempotent_across_store_instances(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    ledger = tmp_path / "ledger"
    store = LifecycleStore(ledger)
    base = store.ingest(source, "local/source")
    manifest = store.create_manifest(base.version_id)
    _publish(store, manifest.manifest_id)

    def plan():
        worker_store = LifecycleStore(ledger)
        worker_manifest = worker_store.get_manifest(manifest.manifest_id)
        profile = worker_store.default_materialization_profile()
        return worker_store.plan_materialization(
            worker_manifest,
            profile,
            tmp_path / "curated",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        runs = [future.result() for future in [executor.submit(plan), executor.submit(plan)]]
    assert runs[0].materialization_id == runs[1].materialization_id
    assert runs[0].idempotency_key == runs[1].idempotency_key
    assert len(store.list_materializations(manifest_id=manifest.manifest_id)) == 1


def test_materialization_recovers_after_rename_before_database_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    _make_dataset(source)
    store = LifecycleStore(tmp_path / "ledger")
    base = store.ingest(source, "local/source")
    manifest = store.create_manifest(base.version_id)
    _publish(store, manifest.manifest_id)
    output = tmp_path / "curated"
    original_ingest = store.ingest
    injected = {"raised": False}

    def fail_once(*args, **kwargs):
        if kwargs.get("operation") == "materialize" and not injected["raised"]:
            injected["raised"] = True
            raise RuntimeError("injected database commit failure")
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(store, "ingest", fail_once)
    with pytest.raises(RuntimeError, match="injected database commit failure"):
        materialize_manifest(store, manifest.manifest_id, output)
    assert output.is_dir()
    failed = store.list_materializations(manifest_id=manifest.manifest_id)[0]
    assert failed["status"] == "failed"

    monkeypatch.setattr(store, "ingest", original_ingest)
    version, recovered = materialize_manifest(store, manifest.manifest_id, output)
    assert recovered["status"] == "committed"
    assert recovered["output_dataset_version_id"] == version.version_id


def test_full_fingerprint_detects_middle_of_video_mutation(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    video = source / "videos" / "chunk-000" / "camera" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"a" * 131072 + b"middle" + b"z" * 131072)
    store = LifecycleStore(tmp_path / "ledger")
    version = store.ingest(source, "local/source")
    payload = bytearray(video.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    video.write_bytes(payload)
    with pytest.raises(ValueError, match="stale"):
        store.assert_version_current(version)


def test_legacy_json_ledger_migration_is_idempotent(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    seed_store = LifecycleStore(tmp_path / "seed-ledger")
    seed = seed_store.ingest(source, "local/source")
    legacy_payload = seed.to_dict()
    for field in (
        "identity_artifact_uri",
        "identity_artifact_digest",
        "identity_confidence",
        "profile_id",
        "profile_digest",
        "executor_digest",
    ):
        legacy_payload.pop(field, None)
    legacy_payload["schema_version"] = 1
    legacy_root = tmp_path / "legacy-ledger"
    version_path = legacy_root / "versions" / f"{seed.version_id}.json"
    version_path.parent.mkdir(parents=True)
    version_path.write_text(json.dumps(legacy_payload))

    migrated = LifecycleStore(legacy_root)
    assert (legacy_root / "lifecycle.db").is_file()
    upgraded = migrated.get_version(seed.version_id)
    assert upgraded.identity_confidence == "legacy_inferred"
    assert Path(upgraded.identity_artifact_uri).is_file()
    assert len(LifecycleStore(legacy_root).list_versions()) == 1


def test_v3_multi_episode_shard_materializes_with_full_validation(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)
    store = LifecycleStore(tmp_path / "ledger")
    raw = store.ingest(source, "local/source")
    v3_root = tmp_path / "source-v3"
    conversion = run_convert_v3(source, v3_root, workers=1)
    base = store.register_derived(
        v3_root,
        "local/source-v3",
        parent_version_ids=[raw.version_id],
        operation="convert_v3",
        stage="standard",
        episode_lineage=[
            {**item, "source_dataset_version_id": raw.version_id}
            for item in conversion.episode_lineage
        ],
    )
    assert base.format_variant_of == raw.version_id
    assert base.logical_snapshot_id == raw.logical_snapshot_id
    assert base.semantic_episode_set_digest == raw.semantic_episode_set_digest
    manifest = store.create_manifest(
        base.version_id,
        exclude=[asdict(EpisodeRef(base.version_id, base.uid_by_index()[0]))],
    )
    _publish(store, manifest.manifest_id)
    curated, run = materialize_manifest(store, manifest.manifest_id, tmp_path / "curated-v3", workers=1)
    assert curated.dataset_format_version.startswith("v3")
    assert curated.uid_by_index() == {0: base.uid_by_index()[1]}
    assert run["status"] == "committed"
    assert run["selected_episode_count"] == 1
