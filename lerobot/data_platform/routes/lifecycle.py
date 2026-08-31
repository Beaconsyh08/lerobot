from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import jsonify, request

from lerobot.data_platform.cli import get_default_output_dir
from lerobot.data_platform.lifecycle import (
    MANIFEST_PUBLISHED,
    EpisodeRef,
    LifecycleStore,
    collect_curation_sidecars,
    execute_preprocessing_profile,
    materialize_manifest,
)


def register_lifecycle_routes(app, ctx) -> None:
    def _store() -> LifecycleStore:
        provider = getattr(ctx, "lifecycle_store", None)
        if provider is None:
            raise RuntimeError("lifecycle store is unavailable")
        return provider() if callable(provider) else provider

    def _dataset_context(body: dict):
        dataset_key = ctx.dataset_key_from_body(body)
        dataset_obj, ds_static = ctx.ensure_dataset_loaded(dataset_key)
        return dataset_key, dataset_obj, Path(ds_static)

    def _new_job(job_type: str, dataset_key: str, total: int = 100, output_root: str | None = None):
        job = {
            "id": uuid.uuid4().hex,
            "job_type": job_type,
            "dataset_key": dataset_key,
            "status": "queued",
            "current": 0,
            "total": max(1, int(total)),
            "progress": 0,
            "message": "Queued",
            "logs": [],
        }
        if output_root:
            job["output_root"] = output_root
        with ctx.jobs_lock:
            ctx.jobs_registry[job["id"]] = job
        return job

    def _latest_version(store: LifecycleStore, dataset_key: str):
        versions = store.list_versions(dataset_key=dataset_key)
        if not versions:
            raise ValueError("dataset has not been ingested into the version ledger")
        return versions[0]

    def _refs_from_indices(version, values) -> list[dict]:
        indices = ctx.parse_int_list(values)
        if indices is None:
            return []
        uid_by_index = version.uid_by_index()
        missing = sorted(set(indices) - set(uid_by_index))
        if missing:
            raise ValueError(f"episodes not found in base dataset version: {missing}")
        return [asdict(EpisodeRef(version.version_id, uid_by_index[index])) for index in indices]

    def _normalize_patches(version, patches: list[dict]) -> list[dict]:
        normalized = []
        uid_by_index = version.uid_by_index()
        for patch in patches:
            item = dict(patch)
            if "episode_ref" not in item and "episode_index" in item:
                episode_index = int(item.pop("episode_index"))
                if episode_index not in uid_by_index:
                    raise ValueError(f"episode not found in base dataset version: {episode_index}")
                item["episode_ref"] = asdict(EpisodeRef(version.version_id, uid_by_index[episode_index]))
            normalized.append(item)
        return normalized

    def _normalize_decisions(version, decisions: list[dict]) -> list[dict]:
        normalized = []
        uid_by_index = version.uid_by_index()
        for decision in decisions:
            item = dict(decision)
            if "episode_ref" not in item and "episode_index" in item:
                episode_index = int(item.pop("episode_index"))
                if episode_index not in uid_by_index:
                    raise ValueError(f"episode not found in base dataset version: {episode_index}")
                item["episode_ref"] = asdict(
                    EpisodeRef(version.version_id, uid_by_index[episode_index])
                )
            normalized.append(item)
        return normalized

    def _normalize_repairs(version, repairs: list[dict]) -> list[dict]:
        normalized = []
        for repair in repairs:
            item = dict(repair)
            if "episode_refs" not in item and "episode_ids" in item:
                item["episode_refs"] = _refs_from_indices(version, item.pop("episode_ids"))
            normalized.append(item)
        return normalized

    def _audit(ds_static: Path, operation: str, dataset_key, **details) -> None:
        if getattr(ctx, "append_operation_log", None) is not None:
            ctx.append_operation_log(
                ds_static,
                operation,
                dataset_key=dataset_key,
                details=details,
            )

    def _ingest(body: dict):
        dataset_key, dataset_obj, ds_static = _dataset_context(body)
        options = body.get("options") or body
        store = _store()
        dataset_root = getattr(dataset_obj, "root", None)
        if dataset_root is None:
            raise ValueError("source dataset is unavailable; cache-only entries cannot be ingested")
        try:
            version = store.ingest(
                Path(dataset_root),
                ctx.repo_id_from_key(dataset_key),
                stage=str(options.get("stage") or "raw"),
                operation="ingest",
                profile=dict(options.get("profile") or {}),
                source_batch_ids=(
                    list(options.get("source_batch_ids") or [])
                    or ([str(options["source_batch_id"])] if options.get("source_batch_id") else [])
                ),
                data_dimensions=dict(options.get("data_dimensions") or {}),
                identity_artifact=(
                    options.get("identity_artifact") or options.get("identity_artifact_path")
                ),
            )
        except Exception as exc:
            quarantine = store.record_quarantine(
                dataset_key=ctx.repo_id_from_key(dataset_key),
                root=Path(dataset_root),
                error=exc,
                operation="ingest",
            )
            _audit(
                ds_static,
                "lifecycle_ingest_quarantine",
                dataset_key,
                quarantine_id=quarantine["quarantine_id"],
                error=str(exc),
            )
            raise
        _audit(
            ds_static,
            "lifecycle_ingest",
            dataset_key,
            dataset_version_id=version.version_id,
            fingerprint=version.fingerprint,
        )
        return version

    @app.route("/api/lifecycle/ingest", methods=["POST"])
    def api_lifecycle_ingest():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify({"dataset_version": _ingest(body).to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            logging.exception("Lifecycle ingest failed")
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/ingest/start", methods=["POST"])
    def api_lifecycle_ingest_start():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key = ctx.dataset_key_from_body(body)
            ctx.ensure_dataset_loaded(dataset_key)
        except KeyError:
            return jsonify({"error": "dataset is not registered"}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        job = _new_job("lifecycle_ingest", ctx.repo_id_from_key(dataset_key))

        def run() -> None:
            try:
                ctx.update_job(job, {"status": "running", "message": "Fingerprinting dataset"})
                version = _ingest(body)
                ctx.finish_job(
                    job,
                    f"Registered immutable dataset version {version.version_id}",
                    dataset_version_id=version.version_id,
                )
            except Exception as exc:
                logging.exception("Lifecycle ingest job failed")
                ctx.fail_job(job, "Lifecycle ingest failed", exc)

        threading.Thread(target=run, name=f"lifecycle-ingest-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/lifecycle/versions")
    def api_lifecycle_versions():
        dataset_key = request.args.get("dataset_key") or None
        versions = _store().list_versions(dataset_key=dataset_key)
        return jsonify({"versions": [item.to_dict() for item in versions]})

    @app.route("/api/lifecycle/versions/<string:version_id>/identity")
    def api_lifecycle_version_identity(version_id: str):
        try:
            version = _store().get_version(version_id)
            if not version.identity_artifact_uri:
                raise ValueError("dataset version has no identity artifact")
            return jsonify({"identity_artifact": json.loads(Path(version.identity_artifact_uri).read_text())})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/versions/<string:version_id>/identity/export", methods=["POST"])
    def api_lifecycle_version_identity_export(version_id: str):
        body = request.get_json(silent=True) or {}
        try:
            raw_path = str(body.get("out_path") or "").strip()
            if not raw_path:
                raise ValueError("out_path is required")
            path = _store().export_identity_artifact(version_id, Path(raw_path))
            return jsonify({"out_path": str(path)})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/replicas")
    def api_lifecycle_replicas():
        version_id = request.args.get("dataset_version_id") or None
        return jsonify(
            {"replicas": [item.to_dict() for item in _store().list_replicas(dataset_version_id=version_id)]}
        )

    @app.route("/api/lifecycle/source-batches", methods=["GET"])
    def api_lifecycle_source_batches_list():
        return jsonify(
            {"source_batches": [item.to_dict() for item in _store().list_source_batches()]}
        )

    @app.route("/api/lifecycle/source-batches", methods=["POST"])
    def api_lifecycle_source_batch_create():
        body = request.get_json(silent=True) or {}
        try:
            batch = _store().create_source_batch(
                str(body.get("name") or ""),
                source_kind=str(body.get("source_kind") or "imported"),
                source_uri=str(body.get("source_uri") or ""),
                robot_profile=str(body.get("robot_profile") or "unknown"),
                signal_schema=str(body.get("signal_schema") or "unknown"),
                dataset_format=str(body.get("dataset_format") or "unknown"),
                retention_class=str(body.get("retention_class") or "protected_source"),
                expected_episode_count=(
                    int(body["expected_episode_count"])
                    if body.get("expected_episode_count") not in (None, "")
                    else None
                ),
                metadata=dict(body.get("metadata") or {}),
                created_by=str(body.get("created_by") or "local-user"),
            )
            return jsonify({"source_batch": batch.to_dict()})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/reconciliations")
    def api_lifecycle_reconciliations_list():
        reports = _store().list_reconciliations(
            dataset_version_id=request.args.get("dataset_version_id") or None,
            source_batch_id=request.args.get("source_batch_id") or None,
        )
        return jsonify({"reconciliations": [item.to_dict() for item in reports]})

    @app.route("/api/curation/dataset-profiles", methods=["GET"])
    def api_curation_dataset_profiles_list():
        profiles = _store().list_dataset_profiles(
            dataset_version_id=request.args.get("dataset_version_id") or None,
        )
        return jsonify({"dataset_profiles": [item.to_dict() for item in profiles]})

    @app.route("/api/curation/dataset-profiles", methods=["POST"])
    def api_curation_dataset_profile_create():
        body = request.get_json(silent=True) or {}
        try:
            profile = _store().create_dataset_profile(
                str(body.get("dataset_version_id") or ""),
                distributions=dict(body.get("distributions") or {}),
                created_by=str(body.get("created_by") or "local-user"),
            )
            return jsonify({"dataset_profile": profile.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/cohorts/resolve", methods=["POST"])
    def api_curation_cohort_resolve():
        body = request.get_json(silent=True) or {}
        try:
            cohort = _store().resolve_cohort(
                str(body.get("dataset_version_id") or ""),
                dict(body.get("query") or {}),
            )
            return jsonify({"cohort": cohort})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/requirements", methods=["GET"])
    def api_curation_requirements_list():
        return jsonify(
            {"requirements": [item.to_dict() for item in _store().list_requirements()]}
        )

    @app.route("/api/curation/requirements", methods=["POST"])
    def api_curation_requirement_create():
        body = request.get_json(silent=True) or {}
        try:
            requirement = _store().create_requirement(
                str(body.get("name") or ""),
                target_episode_count=(
                    int(body["target_episode_count"])
                    if body.get("target_episode_count") not in (None, "")
                    else None
                ),
                dimensions=dict(body.get("dimensions") or {}),
                quality_constraints=dict(body.get("quality_constraints") or {}),
                composition_constraints=dict(body.get("composition_constraints") or {}),
                reason=body.get("reason"),
                created_by=str(body.get("created_by") or "local-user"),
            )
            return jsonify({"requirement": requirement.to_dict()})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/recipes", methods=["GET"])
    def api_curation_recipes_list():
        recipes = _store().list_recipes(
            base_dataset_version_id=request.args.get("base_dataset_version_id") or None,
        )
        return jsonify({"recipes": [item.to_dict() for item in recipes]})

    @app.route("/api/curation/recipes", methods=["POST"])
    def api_curation_recipe_create():
        body = request.get_json(silent=True) or {}
        try:
            store = _store()
            base = store.get_version(str(body.get("base_dataset_version_id") or ""))
            cohort_query = dict(body.get("cohort_query") or {})
            cohort_snapshot = (
                store.resolve_cohort(base.version_id, cohort_query) if cohort_query else {}
            )
            recipe = store.create_recipe(
                str(body.get("name") or ""),
                base.version_id,
                requirement_id=body.get("requirement_id") or None,
                cohort_query_snapshot=cohort_snapshot,
                include=_refs_from_indices(base, body.get("include_episode_ids")),
                exclude=_refs_from_indices(base, body.get("exclude_episode_ids")),
                composition=dict(body.get("composition") or {}),
                deduplication=dict(body.get("deduplication") or {}),
                random_seed=int(body.get("random_seed") or 0),
                created_by=str(body.get("created_by") or "local-user"),
            )
            return jsonify({"recipe": recipe.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/recipes/<string:recipe_id>/compile", methods=["POST"])
    def api_curation_recipe_compile(recipe_id: str):
        body = request.get_json(silent=True) or {}
        try:
            workspace = _store().compile_recipe(
                recipe_id,
                owner=str(body.get("owner") or "local-user"),
            )
            return jsonify({"workspace": workspace.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/recipes/<string:recipe_id>/validate", methods=["POST"])
    def api_curation_recipe_validate(recipe_id: str):
        try:
            validation = _store().validate_recipe(recipe_id)
            return jsonify({"validation": validation}), (200 if validation["valid"] else 409)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/quarantine")
    def api_lifecycle_quarantine():
        dataset_key = request.args.get("dataset_key") or None
        return jsonify({"quarantine": _store().list_quarantine(dataset_key=dataset_key)})

    @app.route("/api/lifecycle/quarantine/<string:quarantine_id>/transition", methods=["POST"])
    def api_lifecycle_quarantine_transition(quarantine_id: str):
        body = request.get_json(silent=True) or {}
        try:
            record = _store().transition_quarantine(
                quarantine_id,
                str(body.get("status") or ""),
                retry_job_id=body.get("retry_job_id"),
                resolved_dataset_version_id=body.get("resolved_dataset_version_id"),
            )
            return jsonify({"quarantine": record})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/profiles", methods=["GET"])
    def api_lifecycle_profiles_list():
        kind = request.args.get("kind") or None
        store = _store()
        if kind in (None, "materialization"):
            store.default_materialization_profile()
        return jsonify({"profiles": [item.to_dict() for item in store.list_profiles(kind=kind)]})

    @app.route("/api/lifecycle/profiles", methods=["POST"])
    def api_lifecycle_profile_create():
        body = request.get_json(silent=True) or {}
        try:
            profile = _store().create_profile(
                str(body.get("kind") or ""),
                str(body.get("name") or ""),
                steps=list(body.get("steps") or []),
                content_options=dict(body.get("content_options") or {}),
                created_by=str(body.get("created_by") or "local-user"),
            )
            return jsonify({"profile": profile.to_dict()})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/preprocess/start", methods=["POST"])
    def api_lifecycle_preprocess_start():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key, _dataset_obj, _ds_static = _dataset_context(body)
            store = _store()
            base = (
                store.get_version(str(body["base_dataset_version_id"]))
                if body.get("base_dataset_version_id")
                else _latest_version(store, ctx.repo_id_from_key(dataset_key))
            )
            selected_repo_id = ctx.repo_id_from_key(dataset_key)
            if not store.version_has_dataset_key(base.version_id, selected_repo_id):
                raise ValueError("base dataset version does not belong to the selected dataset")
            profile = store.get_profile(str(body.get("profile_id") or ""), kind="preprocessing")
            raw_output = str(body.get("out_root") or "").strip()
            out_root = (
                Path(raw_output).expanduser()
                if raw_output
                else Path(base.root).parent / f"{Path(base.root).name}_{profile.name}_v{profile.version}"
            )
            if out_root.exists():
                raise FileExistsError(f"Output dataset already exists: {out_root}")
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        job = _new_job("lifecycle_preprocess", selected_repo_id, output_root=str(out_root))

        def run() -> None:
            try:
                ctx.update_job(job, {"status": "running", "message": f"Executing profile {profile.profile_id}"})
                version, summary = execute_preprocessing_profile(
                    _store(),
                    base.version_id,
                    profile.profile_id,
                    out_root,
                    execution_options=dict(body.get("execution_options") or {}),
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                dataset = ctx.meta_only_dataset_cls(version.dataset_key, root=out_root)
                output_key = ctx.register_dataset(dataset, get_default_output_dir(out_root))
                ctx.finish_job(
                    job,
                    f"Published standard dataset version {version.version_id}",
                    dataset_version_id=version.version_id,
                    output_dataset_key=ctx.repo_id_from_key(output_key),
                    output_root=str(out_root),
                    profile_summary=summary,
                    viewer_url=f"/{ctx.repo_id_from_key(output_key)}/episode_0",
                )
            except Exception as exc:
                logging.exception("Lifecycle preprocessing failed")
                ctx.fail_job(job, "Lifecycle preprocessing failed", exc)

        threading.Thread(target=run, name=f"lifecycle-preprocess-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/curation/workspaces", methods=["GET"])
    def api_curation_workspaces_list():
        base_version_id = request.args.get("base_dataset_version_id") or None
        return jsonify(
            {
                "workspaces": [
                    item.to_dict()
                    for item in _store().list_workspaces(base_dataset_version_id=base_version_id)
                ]
            }
        )

    @app.route("/api/curation/workspaces", methods=["POST"])
    def api_curation_workspace_create():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key, _dataset_obj, ds_static = _dataset_context(body)
            store = _store()
            base = (
                store.get_version(str(body["base_dataset_version_id"]))
                if body.get("base_dataset_version_id")
                else _latest_version(store, ctx.repo_id_from_key(dataset_key))
            )
            if not store.version_has_dataset_key(
                base.version_id,
                ctx.repo_id_from_key(dataset_key),
            ):
                raise ValueError("base dataset version does not belong to the selected dataset")
            patches = _normalize_patches(base, list(body.get("annotation_patches") or []))
            evidence = list(body.get("evidence") or [])
            if bool(body.get("collect_current_artifacts", False)):
                collected_patches, collected_evidence = collect_curation_sidecars(ds_static, base)
                patches = [*collected_patches, *patches]
                evidence = [*collected_evidence, *evidence]
            decisions = _normalize_decisions(base, list(body.get("decisions") or []))
            decisions.extend(
                {"episode_ref": item, "decision": "keep", "reason": str(body.get("reason") or "")}
                for item in _refs_from_indices(base, body.get("include_episode_ids"))
            )
            decisions.extend(
                {"episode_ref": item, "decision": "exclude", "reason": str(body.get("reason") or "")}
                for item in _refs_from_indices(base, body.get("exclude_episode_ids"))
            )
            workspace = store.create_workspace(
                base.version_id,
                owner=str(body.get("owner") or "local-user"),
                decisions=decisions,
                annotation_patches=patches,
                repair_recipes=_normalize_repairs(base, list(body.get("repair_recipes") or [])),
                cohort_query_snapshot=dict(body.get("cohort_query_snapshot") or {}),
                evidence=evidence,
                rule_versions=dict(body.get("rule_versions") or {}),
                model_versions=dict(body.get("model_versions") or {}),
            )
            return jsonify({"workspace": workspace.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/workspaces/<string:workspace_id>", methods=["PATCH"])
    def api_curation_workspace_update(workspace_id: str):
        body = request.get_json(silent=True) or {}
        try:
            store = _store()
            workspace = store.get_workspace(workspace_id)
            base = store.get_version(workspace.base_dataset_version_id)
            changes = dict(body.get("changes") or {})
            if "decisions" in changes:
                changes["decisions"] = _normalize_decisions(base, list(changes["decisions"] or []))
            if "annotation_patches" in changes:
                changes["annotation_patches"] = _normalize_patches(
                    base, list(changes["annotation_patches"] or [])
                )
            if "repair_recipes" in changes:
                changes["repair_recipes"] = _normalize_repairs(
                    base, list(changes["repair_recipes"] or [])
                )
            updated = store.update_workspace(
                workspace_id,
                expected_revision=int(body.get("expected_revision") or 0),
                changes=changes,
            )
            return jsonify({"workspace": updated.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            status = 409 if "revision conflict" in str(exc) else 400
            return jsonify({"error": str(exc)}), status

    @app.route("/api/curation/workspaces/<string:workspace_id>/validate", methods=["POST"])
    def api_curation_workspace_validate(workspace_id: str):
        try:
            return jsonify({"validation": _store().validate_workspace(workspace_id)})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/workspaces/<string:workspace_id>/rebase", methods=["POST"])
    def api_curation_workspace_rebase(workspace_id: str):
        body = request.get_json(silent=True) or {}
        try:
            result = _store().rebase_workspace(
                workspace_id,
                str(body.get("base_dataset_version_id") or ""),
                expected_revision=int(body.get("expected_revision") or 0),
            )
            return jsonify(result), (200 if result.get("rebased") else 409)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/workspaces/<string:workspace_id>/publish", methods=["POST"])
    def api_curation_workspace_publish(workspace_id: str):
        body = request.get_json(silent=True) or {}
        try:
            manifest = _store().publish_workspace(
                workspace_id,
                expected_revision=int(body.get("expected_revision") or 0),
                reviewer=str(body.get("reviewer") or ""),
                reason=body.get("reason"),
            )
            return jsonify({"manifest": manifest.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            status = 409 if "revision conflict" in str(exc) else 400
            return jsonify({"error": str(exc)}), status

    @app.route("/api/curation/manifests", methods=["GET"])
    def api_curation_manifests():
        base_version_id = request.args.get("base_dataset_version_id") or None
        dataset_key = request.args.get("dataset_key") or None
        store = _store()
        if dataset_key and not base_version_id:
            version_ids = {item.version_id for item in store.list_versions(dataset_key=dataset_key)}
            manifests = [
                item for item in store.list_manifests() if item["base_dataset_version_id"] in version_ids
            ]
        else:
            manifests = store.list_manifests(base_dataset_version_id=base_version_id)
        return jsonify({"manifests": manifests})

    @app.route("/api/curation/manifests", methods=["POST"])
    def api_curation_manifest_create():
        body = request.get_json(silent=True) or {}
        try:
            dataset_key, _dataset_obj, ds_static = _dataset_context(body)
            store = _store()
            base = (
                store.get_version(str(body["base_dataset_version_id"]))
                if body.get("base_dataset_version_id")
                else _latest_version(store, ctx.repo_id_from_key(dataset_key))
            )
            if not store.version_has_dataset_key(
                base.version_id,
                ctx.repo_id_from_key(dataset_key),
            ):
                raise ValueError("base dataset version does not belong to the selected dataset")
            patches = _normalize_patches(base, list(body.get("annotation_patches") or []))
            evidence = list(body.get("evidence") or [])
            if bool(body.get("collect_current_artifacts", True)):
                collected_patches, collected_evidence = collect_curation_sidecars(ds_static, base)
                patches = [*collected_patches, *patches]
                evidence = [*collected_evidence, *evidence]
            manifest = store.create_manifest(
                base.version_id,
                include=_refs_from_indices(base, body.get("include_episode_ids")),
                exclude=_refs_from_indices(base, body.get("exclude_episode_ids")),
                annotation_patches=patches,
                repair_recipes=_normalize_repairs(base, list(body.get("repair_recipes") or [])),
                evidence=evidence,
                rule_versions=dict(body.get("rule_versions") or {}),
                model_versions=dict(body.get("model_versions") or {}),
                created_by=str(body.get("created_by") or "local-user"),
                reason=body.get("reason"),
                supersedes_manifest_id=body.get("supersedes_manifest_id"),
            )
            _audit(
                ds_static,
                "curation_manifest_create",
                dataset_key,
                manifest_id=manifest.manifest_id,
                base_dataset_version_id=base.version_id,
            )
            return jsonify({"manifest": manifest.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            logging.exception("Curation manifest creation failed")
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/curation/manifests/<string:manifest_id>/transition", methods=["POST"])
    def api_curation_manifest_transition(manifest_id: str):
        body = request.get_json(silent=True) or {}
        try:
            manifest = _store().transition_manifest(
                manifest_id,
                str(body.get("status") or ""),
                reviewer=body.get("reviewer"),
            )
            return jsonify({"manifest": manifest.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/lifecycle/materialize/start", methods=["POST"])
    def api_lifecycle_materialize_start():
        body = request.get_json(silent=True) or {}
        try:
            manifest = _store().get_manifest(str(body.get("manifest_id") or ""))
            if manifest.status != MANIFEST_PUBLISHED:
                raise ValueError("only published manifests can be materialized")
            base = _store().get_version(manifest.base_dataset_version_id)
            out_root = Path(body.get("out_root") or "").expanduser()
            if not str(body.get("out_root") or "").strip():
                out_root = (
                    Path(base.root).parent / f"{Path(base.root).name}_curated_v{manifest.manifest_version}"
                )
            workers = max(1, int(body.get("workers") or 8))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        job = _new_job(
            "lifecycle_materialize",
            base.dataset_key,
            output_root=str(out_root),
        )

        def run() -> None:
            try:
                ctx.update_job(
                    job,
                    {"status": "running", "message": f"Materializing {manifest.manifest_id}"},
                )
                version, materialization = materialize_manifest(
                    _store(),
                    manifest.manifest_id,
                    out_root,
                    profile_id=(body.get("profile_id") or body.get("profile_version_id")),
                    profile=dict(body.get("profile") or {}),
                    workers=workers,
                    progress_callback=lambda payload: ctx.update_job(job, payload),
                )
                refreshed = ctx.meta_only_dataset_cls(version.dataset_key, root=out_root)
                dataset_key = ctx.register_dataset(refreshed, get_default_output_dir(out_root))
                ds_static = get_default_output_dir(out_root) / "static"
                _audit(
                    ds_static,
                    "lifecycle_materialize",
                    dataset_key,
                    manifest_id=manifest.manifest_id,
                    dataset_version_id=version.version_id,
                    materialization_id=materialization["materialization_id"],
                )
                ctx.finish_job(
                    job,
                    f"Published curated dataset version {version.version_id}",
                    dataset_version_id=version.version_id,
                    materialization_id=materialization["materialization_id"],
                    viewer_url=f"/{ctx.repo_id_from_key(dataset_key)}/episode_0",
                )
            except Exception as exc:
                logging.exception("Lifecycle materialization failed")
                ctx.fail_job(job, "Lifecycle materialization failed", exc)

        threading.Thread(target=run, name=f"lifecycle-materialize-{job['id']}", daemon=True).start()
        return jsonify({"job": ctx.serialize_job(job)})

    @app.route("/api/lifecycle/materializations/<string:materialization_id>")
    def api_lifecycle_materialization_get(materialization_id: str):
        try:
            run = _store().get_materialization(materialization_id)
            return jsonify({"materialization": run.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/api/curation/feedback", methods=["GET"])
    def api_curation_feedback_list():
        version_id = request.args.get("dataset_version_id") or None
        reports = _store().list_feedback(dataset_version_id=version_id)
        return jsonify({"feedback": [item.to_dict() for item in reports]})

    @app.route("/api/curation/feedback", methods=["POST"])
    def api_curation_feedback_publish():
        body = request.get_json(silent=True) or {}
        try:
            version = _store().get_version(str(body.get("dataset_version_id") or ""))
            uid_by_index = version.uid_by_index()
            failures = []
            for failure in body.get("failures") or []:
                item = dict(failure)
                if "episode_ref" not in item and "episode_index" in item:
                    episode_index = int(item.pop("episode_index"))
                    if episode_index not in uid_by_index:
                        raise ValueError(f"failure episode not found: {episode_index}")
                    item["episode_ref"] = asdict(EpisodeRef(version.version_id, uid_by_index[episode_index]))
                failures.append(item)
            report = _store().publish_feedback(
                version.version_id,
                training_run=dict(body.get("training_run") or {}),
                failures=failures,
                coverage_gaps=list(body.get("coverage_gaps") or []),
                collection_brief=dict(body.get("collection_brief") or {}),
                created_by=str(body.get("created_by") or "local-user"),
                manifest_id=body.get("manifest_id"),
            )
            return jsonify({"feedback": report.to_dict()})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
