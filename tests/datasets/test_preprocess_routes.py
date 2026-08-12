import json
import threading
from pathlib import Path
from types import SimpleNamespace

from flask import Flask

from lerobot.data_platform.precompute.preprocess.common import PreprocessResult
from lerobot.data_platform.routes import preprocess as preprocess_routes


class _ImmediateThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


def _route_context(src_root: Path):
    jobs = {}
    lock = threading.Lock()

    def update_job(job, payload):
        job.update(payload)
        total = job.get("total") or 0
        current = job.get("current") or 0
        job["progress"] = int(current / total * 100) if total else 0
        if payload.get("message"):
            append_job_log(job, payload["message"])

    def finish_job(job, message, **updates):
        job.update(status="done", progress=100, message=message, **updates)
        append_job_log(job, message)

    def fail_job(job, message, exc):
        job.update(status="error", message=message, error=str(exc))

    def append_job_log(job, message):
        job.setdefault("logs", []).append({"time": "00:00:00", "message": message})

    return SimpleNamespace(
        datasets_index={
            ("local", "source"): {
                "repo_id": "local/source",
                "root": str(src_root),
                "output_dir": str(src_root / "vis"),
            }
        },
        jobs_registry=jobs,
        jobs_lock=lock,
        dataset_key_from_body=lambda body: tuple(body["dataset_key"].split("/", 1)),
        repo_id_from_key=lambda key: "/".join(key),
        bool_option=lambda options, key, default: bool(options.get(key, default)),
        update_job=update_job,
        finish_job=finish_job,
        fail_job=fail_job,
        append_job_log=append_job_log,
        serialize_job=lambda job: dict(job),
    )


def test_convert_v3_route_uses_indexed_root_without_legacy_dataset_load(tmp_path: Path, monkeypatch):
    src_root = tmp_path / "source"
    (src_root / "meta").mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_episodes": 2})
    )
    out_root = tmp_path / "source_v3"
    calls = {}

    def fake_convert(src, **kwargs):
        calls.update(src=src, **kwargs)
        kwargs["progress_callback"](
            {"status": "done", "current": 2, "total": 2, "message": "Converted test dataset"}
        )
        return PreprocessResult(
            op="convert_v3",
            src_roots=[src],
            out_root=out_root,
            repo_id="local/source_v3",
            total_episodes=2,
            total_frames=5,
            summary={"action": "convert", "target_version": "v3.0"},
        )

    monkeypatch.setattr(preprocess_routes, "run_convert_v3", fake_convert)
    monkeypatch.setattr(preprocess_routes.threading, "Thread", _ImmediateThread)
    app = Flask(__name__)
    ctx = _route_context(src_root)
    preprocess_routes.register_preprocess_routes(app, ctx)

    response = app.test_client().post(
        "/api/preprocess/convert_v3/start",
        json={
            "dataset_key": "local/source",
            "options": {
                "out_root": str(out_root),
                "data_file_size_in_mb": 123,
                "video_file_size_in_mb": 234,
                "workers": 6,
                "image_video_mode": "rgb_lossless",
            },
        },
    )

    assert response.status_code == 200
    job = response.get_json()["job"]
    assert job["status"] == "done"
    assert job["output_root"] == str(out_root)
    assert calls["src"] == src_root
    assert calls["data_file_size_in_mb"] == 123
    assert calls["video_file_size_in_mb"] == 234
    assert calls["workers"] == 6
    assert calls["image_video_mode"] == "rgb_lossless"
    assert calls["overwrite"] is False
    assert calls["dry_run"] is False
    assert any("not auto-registered" in entry["message"] for entry in job["logs"])


def test_convert_v3_route_rejects_nonpositive_file_limit(tmp_path: Path):
    src_root = tmp_path / "source"
    (src_root / "meta").mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_episodes": 2})
    )
    app = Flask(__name__)
    preprocess_routes.register_preprocess_routes(app, _route_context(src_root))

    response = app.test_client().post(
        "/api/preprocess/convert_v3/start",
        json={
            "dataset_key": "local/source",
            "options": {"data_file_size_in_mb": 0},
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "data/video file size limits must be positive"


def test_convert_v3_route_rejects_nonpositive_workers(tmp_path: Path):
    src_root = tmp_path / "source"
    (src_root / "meta").mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_episodes": 2})
    )
    app = Flask(__name__)
    preprocess_routes.register_preprocess_routes(app, _route_context(src_root))

    response = app.test_client().post(
        "/api/preprocess/convert_v3/start",
        json={
            "dataset_key": "local/source",
            "options": {"workers": 0},
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "workers must be positive"


def test_convert_v3_route_rejects_unknown_image_video_mode(tmp_path: Path):
    src_root = tmp_path / "source"
    (src_root / "meta").mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_episodes": 2})
    )
    app = Flask(__name__)
    preprocess_routes.register_preprocess_routes(app, _route_context(src_root))

    response = app.test_client().post(
        "/api/preprocess/convert_v3/start",
        json={
            "dataset_key": "local/source",
            "options": {"image_video_mode": "unknown_mode"},
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == ("image_video_mode must be one of: lerobot_official, rgb_lossless")


def test_convert_v3_route_requests_confirmation_before_overwrite(
    tmp_path: Path,
    monkeypatch,
):
    src_root = tmp_path / "source"
    (src_root / "meta").mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_episodes": 2})
    )
    expected_out = tmp_path / "source_v3"
    expected_out.mkdir()
    app = Flask(__name__)
    preprocess_routes.register_preprocess_routes(app, _route_context(src_root))

    response = app.test_client().post(
        "/api/preprocess/convert_v3/start",
        json={"dataset_key": "local/source", "options": {}},
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["requires_overwrite_confirmation"] is True
    assert payload["output_root"] == str(expected_out)
    assert payload["error"] == f"Output dataset already exists: {expected_out}"

    calls = {}

    def fake_convert(src, **kwargs):
        calls.update(src=src, **kwargs)
        return PreprocessResult(
            op="convert_v3",
            src_roots=[src],
            out_root=expected_out,
            repo_id="local/source_v3",
            total_episodes=2,
            total_frames=5,
            summary={"action": "convert", "target_version": "v3.0"},
        )

    monkeypatch.setattr(preprocess_routes, "run_convert_v3", fake_convert)
    monkeypatch.setattr(preprocess_routes.threading, "Thread", _ImmediateThread)
    confirmed = app.test_client().post(
        "/api/preprocess/convert_v3/start",
        json={
            "dataset_key": "local/source",
            "options": {"overwrite": True},
        },
    )
    assert confirmed.status_code == 200
    assert confirmed.get_json()["job"]["status"] == "done"
    assert calls["out_root"] == expected_out
    assert calls["overwrite"] is True


def test_convert_v3_route_reports_existing_v3_without_output(tmp_path: Path, monkeypatch):
    src_root = tmp_path / "source_v3"
    (src_root / "meta").mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": 2})
    )
    requested_out = tmp_path / "unused_copy"
    calls = {}

    def fake_convert(src, **kwargs):
        calls.update(src=src, **kwargs)
        return PreprocessResult(
            op="convert_v3",
            src_roots=[src],
            out_root=src,
            repo_id="local/source_v3",
            total_episodes=2,
            total_frames=5,
            summary={"action": "already_v3", "already_v3": True, "target_version": "v3.0"},
        )

    monkeypatch.setattr(preprocess_routes, "run_convert_v3", fake_convert)
    monkeypatch.setattr(preprocess_routes.threading, "Thread", _ImmediateThread)
    app = Flask(__name__)
    ctx = _route_context(src_root)
    app_ctx_entry = ctx.datasets_index.pop(("local", "source"))
    ctx.datasets_index[("local", "source_v3")] = app_ctx_entry
    preprocess_routes.register_preprocess_routes(app, ctx)

    response = app.test_client().post(
        "/api/preprocess/convert_v3/start",
        json={
            "dataset_key": "local/source_v3",
            "options": {
                "out_root": str(requested_out),
                "data_file_size_in_mb": 0,
                "video_file_size_in_mb": 0,
                "image_video_mode": "unknown_mode",
            },
        },
    )

    assert response.status_code == 200
    job = response.get_json()["job"]
    assert job["status"] == "done"
    assert job["output_root"] == str(src_root)
    assert "already v3.0" in job["message"]
    assert calls["out_root"] is None
    assert not requested_out.exists()
    assert not any("not auto-registered" in entry["message"] for entry in job["logs"])


def test_repair_v3_video_timestamps_route(tmp_path: Path, monkeypatch):
    src_root = tmp_path / "source_v3"
    (src_root / "meta").mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": 2})
    )
    calls = {}

    def fake_repair(root, **kwargs):
        calls.update(root=root, **kwargs)
        return PreprocessResult(
            op="repair_v3_video_timestamps",
            src_roots=[root],
            out_root=root,
            repo_id="local/source_v3",
            total_episodes=2,
            total_frames=5,
            summary={"videos_reencoded": False},
        )

    monkeypatch.setattr(
        preprocess_routes,
        "repair_v3_video_timestamps",
        fake_repair,
    )
    monkeypatch.setattr(preprocess_routes.threading, "Thread", _ImmediateThread)
    app = Flask(__name__)
    ctx = _route_context(src_root)
    app_ctx_entry = ctx.datasets_index.pop(("local", "source"))
    ctx.datasets_index[("local", "source_v3")] = app_ctx_entry
    preprocess_routes.register_preprocess_routes(app, ctx)

    response = app.test_client().post(
        "/api/preprocess/repair_v3_video_timestamps/start",
        json={
            "dataset_key": "local/source_v3",
            "options": {"dry_run": False},
        },
    )

    assert response.status_code == 200
    job = response.get_json()["job"]
    assert job["status"] == "done"
    assert "without re-encoding" in job["message"]
    assert calls["root"] == src_root
    assert calls["dry_run"] is False


def test_delete_all_flagged_refreshes_viewer_episode_state_before_job_finishes(
    tmp_path: Path,
    monkeypatch,
):
    src_root = tmp_path / "source"
    static_dir = src_root / "vis" / "static"
    (src_root / "meta").mkdir(parents=True)
    static_dir.mkdir(parents=True)
    (src_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_episodes": 3, "features": {}})
    )
    (static_dir / "flagged_episodes.json").write_text(json.dumps({"flagged_episodes": [1]}))
    dataset = SimpleNamespace(
        root=src_root,
        repo_id="local/source",
        meta=SimpleNamespace(episodes={0: {}, 1: {}, 2: {}}),
        total_episodes=3,
    )
    events = []

    def fake_delete(dataset_obj, episode_ids, **_kwargs):
        events.append("delete")
        assert episode_ids == [1]
        dataset_obj.meta.episodes = {0: {}, 1: {}}
        dataset_obj.total_episodes = 2
        return {"deleted_episode_ids": [1], "new_total_episodes": 2, "next_episode": 1}

    def refresh_viewer(dataset_key, dataset_obj, refreshed_static_dir):
        events.append("refresh")
        assert dataset_key == ("local", "source")
        assert dataset_obj is dataset
        assert refreshed_static_dir == static_dir
        return [0, 1]

    monkeypatch.setattr(preprocess_routes, "delete_episodes_inplace", fake_delete)
    monkeypatch.setattr(preprocess_routes.threading, "Thread", _ImmediateThread)
    app = Flask(__name__)
    ctx = _route_context(src_root)
    ctx.ensure_dataset_loaded = lambda _key: (dataset, static_dir)
    ctx.parse_int_list = lambda value: value
    ctx.clear_dataset_caches = None
    ctx.refresh_dataset_after_episode_delete = refresh_viewer
    ctx.append_operation_log = None
    original_finish_job = ctx.finish_job

    def finish_job(job, message, **updates):
        events.append("finish")
        original_finish_job(job, message, **updates)

    ctx.finish_job = finish_job
    preprocess_routes.register_preprocess_routes(app, ctx)

    response = app.test_client().post(
        "/api/preprocess/flag_fixes/start",
        json={
            "dataset_key": "local/source",
            "options": {"fix_kind": "delete_all_flagged"},
        },
    )

    assert response.status_code == 200
    job = response.get_json()["job"]
    assert job["status"] == "done"
    assert job["viewer_url"] == "/local/source/episode_0"
    assert events == ["delete", "refresh", "finish"]


def test_convert_v3_homepage_wiring():
    template = (
        Path(__file__).parents[2]
        / "lerobot"
        / "data_platform"
        / "templates"
        / "visualize_dataset_homepage.html"
    ).read_text()

    assert 'value="convert_v3"' in template
    assert "/api/preprocess/convert_v3/start" in template
    assert "preprocess.v3_data_file_size_mb" in template
    assert "preprocess.v3_video_file_size_mb" in template
    assert "preprocess.v3_workers" in template
    assert "preprocess.v3_image_video_mode" in template
    assert "LeRobot official — AV1 / YUV420 / CRF 30" in template
    assert "RGB lossless — H.264 RGB / CRF 0" in template
    assert "auto: <src>_v3" in template
    assert "auto: <src>_v3_<timestamp>" not in template
    assert "requires_overwrite_confirmation" in template
    assert "payload.options.overwrite = true" in template
    assert 'value="repair_v3_video_timestamps"' in template
    assert "/api/preprocess/repair_v3_video_timestamps/start" in template
    assert "Videos are not re-encoded." in template
    assert "224×224" not in template
    assert "activeJob.output_root" in template
    assert "Dataset is already v3.0; no conversion is needed." in template
    assert "selectedDatasetIsV3()" in template
    assert "v3.0 viewer support uses a read-only adapter in this environment." in template
    assert "Prepare v3.0 viewer cache first" in template
