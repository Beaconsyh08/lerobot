import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from flask import Flask

from lerobot.data_platform.precompute.dataset_io import V3DatasetMetadata, read_episode_table
from lerobot.data_platform.precompute.preprocess import run_convert_v3, run_value_edits
from lerobot.data_platform.precompute.preprocess.common import PreprocessResult
from lerobot.data_platform.routes import preprocess as preprocess_routes


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_dataset(root: Path) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    info = {
        "robot_type": "test",
        "fps": 10,
        "codebase_version": "v2.1",
        "total_episodes": 2,
        "total_frames": 4,
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
            "action": {"dtype": "float32", "shape": [3], "names": ["a0", "a1", "a2"]},
            "state": {"dtype": "float32", "shape": [3], "names": ["s0", "s1", "s2"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text("{}")
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "test"}])
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["test"], "length": 2},
            {"episode_index": 1, "tasks": ["test"], "length": 2},
        ],
    )
    _write_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        [
            {"episode_index": 0, "stats": {}},
            {"episode_index": 1, "stats": {}},
        ],
    )
    offset = 0
    for episode_index, base in ((0, 0.0), (1, 2.0)):
        frame = pd.DataFrame(
            {
                "episode_index": [episode_index, episode_index],
                "frame_index": [0, 1],
                "index": [offset, offset + 1],
                "timestamp": [0.0, 0.1],
                "task_index": [0, 0],
                "action": [[base, base + 1, base + 2], [base + 1, base + 2, base + 3]],
                "state": [[base + 4, base + 5, base + 6], [base + 5, base + 6, base + 7]],
            }
        )
        frame.to_parquet(
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            index=False,
        )
        offset += 2


def _episode(root: Path, episode_index: int) -> pd.DataFrame:
    return pd.read_parquet(root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet")


def test_multiple_value_edits_create_sibling_and_preserve_source(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "edited"
    _make_dataset(source)

    result = run_value_edits(
        source,
        edits=[
            {"field": "action", "dimension": 0, "value": 9.0},
            {"field": "state", "dimension": 1, "value": -2.0},
        ],
        episode_ids=[1],
        out_root=output,
    )

    assert result.out_root == output
    assert result.summary["edit_count"] == 2
    assert result.summary["episode_count"] == 1
    assert result.summary["matched_frames"] == 2
    assert _episode(source, 1).iloc[0]["action"][0] == 2.0
    assert _episode(output, 0).iloc[0]["action"][0] == 0.0
    assert [row[0] for row in _episode(output, 1)["action"]] == [9.0, 9.0]
    assert [row[1] for row in _episode(output, 1)["state"]] == [-2.0, -2.0]

    stats_rows = {
        row["episode_index"]: row["stats"]
        for row in (
            json.loads(line) for line in (output / "meta" / "episodes_stats.jsonl").read_text().splitlines()
        )
    }
    assert stats_rows[1]["action"]["mean"][0] == 9.0
    assert stats_rows[1]["state"]["mean"][1] == -2.0
    global_stats = json.loads((output / "meta" / "stats.json").read_text())
    assert global_stats["action"]["mean"][0] == pytest.approx(4.75)


def test_value_edits_support_in_place_all_and_reject_duplicates(tmp_path: Path):
    source = tmp_path / "source"
    _make_dataset(source)

    run_value_edits(
        source,
        edits=[
            {"field": "action", "dimension": 2, "value": 1.25},
            {"field": "state", "dimension": 0, "value": -0.5},
        ],
        episode_ids=None,
        in_place=True,
    )

    for episode_index in (0, 1):
        frame = _episode(source, episode_index)
        assert [row[2] for row in frame["action"]] == [1.25, 1.25]
        assert [row[0] for row in frame["state"]] == [-0.5, -0.5]

    with pytest.raises(ValueError, match=r"duplicate edit for action\[0\]"):
        run_value_edits(
            source,
            edits=[
                {"field": "action", "dimension": 0, "value": 1.0},
                {"field": "action", "dimension": 0, "value": 2.0},
            ],
            in_place=True,
        )


def test_value_edits_preserve_v3_shard_siblings(tmp_path: Path):
    source = tmp_path / "source"
    v3_root = tmp_path / "source_v3"
    _make_dataset(source)
    run_convert_v3(source, v3_root, workers=1)

    run_value_edits(
        v3_root,
        edits=[
            {"field": "action", "dimension": 0, "value": 8.0},
            {"field": "action", "dimension": 1, "value": 7.0},
        ],
        episode_ids=[1],
        in_place=True,
    )

    meta = V3DatasetMetadata("local/source_v3", v3_root)
    episode_zero = read_episode_table(v3_root, meta, 0)
    episode_one = read_episode_table(v3_root, meta, 1)
    assert episode_zero["action"][0].as_py()[:2] == [0.0, 1.0]
    assert episode_one["action"][0].as_py()[:2] == [8.0, 7.0]
    assert meta.episodes_stats[1]["action"]["mean"][:2] == [8.0, 7.0]


class _ImmediateThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


def test_value_edit_route_forwards_multiple_edits_and_refreshes_in_place_cache(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "source"
    static_dir = tmp_path / "vis" / "static"
    csv_dir = static_dir / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / "episode_000000_ds1.csv").write_text("keep")
    (csv_dir / "episode_000001_ds1.csv").write_text("stale")
    dataset = SimpleNamespace(root=root, total_episodes=2)
    calls = {}

    def fake_run(src_root, **kwargs):
        calls.update(src_root=src_root, **kwargs)
        return PreprocessResult(
            op="set_value",
            src_roots=[src_root],
            out_root=src_root,
            repo_id="local/source",
            total_episodes=1,
            total_frames=2,
            summary={"edit_count": 2},
        )

    jobs = {}
    lock = threading.Lock()

    def update_job(job, payload):
        job.update(payload)

    def finish_job(job, message, **updates):
        job.update(status="done", message=message, **updates)

    def fail_job(job, message, exc):
        job.update(status="error", message=message, error=str(exc))

    def parse_int_list(value):
        return [int(token) for token in str(value).split(",") if token]

    class _MetaOnly:
        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = root

    ctx = SimpleNamespace(
        jobs_registry=jobs,
        jobs_lock=lock,
        dataset_key_from_body=lambda body: tuple(body["dataset_key"].split("/", 1)),
        ensure_dataset_loaded=lambda _key: (dataset, static_dir),
        repo_id_from_key=lambda key: "/".join(key),
        parse_int_list=parse_int_list,
        bool_option=lambda options, key, default: bool(options.get(key, default)),
        update_job=update_job,
        finish_job=finish_job,
        fail_job=fail_job,
        append_job_log=lambda job, message: job.setdefault("logs", []).append({"message": message}),
        serialize_job=lambda job: dict(job),
        clear_dataset_caches=lambda key: calls.update(cleared_key=key),
        meta_only_dataset_cls=_MetaOnly,
        register_dataset=lambda refreshed, output_dir: calls.update(
            refreshed=refreshed,
            output_dir=output_dir,
        ),
    )
    monkeypatch.setattr(preprocess_routes, "run_value_edits", fake_run)
    monkeypatch.setattr(preprocess_routes.threading, "Thread", _ImmediateThread)
    app = Flask(__name__)
    preprocess_routes.register_preprocess_routes(app, ctx)

    edits = [
        {"field": "action", "dimension": 0, "value": 1.0},
        {"field": "state", "dimension": 2, "value": -1.0},
    ]
    response = app.test_client().post(
        "/api/preprocess/value_edit/start",
        json={
            "dataset_key": "local/source",
            "options": {
                "edits": edits,
                "episode_scope": "selected",
                "episodes": "1",
                "output_mode": "in_place",
            },
        },
    )

    assert response.status_code == 200
    assert response.get_json()["job"]["status"] == "done"
    assert calls["src_root"] == root
    assert calls["edits"] == edits
    assert calls["episode_ids"] == [1]
    assert calls["in_place"] is True
    assert calls["cleared_key"] == ("local", "source")
    assert (csv_dir / "episode_000000_ds1.csv").is_file()
    assert not (csv_dir / "episode_000001_ds1.csv").exists()


def test_value_edit_homepage_exposes_multiple_edits_scopes_and_output_modes():
    template = (
        Path(__file__).parents[2]
        / "lerobot"
        / "data_platform"
        / "templates"
        / "visualize_dataset_homepage.html"
    ).read_text()

    assert "data_modification" in template
    assert "/api/preprocess/value_edit/start" in template
    assert "preprocess.value_edits" in template
    assert "addValueEdit()" in template
    assert 'value="selected" x-model="preprocess.value_episode_scope"' in template
    assert 'value="all" x-model="preprocess.value_episode_scope"' in template
    assert 'value="new_dataset" x-model="preprocess.value_output_mode"' in template
    assert 'value="in_place" x-model="preprocess.value_output_mode"' in template
