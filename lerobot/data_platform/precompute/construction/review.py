from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_platform.precompute.construction.writer import (
    _data_path,
    _episode_chunk,
    _stats_from_table,
    _video_path,
)
from lerobot.data_platform.precompute.dataset_io import (
    is_v3_dataset,
    load_episode_records,
)


def _plan_path(out_root: Path) -> Path:
    return Path(out_root) / "meta" / "construction_plan.json"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_plan_doc(out_root: Path) -> dict:
    path = _plan_path(out_root)
    if not path.is_file():
        return {"records": []}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"records": data}
    return data


def _save_plan_doc(out_root: Path, data: dict) -> None:
    _plan_path(out_root).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_construction_records(out_root: Path) -> list[dict]:
    return list(_load_plan_doc(out_root).get("records", []))


def load_construction_doc(out_root: Path) -> dict:
    return _load_plan_doc(out_root)


def save_decision(out_root: Path, new_idx: int, decision: str, reason: str | None = None) -> dict:
    data = _load_plan_doc(out_root)
    decision = str(decision).lower()
    if decision not in {"accept", "reject"}:
        raise ValueError("decision must be accept or reject")

    for record in data.get("records", []):
        if int(record["new_episode_index"]) != int(new_idx):
            continue
        record["decision"] = decision
        record["rejected"] = decision == "reject"
        record["reject_reason"] = reason if decision == "reject" else None
        _save_plan_doc(out_root, data)
        return record
    raise KeyError(new_idx)


def _set_column(table: pa.Table, name: str, values, arrow_type: pa.DataType) -> pa.Table:
    field = pa.field(name, arrow_type)
    column = pa.array(values, type=arrow_type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), field, column)
    return table.append_column(field, column)


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def finalize(out_root: Path) -> dict:
    out_root = Path(out_root)
    info_path = out_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    plan_doc = _load_plan_doc(out_root)
    rejected = {
        int(record["new_episode_index"])
        for record in plan_doc.get("records", [])
        if bool(record.get("rejected"))
    }
    if is_v3_dataset(out_root):
        old_episodes = load_episode_records(out_root)
        if not rejected:
            return {"removed": 0, "remaining": len(old_episodes)}
        kept = [row for row in old_episodes if int(row["episode_index"]) not in rejected]
        if not kept:
            raise ValueError("Construction review cannot reject every episode")
        old_to_new = {int(row["episode_index"]): new_index for new_index, row in enumerate(kept)}
        new_records = []
        for record in plan_doc.get("records", []):
            old_index = int(record["new_episode_index"])
            if old_index in rejected:
                continue
            if old_index in old_to_new:
                updated = dict(record)
                updated["new_episode_index"] = old_to_new[old_index]
                updated["rejected"] = False
                updated["reject_reason"] = None
                updated["decision"] = "accept"
                new_records.append(updated)
        plan_doc["records"] = new_records

        from lerobot.data_platform.precompute.preprocess.dataset_merge import run_merge

        with tempfile.TemporaryDirectory(
            prefix=f".{out_root.name}.construction-finalize-",
            dir=out_root.parent,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            rebuilt_root = temp_root / "rebuilt"
            run_merge(
                [out_root],
                out_root=rebuilt_root,
                workers=8,
                exclude_episodes=[rejected],
                _allow_single_source=True,
                _op="construction_finalize",
                _default_op="construction_finalize",
            )
            generated_meta = rebuilt_root / "meta"
            for child in (out_root / "meta").iterdir():
                if child.name in {
                    "episodes",
                    "info.json",
                    "stats.json",
                    "tasks.parquet",
                    "construction_plan.json",
                }:
                    continue
                destination = generated_meta / child.name
                if destination.exists():
                    continue
                if child.is_dir():
                    shutil.copytree(child, destination, symlinks=True)
                else:
                    shutil.copy2(child, destination, follow_symlinks=False)
            (generated_meta / "construction_plan.json").write_text(
                json.dumps(plan_doc, indent=2, ensure_ascii=False) + "\n"
            )

            backup_root = temp_root / "backup"
            backup_root.mkdir()
            replaced = []
            try:
                for name in ("data", "meta", "videos"):
                    current = out_root / name
                    if current.exists():
                        current.rename(backup_root / name)
                    replaced.append(name)
                    generated = rebuilt_root / name
                    if generated.exists():
                        generated.rename(current)
            except Exception:
                for name in reversed(replaced):
                    current = out_root / name
                    if current.is_dir():
                        shutil.rmtree(current)
                    backup = backup_root / name
                    if backup.exists():
                        backup.rename(current)
                raise
        return {"removed": len(rejected), "remaining": len(kept)}
    if not rejected:
        return {"removed": 0, "remaining": len(_read_jsonl(out_root / "meta" / "episodes.jsonl"))}

    old_episodes = _read_jsonl(out_root / "meta" / "episodes.jsonl")
    old_tasks = _read_jsonl(out_root / "meta" / "tasks.jsonl")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in old_tasks}
    kept = [row for row in old_episodes if int(row["episode_index"]) not in rejected]
    task_to_new_index: dict[str, int] = {}
    new_task_rows: list[dict] = []

    def task_index_for(task: str) -> int:
        if task not in task_to_new_index:
            task_to_new_index[task] = len(task_to_new_index)
            new_task_rows.append({"task_index": task_to_new_index[task], "task": task})
        return task_to_new_index[task]

    tmp_data = out_root / "data.__finalize_tmp__"
    tmp_videos = out_root / "videos.__finalize_tmp__"
    if tmp_data.exists():
        shutil.rmtree(tmp_data)
    if tmp_videos.exists():
        shutil.rmtree(tmp_videos)

    new_episode_rows = []
    new_stats_rows = []
    old_to_new: dict[int, int] = {}
    global_offset = 0
    video_keys = [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]

    for new_idx, row in enumerate(kept):
        old_idx = int(row["episode_index"])
        old_to_new[old_idx] = new_idx
        table = pq.read_table(out_root / _data_path(info, old_idx))
        task = (row.get("tasks") or [None])[0]
        if task is None and "task_index" in table.column_names:
            task = task_by_index.get(int(table["task_index"][0].as_py()), "")
        task_idx = task_index_for(str(task))

        table = _set_column(table, "episode_index", [new_idx] * table.num_rows, pa.int64())
        table = _set_column(
            table, "index", list(range(global_offset, global_offset + table.num_rows)), pa.int64()
        )
        table = _set_column(table, "task_index", [task_idx] * table.num_rows, pa.int64())
        out_data_path = tmp_data / _data_path(info, new_idx).relative_to("data")
        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_data_path)

        for video_key in video_keys:
            old_video = out_root / (_video_path(info, old_idx, video_key) or Path())
            new_video_rel = _video_path(info, new_idx, video_key)
            if new_video_rel is not None:
                _link_or_copy(old_video, tmp_videos / new_video_rel.relative_to("videos"))

        new_episode_rows.append({"episode_index": new_idx, "tasks": [str(task)], "length": table.num_rows})
        new_stats_rows.append({"episode_index": new_idx, "stats": _stats_from_table(table, info["features"])})
        global_offset += table.num_rows

    old_data = out_root / "data"
    if old_data.exists():
        shutil.rmtree(old_data)
    tmp_data.rename(old_data)

    if tmp_videos.exists():
        old_videos = out_root / "videos"
        if old_videos.exists():
            shutil.rmtree(old_videos)
        tmp_videos.rename(old_videos)

    info["total_episodes"] = len(new_episode_rows)
    info["total_frames"] = global_offset
    info["total_tasks"] = len(new_task_rows)
    info["total_chunks"] = max(1, _episode_chunk(info, max(0, len(new_episode_rows) - 1)) + 1)
    info["total_videos"] = len(new_episode_rows) * len(video_keys)
    info["splits"] = {"train": f"0:{len(new_episode_rows)}"}
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    _write_jsonl(out_root / "meta" / "tasks.jsonl", new_task_rows)
    _write_jsonl(out_root / "meta" / "episodes.jsonl", new_episode_rows)
    _write_jsonl(out_root / "meta" / "episodes_stats.jsonl", new_stats_rows)

    new_records = []
    for record in plan_doc.get("records", []):
        old_idx = int(record["new_episode_index"])
        if old_idx in rejected:
            continue
        if old_idx in old_to_new:
            record["new_episode_index"] = old_to_new[old_idx]
            record["rejected"] = False
            record["reject_reason"] = None
            record["decision"] = "accept"
            new_records.append(record)
    plan_doc["records"] = new_records
    _save_plan_doc(out_root, plan_doc)
    return {"removed": len(rejected), "remaining": len(new_episode_rows)}
