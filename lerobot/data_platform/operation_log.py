"""Append-only operation audit logs for the Data Platform."""

from __future__ import annotations

import functools
import getpass
import hashlib
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import fcntl
except ImportError:  # pragma: no cover - Data Platform is normally run on Linux.
    fcntl = None


LOG_FILENAME = "operation_log.jsonl"
SCHEMA_VERSION = 1
_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "authorization", "cookie", "password", "secret", "token")
_WRITE_LOCK = threading.Lock()


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def sanitize_for_log(value: Any, *, _depth: int = 0) -> Any:
    """Make a bounded JSON value while redacting credentials."""
    if _depth >= 8:
        return "<max-depth>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        out = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 200:
                out["<truncated>"] = f"{len(value) - index} more fields"
                break
            out[str(key)] = (
                "<redacted>" if _is_sensitive_key(key) else sanitize_for_log(item, _depth=_depth + 1)
            )
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized = [sanitize_for_log(item, _depth=_depth + 1) for item in items[:500]]
        if len(items) > 500:
            sanitized.append(f"<{len(items) - 500} more items>")
        return sanitized
    if isinstance(value, str):
        return value if len(value) <= 4000 else value[:4000] + "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def local_actor() -> dict[str, str]:
    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"
    return {"username": username or "unknown", "hostname": hostname or "unknown"}


def _normalized_paths(log_dirs: Path | str | Iterable[Path | str | None] | None) -> list[Path]:
    if log_dirs is None:
        return []
    values = [log_dirs] if isinstance(log_dirs, (str, Path)) else list(log_dirs)
    paths = []
    seen = set()
    for value in values:
        if value is None:
            continue
        path = Path(value).expanduser()
        path = path if path.name == LOG_FILENAME else path / LOG_FILENAME
        key = str(path)
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def build_operation_event(
    operation: str,
    *,
    status: str,
    phase: str = "result",
    source: str = "web",
    dataset_keys: Iterable[str] | None = None,
    dataset_roots: Iterable[Path | str] | None = None,
    episode_ids: Iterable[int] | None = None,
    details: dict | None = None,
    actor: dict | None = None,
    client: dict | None = None,
    event_id: str | None = None,
    parent_event_id: str | None = None,
) -> dict:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    keys = [str(value) for value in (dataset_keys or []) if value not in (None, "")]
    roots = [str(value) for value in (dataset_roots or []) if value not in (None, "")]
    episodes = sorted({int(value) for value in (episode_ids or [])})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or uuid.uuid4().hex,
        "parent_event_id": parent_event_id,
        "timestamp": timestamp,
        "operation": str(operation),
        "phase": str(phase),
        "status": str(status),
        "source": str(source),
        "actor": sanitize_for_log(actor or local_actor()),
        "client": sanitize_for_log(client or {}),
        "dataset_keys": list(dict.fromkeys(keys)),
        "dataset_roots": list(dict.fromkeys(roots)),
        "episode_ids": episodes,
        "details": sanitize_for_log(details or {}),
    }
    # Keep the original reader contract for existing integrations.
    payload.update(
        {
            "time": timestamp,
            "op": payload["operation"],
            "dataset_key": payload["dataset_keys"][0] if payload["dataset_keys"] else None,
            "dataset_root": payload["dataset_roots"][0] if payload["dataset_roots"] else None,
        }
    )
    return payload


def append_operation_event(
    log_dirs: Path | str | Iterable[Path | str | None] | None,
    operation: str,
    **event_fields,
) -> dict:
    """Append one identical event to each requested ledger."""
    payload = build_operation_event(operation, **event_fields)
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    for path in _normalized_paths(log_dirs):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK, path.open("ab") as stream:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(encoded)
                stream.flush()
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            logging.warning("Could not append operation audit log %s: %s", path, exc)
    return payload


def _event_timestamp(event: dict) -> str:
    return str(event.get("timestamp") or event.get("time") or "")


def read_operation_events(
    log_dirs: Path | str | Iterable[Path | str | None] | None,
    *,
    limit: int = 200,
    dataset_key: str | None = None,
    status: str | None = None,
    operation: str | None = None,
) -> list[dict]:
    """Read newest matching events, deduplicating mirrored ledgers."""
    limit = max(1, min(5000, int(limit)))
    candidates: list[dict] = []
    seen = set()
    status_aliases = {"success": {"success", "ok"}, "failed": {"failed", "error"}}
    accepted_statuses = status_aliases.get(status, {status}) if status else None
    for path in _normalized_paths(log_dirs):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as stream:
                lines = deque(stream, maxlen=max(5000, limit * 20))
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = event.get("event_id")
            if event_id:
                dedupe_key = str(event_id)
            else:
                canonical = json.dumps(event, ensure_ascii=False, sort_keys=True)
                dedupe_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if dedupe_key in seen:
                continue
            keys = event.get("dataset_keys") or (
                [event.get("dataset_key")] if event.get("dataset_key") else []
            )
            if dataset_key and dataset_key not in keys:
                continue
            if accepted_statuses and str(event.get("status") or "") not in accepted_statuses:
                continue
            event_operation = str(event.get("operation") or event.get("op") or "")
            if operation and operation.lower() not in event_operation.lower():
                continue
            event.setdefault("operation", event_operation)
            event.setdefault("timestamp", _event_timestamp(event))
            candidates.append(event)
            seen.add(dedupe_key)
    candidates.sort(key=_event_timestamp, reverse=True)
    return candidates[:limit]


def _cli_value(argv: list[str], option: str) -> str | None:
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return None


def _sanitize_cli_argv(argv: list[str]) -> list[str]:
    sanitized = []
    redact_next = False
    for value in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("--"):
            option, separator, _raw = value.partition("=")
            if _is_sensitive_key(option):
                sanitized.append(option + ("=<redacted>" if separator else ""))
                redact_next = not separator
                continue
        sanitized.append(value if len(value) <= 4000 else value[:4000] + "<truncated>")
    return sanitized


def audit_cli_main(func: Callable) -> Callable:
    """Record the complete lifecycle of a Data Platform CLI invocation."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        argv = list(sys.argv[1:])
        root_value = _cli_value(argv, "--root")
        output_value = _cli_value(argv, "--output-dir")
        run_visualize_value = str(_cli_value(argv, "--run-visualize") or "0").lower()
        root = Path(root_value).expanduser() if root_value else None
        if output_value:
            log_dir = Path(output_value).expanduser() / "static"
        elif root is not None and run_visualize_value not in {"0", "false", "no", "off"}:
            log_dir = root / "vis" / "_console" / "static"
        elif root is not None:
            log_dir = root.parent / "vis" / f"local_vis_{root.name or 'dataset'}" / "static"
        else:
            log_dir = None
        event_id = uuid.uuid4().hex
        started = time.perf_counter()
        details = {
            "argv": _sanitize_cli_argv(argv),
            "pid": os.getpid(),
            "actions": [value for value in argv if value.startswith("--") and value != "--root"],
        }
        append_operation_event(
            log_dir,
            "cli_invocation",
            status="started",
            phase="request",
            source="cli",
            dataset_roots=[root] if root is not None else [],
            details=details,
            event_id=event_id,
        )
        try:
            result = func(*args, **kwargs)
        except SystemExit as exc:
            status = "success" if exc.code in (None, 0) else "failed"
            append_operation_event(
                log_dir,
                "cli_invocation",
                status=status,
                phase="result",
                source="cli",
                dataset_roots=[root] if root is not None else [],
                details={
                    **details,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "exit_code": exc.code,
                },
                parent_event_id=event_id,
            )
            raise
        except BaseException as exc:
            append_operation_event(
                log_dir,
                "cli_invocation",
                status="failed",
                phase="result",
                source="cli",
                dataset_roots=[root] if root is not None else [],
                details={
                    **details,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "error": str(exc),
                },
                parent_event_id=event_id,
            )
            raise
        append_operation_event(
            log_dir,
            "cli_invocation",
            status="success",
            phase="result",
            source="cli",
            dataset_roots=[root] if root is not None else [],
            details={**details, "duration_ms": round((time.perf_counter() - started) * 1000)},
            parent_event_id=event_id,
        )
        return result

    return wrapper
