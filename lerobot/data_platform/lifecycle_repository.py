"""Multi-process persistence for the local dataset lifecycle control plane."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol


_INITIALIZE_LOCK = threading.Lock()


class LifecycleRepository(Protocol):
    def get(
        self,
        kind: str,
        record_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict | None: ...

    def list(
        self,
        kind: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> list[dict]: ...

    def put(
        self,
        kind: str,
        record_id: str,
        payload: dict,
        *,
        immutable: bool = False,
        connection: sqlite3.Connection | None = None,
    ) -> None: ...


class SQLiteLifecycleRepository:
    """SQLite-backed append-oriented repository with legacy JSON import."""

    _LEGACY_KINDS = (
        "versions",
        "replicas",
        "identities",
        "profiles",
        "workspaces",
        "manifests",
        "materializations",
        "feedback",
        "quarantine",
    )

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "lifecycle.db"
        self._initialize()
        self.import_legacy_json()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with _INITIALIZE_LOCK:
            for attempt in range(60):
                try:
                    self._initialize_once()
                    return
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 59:
                        raise
                    time.sleep(0.05)

    def _initialize_once(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS lifecycle_records (
                    kind TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (kind, record_id)
                );
                CREATE INDEX IF NOT EXISTS lifecycle_records_kind_created
                    ON lifecycle_records(kind, created_at);

                CREATE TABLE IF NOT EXISTS lifecycle_migrations (
                    migration_id TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lifecycle_job_leases (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    lease_until REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    @staticmethod
    def _encode(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def get(
        self,
        kind: str,
        record_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict | None:
        def _read(active: sqlite3.Connection):
            return active.execute(
                "SELECT payload_json FROM lifecycle_records WHERE kind = ? AND record_id = ?",
                (str(kind), str(record_id)),
            ).fetchone()
        if connection is not None:
            row = _read(connection)
        else:
            with self.connect() as active:
                row = _read(active)
        return json.loads(row["payload_json"]) if row is not None else None

    def list(
        self,
        kind: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> list[dict]:
        def _read(active: sqlite3.Connection):
            return active.execute(
                "SELECT payload_json FROM lifecycle_records WHERE kind = ? ORDER BY created_at",
                (str(kind),),
            ).fetchall()
        if connection is not None:
            rows = _read(connection)
        else:
            with self.connect() as active:
                rows = _read(active)
        return [json.loads(row["payload_json"]) for row in rows]

    def put(
        self,
        kind: str,
        record_id: str,
        payload: dict,
        *,
        immutable: bool = False,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        encoded = self._encode(payload)
        created_at = str(payload.get("created_at") or payload.get("updated_at") or time.time())

        def _write(active: sqlite3.Connection) -> None:
            existing = active.execute(
                "SELECT payload_json FROM lifecycle_records WHERE kind = ? AND record_id = ?",
                (str(kind), str(record_id)),
            ).fetchone()
            if existing is not None and immutable:
                if existing["payload_json"] != encoded:
                    raise ValueError(f"immutable lifecycle record already exists: {kind}/{record_id}")
                return
            active.execute(
                """
                INSERT INTO lifecycle_records(kind, record_id, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(kind, record_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (str(kind), str(record_id), encoded, created_at, str(time.time())),
            )

        if connection is not None:
            _write(connection)
            return
        with self.transaction() as active:
            _write(active)

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def import_legacy_json(self) -> dict:
        """Idempotently copy the old directory ledger into SQLite without deleting it."""
        imported = 0
        with self.transaction() as connection:
            for kind in self._LEGACY_KINDS:
                directory = self.root / kind
                if not directory.is_dir():
                    continue
                for path in sorted(directory.glob("*.json")):
                    try:
                        payload = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    record_id = self._record_id(kind, payload, path.stem)
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO lifecycle_records(
                            kind, record_id, payload_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            kind,
                            record_id,
                            self._encode(payload),
                            str(payload.get("created_at") or path.stat().st_mtime),
                            str(path.stat().st_mtime),
                        ),
                    )
                    imported += max(0, cursor.rowcount)
            connection.execute(
                """
                INSERT INTO lifecycle_migrations(migration_id, applied_at, details_json)
                VALUES ('legacy-json-v1', datetime('now'), ?)
                ON CONFLICT(migration_id) DO UPDATE SET
                    applied_at = excluded.applied_at,
                    details_json = excluded.details_json
                """,
                (self._encode({"last_imported": imported}),),
            )
        return {"imported": imported}

    @staticmethod
    def _record_id(kind: str, payload: dict, fallback: str) -> str:
        fields = {
            "versions": "version_id",
            "replicas": "replica_id",
            "identities": "artifact_digest",
            "profiles": "profile_id",
            "workspaces": "workspace_id",
            "manifests": "manifest_id",
            "materializations": "materialization_id",
            "feedback": "feedback_id",
            "quarantine": "quarantine_id",
        }
        return str(payload.get(fields.get(kind, "")) or fallback)

    def claim_job(
        self,
        job_id: str,
        job_type: str,
        *,
        payload: dict | None = None,
        owner: str | None = None,
        lease_seconds: float = 60.0,
    ) -> bool:
        owner = str(owner or f"pid:{os.getpid()}")
        now = time.time()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT owner, lease_until FROM lifecycle_job_leases WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
            if row is not None and float(row["lease_until"]) > now and row["owner"] != owner:
                return False
            connection.execute(
                """
                INSERT INTO lifecycle_job_leases(
                    job_id, job_type, owner, lease_until, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type = excluded.job_type,
                    owner = excluded.owner,
                    lease_until = excluded.lease_until,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(job_id),
                    str(job_type),
                    owner,
                    now + max(1.0, float(lease_seconds)),
                    self._encode(payload or {}),
                    now,
                ),
            )
        return True

    def heartbeat_job(self, job_id: str, *, owner: str, lease_seconds: float = 60.0) -> bool:
        now = time.time()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE lifecycle_job_leases
                SET lease_until = ?, updated_at = ?
                WHERE job_id = ? AND owner = ?
                """,
                (now + max(1.0, float(lease_seconds)), now, str(job_id), str(owner)),
            )
        return cursor.rowcount == 1

    def release_job(self, job_id: str, *, owner: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM lifecycle_job_leases WHERE job_id = ? AND owner = ?",
                (str(job_id), str(owner)),
            )
        return cursor.rowcount == 1
