#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local password authentication for the single-user Data Platform admin mode."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
from pathlib import Path


ADMIN_SESSION_COOKIE = "data_platform_admin_session"
_PASSWORD_ITERATIONS = 310_000
_SCHEMA_VERSION = 1
_MAX_SESSIONS = 20


def _encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _password_digest(password: str, salt: bytes, iterations: int) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return _encode_bytes(digest)


def _session_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AdminAuthStore:
    """Persist one local admin password hash and active session token digests."""

    _lock = threading.RLock()

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()

    @staticmethod
    def validate_password(password: str) -> str:
        value = str(password)
        if len(value) < 8:
            raise ValueError("admin password must contain at least 8 characters")
        if len(value) > 256:
            raise ValueError("admin password must contain at most 256 characters")
        if not value.strip():
            raise ValueError("admin password cannot contain only whitespace")
        return value

    def is_configured(self) -> bool:
        with self._lock:
            return bool(self._read().get("password_digest"))

    def setup_password(self, password: str) -> str:
        password = self.validate_password(password)
        with self._lock:
            state = self._read()
            if state.get("password_digest"):
                raise ValueError("admin password is already configured")
            state = self._password_state(password)
            token = self._add_session(state)
            self._write(state)
            return token

    def authenticate(self, password: str) -> str:
        with self._lock:
            state = self._read()
            if not state.get("password_digest"):
                raise ValueError("admin password is not configured")
            if not self._password_matches(state, str(password)):
                raise PermissionError("invalid admin password")
            token = self._add_session(state)
            self._write(state)
            return token

    def verify_session(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            state = self._read()
            expected = _session_digest(str(token))
            return any(hmac.compare_digest(expected, item) for item in state.get("sessions", []))

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            state = self._read()
            digest = _session_digest(str(token))
            sessions = [
                item for item in state.get("sessions", []) if not hmac.compare_digest(item, digest)
            ]
            if sessions != state.get("sessions", []):
                state["sessions"] = sessions
                self._write(state)

    def change_password(self, current_password: str, new_password: str) -> str:
        new_password = self.validate_password(new_password)
        with self._lock:
            state = self._read()
            if not state.get("password_digest"):
                raise ValueError("admin password is not configured")
            if not self._password_matches(state, str(current_password)):
                raise PermissionError("invalid admin password")
            state = self._password_state(new_password)
            token = self._add_session(state)
            self._write(state)
            return token

    def _read(self) -> dict:
        if not self.path.is_file():
            return {"schema_version": _SCHEMA_VERSION, "sessions": []}
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"failed to read admin authentication state: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("unsupported admin authentication state")
        return payload

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(6)}.tmp")
        try:
            temporary.write_text(json.dumps(state, indent=2, sort_keys=True))
            os.chmod(temporary, 0o600)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _password_state(password: str) -> dict:
        salt = secrets.token_bytes(16)
        return {
            "schema_version": _SCHEMA_VERSION,
            "password_salt": _encode_bytes(salt),
            "password_digest": _password_digest(password, salt, _PASSWORD_ITERATIONS),
            "password_iterations": _PASSWORD_ITERATIONS,
            "sessions": [],
        }

    @staticmethod
    def _password_matches(state: dict, password: str) -> bool:
        salt_text = str(state.get("password_salt") or "")
        padding = "=" * (-len(salt_text) % 4)
        try:
            salt = base64.urlsafe_b64decode(salt_text + padding)
            iterations = int(state.get("password_iterations") or _PASSWORD_ITERATIONS)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("invalid admin password state") from exc
        observed = _password_digest(password, salt, iterations)
        return hmac.compare_digest(observed, str(state.get("password_digest") or ""))

    @staticmethod
    def _add_session(state: dict) -> str:
        token = secrets.token_urlsafe(32)
        sessions = [*state.get("sessions", []), _session_digest(token)]
        state["sessions"] = sessions[-_MAX_SESSIONS:]
        return token
