from pathlib import Path

import pytest

from lerobot.data_platform.admin_auth import AdminAuthStore


def test_admin_password_is_hashed_and_sessions_persist_until_logout(tmp_path: Path):
    path = tmp_path / "admin_auth.json"
    store = AdminAuthStore(path)

    token = store.setup_password("correct horse battery staple")

    assert path.is_file()
    assert "correct horse battery staple" not in path.read_text()
    assert store.verify_session(token) is True
    assert AdminAuthStore(path).verify_session(token) is True
    with pytest.raises(PermissionError, match="invalid admin password"):
        store.authenticate("wrong password")

    store.logout(token)
    assert store.verify_session(token) is False


def test_changing_admin_password_revokes_existing_sessions(tmp_path: Path):
    store = AdminAuthStore(tmp_path / "admin_auth.json")
    old_token = store.setup_password("old password")

    new_token = store.change_password("old password", "new password")

    assert store.verify_session(old_token) is False
    assert store.verify_session(new_token) is True
    with pytest.raises(PermissionError, match="invalid admin password"):
        store.authenticate("old password")
    assert store.verify_session(store.authenticate("new password")) is True


@pytest.mark.parametrize("password", ["short", "        "])
def test_admin_password_rejects_weak_values(tmp_path: Path, password: str):
    with pytest.raises(ValueError, match="admin password"):
        AdminAuthStore(tmp_path / "admin_auth.json").setup_password(password)
