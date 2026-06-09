"""T1a — manage_users CLI tests.

CLI 是 admin 的唯一 user-management 入口。测试 contract，不测 argparse 细节。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
CLI = REPO / "scripts" / "manage_users.py"


def run_cli(*args: str, db_path: Path) -> subprocess.CompletedProcess:
    env = {"KB_USERS_DB_URL": f"sqlite+aiosqlite:///{db_path}"}
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True, text=True,
        cwd=REPO, env={**os.environ, **env},
    )


def test_create_prints_plaintext_once(tmp_path):
    result = run_cli("create", "alice", "--role", "member", db_path=tmp_path / "u.db")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "kb_alice_" in out  # plaintext key
    assert "save this" in out.lower() or "shown once" in out.lower()  # 提醒文案


def test_create_duplicate_fails_with_clear_message(tmp_path):
    db = tmp_path / "u.db"
    run_cli("create", "alice", "--role", "member", db_path=db)
    result = run_cli("create", "alice", "--role", "admin", db_path=db)
    assert result.returncode != 0
    assert "already exists" in (result.stdout + result.stderr).lower()


def test_list_shows_users_with_prefix_not_hash(tmp_path):
    db = tmp_path / "u.db"
    run_cli("create", "alice", "--role", "member", db_path=db)
    run_cli("create", "bob", "--role", "admin", db_path=db)
    result = run_cli("list", db_path=db)
    assert "alice" in result.stdout
    assert "bob" in result.stdout
    assert "kb_alice_" in result.stdout  # prefix shown
    # sha256 hash (64 hex chars) should NOT leak
    import re
    assert not re.search(r"\b[0-9a-f]{64}\b", result.stdout)


def test_disable_marks_user_disabled(tmp_path):
    db = tmp_path / "u.db"
    run_cli("create", "alice", "--role", "member", db_path=db)
    result = run_cli("disable", "alice", db_path=db)
    assert result.returncode == 0
    listed = run_cli("list", "--all", db_path=db)
    assert "disabled" in listed.stdout.lower()


def test_set_role_changes_role(tmp_path):
    db = tmp_path / "u.db"
    run_cli("create", "alice", "--role", "member", db_path=db)
    result = run_cli("set-role", "alice", "--role", "admin", db_path=db)
    assert result.returncode == 0
    listed = run_cli("list", db_path=db)
    # crude check: 'admin' appears next to alice
    lines = [l for l in listed.stdout.splitlines() if "alice" in l]
    assert lines and "admin" in lines[0]


def test_reset_key_prints_new_plaintext(tmp_path):
    db = tmp_path / "u.db"
    create = run_cli("create", "alice", "--role", "member", db_path=db)
    old_key = next(
        word for line in create.stdout.splitlines() for word in line.split()
        if word.startswith("kb_alice_")
    )
    reset = run_cli("reset-key", "alice", db_path=db)
    assert reset.returncode == 0
    new_key = next(
        word for line in reset.stdout.splitlines() for word in line.split()
        if word.startswith("kb_alice_")
    )
    assert new_key != old_key
