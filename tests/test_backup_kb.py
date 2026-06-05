"""Backup script — pause → cp (online SQLite + qdrant tree) → reset
flag in copy → resume → manifest (T5-8).

These tests cover the four review fixes above. Integration test
(spin up server, mutate, restore) deferred to E2E smoke in Task 10."""
import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_sqlite_with_table(path: Path, value: str) -> None:
    """Create a tiny WAL-mode SQLite DB so we can prove the backup
    captures committed data even when WAL hasn't been checkpointed."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (k TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def mock_set_mode():
    """Replace _set_maintenance_mode so tests don't need a live server."""
    from scripts import backup_kb
    called = {"on": [], "off": []}
    async def fake(*args, on: bool, **kwargs):
        called["on" if on else "off"].append(True)
        return {"on": on, "drained": True}
    with patch.object(backup_kb, "_set_maintenance_mode", new=fake):
        yield called


@pytest.mark.asyncio
async def test_backup_uses_sqlite_backup_api_captures_uncheckpointed_wal(tmp_path, mock_set_mode):
    """P0 #3a: WAL-mode SQLite with uncheckpointed data. shutil.copy
    on the .db file alone would miss that data. backup() must use the
    .backup() API so the destination has everything."""
    from scripts import backup_kb
    src_dir = tmp_path / "kbdata"
    src_dir.mkdir()
    db_path = src_dir / "kb_metadata.db"
    _make_sqlite_with_table(db_path, "important-uncheckpointed")
    backup_dir = tmp_path / "backup"

    await backup_kb.backup(
        qdrant_path=str(src_dir),
        image_storage_path=str(tmp_path / "images"),
        backup_dir=str(backup_dir),
        server_url="http://127.0.0.1:65535",
        admin_key="fake",
        strict=True,
    )

    bak_conn = sqlite3.connect(str(backup_dir / "kb_metadata.db.bak"))
    try:
        rows = bak_conn.execute("SELECT k FROM t").fetchall()
    finally:
        bak_conn.close()
    assert rows == [("important-uncheckpointed",)]


@pytest.mark.asyncio
async def test_backup_qdrant_excludes_kb_sqlite_files(tmp_path, mock_set_mode):
    """P0 #3b: qdrant_path contains both KB SQLite DBs AND Qdrant
    storage. The qdrant tree copy must include Qdrant files but
    exclude the KB DBs (those are copied separately via .backup())
    and lock files."""
    from scripts import backup_kb
    src_dir = tmp_path / "kbdata"
    (src_dir / "collection" / "kb_text").mkdir(parents=True)
    (src_dir / "collection" / "kb_text" / "storage.bin").write_bytes(b"vec1")
    (src_dir / "aliases").mkdir()
    (src_dir / "aliases" / "alias.json").write_text("{}", encoding="utf-8")
    (src_dir / "meta.json").write_text('{"v":1}', encoding="utf-8")
    (src_dir / "kb_metadata.db").touch()
    (src_dir / "ingestion_jobs.db").touch()
    (src_dir / ".lock").write_text("pid", encoding="utf-8")

    backup_dir = tmp_path / "backup"
    await backup_kb.backup(
        qdrant_path=str(src_dir),
        image_storage_path=str(tmp_path / "images"),
        backup_dir=str(backup_dir),
        server_url="http://127.0.0.1:65535",
        admin_key="fake",
        strict=True,
    )

    qbak = backup_dir / "qdrant"
    # Included: Qdrant's own data
    assert (qbak / "collection" / "kb_text" / "storage.bin").read_bytes() == b"vec1"
    assert (qbak / "aliases" / "alias.json").exists()
    assert (qbak / "meta.json").exists()
    # Excluded: KB SQLite DBs + lock file
    assert not (qbak / "kb_metadata.db").exists()
    assert not (qbak / "ingestion_jobs.db").exists()
    assert not (qbak / ".lock").exists()


@pytest.mark.asyncio
async def test_backup_resets_maintenance_flag_in_copy(tmp_path, mock_set_mode):
    """P1.2: the backup is taken AFTER set_on, so the snapshot of
    kb_metadata.db has maintenance_state.on=1. Restoring this backup
    would bring the server up in maintenance. After copying, backup()
    must open the snapshot and reset that row to on=0."""
    from scripts import backup_kb

    src_dir = tmp_path / "kbdata"
    src_dir.mkdir()
    db_path = src_dir / "kb_metadata.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'CREATE TABLE maintenance_state (id INTEGER PRIMARY KEY, "on" INTEGER, '
            'set_at TIMESTAMP, set_by_user_id TEXT)'
        )
        conn.execute(
            'INSERT INTO maintenance_state (id, "on", set_at, set_by_user_id) '
            "VALUES (1, 1, '2026-05-20', 'u-admin')"
        )
        conn.commit()
    finally:
        conn.close()

    backup_dir = tmp_path / "backup"
    await backup_kb.backup(
        qdrant_path=str(src_dir),
        image_storage_path=str(tmp_path / "images"),
        backup_dir=str(backup_dir),
        server_url="http://127.0.0.1:65535",
        admin_key="fake",
        strict=True,
    )

    bak = sqlite3.connect(str(backup_dir / "kb_metadata.db.bak"))
    try:
        on_val = bak.execute('SELECT "on" FROM maintenance_state WHERE id=1').fetchone()[0]
    finally:
        bak.close()
    assert on_val == 0, "backup copy must have maintenance_state.on=0"


@pytest.mark.asyncio
async def test_backup_strict_default_aborts_on_drain_timeout(tmp_path):
    """P1.4: --strict is the DEFAULT. drained:false → abort + resume."""
    from scripts import backup_kb
    src_dir = tmp_path / "kbdata"; src_dir.mkdir()
    db_path = src_dir / "kb_metadata.db"
    _make_sqlite_with_table(db_path, "x")

    calls = {"on": 0, "off": 0}
    async def fake(*a, on: bool, **kw):
        calls["on" if on else "off"] += 1
        return {"on": on, "drained": False if on else True, "still_active": 1}

    with patch.object(backup_kb, "_set_maintenance_mode", new=fake):
        with pytest.raises(RuntimeError, match="did not drain"):
            await backup_kb.backup(
                qdrant_path=str(src_dir),
                image_storage_path=str(tmp_path / "images"),
                backup_dir=str(tmp_path / "backup"),
                server_url="http://127.0.0.1:65535",
                admin_key="fake",
            )
    assert calls["off"] == 1


@pytest.mark.asyncio
async def test_backup_best_effort_continues_on_drain_timeout(tmp_path):
    """P1.4 inverse: strict=False (--best-effort) lets a stuck worker
    pass; the snapshot may be inconsistent but the operator opted in."""
    from scripts import backup_kb
    src_dir = tmp_path / "kbdata"; src_dir.mkdir()
    db_path = src_dir / "kb_metadata.db"
    _make_sqlite_with_table(db_path, "x")

    async def fake(*a, on: bool, **kw):
        return {"on": on, "drained": False if on else True, "still_active": 1}

    with patch.object(backup_kb, "_set_maintenance_mode", new=fake):
        manifest = await backup_kb.backup(
            qdrant_path=str(src_dir),
            image_storage_path=str(tmp_path / "images"),
            backup_dir=str(tmp_path / "backup"),
            server_url="http://127.0.0.1:65535",
            admin_key="fake",
            strict=False,
        )
    assert manifest["drained"] is False


@pytest.mark.asyncio
async def test_backup_honors_custom_nav_db_path(tmp_path, mock_set_mode):
    """If nav_db_path is set outside qdrant_path, backup must still
    capture it via the SQLite online-backup loop. Without this,
    customized deployments silently lose their nav DB."""
    from scripts import backup_kb
    qdrant = tmp_path / "qdata"; qdrant.mkdir()
    # kb_metadata.db lives inside qdrant_path (default layout)
    _make_sqlite_with_table(qdrant / "kb_metadata.db", "kbmeta")
    # nav_index.db lives ELSEWHERE — custom NAV_DB_PATH
    nav_elsewhere = tmp_path / "other-disk" / "nav_index.db"
    nav_elsewhere.parent.mkdir(parents=True)
    _make_sqlite_with_table(nav_elsewhere, "nav-data")

    backup_dir = tmp_path / "backup"
    await backup_kb.backup(
        qdrant_path=str(qdrant),
        image_storage_path=str(tmp_path / "images"),
        backup_dir=str(backup_dir),
        server_url="http://127.0.0.1:65535",
        admin_key="fake",
        strict=True,
        nav_db_path=str(nav_elsewhere),
    )

    # Nav DB content is in the backup, despite being outside qdrant_path.
    bak = sqlite3.connect(str(backup_dir / "nav_index.db.bak"))
    try:
        rows = bak.execute("SELECT k FROM t").fetchall()
    finally:
        bak.close()
    assert rows == [("nav-data",)]


@pytest.mark.asyncio
async def test_backup_always_resumes_on_failure(tmp_path, mock_set_mode):
    """Even if cp fails mid-way, finally block must call off."""
    from scripts import backup_kb
    with pytest.raises(Exception):
        await backup_kb.backup(
            qdrant_path="/does/not/exist",
            image_storage_path="/does/not/exist",
            backup_dir=str(tmp_path / "backup"),
            server_url="http://127.0.0.1:65535",
            admin_key="fake",
        )
    assert mock_set_mode["off"] == [True]
