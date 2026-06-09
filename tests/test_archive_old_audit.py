"""archive_old_audit.py — archive >retention-days audit rows to JSONL."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
async def seeded_audit(tmp_path):
    """Seed audit_log with old + recent rows. Returns (audit, db_url, archive_dir)."""
    from kb.audit.log import AuditLog
    from kb.storage.metadata_db import AuditLogRecord
    from sqlalchemy import update
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import NullPool

    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    audit = AuditLog(db_url=db_url, fallback_path=tmp_path / "fb.jsonl")
    await audit.init()

    # Write 3 OLD (>400 days ago, year 2024) + 2 RECENT (< 100 days ago)
    for i in range(3):
        await audit.write(
            "doc.ingest", user_id="u-old", target_kind="document", target_id=f"old-{i}",
        )
    for i in range(2):
        await audit.write(
            "doc.ingest", user_id="u-new", target_kind="document", target_id=f"new-{i}",
        )

    # Backdate the first 3 rows to 2024-03-15 (well over 400 days ago from 2026-05-21).
    old_ts = datetime(2024, 3, 15, 10, 0, 0)
    engine = create_async_engine(db_url, poolclass=NullPool)
    sf = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sf() as session:
        result = await session.execute(
            __import__("sqlalchemy").select(AuditLogRecord)
            .where(AuditLogRecord.user_id == "u-old")
        )
        old_ids = [r.event_id for r in result.scalars().all()]
        await session.execute(
            update(AuditLogRecord)
            .where(AuditLogRecord.event_id.in_(old_ids))
            .values(ts=old_ts)
        )
        await session.commit()
    await engine.dispose()

    archive_dir = tmp_path / "audit_archive"
    yield audit, db_url, archive_dir
    await audit.aclose()


@pytest.mark.asyncio
async def test_dry_run_lists_candidates_but_touches_nothing(seeded_audit):
    """Default (confirm=False) returns counts and would-write paths but
    does NOT create files or DELETE rows."""
    from scripts.archive_old_audit import archive_main
    audit, db_url, archive_dir = seeded_audit

    result = await archive_main(
        retention_days=365,
        confirm=False,
        kb_metadata_url=db_url,
        archive_dir=archive_dir,
    )
    assert result["candidates"] == 3
    assert result["archived"] == 0
    assert result["deleted"] == 0
    assert result["by_year"] == {2024: 3}
    # No file created on dry run.
    assert not archive_dir.exists() or not list(archive_dir.iterdir())
    # All 5 rows still in hot table.
    rows = await audit.query(limit=100)
    assert len(rows) == 5


@pytest.mark.asyncio
async def test_confirm_writes_jsonl_and_deletes_hot_rows(seeded_audit):
    """confirm=True: append year-bucket JSONL, fsync, DELETE the 3 rows."""
    from scripts.archive_old_audit import archive_main
    audit, db_url, archive_dir = seeded_audit

    result = await archive_main(
        retention_days=365,
        confirm=True,
        kb_metadata_url=db_url,
        archive_dir=archive_dir,
    )
    assert result["archived"] == 3
    assert result["deleted"] == 3
    archive_file = archive_dir / "audit_archive_2024.jsonl"
    assert archive_file.exists()
    lines = archive_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    # Each line is valid JSON with the shape AuditLog.query() returns.
    for line in lines:
        row = json.loads(line)
        assert row["user_id"] == "u-old"
        assert row["action"] == "doc.ingest"
        assert row["ts"].startswith("2024-")
        assert "event_id" in row
    # Hot table: only 2 recent rows left.
    remaining = await audit.query(limit=100)
    assert len(remaining) == 2
    assert all(r["user_id"] == "u-new" for r in remaining)


@pytest.mark.asyncio
async def test_archive_is_append_not_truncate(seeded_audit, tmp_path):
    """Running archive twice MUST NOT overwrite the year-bucket file.
    The second run finds nothing to archive (already moved), but if
    someone manually re-inserts an old row later, it appends not
    truncates."""
    from scripts.archive_old_audit import archive_main
    audit, db_url, archive_dir = seeded_audit

    # First pass: archive the 3 rows
    await archive_main(retention_days=365, confirm=True,
                        kb_metadata_url=db_url, archive_dir=archive_dir)
    archive_file = archive_dir / "audit_archive_2024.jsonl"
    first_run_size = archive_file.stat().st_size

    # Pretend a new old row arrived (simulate a slow audit drain that
    # delivered after archive ran). Insert directly.
    from kb.storage.metadata_db import AuditLogRecord
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(db_url, poolclass=NullPool)
    sf = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sf() as session:
        session.add(AuditLogRecord(
            user_id="u-late", action="doc.ingest",
            target_kind="document", target_id="late-1",
            ts=datetime(2024, 6, 1), payload_json="{}",
        ))
        await session.commit()
    await engine.dispose()

    # Second archive pass: appends the late row WITHOUT rewriting file.
    result = await archive_main(retention_days=365, confirm=True,
                                  kb_metadata_url=db_url, archive_dir=archive_dir)
    assert result["archived"] == 1
    second_run_size = archive_file.stat().st_size
    assert second_run_size > first_run_size  # appended, not truncated

    lines = archive_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4   # 3 from first run + 1 from second


@pytest.mark.asyncio
async def test_year_bucket_split(tmp_path):
    """Rows from different years go to different JSONL files."""
    from kb.audit.log import AuditLog
    from kb.storage.metadata_db import AuditLogRecord
    from scripts.archive_old_audit import archive_main
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import NullPool

    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    audit = AuditLog(db_url=db_url, fallback_path=tmp_path / "fb.jsonl")
    await audit.init()
    try:
        engine = create_async_engine(db_url, poolclass=NullPool)
        sf = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with sf() as session:
            session.add_all([
                AuditLogRecord(user_id="u", action="doc.ingest",
                               target_kind="document", target_id="d-2023",
                               ts=datetime(2023, 5, 1), payload_json="{}"),
                AuditLogRecord(user_id="u", action="doc.ingest",
                               target_kind="document", target_id="d-2024",
                               ts=datetime(2024, 5, 1), payload_json="{}"),
                AuditLogRecord(user_id="u", action="doc.ingest",
                               target_kind="document", target_id="d-2024b",
                               ts=datetime(2024, 11, 1), payload_json="{}"),
            ])
            await session.commit()
        await engine.dispose()

        archive_dir = tmp_path / "audit_archive"
        result = await archive_main(
            retention_days=365, confirm=True,
            kb_metadata_url=db_url, archive_dir=archive_dir,
        )
        assert result["by_year"] == {2023: 1, 2024: 2}
        assert (archive_dir / "audit_archive_2023.jsonl").exists()
        assert (archive_dir / "audit_archive_2024.jsonl").exists()
        # Each file has the right count.
        f23 = (archive_dir / "audit_archive_2023.jsonl").read_text(encoding="utf-8").strip().splitlines()
        f24 = (archive_dir / "audit_archive_2024.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(f23) == 1
        assert len(f24) == 2
    finally:
        await audit.aclose()


@pytest.mark.asyncio
async def test_amain_refuses_when_maintenance_on(seeded_audit, monkeypatch):
    """amain() exits 3 when maintenance probe says on."""
    from scripts import archive_old_audit
    audit, db_url, _ = seeded_audit
    monkeypatch.setenv("KB_ADMIN_KEY", "fake")

    async def fake_probe(*a, **kw):
        return True
    monkeypatch.setattr(archive_old_audit, "_check_maintenance_via_http", fake_probe)
    # archive_main uses Settings() → settings.qdrant_path; force test DB.
    monkeypatch.setattr(archive_old_audit, "archive_main", AsyncMock())

    rc = await archive_old_audit.amain([])
    assert rc == 3


@pytest.mark.asyncio
async def test_amain_dry_run_returns_4_when_work_pending(seeded_audit, monkeypatch):
    """Without --confirm, when there ARE candidates, exit code 4 so
    cron / CI flags 'you forgot --confirm' instead of silently 0."""
    from scripts import archive_old_audit

    async def fake_probe(*a, **kw):
        return False
    monkeypatch.setattr(archive_old_audit, "_check_maintenance_via_http", fake_probe)

    _, db_url, archive_dir = seeded_audit
    async def fake_archive(**kw):
        return {"candidates": 3, "archived": 0, "deleted": 0,
                "by_year": {2024: 3}, "files": []}
    monkeypatch.setattr(archive_old_audit, "archive_main", fake_archive)
    monkeypatch.setenv("KB_ADMIN_KEY", "fake")

    rc = await archive_old_audit.amain([])
    assert rc == 4


@pytest.mark.asyncio
async def test_amain_returns_0_when_nothing_to_archive(monkeypatch):
    """Dry run with no candidates → exit 0 (clean state)."""
    from scripts import archive_old_audit

    async def fake_probe(*a, **kw):
        return False
    monkeypatch.setattr(archive_old_audit, "_check_maintenance_via_http", fake_probe)

    async def fake_archive(**kw):
        return {"candidates": 0, "archived": 0, "deleted": 0,
                "by_year": {}, "files": []}
    monkeypatch.setattr(archive_old_audit, "archive_main", fake_archive)
    monkeypatch.setenv("KB_ADMIN_KEY", "fake")

    rc = await archive_old_audit.amain([])
    assert rc == 0
