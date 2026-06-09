"""Doctor team view aggregation (T4-4).

Tests focus on the aggregation/rendering pipeline. The data sources
themselves (audit_log, job_store, meta_db) have their own tests.
"""
from datetime import datetime, timedelta, timezone

import pytest

from kb.ops.doctor_team import gather_team_view, render_team_text, team_view_to_dict


@pytest.fixture
async def populated_stores(tmp_path):
    """Seed audit_log, job_store, meta_db with a known multi-user scenario."""
    from kb.audit.log import AuditLog
    from kb.jobs.sqlite_store import SQLiteJobStore
    from kb.storage.metadata_db import MetadataDB

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    await meta.init()
    audit = AuditLog(
        db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db",
        fallback_path=tmp_path / "audit_fallback.jsonl",
    )
    await audit.init()
    jobs = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/ingestion_jobs.db")
    await jobs.init()

    # Seed audit rows: alice 8 ingests + 0 cancels + 1 failed; bob 2 ingests
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for i in range(8):
        await audit.write(
            "doc.ingest",
            user_id="u-alice",
            target_kind="document",
            target_id=f"doc-alice-{i}",
            payload={"doc_name": f"a{i}.pdf"},
        )
    await audit.write(
        "job.failed",
        user_id="u-alice",
        target_kind="job",
        target_id="job-failed",
        payload={"item_id": "i", "error_type": "X", "error_msg": "x"},
    )
    for i in range(2):
        await audit.write(
            "doc.ingest",
            user_id="u-bob",
            target_kind="document",
            target_id=f"doc-bob-{i}",
            payload={"doc_name": f"b{i}.pdf"},
        )
    # 3 trash docs
    for did, owner in [
        ("trash1", "u-alice"), ("trash2", "u-bob"), ("trash3", "u-alice"),
    ]:
        await meta.upsert_document(
            doc_id=did, doc_name=f"{did}.pdf",
            version=None, effective_date=None,
            expiry_date=None, supersedes=None,
            owner_id=owner,
        )
        async with meta._engine.begin() as conn:
            await conn.exec_driver_sql(
                "UPDATE documents SET status='deleted' WHERE doc_id=?", (did,)
            )
    # 2 orphan-owner active docs
    for did, owner in [("legacy1", None), ("legacy2", "_system")]:
        await meta.upsert_document(
            doc_id=did, doc_name=f"{did}.pdf",
            version=None, effective_date=None,
            expiry_date=None, supersedes=None,
            owner_id=owner,
        )

    # Bob queues 1 item, then alice queues 5 → alice 5 queued, bob 1
    await jobs.create_job(paths=["/b.pdf"], config={}, owner_id="u-bob")
    await jobs.create_job(
        paths=[f"/a{i}.pdf" for i in range(5)], config={}, owner_id="u-alice",
    )

    yield {"meta": meta, "audit": audit, "jobs": jobs, "retention_days": 30}

    await jobs.close()
    await audit.aclose()
    await meta.aclose()


@pytest.mark.asyncio
async def test_gather_team_view_aggregates_per_user_activity(populated_stores):
    view = await gather_team_view(**populated_stores)
    activity = {row["user_id"]: row for row in view["activity_24h"]}
    assert activity["u-alice"]["ingest"] == 8
    assert activity["u-alice"]["failed"] == 1
    assert activity["u-bob"]["ingest"] == 2


@pytest.mark.asyncio
async def test_gather_team_view_queue_grouping(populated_stores):
    view = await gather_team_view(**populated_stores)
    queue = view["queue_by_owner"]
    assert queue.get("u-alice") == 5
    assert queue.get("u-bob") == 1


@pytest.mark.asyncio
async def test_gather_team_view_trash_inventory(populated_stores):
    view = await gather_team_view(**populated_stores)
    trash = view["trash_inventory"]
    assert len(trash) == 3
    assert {d["doc_id"] for d in trash} == {"trash1", "trash2", "trash3"}
    # Each entry carries owner_id + retention_days_remaining (int >= 0)
    assert all("retention_days_remaining" in d for d in trash)


@pytest.mark.asyncio
async def test_gather_team_view_orphan_owners(populated_stores):
    view = await gather_team_view(**populated_stores)
    orphans = view["orphan_owners"]
    assert {d["doc_id"] for d in orphans} == {"legacy1", "legacy2"}


@pytest.mark.asyncio
async def test_render_team_text_renders_all_sections(populated_stores):
    view = await gather_team_view(**populated_stores)
    out = render_team_text(view)
    # spot-check section markers
    assert "Today's Activity" in out
    assert "Queue Status" in out
    assert "Soft-Deleted Inventory" in out
    assert "Orphan Doc Owners" in out
    # Loose substring asserts — avoids brittle coupling to exact format
    # changes (column widths, separator chars). The structural section
    # markers above are the stable contract.
    assert "alice" in out and "8" in out
    # Trash count
    assert "3" in out  # 3 trash docs


@pytest.mark.asyncio
async def test_team_view_to_dict_is_json_serializable(populated_stores):
    import json
    view = await gather_team_view(**populated_stores)
    d = team_view_to_dict(view)
    s = json.dumps(d, default=str)
    assert "u-alice" in s
    assert "trash1" in s


# ── T4 review fixes ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gather_team_view_handles_jobs_none(populated_stores):
    """T4 review P1#2: ingestion_jobs.db may be missing on a fresh
    install — scripts/doctor.py pre-checks and passes jobs=None.
    gather_team_view must mark the queue section unavailable without
    raising or logging a warning (None is intentional, not erroneous).
    """
    stores = {**populated_stores, "jobs": None}
    view = await gather_team_view(**stores)
    assert view["active_workers"] is None
    assert view["stale_claims"] is None
    assert view["queue_by_owner"] is None
    # Other sections still populated
    assert view["activity_24h"] is not None
    assert view["trash_inventory"] is not None
    assert view["orphan_owners"] is not None


@pytest.mark.asyncio
async def test_gather_team_view_splits_stale_claims_from_active(tmp_path):
    """T4 review P1#1: a dead worker (lease expired) appears in
    stale_claims, not active_workers, so doctor doesn't mislead admin."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update

    from kb.audit.log import AuditLog
    from kb.jobs.sqlite_store import SQLiteJobStore, JobItemRecord
    from kb.storage.metadata_db import MetadataDB

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    await meta.init()
    audit = AuditLog(
        db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db",
        fallback_path=tmp_path / "audit_fallback.jsonl",
    )
    await audit.init()
    jobs = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/ingestion_jobs.db")
    await jobs.init()
    try:
        await jobs.create_job(paths=["/a.pdf", "/b.pdf"], config={}, owner_id="u-x")
        active = await jobs.claim_next_item(worker_id="w-alive", lease_seconds=3600)
        stale = await jobs.claim_next_item(worker_id="w-dead",  lease_seconds=3600)
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
        async with jobs._engine.begin() as conn:
            await conn.execute(
                update(JobItemRecord)
                .where(JobItemRecord.item_id == stale.item_id)
                .values(locked_until=past)
            )

        view = await gather_team_view(meta=meta, audit=audit, jobs=jobs, retention_days=30)
        active_ids = {a["item_id"] for a in view["active_workers"]}
        stale_ids = {s["item_id"] for s in view["stale_claims"]}
        assert active.item_id in active_ids
        assert stale.item_id in stale_ids
        assert active.item_id not in stale_ids
        assert stale.item_id not in active_ids
    finally:
        await jobs.close()
        await audit.aclose()
        await meta.aclose()


@pytest.mark.asyncio
async def test_gather_team_view_trash_uses_newest_delete_ts(tmp_path):
    """T4 review P2#2: when the same doc has multiple doc.delete.completed
    audit rows (rare in T4, common once T5 adds restore), the trash
    section's deleted_at must be the NEWEST timestamp, not the oldest.
    audit.query returns ts DESC, so setdefault keeps the first-seen
    (newest) row per target_id.
    """
    import asyncio

    from kb.audit.log import AuditLog
    from kb.storage.metadata_db import MetadataDB

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    await meta.init()
    audit = AuditLog(
        db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db",
        fallback_path=tmp_path / "audit_fallback.jsonl",
    )
    await audit.init()
    try:
        await meta.upsert_document(
            doc_id="d", doc_name="d.pdf", version=None,
            effective_date=None, expiry_date=None, supersedes=None,
            owner_id="u-x",
        )
        async with meta._engine.begin() as conn:
            await conn.exec_driver_sql(
                "UPDATE documents SET status='deleted' WHERE doc_id='d'"
            )
        # doctor queries on the two-phase completed action name.
        await audit.write_completed(
            "doc.delete", user_id="u-x", target_kind="document", target_id="d",
        )
        # Measurable ts gap so DESC ordering is unambiguous.
        await asyncio.sleep(0.05)
        await audit.write_completed(
            "doc.delete", user_id="u-x", target_kind="document", target_id="d",
        )

        delete_rows = await audit.query(action="doc.delete.completed", limit=10)
        assert len(delete_rows) >= 2
        newest_ts = delete_rows[0]["ts"]
        oldest_ts = delete_rows[-1]["ts"]
        assert newest_ts != oldest_ts

        view = await gather_team_view(meta=meta, audit=audit, jobs=None, retention_days=30)
        trash = view["trash_inventory"]
        assert len(trash) == 1
        entry = trash[0]
        # gather_team_view parses ts strings via fromisoformat → datetime.
        # Compare with the same parse on newest_ts.
        from datetime import datetime, timezone as _tz
        expected = datetime.fromisoformat(newest_ts.replace("Z", "+00:00"))
        if expected.tzinfo:
            expected = expected.astimezone(_tz.utc).replace(tzinfo=None)
        assert entry["deleted_at"] == expected
    finally:
        await audit.aclose()
        await meta.aclose()


@pytest.mark.asyncio
async def test_maintenance_section_reports_off_when_not_set(populated_stores, tmp_path):
    """T5 carry-forward: doctor team panel surfaces maintenance flag.
    Default state is off, set_at None, set_by_user_id None."""
    from kb.admin.maintenance import MaintenanceState
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    maint = MaintenanceState(db_url=db_url)
    await maint.init()
    try:
        view = await gather_team_view(**populated_stores, maintenance=maint)
        assert view["maintenance"] == {
            "on": False, "set_at": None, "set_by_user_id": None,
        }
    finally:
        await maint.aclose()


@pytest.mark.asyncio
async def test_maintenance_section_reports_on_with_attribution(populated_stores, tmp_path):
    """Flipping ON must surface ON + set_by_user_id + 'writes blocked' hint."""
    from kb.admin.maintenance import MaintenanceState
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    maint = MaintenanceState(db_url=db_url)
    await maint.init()
    try:
        await maint.set_on(user_id="u-alice")
        view = await gather_team_view(**populated_stores, maintenance=maint)
        m = view["maintenance"]
        assert m["on"] is True
        assert m["set_by_user_id"] == "u-alice"
        assert m["set_at"] is not None
        text = render_team_text(view)
        assert "Maintenance" in text
        assert "ON" in text
        assert "u-alice" in text
        assert "writes blocked" in text
    finally:
        await maint.aclose()


@pytest.mark.asyncio
async def test_maintenance_section_absent_when_not_wired(populated_stores):
    """Backward-compat: pre-T5 callers that don't pass maintenance still
    work — section renders 'data unavailable', no AttributeError."""
    view = await gather_team_view(**populated_stores)  # no maintenance kwarg
    assert view["maintenance"] is None
    text = render_team_text(view)
    assert "Maintenance" in text
    assert "data unavailable" in text


def test_team_subparser_accepts_json_after_keyword():
    """T4 review P2#1: `doctor.py team --json` (subcommand before flag)
    must work the same as `doctor.py --json team`."""
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).parent.parent / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    import doctor  # type: ignore[import-not-found]

    parser = doctor.build_parser()
    # Both orderings must leave `args.json or args.team_json` truthy.
    ns_before = parser.parse_args(["--json", "team"])
    ns_after = parser.parse_args(["team", "--json"])
    assert (ns_before.json or ns_before.team_json) is True
    assert (ns_after.json or ns_after.team_json) is True
    ns_neither = parser.parse_args(["team"])
    assert (ns_neither.json or ns_neither.team_json) is False
