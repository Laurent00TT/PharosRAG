import pytest


@pytest.mark.asyncio
async def test_sqlite_job_store_create_and_claim(tmp_path):
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    job = await store.create_job(paths=["a.pdf"], config={"max_attempts": 3})

    item = await store.claim_next_item(worker_id="w1", lease_seconds=60)

    assert item is not None
    assert item.job_id == job.job_id
    assert item.path == "a.pdf"
    assert item.attempt == 1


@pytest.mark.asyncio
async def test_sqlite_job_store_complete_item_updates_job_counts(tmp_path):
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    job = await store.create_job(paths=["a.pdf"], config={})
    item = await store.claim_next_item(worker_id="w1", lease_seconds=60)

    await store.complete_item(item.item_id, {"doc_id": "d"})
    refreshed = await store.get_job(job.job_id)

    assert refreshed.succeeded_items == 1
    assert refreshed.status == "succeeded"


@pytest.mark.asyncio
async def test_sqlite_job_store_reclaims_expired_claim(tmp_path):
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    await store.create_job(paths=["a.pdf"], config={})

    first = await store.claim_next_item(worker_id="w1", lease_seconds=-1)
    second = await store.claim_next_item(worker_id="w2", lease_seconds=60)

    assert second is not None
    assert second.item_id == first.item_id
    assert second.attempt == 2


@pytest.mark.asyncio
async def test_sqlite_job_store_heartbeat_extends_worker_lease(tmp_path):
    from kb.jobs.sqlite_store import JobItemRecord, SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    await store.create_job(paths=["a.pdf"], config={})
    item = await store.claim_next_item(worker_id="w1", lease_seconds=30)

    ok = await store.heartbeat(
        item_id=item.item_id,
        worker_id="w1",
        lease_seconds=120,
    )

    assert ok is True
    async with store.session() as session:
        rec = await session.get(JobItemRecord, item.item_id)
    assert rec.locked_by == "w1"
    assert rec.locked_until is not None


@pytest.mark.asyncio
async def test_sqlite_job_store_request_cancel_cancels_queued_items(tmp_path):
    from kb.jobs.models import ITEM_CANCELLED, JOB_CANCELLED
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    job = await store.create_job(paths=["a.pdf", "b.pdf"], config={})

    await store.request_cancel(job.job_id)

    refreshed = await store.get_job(job.job_id)
    items = await store.list_job_items(job.job_id)
    assert refreshed.cancel_requested is True
    assert refreshed.status == JOB_CANCELLED
    assert {item.status for item in items} == {ITEM_CANCELLED}


@pytest.mark.asyncio
async def test_is_cancel_requested_reflects_flag(tmp_path):
    """``is_cancel_requested`` is the cheap polled read used by the
    pipeline at page boundaries. Must reflect the current flag and
    return False for unknown job ids without raising."""
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    job = await store.create_job(paths=["a.pdf"], config={})

    assert await store.is_cancel_requested(job.job_id) is False
    assert await store.is_cancel_requested("nonexistent") is False

    await store.request_cancel(job.job_id)
    assert await store.is_cancel_requested(job.job_id) is True


@pytest.mark.asyncio
async def test_cancel_item_transitions_to_cancelled_without_retry(tmp_path):
    """``cancel_item`` must mark the item ITEM_CANCELLED with no retry
    scheduled and a structured reason in error_json. Distinct from
    fail_item, which would queue a retry attempt."""
    import json
    from kb.jobs.models import ITEM_CANCELLED
    from kb.jobs.sqlite_store import JobItemRecord, SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    await store.create_job(paths=["a.pdf"], config={})
    item = await store.claim_next_item(worker_id="w1", lease_seconds=60)

    await store.cancel_item(item.item_id, "cancelled at page 5")

    async with store.session() as session:
        rec = await session.get(JobItemRecord, item.item_id)
    assert rec.status == ITEM_CANCELLED
    assert rec.locked_by is None
    assert rec.locked_until is None
    err = json.loads(rec.error_json)
    assert err["cancelled"] is True
    assert err["reason"] == "cancelled at page 5"


@pytest.mark.asyncio
async def test_cancel_item_for_unknown_id_is_noop(tmp_path):
    """``cancel_item`` for a non-existent item must NOT raise. The
    runner might race with manual cleanup."""
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    # Should silently no-op rather than raising
    await store.cancel_item("ghost", "whatever")


@pytest.mark.asyncio
async def test_sqlite_job_store_retry_failed_items(tmp_path):
    from kb.jobs.errors import JobError
    from kb.jobs.models import ITEM_QUEUED, JOB_QUEUED
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    job = await store.create_job(paths=["a.pdf"], config={"max_attempts": 1})
    item = await store.claim_next_item(worker_id="w1", lease_seconds=60)
    await store.fail_item(
        item.item_id,
        JobError(error_type="ParseError", message="bad", retryable=False),
    )

    retried = await store.retry_failed_items(job.job_id)

    refreshed = await store.get_job(job.job_id)
    items = await store.list_job_items(job.job_id)
    assert retried == 1
    assert refreshed.status == JOB_QUEUED
    assert items[0].status == ITEM_QUEUED
    assert items[0].attempt == 0


@pytest.mark.asyncio
async def test_sqlite_job_store_lists_events_after_id(tmp_path):
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    job = await store.create_job(paths=["a.pdf"], config={})
    await store.append_event(job.job_id, "first", "one")
    await store.append_event(job.job_id, "second", "two", payload={"ok": True})
    events = await store.list_events(job.job_id)

    later = await store.list_events(job.job_id, after_id=events[0].event_id)

    assert [event.event for event in later] == ["second"]
    assert later[0].payload == {"ok": True}



@pytest.mark.asyncio
async def test_sqlite_job_store_concurrent_claim_no_double_assignment(tmp_path):
    """Two workers calling claim_next_item concurrently must NOT both get the
    same item. With BEGIN IMMEDIATE, one acquires the write lock first; the
    other sees the row already claimed and either gets the next item or None.
    """
    import asyncio
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    # Create a single-item job; if claim is racy, both workers would each
    # think they own it.
    await store.create_job(paths=["only.pdf"], config={"max_attempts": 3})

    a, b = await asyncio.gather(
        store.claim_next_item(worker_id="worker-a", lease_seconds=60),
        store.claim_next_item(worker_id="worker-b", lease_seconds=60),
        return_exceptions=True,
    )

    # Tolerate either ordering and either an OperationalError "database is
    # locked" or a clean None — both prove the race is closed. What must NOT
    # happen: both return an IngestionJobItem.
    items = [r for r in (a, b) if not isinstance(r, BaseException) and r is not None]
    assert len(items) == 1, (
        f"Expected exactly one worker to claim the item; got {len(items)}. "
        f"a={a!r} b={b!r}"
    )


@pytest.mark.asyncio
async def test_sqlite_job_store_concurrent_claim_distinct_items(tmp_path):
    """When there are enough items for all workers, each worker should get a
    distinct item (no double-assignment, no starvation)."""
    import asyncio
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    await store.create_job(
        paths=["a.pdf", "b.pdf", "c.pdf", "d.pdf"], config={"max_attempts": 3}
    )

    results = await asyncio.gather(
        *[store.claim_next_item(worker_id=f"w{i}", lease_seconds=60) for i in range(4)],
        return_exceptions=True,
    )
    items = [r for r in results if not isinstance(r, BaseException) and r is not None]
    item_ids = {item.item_id for item in items}
    assert len(item_ids) == len(items), (
        f"Workers claimed overlapping items: {[i.item_id for i in items]}"
    )


@pytest.mark.asyncio
async def test_readonly_store_does_not_block_concurrent_write(tmp_path):
    """P0-1 review fix: SQLiteJobStore(readonly=True) must not take
    BEGIN IMMEDIATE write-intent locks. A doctor SELECT running in
    parallel with a worker claim should not deadlock or serialize."""
    import asyncio
    from kb.jobs.sqlite_store import SQLiteJobStore

    writer = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db")
    await writer.init()
    reader = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db", readonly=True)
    # reader does NOT need init() — the table already exists; the
    # readonly constructor must not require schema creation either.
    try:
        await writer.create_job(paths=["/a.pdf"], config={}, owner_id="u-a")

        # Concurrent: reader.list_claimed_items() while writer claims.
        # If readonly mode takes a write lock, this would serialize and
        # one of them would block past the timeout.
        async def reader_task():
            return await reader.list_claimed_items()
        async def writer_task():
            return await writer.claim_next_item(worker_id="w-1", lease_seconds=60)

        results = await asyncio.wait_for(
            asyncio.gather(reader_task(), writer_task()),
            timeout=5.0,
        )
        _, claimed = results
        assert claimed is not None
        # Reader saw either 0 or 1 claimed items depending on scheduling
        # — both are valid; the assertion is just "no timeout".
    finally:
        await reader.close()
        await writer.close()


@pytest.mark.asyncio
async def test_list_claimed_items_returns_dicts_with_lock_fields(tmp_path):
    """P0-2 review fix: list_claimed_items returns list[dict] (not
    IngestionJobItem) so it can include runtime lock fields without
    polluting the main dataclass."""
    from kb.jobs.sqlite_store import SQLiteJobStore
    s = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db")
    await s.init()
    try:
        await s.create_job(paths=["/a.pdf", "/b.pdf"], config={}, owner_id="u-a")
        await s.claim_next_item(worker_id="w-1", lease_seconds=60)
        claimed = await s.list_claimed_items()
        assert len(claimed) == 1
        row = claimed[0]
        # Shape contract — keys doctor depends on:
        for k in (
            "item_id", "job_id", "path", "status", "attempt",
            "owner_id", "locked_by", "locked_until", "claimed_at",
        ):
            assert k in row, f"missing key: {k}"
        assert row["locked_by"] == "w-1"
        assert row["claimed_at"] is not None
        assert row["owner_id"] == "u-a"
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_count_queued_by_owner_groups_correctly(tmp_path):
    """Doctor: 'alice has 5 queued, bob has 2' bar chart."""
    from kb.jobs.sqlite_store import SQLiteJobStore
    s = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db")
    await s.init()
    try:
        await s.create_job(paths=["/a1.pdf", "/a2.pdf", "/a3.pdf"], config={}, owner_id="u-alice")
        await s.create_job(paths=["/b1.pdf"], config={}, owner_id="u-bob")
        await s.create_job(paths=["/old.pdf"], config={}, owner_id=None)
        # Claim one item. Cold start (no prior claims) → fairness sorts by
        # rowid ASC → alice's first item wins (she seeded first). The count
        # assertions below rely on this; if seed order changes, update them.
        await s.claim_next_item(worker_id="w-1", lease_seconds=60)
        counts = await s.count_queued_by_owner()
        assert counts.get("u-alice") == 2   # one of three claimed
        assert counts.get("u-bob") == 1
        assert counts.get(None) == 1        # CLI-style ingest
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_list_recent_claims_by_owner_returns_datetime(tmp_path):
    """P2-7 review fix: SQLAlchemy ORM expression (not raw SQL) so
    SQLite datetime values come back as datetime, not string."""
    from datetime import datetime
    from kb.jobs.sqlite_store import SQLiteJobStore
    s = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db")
    await s.init()
    try:
        await s.create_job(paths=["/a.pdf"], config={}, owner_id="u-alice")
        await s.claim_next_item(worker_id="w-1", lease_seconds=60)
        recent = await s.list_recent_claims_by_owner(window_seconds=3600)
        assert "u-alice" in recent
        assert isinstance(recent["u-alice"], datetime), (
            f"expected datetime, got {type(recent['u-alice'])}"
        )
    finally:
        await s.close()


# ── T4 review fixes: stale claim classification ─────────────────────


@pytest.mark.asyncio
async def test_list_claimed_items_marks_stale_when_lease_expired(tmp_path):
    """T4 review P1#1: items where locked_until <= now are flagged
    is_stale=True so doctor can render them separately from active
    workers (a dead worker shouldn't masquerade as running)."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    from kb.jobs.sqlite_store import SQLiteJobStore, JobItemRecord

    s = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db")
    await s.init()
    try:
        await s.create_job(paths=["/a.pdf", "/b.pdf"], config={}, owner_id="u-x")
        # Claim one with a normal lease — active
        active = await s.claim_next_item(worker_id="w-active", lease_seconds=3600)
        # Claim another with a normal lease, then backdate to expired
        stale = await s.claim_next_item(worker_id="w-dead", lease_seconds=3600)
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
        async with s._engine.begin() as conn:
            await conn.execute(
                update(JobItemRecord)
                .where(JobItemRecord.item_id == stale.item_id)
                .values(locked_until=past)
            )

        claimed = await s.list_claimed_items()
        by_id = {row["item_id"]: row for row in claimed}
        assert by_id[active.item_id]["is_stale"] is False
        assert by_id[stale.item_id]["is_stale"] is True
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_count_queued_by_owner_include_reclaimable_counts_stale(tmp_path):
    """T4 review P1#1: include_reclaimable=True also counts stale-claim
    items so doctor's queue figure matches what the next claim round
    will pick up. Default behavior unchanged (strict QUEUED only)."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import update
    from kb.jobs.sqlite_store import SQLiteJobStore, JobItemRecord

    s = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db")
    await s.init()
    try:
        await s.create_job(paths=["/a.pdf", "/b.pdf"], config={}, owner_id="u-alice")
        # Claim alice's first item and backdate the lease — stale-reclaimable.
        stale = await s.claim_next_item(worker_id="w-dead", lease_seconds=3600)
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
        async with s._engine.begin() as conn:
            await conn.execute(
                update(JobItemRecord)
                .where(JobItemRecord.item_id == stale.item_id)
                .values(locked_until=past)
            )

        strict = await s.count_queued_by_owner()
        loose = await s.count_queued_by_owner(include_reclaimable=True)
        # Strict: only the remaining QUEUED item counts (1).
        assert strict.get("u-alice") == 1
        # Loose: the stale-claimed one is reclaimable; counts as 2.
        assert loose.get("u-alice") == 2
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_claim_next_item_respects_maintenance_check(tmp_path):
    """T5: when maintenance_check returns True, claim_next_item
    returns None immediately — no items consumed."""
    from kb.jobs.sqlite_store import SQLiteJobStore

    on = False
    async def check():
        return on

    s = SQLiteJobStore(
        db_url=f"sqlite+aiosqlite:///{tmp_path}/j.db",
        maintenance_check=check,
    )
    await s.init()
    try:
        await s.create_job(paths=["/a.pdf"], config={}, owner_id="u-x")

        # Off → normal claim
        item = await s.claim_next_item(worker_id="w-1", lease_seconds=60)
        assert item is not None

        # Complete it so we can claim again from a clean state
        await s.complete_item(item.item_id, result={})

        # Flip and seed another job
        on = True
        await s.create_job(paths=["/b.pdf"], config={}, owner_id="u-x")
        item2 = await s.claim_next_item(worker_id="w-1", lease_seconds=60)
        assert item2 is None  # maintenance gate blocks
    finally:
        await s.close()
