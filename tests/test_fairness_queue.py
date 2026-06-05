"""Fairness-aware claim_next_item (T4 / I-10).

The invariant: when multiple owners have queued items, the worker picks
the item whose owner has been idle the longest (LRU on most recent
claim within a 1-hour window). Within a single owner, ties break on
items.rowid ASC (insertion order; the table has no explicit created_at
column).
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from kb.jobs.sqlite_store import SQLiteJobStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await s.init()
    yield s
    await s.close()


async def _seed_job(store, owner_id: str | None, n_items: int) -> str:
    """Helper: create a job with N items and known owner_id."""
    job = await store.create_job(
        paths=[f"/fake-{owner_id}-{i}.pdf" for i in range(n_items)],
        config={},
        owner_id=owner_id,
    )
    return job.job_id


@pytest.mark.asyncio
async def test_claim_writes_claimed_at(store):
    """C-1: claimed_at must be written in the same UPDATE as status."""
    await _seed_job(store, owner_id="u-alice", n_items=1)
    item = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    assert item is not None
    assert item.claimed_at is not None
    assert (datetime.now(timezone.utc).replace(tzinfo=None) - item.claimed_at) < timedelta(seconds=5)


@pytest.mark.asyncio
async def test_single_owner_preserves_fifo(store):
    """Within one owner, items.rowid ASC tiebreak (insertion order; no
    explicit created_at column) yields FIFO."""
    await _seed_job(store, owner_id="u-alice", n_items=3)
    claimed_paths = []
    for _ in range(3):
        item = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
        claimed_paths.append(item.path)
        await store.complete_item(item.item_id, result={})
    assert claimed_paths == [
        "/fake-u-alice-0.pdf", "/fake-u-alice-1.pdf", "/fake-u-alice-2.pdf",
    ]


@pytest.mark.asyncio
async def test_two_owners_interleave_after_first_claim(store):
    """Spec scenario: alice submits 5 items, bob submits 1 AFTER alice's
    first item is claimed. Bob's item should claim BEFORE alice's #2.
    """
    await _seed_job(store, owner_id="u-alice", n_items=5)
    item_a1 = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    assert item_a1.path.startswith("/fake-u-alice")
    await _seed_job(store, owner_id="u-bob", n_items=1)
    await store.complete_item(item_a1.item_id, result={})
    item_next = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    assert item_next.path.startswith("/fake-u-bob"), (
        f"expected bob's item to jump the queue, got {item_next.path}"
    )


@pytest.mark.asyncio
async def test_failed_items_count_for_fairness(store):
    """C-3: alice's failed items still push her MAX(claimed_at) — she
    cannot batch-fail to reset her bucket and dominate the queue.
    """
    from kb.jobs.errors import JobError

    await _seed_job(store, owner_id="u-alice", n_items=2)
    a1 = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    await store.fail_item(
        a1.item_id,
        JobError(error_type="X", message="x", retryable=False),
        None,
    )
    await _seed_job(store, owner_id="u-bob", n_items=1)
    nxt = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    assert nxt.path.startswith("/fake-u-bob")


@pytest.mark.asyncio
async def test_cold_start_degenerates_to_fifo(store):
    """C-2: with no recent claims, fairness sort degenerates to insertion order."""
    await _seed_job(store, owner_id="u-bob",   n_items=1)
    await asyncio.sleep(0.01)
    await _seed_job(store, owner_id="u-alice", n_items=1)
    item = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    assert item.path == "/fake-u-bob-0.pdf", (
        "cold start: oldest insertion order should win (bob created first)"
    )


@pytest.mark.asyncio
async def test_stale_claim_outside_window_treats_owner_as_cold(store):
    """C-2 boundary: an owner whose most recent claim is older than the
    fairness window (1 hour) is treated identically to an owner with no
    claims at all. Both have NULL recent.last_claim and sort by rowid.
    """
    from kb.jobs.sqlite_store import JobItemRecord
    from sqlalchemy import update

    # alice has 2 items, claims #1
    await _seed_job(store, owner_id="u-alice", n_items=2)
    a1 = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    assert a1 is not None
    await store.complete_item(a1.item_id, result={})

    # Backdate alice's claimed_at on item #1 to 90 minutes ago
    stale_ts = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=90)
    async with store._engine.begin() as conn:
        await conn.execute(
            update(JobItemRecord)
            .where(JobItemRecord.item_id == a1.item_id)
            .values(claimed_at=stale_ts)
        )

    # bob arrives with 1 item AFTER alice's stale claim
    await _seed_job(store, owner_id="u-bob", n_items=1)

    # Next claim: alice's stale claim is outside the window, so alice's
    # recent.last_claim is NULL — same as bob's. Tiebreak by rowid: alice's
    # remaining item was inserted before bob's, so alice wins.
    nxt = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
    assert nxt is not None
    assert nxt.path.startswith("/fake-u-alice"), (
        f"expected alice's #2 to win (stale claim → cold → rowid order), "
        f"got {nxt.path}"
    )


@pytest.mark.asyncio
async def test_null_and_system_owners_can_be_claimed(store):
    """C-4: NULL owner_id and '_system' owner_id are both valid claim
    targets — the LEFT JOIN's NULL-safe ``IS`` match must not filter
    them out, otherwise CLI ingest and T1a backfill rows would be
    permanently stranded.

    Verifies the JOIN doesn't crash AND both buckets eventually drain.
    """
    await _seed_job(store, owner_id=None,       n_items=1)
    await _seed_job(store, owner_id="_system",  n_items=1)
    await _seed_job(store, owner_id="u-alice",  n_items=1)

    claimed_owners: list[str | None] = []
    for _ in range(3):
        item = await store.claim_next_item(worker_id="w-1", lease_seconds=60)
        assert item is not None, "claim returned None despite queued items"
        job = await store.get_job(item.job_id)
        claimed_owners.append(job.owner_id if job else "?")
        await store.complete_item(item.item_id, result={})

    assert set(claimed_owners) == {None, "_system", "u-alice"}
