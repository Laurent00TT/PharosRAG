import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_job_runner_claims_and_executes_one_item():
    from kb.jobs.runner import IngestionJobRunner

    item = SimpleNamespace(item_id="i1", job_id="j1")
    store = MagicMock()
    store.claim_next_item = AsyncMock(return_value=item)
    store.heartbeat = AsyncMock(return_value=True)
    executor = MagicMock()
    executor.execute_item = AsyncMock()

    runner = IngestionJobRunner(
        store=store,
        executor=executor,
        poll_interval_s=0.01,
        lease_seconds=60,
    )
    did_work = await runner.run_once()

    assert did_work is True
    executor.execute_item.assert_awaited_once_with(item, runner.worker_id)


@pytest.mark.asyncio
async def test_job_runner_returns_false_when_no_item():
    from kb.jobs.runner import IngestionJobRunner

    store = MagicMock()
    store.claim_next_item = AsyncMock(return_value=None)
    executor = MagicMock()
    executor.execute_item = AsyncMock()

    runner = IngestionJobRunner(store=store, executor=executor, poll_interval_s=0)

    assert await runner.run_once() is False
    executor.execute_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_job_runner_heartbeats_while_executor_runs():
    from kb.jobs.runner import IngestionJobRunner

    item = SimpleNamespace(item_id="i1", job_id="j1")
    store = MagicMock()
    store.claim_next_item = AsyncMock(return_value=item)
    store.heartbeat = AsyncMock(return_value=True)

    async def execute_item(_item, _worker_id):
        await asyncio.sleep(0.03)

    executor = MagicMock()
    executor.execute_item = AsyncMock(side_effect=execute_item)
    runner = IngestionJobRunner(
        store=store,
        executor=executor,
        poll_interval_s=0.01,
        lease_seconds=60,
        heartbeat_interval_s=0.01,
    )

    await runner.run_once()

    assert store.heartbeat.await_count >= 1


@pytest.mark.asyncio
async def test_job_runner_stop_prevents_new_claims():
    from kb.jobs.runner import IngestionJobRunner

    store = MagicMock()
    store.claim_next_item = AsyncMock()
    executor = MagicMock()
    runner = IngestionJobRunner(store=store, executor=executor)

    runner.stop()
    did_work = await runner.run_once()

    assert did_work is False
    store.claim_next_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_forever_survives_run_once_exception():
    """Critical long-running invariant: a single unexpected exception
    from run_once (e.g. transient SQLite I/O error) MUST NOT kill the
    worker loop. Earlier behavior re-raised and ended run_forever, so
    long-running workers silently died at the first anomaly."""
    from kb.jobs.runner import IngestionJobRunner

    call_count = {"n": 0}
    store = MagicMock()

    async def claim_then_recover(*_args, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated transient hiccup")
        # Second call: signal "no work" so the loop sleeps briefly
        return None

    store.claim_next_item = AsyncMock(side_effect=claim_then_recover)
    executor = MagicMock()
    runner = IngestionJobRunner(
        store=store, executor=executor,
        poll_interval_s=0.01, lease_seconds=60,
    )

    # Drive the loop just long enough to prove it survived the first call
    task = asyncio.create_task(runner.run_forever())
    await asyncio.sleep(0.1)
    runner.stop()
    await asyncio.wait_for(task, timeout=1.0)

    # First call crashed, subsequent calls succeeded → call_count > 1
    assert call_count["n"] >= 2, (
        "run_forever should have kept looping after the first exception"
    )


@pytest.mark.asyncio
async def test_run_forever_propagates_cancelled_error():
    """Cancellation is the one exception we deliberately let propagate
    — it's the mechanism for graceful shutdown via task.cancel()."""
    from kb.jobs.runner import IngestionJobRunner

    store = MagicMock()
    store.claim_next_item = AsyncMock(side_effect=asyncio.CancelledError())
    executor = MagicMock()
    runner = IngestionJobRunner(
        store=store, executor=executor, poll_interval_s=0.01,
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run_forever()


@pytest.mark.asyncio
async def test_runner_marks_item_failed_when_executor_raises_unhandled(tmp_path):
    """Edge case: if execute_item raises (e.g. executor's own fail_item failed),
    runner's defensive layer should still call fail_item so the item doesn't
    stay stuck in CLAIMED."""
    from kb.jobs.runner import IngestionJobRunner
    from kb.jobs.sqlite_store import SQLiteJobStore

    store = SQLiteJobStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db")
    await store.init()
    job = await store.create_job(paths=["a.pdf"], config={"max_attempts": 1})

    class _BadExecutor:
        async def execute_item(self, item, worker_id):
            raise RuntimeError("simulated unhandled exception")

    runner = IngestionJobRunner(
        store=store,
        executor=_BadExecutor(),
        poll_interval_s=0.01,
        lease_seconds=60,
    )

    with pytest.raises(RuntimeError, match="simulated unhandled"):
        await runner.run_once()

    # Item should NOT be in CLAIMED; runner's defensive fail_item put it in
    # FAILED (with attempt=1, max_attempts=1 → terminal failure, not retried).
    refreshed = await store.get_job(job.job_id)
    assert refreshed.failed_items == 1
