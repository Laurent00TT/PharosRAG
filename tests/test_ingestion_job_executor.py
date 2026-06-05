from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_executor_completes_item_on_pipeline_success():
    from kb.ingestion.job_executor import IngestionJobExecutor

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    # T3: execute_item fetches the job to read owner_id; mock returns
    # None which the executor treats as "no owner" — pipeline.ingest is
    # still called, just with owner_id=None.
    store.get_job = AsyncMock(return_value=None)
    store.complete_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(return_value={"doc_id": "d", "skipped": False})

    executor = IngestionJobExecutor(
        store=store,
        pipeline_factory=lambda: pipeline,
    )
    await executor.execute_item(item, worker_id="w")

    pipeline.ingest.assert_awaited_once_with(Path("a.pdf"), owner_id=None)
    store.complete_item.assert_awaited_once_with(
        "i1",
        {"doc_id": "d", "skipped": False},
    )
    assert store.append_event.await_count == 2


@pytest.mark.asyncio
async def test_executor_retries_timeout_errors():
    from kb.ingestion.job_executor import IngestionJobExecutor

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.fail_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(side_effect=TimeoutError("slow"))

    executor = IngestionJobExecutor(
        store=store,
        pipeline_factory=lambda: pipeline,
        retry_backoff_seconds=5,
    )
    await executor.execute_item(item, worker_id="w")

    _, error, retry_at = store.fail_item.await_args.args
    assert error.retryable is True
    assert error.error_type == "TimeoutError"
    assert retry_at is not None


@pytest.mark.asyncio
async def test_executor_marks_known_parse_limits_non_retryable():
    from kb.ingestion.job_executor import IngestionJobExecutor
    from kb.parser.base import ParseError

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.fail_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(
        side_effect=ParseError("file exceeds MinerU API limit")
    )

    executor = IngestionJobExecutor(
        store=store,
        pipeline_factory=lambda: pipeline,
    )
    await executor.execute_item(item, worker_id="w")

    _, error, retry_at = store.fail_item.await_args.args
    assert error.retryable is False
    assert retry_at is None


@pytest.mark.asyncio
async def test_executor_routes_jobcancelled_to_cancel_item_not_fail_item():
    """Critical contract: cooperative-cancel must NOT go through
    fail_item (which schedules retry + bumps attempt counter) — it must
    transition the item to ITEM_CANCELLED via cancel_item so the doctor
    view stays clean."""
    from kb.ingestion.job_executor import IngestionJobExecutor
    from kb.jobs.cancel import JobCancelled

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.cancel_item = AsyncMock()
    store.fail_item = AsyncMock()
    store.complete_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(side_effect=JobCancelled("cancelled at page 5"))

    executor = IngestionJobExecutor(
        store=store,
        pipeline_factory=lambda: pipeline,
    )
    await executor.execute_item(item, worker_id="w")

    store.cancel_item.assert_awaited_once_with("i1", "cancelled at page 5")
    store.fail_item.assert_not_called()
    store.complete_item.assert_not_called()
    # The trailing event must signal cancellation, not failure
    event_names = [call.args[1] for call in store.append_event.await_args_list]
    assert "ingest.item.cancelled" in event_names
    assert "ingest.item.failed" not in event_names


@pytest.mark.asyncio
async def test_executor_injects_cancel_check_into_factory_that_accepts_it():
    """Factory accepting cancel_check kwarg should receive a bound closure
    that polls the store for the item's owning job_id. Pre-existing
    no-arg factories must still work unchanged."""
    from kb.ingestion.job_executor import IngestionJobExecutor

    item = SimpleNamespace(item_id="i1", job_id="job-xyz", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.complete_item = AsyncMock()
    store.append_event = AsyncMock()
    store.is_cancel_requested = AsyncMock(return_value=False)
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(return_value={"doc_id": "d"})

    received = {}

    def factory(*, cancel_check=None):
        received["cancel_check"] = cancel_check
        return pipeline

    executor = IngestionJobExecutor(store=store, pipeline_factory=factory)
    await executor.execute_item(item, worker_id="w")

    assert received["cancel_check"] is not None
    # Polling the closure should hit is_cancel_requested with the right job_id
    await received["cancel_check"]()
    store.is_cancel_requested.assert_awaited_once_with("job-xyz")


@pytest.mark.asyncio
async def test_executor_acloses_pipeline_after_success():
    """Long-running workers create one pipeline per item — pipeline.aclose()
    MUST run in a finally so we don't leak HTTP clients / SQLite handles
    across thousands of jobs."""
    from kb.ingestion.job_executor import IngestionJobExecutor

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.complete_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(return_value={"doc_id": "d"})
    pipeline.aclose = AsyncMock()

    executor = IngestionJobExecutor(
        store=store, pipeline_factory=lambda: pipeline,
    )
    await executor.execute_item(item, worker_id="w")
    pipeline.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_acloses_pipeline_after_failure():
    """Even when ingest raises, aclose must still run — otherwise a
    failing-job loop leaks resources fastest."""
    from kb.ingestion.job_executor import IngestionJobExecutor

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.fail_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(side_effect=TimeoutError("slow"))
    pipeline.aclose = AsyncMock()

    executor = IngestionJobExecutor(
        store=store, pipeline_factory=lambda: pipeline,
    )
    await executor.execute_item(item, worker_id="w")
    pipeline.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_acloses_pipeline_after_jobcancelled():
    """Cancel path is the third exit branch — must close too."""
    from kb.ingestion.job_executor import IngestionJobExecutor
    from kb.jobs.cancel import JobCancelled

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.cancel_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(side_effect=JobCancelled("cancelled"))
    pipeline.aclose = AsyncMock()

    executor = IngestionJobExecutor(
        store=store, pipeline_factory=lambda: pipeline,
    )
    await executor.execute_item(item, worker_id="w")
    pipeline.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_swallows_aclose_errors():
    """aclose is best-effort. If it raises, the original ingest result
    (or original exception) must still be the executor's outcome —
    aclose noise is logged, not propagated."""
    from kb.ingestion.job_executor import IngestionJobExecutor

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.complete_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(return_value={"doc_id": "d"})
    pipeline.aclose = AsyncMock(side_effect=RuntimeError("conn already gone"))

    executor = IngestionJobExecutor(
        store=store, pipeline_factory=lambda: pipeline,
    )
    # Must complete normally despite aclose blowup
    await executor.execute_item(item, worker_id="w")
    store.complete_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_executor_skips_cancel_check_for_legacy_noarg_factory():
    """Older factories (e.g. the test_executor_completes_item_on_pipeline_success
    above, or third-party callers) take no kwargs. The executor must
    detect that and avoid passing cancel_check, otherwise the factory
    call would raise TypeError."""
    from kb.ingestion.job_executor import IngestionJobExecutor

    item = SimpleNamespace(item_id="i1", job_id="j1", path="a.pdf")
    store = MagicMock()
    store.get_job = AsyncMock(return_value=None)
    store.complete_item = AsyncMock()
    store.append_event = AsyncMock()
    pipeline = MagicMock()
    pipeline.ingest = AsyncMock(return_value={"doc_id": "d"})

    # Strict no-arg factory; raises if cancel_check is passed.
    executor = IngestionJobExecutor(
        store=store,
        pipeline_factory=lambda: pipeline,
    )
    await executor.execute_item(item, worker_id="w")
    store.complete_item.assert_awaited_once()
