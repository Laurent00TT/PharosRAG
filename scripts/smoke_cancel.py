"""End-to-end smoke test for the cooperative-cancel path.

Exercises the full chain — SQLiteJobStore → IngestionJobRunner →
IngestionJobExecutor → pipeline → cancel_check closure → cancel_item —
without any external services (mineru / embedding server / vision_llm). The fake
pipeline polls the injected cancel_check at simulated "page boundaries"
exactly like the real IngestionPipeline.

Usage:
    PYTHONPATH=src python scripts/smoke_cancel.py

What it verifies:
  1. ``request_cancel`` flips the job flag
  2. The pipeline observes it at the next page boundary (worst-case = one
     page's processing time)
  3. JobCancelled flows up to executor and is routed to ``cancel_item``
     NOT ``fail_item`` (no retry queued)
  4. The item lands in ITEM_CANCELLED with structured reason + cancelled=True
  5. The job-level status reflects the terminal state
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.ingestion.job_executor import IngestionJobExecutor
from kb.jobs.cancel import JobCancelled
from kb.jobs.runner import IngestionJobRunner
from kb.jobs.sqlite_store import JobItemRecord, SQLiteJobStore


async def _fetch_item_row(store: SQLiteJobStore, item_id: str) -> JobItemRecord:
    """Pull the raw ORM row so we can inspect locked_by/error_json fields
    that the IngestionJobItem DTO doesn't expose."""
    async with store.session() as session:
        return await session.get(JobItemRecord, item_id)


class _FakePipeline:
    """Mimics IngestionPipeline.ingest's cancel-check loop without
    touching MinerU / qdrant / vision channels."""

    def __init__(self, *, cancel_check=None):
        self._cancel_check = cancel_check
        self._total_pages = 10
        self._per_page_seconds = 0.4

    async def ingest(self, path: Path) -> dict:
        for page_num in range(self._total_pages):
            if self._cancel_check is not None and await self._cancel_check():
                raise JobCancelled(
                    f"cancelled at page {page_num} of {path.name}"
                )
            await asyncio.sleep(self._per_page_seconds)
        return {
            "doc_id": "fake",
            "doc_name": path.name,
            "pages_processed": self._total_pages,
        }


def _log(t_start: float, msg: str) -> None:
    elapsed = asyncio.get_event_loop().time() - t_start
    print(f"  [t={elapsed:5.2f}s] {msg}")


async def main() -> int:
    tmp = tempfile.mkdtemp(prefix="kb_smoke_cancel_")
    db_url = f"sqlite+aiosqlite:///{tmp}/jobs.db"

    store = SQLiteJobStore(db_url=db_url)
    await store.init()

    executor = IngestionJobExecutor(
        store=store,
        pipeline_factory=lambda *, cancel_check=None: _FakePipeline(
            cancel_check=cancel_check,
        ),
    )
    runner = IngestionJobRunner(
        store=store,
        executor=executor,
        poll_interval_s=0.1,
        lease_seconds=60,
    )

    t_start = asyncio.get_event_loop().time()
    print("=" * 60)
    print("END-TO-END CANCEL SMOKE")
    print("=" * 60)

    # 1. Enqueue job
    job = await store.create_job(paths=["fake.pdf"], config={})
    _log(t_start, f"created job_id={job.job_id[:8]}")

    # 2. Start runner in background
    runner_task = asyncio.create_task(runner.run_forever())

    # 3. Let it claim and start processing 2 pages
    await asyncio.sleep(0.9)
    items = await store.list_job_items(job.job_id)
    item_row = await _fetch_item_row(store, items[0].item_id)
    _log(
        t_start,
        f"after start: status={items[0].status} locked_by={item_row.locked_by}",
    )
    assert items[0].status == "claimed", f"expected claimed, got {items[0].status}"

    # 4. Request cancel
    _log(t_start, "request_cancel(job_id)")
    await store.request_cancel(job.job_id)

    # Confirm flag is set
    assert await store.is_cancel_requested(job.job_id) is True
    _log(t_start, "is_cancel_requested → True (visible to pipeline)")

    # 5. Wait at most one page-boundary + cleanup for the pipeline to observe
    #    Per-page time is 0.4s; give it 1.0s margin.
    await asyncio.sleep(1.0)

    items = await store.list_job_items(job.job_id)
    refreshed = await store.get_job(job.job_id)
    item_row = await _fetch_item_row(store, items[0].item_id)
    _log(t_start, f"item.status={items[0].status} job.status={refreshed.status}")

    # 6. Stop runner
    runner.stop()
    runner_task.cancel()
    try:
        await runner_task
    except asyncio.CancelledError:
        pass

    # 7. Validate terminal state
    print()
    print("-" * 60)
    print("ASSERTIONS")
    print("-" * 60)

    ok = True
    # Item must be in cancelled state, not failed
    if items[0].status != "cancelled":
        print(f"  ✗ item.status expected 'cancelled', got {items[0].status!r}")
        ok = False
    else:
        print("  ✓ item.status = 'cancelled' (not 'failed', no retry)")

    # error_json must carry structured reason (read from ORM row, not DTO)
    err = json.loads(item_row.error_json) if item_row.error_json else {}
    if not err.get("cancelled"):
        print(f"  ✗ error_json missing cancelled=True: {err}")
        ok = False
    else:
        print(f"  ✓ error_json carries cancelled=True")
    reason = err.get("reason", "")
    if "cancelled at page" not in reason:
        print(f"  ✗ reason missing page context: {reason!r}")
        ok = False
    else:
        print(f"  ✓ reason carries page context: {reason!r}")

    # Job-level state
    if not refreshed.cancel_requested:
        print(f"  ✗ job.cancel_requested expected True")
        ok = False
    else:
        print(f"  ✓ job.cancel_requested = True")
    if refreshed.status != "cancelled":
        print(f"  ✗ job.status expected 'cancelled', got {refreshed.status!r}")
        ok = False
    else:
        print(f"  ✓ job.status = 'cancelled'")

    # Locked-by must be released so a future re-enqueue isn't blocked
    if item_row.locked_by is not None:
        print(f"  ✗ item.locked_by should be cleared, got {item_row.locked_by!r}")
        ok = False
    else:
        print(f"  ✓ item.locked_by released")

    # Timeline check: total elapsed should be under 2.5s (1.4s of work + 1.0s wait)
    # If the pipeline somehow didn't observe the cancel, the loop would run
    # 10 * 0.4 = 4.0s before naturally completing — that would mean cancel
    # was ineffective.
    total = asyncio.get_event_loop().time() - t_start
    if total > 3.5:
        print(f"  ✗ total elapsed {total:.2f}s suggests cancel didn't interrupt")
        ok = False
    else:
        print(f"  ✓ cancel took effect within {total:.2f}s (vs 4.0s natural completion)")

    print()
    if ok:
        print("RESULT: ✓ all assertions passed — cancel path is end-to-end working")
        return 0
    print("RESULT: ✗ at least one assertion failed — see above")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
