import asyncio
import contextlib
import logging
import uuid

from kb.jobs.errors import JobError

logger = logging.getLogger(__name__)


class IngestionJobRunner:
    def __init__(
        self,
        store,
        executor,
        *,
        poll_interval_s: float = 2.0,
        lease_seconds: int = 60,
        heartbeat_interval_s: float | None = None,
    ) -> None:
        self._store = store
        self._executor = executor
        self._poll_interval_s = poll_interval_s
        self._lease_seconds = lease_seconds
        self._heartbeat_interval_s = (
            heartbeat_interval_s
            if heartbeat_interval_s is not None
            else max(1.0, lease_seconds / 3)
        )
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self._stopped = asyncio.Event()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def run_once(self) -> bool:
        if self._stopped.is_set():
            return False
        item = await self._store.claim_next_item(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if item is None:
            return False
        heartbeat = asyncio.create_task(self._heartbeat(item.item_id))
        try:
            await self._executor.execute_item(item, self._worker_id)
        except Exception as exc:
            # Defensive fail-safe: the executor's own try/except handles
            # pipeline errors, but if fail_item itself raised (e.g. SQLite
            # disconnect) or a different unhandled error escaped the executor,
            # the item would otherwise stay in CLAIMED state until lease
            # expiry — with attempt counter already incremented and no
            # diagnostic info attached. Catch here as last resort.
            logger.exception(
                "Unhandled exception in executor for item %s", item.item_id
            )
            with contextlib.suppress(Exception):
                await self._store.fail_item(
                    item.item_id,
                    JobError(
                        error_type=f"RunnerUnhandled:{type(exc).__name__}",
                        message=str(exc),
                        retryable=True,
                    ),
                )
            raise
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        return True

    async def run_forever(self) -> None:
        """Main worker loop. Survives unexpected errors from run_once so a
        single anomaly (e.g. a transient SQLite I/O error escaping the
        executor) can't kill the worker process. Without this, a
        long-running worker would die at the first unhandled exception
        and silently stop processing the queue.

        ``asyncio.CancelledError`` is the one exception we deliberately
        let propagate — that's how callers (or asyncio.shield) ask the
        loop to stop.
        """
        while not self._stopped.is_set():
            try:
                did_work = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Log with full traceback then back off. We intentionally
                # don't try to recover state here — run_once handles its
                # own item lifecycle in finally; what reaches us is
                # something exotic (DB connection lost mid-claim, OOM,
                # etc.) and the safest recovery is "wait a beat, try again".
                logger.exception("run_once crashed; backing off and continuing")
                did_work = False
            if not did_work:
                await asyncio.sleep(self._poll_interval_s)

    async def start(self) -> None:
        await self.run_forever()

    def stop(self) -> None:
        self._stopped.set()

    async def _heartbeat(self, item_id: str) -> None:
        while not self._stopped.is_set():
            await asyncio.sleep(self._heartbeat_interval_s)
            ok = await self._store.heartbeat(
                item_id=item_id,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            if not ok:
                return
