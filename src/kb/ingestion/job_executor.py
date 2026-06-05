import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from kb.jobs.cancel import JobCancelled
from kb.jobs.errors import JobError
from kb.jobs.resource_manager import ResourceManager
from kb.parser.base import ParseError


class IngestionJobExecutor:
    def __init__(
        self,
        store,
        pipeline_factory: Callable[..., object],
        *,
        resource_manager: ResourceManager | None = None,
        resources: list[str] | None = None,
        retry_backoff_seconds: int = 30,
        audit_log=None,
        users_store=None,
    ) -> None:
        self._store = store
        self._pipeline_factory = pipeline_factory
        self._resource_manager = resource_manager
        self._resources = resources or []
        self._retry_backoff_seconds = retry_backoff_seconds
        # Best-effort audit sink for terminal (non-retryable) failures.
        # Worker constructs the AuditLog and injects it; legacy call sites
        # (most tests) leave this None and the audit write is skipped.
        self._audit_log = audit_log
        # T3: users_store is the source of truth for materialising the
        # owner User from job.owner_id so execute_item can bind the
        # current_user ContextVar before invoking the pipeline. Legacy
        # callers (most executor unit tests) leave this None — the
        # ContextVar set/reset still runs symmetrically (B-2 strict
        # pairing) but binds None instead of a real user.
        self._users_store = users_store
        # Factories may or may not accept a cancel_check kwarg — older
        # call sites (tests, scripts) pass a no-arg lambda. Detect once
        # so each pipeline build doesn't pay a signature lookup.
        self._factory_accepts_cancel_check = self._factory_accepts_kwarg(
            pipeline_factory, "cancel_check"
        )

    @staticmethod
    def _factory_accepts_kwarg(factory: Callable, name: str) -> bool:
        try:
            sig = inspect.signature(factory)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        if name in params:
            return True
        # Accept **kwargs as "yes, it can pass it through"
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    async def execute_item(self, item, worker_id: str) -> None:
        from kb.auth.context import current_user

        # T3: fetch the job at entry so we can (a) read owner_id for
        # ContextVar binding, (b) forward owner_id to pipeline.ingest,
        # (c) attribute the job.failed audit to the real owner instead
        # of user_id=None. Legacy callers that mocked store without a
        # get_job AsyncMock must update — see test_ingestion_job_executor.py.
        job = await self._store.get_job(item.job_id)
        owner_id = job.owner_id if job is not None else None

        # Materialize owner User from users_store when present. The
        # '_system' sentinel is the T1a backfill marker for pre-T3
        # jobs — treat it identical to NULL (no real user to bind).
        owner_user = None
        if (
            self._users_store is not None
            and owner_id is not None
            and owner_id != "_system"
        ):
            owner_user = await self._users_store.get_by_user_id(owner_id)

        # B-2: always-paired set/reset (try/finally), even when
        # owner_user is None. Setting None explicitly clears any
        # ContextVar value left over from a previous coroutine on the
        # same worker loop; resetting restores the prior token. Same
        # pattern as T1b auth middleware (src/kb/tool_server/app.py:359).
        ctx_token = current_user.set(owner_user)
        try:
            await self._append_event(
                item,
                "ingest.item.started",
                "Ingestion item started",
                worker_id=worker_id,
                path=item.path,
            )
            try:
                result = await self._run_pipeline(item, owner_id=owner_id)
                await self._store.complete_item(item.item_id, result)
                await self._append_event(
                    item,
                    "ingest.item.finished",
                    "Ingestion item finished",
                    worker_id=worker_id,
                    result=result,
                )
            except JobCancelled as cancelled:
                # Cooperative cancel: distinct path from fail_item. The
                # store has its own cancel_item helper that transitions the
                # item to ITEM_CANCELLED without scheduling a retry; this
                # keeps the doctor view clean and avoids the "cancelled
                # then auto-retried" trap.
                reason = str(cancelled) or "cancelled by user"
                await self._store.cancel_item(item.item_id, reason)
                await self._append_event(
                    item,
                    "ingest.item.cancelled",
                    "Ingestion item cancelled",
                    worker_id=worker_id,
                    reason=reason,
                )
            except Exception as exc:
                job_error = _error_from_exception(exc)
                retry_at = (
                    _now() + timedelta(seconds=self._retry_backoff_seconds)
                    if job_error.retryable
                    else None
                )
                await self._store.fail_item(item.item_id, job_error, retry_at)
                await self._append_event(
                    item,
                    "ingest.item.failed",
                    "Ingestion item failed",
                    level="error",
                    worker_id=worker_id,
                    error=job_error.to_dict(),
                    retry_at=retry_at.isoformat() if retry_at else None,
                )
                # Only audit on terminal (non-retryable) failures so transient
                # retryables that may still succeed don't create misleading
                # job.failed rows. T3 fix (P0-4): user_id is the job owner
                # — audit_log.write does NOT read ContextVar, so binding
                # current_user upstream is insufficient; the kwarg must be
                # set explicitly. Re-fetch the job inside the except in case
                # the original failure happened before the initial get_job
                # call (defensive — currently impossible since get_job is
                # the very first await, but cheap and future-proof).
                if not job_error.retryable and self._audit_log is not None:
                    failed_job = job if job is not None else await self._store.get_job(item.job_id)
                    failed_owner = failed_job.owner_id if failed_job is not None else None
                    await self._audit_log.write(
                        "job.failed",
                        user_id=failed_owner,
                        target_kind="job",
                        target_id=item.job_id,
                        payload={
                            "item_id": item.item_id,
                            "error_type": job_error.error_type,
                            "error_msg": job_error.message[:200],
                        },
                    )
        finally:
            current_user.reset(ctx_token)

    async def _run_pipeline(self, item, *, owner_id: str | None = None) -> dict:
        if self._factory_accepts_cancel_check:
            # Closure binds the item's owning job_id so each pipeline
            # build polls cancel state for the correct job.
            async def _cancel_check() -> bool:
                return await self._store.is_cancel_requested(item.job_id)

            pipeline = self._pipeline_factory(cancel_check=_cancel_check)
        else:
            pipeline = self._pipeline_factory()
        # finally aclose: a long-running worker creates one pipeline per
        # item, and IngestionPipeline owns HTTP clients + SQLAlchemy
        # engines + qdrant pool. Without this, every job leaks those
        # handles until the worker process exits.
        try:
            if self._resource_manager is None or not self._resources:
                # T3: forward owner_id so IngestionPipeline writes the
                # documents.owner_id column + qdrant payload owner_id
                # field. Legacy/pre-T3 IngestionJob rows have owner_id
                # None or "_system" — both are valid kwarg values and
                # the pipeline treats them as "no owner".
                return await pipeline.ingest(Path(item.path), owner_id=owner_id)
            async with self._resource_manager.acquire_many(self._resources):
                return await pipeline.ingest(Path(item.path), owner_id=owner_id)
        finally:
            aclose = getattr(pipeline, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception as exc:
                    # aclose is best-effort cleanup; surfacing its
                    # error here would mask the real ingest exception
                    # (if any) raised above. Log and move on.
                    import logging
                    logging.getLogger(__name__).warning(
                        "pipeline.aclose() failed for item %s: %s",
                        item.item_id, exc,
                    )

    async def _append_event(
        self,
        item,
        event: str,
        message: str,
        *,
        level: str = "info",
        **payload,
    ) -> None:
        append_event = getattr(self._store, "append_event", None)
        if append_event is None:
            return
        await append_event(
            item.job_id,
            event,
            message,
            item_id=item.item_id,
            level=level,
            payload=payload,
        )


def _error_from_exception(exc: Exception) -> JobError:
    return JobError(
        error_type=type(exc).__name__,
        message=str(exc),
        retryable=_is_retryable(exc),
    )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ParseError):
        message = str(exc).lower()
        non_retryable_markers = (
            "api limit",
            "exceeds",
            "encrypted",
            "invalid",
            "not found",
            "unsupported",
        )
        return not any(marker in message for marker in non_retryable_markers)
    exc_type = type(exc).__name__
    if exc_type in {"ConnectTimeout", "ReadTimeout", "PoolTimeout", "RequestError"}:
        return True
    if exc_type == "HTTPStatusError":
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", 0)
        return status_code == 429 or status_code >= 500
    return False


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
