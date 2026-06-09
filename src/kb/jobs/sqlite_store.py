import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from sqlalchemy import Boolean, DateTime, Integer, String, Text, event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

from kb.jobs.errors import JobError
from kb.jobs.models import (
    ITEM_CLAIMED,
    ITEM_CANCELLED,
    ITEM_FAILED,
    ITEM_QUEUED,
    ITEM_SKIPPED,
    ITEM_SUCCEEDED,
    JOB_CANCELLED,
    JOB_CANCELLING,
    JOB_FAILED,
    JOB_PARTIALLY_SUCCEEDED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    IngestionJob,
    IngestionJobEvent,
    IngestionJobItem,
)


# Raw SQL rather than SQLAlchemy Core because the LEFT JOIN + GROUP BY
# subquery that powers fairness is awkward to express without string
# interpolation: SQLAlchemy's label()/subquery() API would need explicit
# aliasing for the correlated ON clause, adding noise with no safety gain
# (the parameters :now and :window_start are still bound, so there is no
# injection risk).  Keeping it as a single named text() constant also makes
# the intent — "this is the fairness query" — easier to review than a chain
# of .join()/.group_by() calls.
_CLAIM_SQL = text("""
    SELECT items.item_id
    FROM ingestion_job_items items
    JOIN ingestion_jobs jobs ON items.job_id = jobs.job_id
    LEFT JOIN (
        SELECT j.owner_id, MAX(i.claimed_at) AS last_claim
        FROM ingestion_job_items i
        JOIN ingestion_jobs j ON i.job_id = j.job_id
        WHERE i.claimed_at IS NOT NULL
          AND i.claimed_at > :window_start
        GROUP BY j.owner_id
    ) recent ON jobs.owner_id IS recent.owner_id    -- NULL-safe; covers both NULL=NULL and non-NULL equality in SQLite
    WHERE (
        items.status = 'queued'
        OR (items.status = 'claimed'
            AND items.locked_until IS NOT NULL
            AND items.locked_until <= :now)
    )
    AND (items.run_after IS NULL OR items.run_after <= :now)
    ORDER BY
        recent.last_claim IS NULL DESC,
        recent.last_claim ASC,
        items.rowid ASC
    LIMIT 1
""")

# T4 / I-10: an owner is "active" in the fairness LRU calculation if
# they've claimed an item in this window. Outside the window, the
# owner is treated as cold (NULL last_claim → sorts first via NULLS
# FIRST). 1 hour matches the spec's intuition that "fairness should
# rebalance within a small-team workday."
_FAIRNESS_WINDOW = timedelta(hours=1)


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "ingestion_jobs"
    job_id:           Mapped[str]        = mapped_column(String, primary_key=True)
    status:           Mapped[str]        = mapped_column(String)
    total_items:      Mapped[int]        = mapped_column(Integer)
    succeeded_items:  Mapped[int]        = mapped_column(Integer, default=0)
    failed_items:     Mapped[int]        = mapped_column(Integer, default=0)
    skipped_items:    Mapped[int]        = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool]       = mapped_column(Boolean, default=False)
    config_json:      Mapped[str]        = mapped_column(Text, default="{}")
    owner_id:         Mapped[str | None] = mapped_column(String, nullable=True)  # T1a I-9


class JobItemRecord(Base):
    __tablename__ = "ingestion_job_items"
    item_id:       Mapped[str]               = mapped_column(String, primary_key=True)
    job_id:        Mapped[str]               = mapped_column(String, index=True)
    path:          Mapped[str]               = mapped_column(Text)
    status:        Mapped[str]               = mapped_column(String, index=True)
    attempt:       Mapped[int]               = mapped_column(Integer, default=0)
    max_attempts:  Mapped[int]               = mapped_column(Integer, default=3)
    locked_by:     Mapped[str | None]        = mapped_column(String, nullable=True)
    locked_until:  Mapped[datetime | None]   = mapped_column(DateTime, nullable=True)
    run_after:     Mapped[datetime | None]   = mapped_column(DateTime, nullable=True)
    result_json:   Mapped[str]               = mapped_column(Text, default="{}")
    error_json:    Mapped[str]               = mapped_column(Text, default="{}")
    claimed_at:    Mapped[datetime | None]   = mapped_column(DateTime, nullable=True)  # T1a P1-4


class JobEventRecord(Base):
    __tablename__ = "ingestion_job_events"
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    item_id: Mapped[str | None] = mapped_column(String, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    level: Mapped[str] = mapped_column(String)
    event: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")


class SQLiteJobStore:
    def __init__(
        self,
        db_url: str,
        *,
        readonly: bool = False,
        maintenance_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self._engine = create_async_engine(db_url, poolclass=NullPool)
        self._readonly = readonly
        self._maintenance_check = maintenance_check

        # Note: `readonly` is *advisory* — it only changes the begin-listener
        # behavior; it does NOT enforce read-only at the method level. A caller
        # that uses readonly=True and then calls a write method (create_job,
        # claim_next_item) WILL write, just without the IMMEDIATE lock
        # guarantee. Doctor + monitoring are the documented callers; if a new
        # caller needs hard enforcement, add a guard at the method level.
        if not readonly:
            # Force every transaction to use BEGIN IMMEDIATE so a write intent is
            # taken at the start of the transaction rather than upgraded lazily at
            # the first write. This closes a TOCTOU window in claim_next_item where
            # two workers could both read the same queued row and then race to mark
            # it claimed. SQLite has no row-level lock; without IMMEDIATE the
            # read in claim_next_item does not reserve write access until commit
            # time, by which point both writers may have produced inconsistent state.
            #
            # SQLAlchemy's auto-BEGIN is DEFERRED; the listener replaces it.
            # Read-only consumers (doctor, monitoring) skip this so their
            # SELECTs don't contend with worker writes for the global file
            # lock — review P0-1 fix.
            @event.listens_for(self._engine.sync_engine, "begin")
            def _begin_immediate(conn):
                conn.exec_driver_sql("BEGIN IMMEDIATE")

        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    def session(self) -> AsyncSession:
        return self._session_factory()

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            # T1a idempotent ALTER + backfill — see spec P1-2/I-9/P1-4
            jobs_cols = {
                row[1] for row in
                (await conn.exec_driver_sql("PRAGMA table_info(ingestion_jobs)")).fetchall()
            }
            if "owner_id" not in jobs_cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE ingestion_jobs ADD COLUMN owner_id TEXT"
                )
                # Backfill any pre-existing rows with the system sentinel so
                # downstream code-level "owner_id IS NULL means missing" stays
                # a meaningful check (vs ambiguous old-row-or-bug).
                await conn.exec_driver_sql(
                    "UPDATE ingestion_jobs SET owner_id = '_system' WHERE owner_id IS NULL"
                )

            items_cols = {
                row[1] for row in
                (await conn.exec_driver_sql("PRAGMA table_info(ingestion_job_items)")).fetchall()
            }
            if "claimed_at" not in items_cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE ingestion_job_items ADD COLUMN claimed_at DATETIME"
                )

    async def close(self) -> None:
        await self._engine.dispose()

    async def create_job(
        self,
        paths: list[str],
        config: dict,
        *,
        owner_id: str | None = None,
    ) -> IngestionJob:
        """Create a job row + N item rows. owner_id flows from the calling
        REST handler (POST /ingestion/jobs uses current_user.user_id; CLI
        /scripts pass None and the row remains NULL → admin-only writes
        per I-11)."""
        job_id = uuid.uuid4().hex
        max_attempts = int(config.get("max_attempts", 3))
        async with self._session_factory() as session:
            job = JobRecord(
                job_id=job_id,
                status=JOB_QUEUED,
                total_items=len(paths),
                config_json=json.dumps(config, ensure_ascii=False),
                owner_id=owner_id,
            )
            session.add(job)
            for path in paths:
                session.add(JobItemRecord(
                    item_id=uuid.uuid4().hex,
                    job_id=job_id,
                    path=path,
                    status=ITEM_QUEUED,
                    max_attempts=max_attempts,
                ))
            await session.commit()
        return IngestionJob(
            job_id=job_id,
            status=JOB_QUEUED,
            total_items=len(paths),
            config=config,
            owner_id=owner_id,
        )

    async def claim_next_item(
        self,
        worker_id: str,
        lease_seconds: int,
    ) -> IngestionJobItem | None:
        """Fairness-aware claim (T4 / I-10)."""
        # T5: maintenance gate — backup script flips this ON to drain
        # the queue. None means no gate configured (test/dev fixtures).
        if self._maintenance_check is not None and await self._maintenance_check():
            return None
        now = _now()
        window_start = now - _FAIRNESS_WINDOW
        async with self._session_factory() as session:
            row = (await session.execute(
                _CLAIM_SQL,
                {"now": now, "window_start": window_start},
            )).first()
            if row is None:
                return None
            item_id = row[0]
            rec = await session.get(JobItemRecord, item_id)
            # Defensive: under BEGIN IMMEDIATE this is unreachable unless the row
            # was deleted outside the store API. Keep the guard to fail-soft.
            if rec is None:
                return None
            job = await session.get(JobRecord, rec.job_id)
            if job and job.cancel_requested:
                rec.status = ITEM_CANCELLED
                rec.locked_by = None
                rec.locked_until = None
                await self._update_job_counts(session, rec.job_id)
                await session.commit()
                return None
            rec.status = ITEM_CLAIMED
            rec.attempt += 1
            rec.locked_by = worker_id
            rec.locked_until = now + timedelta(seconds=lease_seconds)
            rec.claimed_at = now            # C-1: write claimed_at
            if job and job.status == JOB_QUEUED:
                job.status = JOB_RUNNING
            await session.commit()
            return _item_from_record(rec)

    async def heartbeat(
        self,
        item_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        async with self._session_factory() as session:
            rec = await session.get(JobItemRecord, item_id)
            if (
                rec is None
                or rec.status != ITEM_CLAIMED
                or rec.locked_by != worker_id
            ):
                return False
            rec.locked_until = _now() + timedelta(seconds=lease_seconds)
            await session.commit()
            return True

    async def complete_item(self, item_id: str, result: dict) -> None:
        async with self._session_factory() as session:
            rec = await session.get(JobItemRecord, item_id)
            if rec is None:
                return
            rec.status = ITEM_SKIPPED if result.get("skipped") else ITEM_SUCCEEDED
            rec.result_json = json.dumps(result, ensure_ascii=False)
            rec.locked_by = None
            rec.locked_until = None
            await self._update_job_counts(session, rec.job_id)
            await session.commit()

    async def fail_item(
        self,
        item_id: str,
        error: JobError,
        retry_at=None,
    ) -> None:
        async with self._session_factory() as session:
            rec = await session.get(JobItemRecord, item_id)
            if rec is None:
                return
            should_retry = error.retryable and rec.attempt < rec.max_attempts
            rec.status = ITEM_QUEUED if should_retry else ITEM_FAILED
            rec.run_after = retry_at
            rec.error_json = json.dumps(error.to_dict(), ensure_ascii=False)
            rec.locked_by = None
            rec.locked_until = None
            await self._update_job_counts(session, rec.job_id)
            await session.commit()

    async def is_cancel_requested(self, job_id: str) -> bool:
        """Cheap polled read used by the pipeline at page boundaries to
        observe cooperative-cancel signals. The job table is small and
        ``job_id`` is the primary key, so this is essentially free.

        Returns False for unknown ``job_id`` rather than raising — the
        pipeline polls this every page; raising on a transiently-missing
        job would turn a normal ingest into a fail-and-retry loop.
        """
        async with self._session_factory() as session:
            job = await session.get(JobRecord, job_id)
            if job is None:
                return False
            return bool(job.cancel_requested)

    async def cancel_item(self, item_id: str, reason: str) -> None:
        """Mark an item as cooperatively cancelled by a running worker.

        Distinct from ``fail_item``: no retry_at, no attempt counter
        bump, status goes directly to ITEM_CANCELLED. The reason string
        is stored in error_json for traceability (so the doctor view
        can show "cancelled at page N" rather than a generic stop)."""
        async with self._session_factory() as session:
            rec = await session.get(JobItemRecord, item_id)
            if rec is None:
                return
            rec.status = ITEM_CANCELLED
            rec.error_json = json.dumps(
                {"reason": reason, "cancelled": True}, ensure_ascii=False,
            )
            rec.locked_by = None
            rec.locked_until = None
            await self._update_job_counts(session, rec.job_id)
            await session.commit()

    async def request_cancel(self, job_id: str) -> None:
        async with self._session_factory() as session:
            job = await session.get(JobRecord, job_id)
            if job is None:
                return
            job.cancel_requested = True
            if job.status not in {
                JOB_SUCCEEDED,
                JOB_PARTIALLY_SUCCEEDED,
                JOB_FAILED,
                JOB_CANCELLED,
            }:
                job.status = JOB_CANCELLING
            result = await session.execute(
                select(JobItemRecord).where(JobItemRecord.job_id == job_id)
            )
            for item in result.scalars().all():
                if item.status == ITEM_QUEUED:
                    item.status = ITEM_CANCELLED
                    item.locked_by = None
                    item.locked_until = None
            await self._update_job_counts(session, job_id)
            await session.commit()

    async def retry_failed_items(self, job_id: str) -> int:
        async with self._session_factory() as session:
            job = await session.get(JobRecord, job_id)
            if job is None:
                return 0
            result = await session.execute(
                select(JobItemRecord)
                .where(JobItemRecord.job_id == job_id)
                .where(JobItemRecord.status == ITEM_FAILED)
            )
            retried = 0
            for item in result.scalars().all():
                item.status = ITEM_QUEUED
                item.attempt = 0
                item.locked_by = None
                item.locked_until = None
                item.run_after = None
                item.error_json = "{}"
                retried += 1
            if retried:
                job.status = JOB_QUEUED
                job.cancel_requested = False
                await self._update_job_counts(session, job_id)
            await session.commit()
            return retried

    async def get_job(self, job_id: str) -> IngestionJob | None:
        async with self._session_factory() as session:
            rec = await session.get(JobRecord, job_id)
            return _job_from_record(rec) if rec else None

    async def list_job_items(self, job_id: str) -> list[IngestionJobItem]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(JobItemRecord).where(JobItemRecord.job_id == job_id)
            )
            return [_item_from_record(row) for row in result.scalars().all()]

    async def list_claimed_items(self) -> list[dict]:
        """Items currently held by a worker (status=CLAIMED).

        Returns dicts (not IngestionJobItem) because doctor needs the
        runtime lock fields (locked_by / locked_until / claimed_at) and
        the job-owner field, but stuffing them into the main dataclass
        would be lock-state pollution on the hot path. P0-2 review fix.

        ``is_stale`` flags items whose lease has expired
        (``locked_until <= now``). These are still ``status=CLAIMED`` in
        the DB but the next ``claim_next_item`` call will reclaim them
        (matches the lease-expired branch in ``_CLAIM_SQL``). Doctor
        renders them as a separate "Stale (reclaimable)" sub-section so
        a dead worker doesn't masquerade as an active one.
        """
        now = _now()
        async with self._session_factory() as session:
            result = await session.execute(
                select(JobItemRecord, JobRecord.owner_id)
                .join(JobRecord, JobItemRecord.job_id == JobRecord.job_id)
                .where(JobItemRecord.status == ITEM_CLAIMED)
            )
            rows = result.all()
            return [
                {
                    "item_id": item.item_id,
                    "job_id": item.job_id,
                    "path": item.path,
                    "status": item.status,
                    "attempt": item.attempt,
                    "owner_id": owner_id,
                    "locked_by": item.locked_by,
                    "locked_until": item.locked_until,
                    "claimed_at": item.claimed_at,
                    "is_stale": bool(
                        item.locked_until is not None
                        and item.locked_until <= now
                    ),
                }
                for item, owner_id in rows
            ]

    async def count_active_claims(self) -> int:
        """Count items with a NON-EXPIRED lease (status=CLAIMED AND
        locked_until > now). Drain loop uses this.

        Review fix P1.3: an earlier draft counted ALL status=CLAIMED
        rows, which includes stale-but-already-reclaimable items
        (locked_until <= now). Stale claims will be picked up by the
        next claim_next_item round; they do NOT represent a worker
        actively writing. Drain should only wait for the active set —
        otherwise a dead worker's expired lease blocks backup forever.

        For doctor's "Active workers" line, T4 already has
        list_claimed_items()[*]['is_stale'] — this helper is for the
        drain count specifically.
        """
        now = _now()
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count(JobItemRecord.item_id))
                .where(JobItemRecord.status == ITEM_CLAIMED)
                .where(JobItemRecord.locked_until.is_not(None))
                .where(JobItemRecord.locked_until > now)
            )
            return result.scalar_one()

    async def count_queued_by_owner(
        self, *, include_reclaimable: bool = False,
    ) -> dict[str | None, int]:
        """Group queued items by job.owner_id → count.

        Includes NULL (CLI ingest) and '_system' (T1a backfill) as
        distinct keys so doctor can render an "orphan queue" entry.

        ``include_reclaimable=True`` also counts CLAIMED items whose
        lease has expired (``locked_until <= now``). This mirrors the
        WHERE clause of ``_CLAIM_SQL`` so doctor's "items waiting"
        figure agrees with what the next claim round will pick up.
        Default keeps the historical strict-queued semantic.
        """
        now = _now()
        async with self._session_factory() as session:
            stmt = (
                select(JobRecord.owner_id, func.count(JobItemRecord.item_id))
                .join(JobItemRecord, JobItemRecord.job_id == JobRecord.job_id)
                .group_by(JobRecord.owner_id)
            )
            if include_reclaimable:
                stmt = stmt.where(
                    (JobItemRecord.status == ITEM_QUEUED)
                    | (
                        (JobItemRecord.status == ITEM_CLAIMED)
                        & (JobItemRecord.locked_until.is_not(None))
                        & (JobItemRecord.locked_until <= now)
                    )
                )
            else:
                stmt = stmt.where(JobItemRecord.status == ITEM_QUEUED)
            result = await session.execute(stmt)
            # GROUP BY owner_id collapses duplicates; dict-comp key collision is impossible here.
            return {row[0]: row[1] for row in result.all()}

    async def list_recent_claims_by_owner(
        self, window_seconds: int = int(_FAIRNESS_WINDOW.total_seconds()),
    ) -> dict[str | None, datetime]:
        """MAX(claimed_at) per owner within the window.

        Used by doctor (fairness debugging). SQLAlchemy ORM expression
        (not raw SQL) so SQLite-returned datetime values come back typed
        — P2-7 review fix.
        """
        cutoff = _now() - timedelta(seconds=window_seconds)
        async with self._session_factory() as session:
            result = await session.execute(
                select(JobRecord.owner_id, func.max(JobItemRecord.claimed_at).label("last_claim"))
                .join(JobItemRecord, JobItemRecord.job_id == JobRecord.job_id)
                .where(JobItemRecord.claimed_at.is_not(None))
                .where(JobItemRecord.claimed_at > cutoff)
                .group_by(JobRecord.owner_id)
            )
            # GROUP BY owner_id collapses duplicates; dict-comp key collision is impossible here.
            return {row[0]: row[1] for row in result.all()}

    async def append_event(
        self,
        job_id: str,
        event: str,
        message: str,
        *,
        item_id: str | None = None,
        level: str = "info",
        payload: dict | None = None,
    ) -> None:
        async with self._session_factory() as session:
            session.add(JobEventRecord(
                job_id=job_id,
                item_id=item_id,
                ts=_now(),
                level=level,
                event=event,
                message=message,
                payload_json=json.dumps(payload or {}, ensure_ascii=False),
            ))
            await session.commit()

    async def list_events(self, job_id: str, after_id: int = 0) -> list[IngestionJobEvent]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(JobEventRecord)
                .where(JobEventRecord.job_id == job_id)
                .where(JobEventRecord.event_id > after_id)
                .order_by(JobEventRecord.event_id)
            )
            return [_event_from_record(row) for row in result.scalars().all()]

    async def _update_job_counts(self, session: AsyncSession, job_id: str) -> None:
        job = await session.get(JobRecord, job_id)
        if job is None:
            return
        result = await session.execute(
            select(JobItemRecord).where(JobItemRecord.job_id == job_id)
        )
        items = list(result.scalars().all())
        job.succeeded_items = sum(1 for item in items if item.status == ITEM_SUCCEEDED)
        job.skipped_items = sum(1 for item in items if item.status == ITEM_SKIPPED)
        job.failed_items = sum(1 for item in items if item.status == ITEM_FAILED)
        cancelled_items = sum(1 for item in items if item.status == ITEM_CANCELLED)
        terminal = {ITEM_SUCCEEDED, ITEM_SKIPPED, ITEM_FAILED, ITEM_CANCELLED}
        if all(item.status in terminal for item in items):
            if job.failed_items and (job.succeeded_items or job.skipped_items):
                job.status = JOB_PARTIALLY_SUCCEEDED
            elif cancelled_items and not (
                job.succeeded_items or job.skipped_items or job.failed_items
            ):
                job.status = JOB_CANCELLED
            elif cancelled_items:
                job.status = JOB_PARTIALLY_SUCCEEDED
            elif job.failed_items:
                job.status = JOB_FAILED
            else:
                job.status = JOB_SUCCEEDED


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _job_from_record(rec: JobRecord) -> IngestionJob:
    return IngestionJob(
        job_id=rec.job_id,
        status=rec.status,
        total_items=rec.total_items,
        succeeded_items=rec.succeeded_items,
        failed_items=rec.failed_items,
        skipped_items=rec.skipped_items,
        cancel_requested=rec.cancel_requested,
        config=json.loads(rec.config_json or "{}"),
        owner_id=rec.owner_id,
    )


def _item_from_record(rec: JobItemRecord) -> IngestionJobItem:
    return IngestionJobItem(
        item_id=rec.item_id,
        job_id=rec.job_id,
        path=rec.path,
        status=rec.status,
        attempt=rec.attempt,
        max_attempts=rec.max_attempts,
        claimed_at=rec.claimed_at,        # T4: surface the new field
        locked_by=rec.locked_by,
        locked_until=rec.locked_until,
    )


def _event_from_record(rec: JobEventRecord) -> IngestionJobEvent:
    return IngestionJobEvent(
        event_id=rec.event_id,
        job_id=rec.job_id,
        item_id=rec.item_id,
        ts=rec.ts,
        level=rec.level,
        event=rec.event,
        message=rec.message,
        payload=json.loads(rec.payload_json or "{}"),
    )
