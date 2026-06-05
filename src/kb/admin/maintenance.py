"""SQLite-backed maintenance flag — visible across processes.

Why SQLite and not asyncio.Event:
  The KB runs in two processes (tool_server + worker). An in-process
  asyncio.Event set via the admin endpoint would only affect the
  tool_server's claim_next_item path; the worker is a separate process
  and would happily keep claiming items. SQLite is the only shared
  state between the two processes today (both write to ingestion_jobs.db
  + kb_metadata.db), so we put the flag in a dedicated single-row table.

Cost: one indexed PK lookup per write request (~0.1ms on local SQLite).
At small-team write rate (< 100/min) this is invisible.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy import DateTime, Integer, String, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class MaintenanceRecord(_Base):
    """Single-row table — id is always 1.

    Stored in kb_metadata.db so both tool_server and worker (both already
    open kb_metadata.db for audit + users) can read without opening yet
    another file. Schema lives in this module rather than metadata_db.py
    so the admin concern stays self-contained and is easy to delete /
    rename if we ever swap the storage backend.
    """
    __tablename__ = "maintenance_state"
    id:              Mapped[int]               = mapped_column(Integer, primary_key=True)
    on:              Mapped[int]               = mapped_column(Integer, default=0)
    set_at:          Mapped[datetime | None]   = mapped_column(DateTime, nullable=True)
    set_by_user_id:  Mapped[str | None]        = mapped_column(String, nullable=True)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MaintenanceState:
    """SQLite-backed cross-process flag.

    Lifecycle:
      MaintenanceState(db_url) → await .init() → use → await .aclose()

    Server constructs once in lifespan, attaches to app.state.maintenance.
    Worker constructs separately in scripts/worker.py, passes
    state.is_on as the maintenance_check callback to SQLiteJobStore.
    """

    def __init__(self, db_url: str) -> None:
        self._engine = create_async_engine(db_url, poolclass=NullPool)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )

    async def init(self) -> None:
        """Idempotent: creates the table + seeds the single row if absent.

        Same pattern as MetadataDB.init() / AuditLog.init() — safe to
        call from multiple processes; whoever wins the race fills row 1,
        the loser's INSERT-OR-IGNORE no-ops.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
            await conn.exec_driver_sql(
                "INSERT OR IGNORE INTO maintenance_state (id, \"on\") VALUES (1, 0)"
            )

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def is_on(self) -> bool:
        """Hot path — every write hits this. Single indexed PK lookup."""
        async with self._session_factory() as session:
            rec = await session.get(MaintenanceRecord, 1)
            return bool(rec and rec.on)

    async def set_on(self, *, user_id: str) -> None:
        """Idempotent: writing `on=1` over `on=1` is fine; second writer
        wins for the set_by_user_id / set_at record."""
        await self._write(on=True, user_id=user_id)

    async def set_off(self, *, user_id: str) -> None:
        await self._write(on=False, user_id=user_id)

    async def _write(self, *, on: bool, user_id: str) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                update(MaintenanceRecord)
                .where(MaintenanceRecord.id == 1)
                .values(on=1 if on else 0, set_at=_now(), set_by_user_id=user_id)
            )
            if result.rowcount == 0:
                # init() seeds row id=1; an UPDATE finding nothing means
                # the caller skipped init(). Fail loud so the bug is
                # immediately visible instead of silently swallowing the flag.
                raise RuntimeError(
                    "maintenance_state row id=1 missing — call MaintenanceState.init() first"
                )
            await session.commit()

    async def get_state(self) -> dict[str, Any]:
        async with self._session_factory() as session:
            rec = await session.get(MaintenanceRecord, 1)
            if rec is None:
                return {"on": False, "set_at": None, "set_by_user_id": None}
            return {
                "on": bool(rec.on),
                "set_at": rec.set_at,
                "set_by_user_id": rec.set_by_user_id,
            }


# ── FastAPI integration ──────────────────────────────────────────────


async def require_no_maintenance(request: Request) -> None:
    """FastAPI dependency: 503 when maintenance is ON.

    Usage:
        @router.post("/...", dependencies=[Depends(require_no_maintenance)])

    Or at router-include time:
        app.include_router(router, dependencies=[Depends(require_no_maintenance)])

    Relies on app.state.maintenance being set by lifespan. Endpoints that
    are MEANT to work during maintenance (the admin maintenance endpoint
    itself, all GET endpoints) MUST NOT include this dependency.

    Review fix P1.1: fail-CLOSED on missing state. If lifespan didn't
    wire up app.state.maintenance, the safest behavior is to refuse the
    write — silently failing open during a misconfigured rollout would
    let writes through during a backup that the admin THOUGHT had the
    server locked down. Tests that need to bypass this guard explicitly
    inject an off-state MaintenanceState into app.state.maintenance.
    """
    state: MaintenanceState | None = getattr(request.app.state, "maintenance", None)
    if state is None:
        logger.error(
            "require_no_maintenance: app.state.maintenance not set — refusing write (fail-closed)",
        )
        raise HTTPException(
            status_code=503,
            detail="maintenance state not initialised — refusing writes",
        )
    if await state.is_on():
        logger.info(
            "require_no_maintenance: maintenance ON — blocking %s %s",
            request.method, request.url.path,
        )
        raise HTTPException(
            status_code=503,
            detail="server in maintenance mode (backup in progress)",
        )
