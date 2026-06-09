"""Cross-process cache-epoch counter (security review P1 #6).

The single-process search cache in ``kb.search.cache`` is invalidated by
``invalidate_all()`` from the write path. That works for one process —
but the project runs in two: tool_server (FastAPI) and worker (the
ingest loop). When the worker finishes an ingest or marks a doc
deprecated, it clears its OWN process's cache, NOT the tool_server's.
For up to ``DEFAULT_TTL_S`` seconds afterward, MCP / REST search calls
served by tool_server return cached results that don't reflect the new
state.

Approach: a single-row SQLite table ``cache_epoch`` in ``kb_metadata.db``
(both processes already open this file). Worker calls ``bump()`` after
any write that changes search-visible state; tool_server reads the
epoch and folds it into the cache key. When the epoch bumps, every old
cache key becomes stale automatically — the next request misses cache
and recomputes against fresh data.

Cost: one indexed PK SELECT per search (~0.1ms on local SQLite). The
epoch is also memoized for the duration of a single search call — we
don't re-read it across the hyde / retrieval / rerank phases.

Persistence: the table is part of ``kb_metadata.db`` so the epoch
survives both restarts. New install starts at epoch=0; the first bump
moves it to 1. There is no upper bound; SQLite INTEGER is 64-bit
signed so it'd take 2^63 bumps to roll over (i.e. never in this
deployment's lifetime).
"""
from __future__ import annotations

from sqlalchemy import Integer, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool


class _Base(DeclarativeBase):
    pass


class CacheEpochRecord(_Base):
    """Single-row table — id is always 1.

    Stored in kb_metadata.db alongside MaintenanceState and AuditLog so
    we don't open yet another file. Schema lives in this module (not
    metadata_db.py) so the cache concern stays self-contained.
    """
    __tablename__ = "cache_epoch"
    id:    Mapped[int] = mapped_column(Integer, primary_key=True)
    epoch: Mapped[int] = mapped_column(Integer, default=0)


class CacheEpochStore:
    """SQLite-backed cross-process epoch counter.

    Lifecycle:
      CacheEpochStore(db_url) → await .init() → use → await .aclose()

    tool_server constructs once in lifespan and stores in app.state.
    Worker constructs separately; both pointed at the same kb_metadata.db.
    """

    def __init__(self, db_url: str) -> None:
        self._engine = create_async_engine(db_url, poolclass=NullPool)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )

    async def init(self) -> None:
        """Idempotent: creates the table + seeds the single row (epoch=0)
        if absent. Safe to call concurrently from both processes; the
        loser's INSERT OR IGNORE no-ops."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
            await conn.exec_driver_sql(
                "INSERT OR IGNORE INTO cache_epoch (id, epoch) VALUES (1, 0)"
            )

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def get_epoch(self) -> int:
        """Hot path — every search reads this. Single indexed PK lookup.
        Returns 0 if init() wasn't called (fail-soft so a misconfigured
        deployment still serves search; just loses cross-process
        invalidation)."""
        async with self._session_factory() as session:
            rec = await session.get(CacheEpochRecord, 1)
            return int(rec.epoch) if rec else 0

    async def bump(self) -> int:
        """Write path — call from any process after a change that
        affects search-visible state (ingest succeeded, doc deprecated
        or deleted, restore). Returns the new epoch value.

        SQLite's UPDATE ... SET epoch = epoch + 1 is atomic without
        any explicit lock; concurrent bumps from worker + tool_server
        are serialized by SQLite's writer lock and both increments
        land. The exact value is opaque to the consumer; what matters
        is that it strictly increases after every meaningful write."""
        async with self._session_factory() as session:
            await session.execute(
                update(CacheEpochRecord)
                .where(CacheEpochRecord.id == 1)
                .values(epoch=CacheEpochRecord.epoch + 1)
            )
            await session.commit()
            rec = await session.get(CacheEpochRecord, 1)
            return int(rec.epoch) if rec else 0
