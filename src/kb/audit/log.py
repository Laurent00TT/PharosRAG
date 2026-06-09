"""AuditLog — three write semantics + JSONL fallback for two-phase completed.

Spec §6.3 / I-4 / T2 plan invariants A-1, A-2, A-3:

- **mandatory single-phase** (user.create / user.disable / user.role_change /
  user.key_reset): caller uses ``make_record(...)`` + adds to its own ORM
  session. One commit, atomic with the user write. NO JSONL fallback —
  if audit can't be written, the user op shouldn't succeed (spec
  "操作即原子").

- **mandatory two-phase** (doc.delete / doc.reassign): caller invokes
  ``write_attempted`` (best-effort, before main op) then ``write_completed``
  (mandatory, after main op). ``write_completed`` falls back to fsync'd
  JSONL on DB failure; both failing raises ``AuditWriteFailure`` (caller
  should map to HTTP 503 — main op happened, audit lost).

- **best-effort** (everything else: doc.ingest, job.cancel, job.failed,
  authz.denied, auth.brute_force): caller invokes ``write(action, ...)``.
  DB failure emits error trace and swallows; main op continues.

``write(action, ...)`` rejects MANDATORY and TWO_PHASE action names
explicitly (assertion error) so a silent-drop bug is impossible.

``init()`` self-contained creates the audit_log table via
``AuditLogRecord.__table__.create(checkfirst=True)`` — does not depend on
``MetadataDB.init()``. CLI / scripts can construct an AuditLog and start
writing immediately.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from kb.observability.trace import emit, emit_error
from kb.storage.metadata_db import AuditLogRecord

logger = logging.getLogger(__name__)


MANDATORY_ACTIONS: frozenset[str] = frozenset({
    "user.create", "user.disable", "user.role_change", "user.key_reset",
})

TWO_PHASE_ACTIONS: frozenset[str] = frozenset({"doc.delete", "doc.reassign"})


class AuditWriteFailure(Exception):
    """Two-phase completed write failed on both DB and JSONL fallback.

    Caller (destructive endpoint) should map to HTTP 503 — main op already
    happened on the data plane but audit row is missing. Admin must
    reconstruct from kb_trace.jsonl (look for ``audit.completed_lost``).
    """


class AuditContractError(ValueError):
    """Caller misused the AuditLog API (wrong action for the chosen method).

    Raised instead of ``assert`` so the contract holds even under
    ``python -O`` (which strips assertions). Caught early — at the
    method entry — before any DB write happens, so a misuse cannot
    accidentally end up in the wrong write semantic (e.g. a TWO_PHASE
    action sneaking through best-effort ``write()``).
    """


def _now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _serialize_payload(payload: dict | None) -> str:
    return json.dumps(payload or {}, ensure_ascii=False, default=str)


class AuditLog:
    def __init__(self, db_url: str, fallback_path: str | Path) -> None:
        self._engine = create_async_engine(db_url, poolclass=NullPool)
        # WAL so concurrent writers (server + worker + CLI on the same
        # kb_metadata.db file) don't serialize behind the global file lock.
        # Mirrors UsersStore.__init__ — important because AuditLog.init()
        # is standalone (P0-2), so this engine can be the FIRST opener of
        # the DB file via the CLI bootstrap path.
        @event.listens_for(self._engine.sync_engine, "connect")
        def _enable_wal(dbapi_conn, _record):  # pragma: no cover - trivial
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self.fallback_path = Path(fallback_path)

    async def init(self) -> None:
        """Self-contained: ensure audit_log table + composite indexes exist
        (P0-2 + P2-6)."""
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._engine.begin() as conn:
            await conn.run_sync(
                lambda sc: AuditLogRecord.__table__.create(sc, checkfirst=True)
            )
            # Idempotent index migration — checkfirst on table creation does
            # NOT re-check indexes when the table already exists, so we
            # explicitly probe + create here. CREATE INDEX IF NOT EXISTS is
            # SQLite's safe form.
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_audit_user_ts "
                "ON audit_log(user_id, ts)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_audit_action_ts "
                "ON audit_log(action, ts)"
            )

    async def aclose(self) -> None:
        await self._engine.dispose()

    # ── A-1: single-phase mandatory factory ──────────────────────────

    def make_record(
        self,
        action: str,
        *,
        user_id: str | None,
        target_kind: str,
        target_id: str | None,
        payload: dict | None = None,
    ) -> AuditLogRecord:
        """Return an unsaved AuditLogRecord. Caller adds it to its own
        SQLAlchemy session and commits together with the main row.

        Restricted to MANDATORY_ACTIONS (user.*) so callers cannot silently
        opt out of the same-session atomicity contract.
        """
        if action not in MANDATORY_ACTIONS:
            raise AuditContractError(
                f"{action!r} is not single-phase mandatory; use "
                f"write_attempted/write_completed (two-phase) or write() (best-effort)"
            )
        return AuditLogRecord(
            user_id=user_id,
            action=action,
            target_kind=target_kind,
            target_id=target_id,
            ts=_now_naive(),
            payload_json=_serialize_payload(payload),
        )

    # ── A-2: two-phase ───────────────────────────────────────────────

    async def write_attempted(
        self,
        action: str,
        *,
        user_id: str | None,
        target_kind: str,
        target_id: str | None,
        payload: dict | None = None,
    ) -> None:
        if action not in TWO_PHASE_ACTIONS:
            raise AuditContractError(f"{action!r} is not two-phase")
        # attempted phase is best-effort by spec.
        try:
            await self._db_insert_best_effort(
                f"{action}.attempted", user_id, target_kind, target_id, payload,
            )
        except Exception as exc:
            emit_error(
                "audit.best_effort_failed", exc,
                action=f"{action}.attempted", target_id=target_id, user_id=user_id,
            )

    async def write_completed(
        self,
        action: str,
        *,
        user_id: str | None,
        target_kind: str,
        target_id: str | None,
        payload: dict | None = None,
    ) -> None:
        """Mandatory: DB -> JSONL fallback -> AuditWriteFailure."""
        if action not in TWO_PHASE_ACTIONS:
            raise AuditContractError(f"{action!r} is not two-phase")
        full_action = f"{action}.completed"
        try:
            await self._db_insert_for_mandatory(
                full_action, user_id, target_kind, target_id, payload,
            )
            return
        except Exception as exc:
            emit_error(
                "audit.mandatory_db_failed", exc,
                action=full_action, target_id=target_id, user_id=user_id,
            )
        try:
            self._append_fallback_jsonl_synced(
                full_action, user_id, target_kind, target_id, payload,
            )
            emit(
                "audit.mandatory_via_fallback",
                action=full_action, target_id=target_id, user_id=user_id,
            )
            return
        except Exception as exc:
            emit_error(
                "audit.mandatory_fallback_failed", exc,
                action=full_action, target_id=target_id, user_id=user_id,
            )
            raise AuditWriteFailure(
                f"audit completed write totally failed for action={full_action!r}"
            ) from exc

    # ── A-3: best-effort write ───────────────────────────────────────

    async def write(
        self,
        action: str,
        *,
        user_id: str | None,
        target_kind: str,
        target_id: str | None,
        payload: dict | None = None,
    ) -> None:
        """Best-effort only. Rejects mandatory + two-phase actions explicitly.

        Also rejects derived two-phase suffixes (``doc.delete.attempted``,
        ``doc.delete.completed``, etc.). If a caller wrote one of those
        through ``write()`` it would bypass the mandatory fallback path
        and silently lose audit on DB failure — caught at call time so
        the failure mode is impossible at runtime.
        """
        if action in TWO_PHASE_ACTIONS:
            raise AuditContractError(
                f"{action!r} is two-phase — call write_attempted/write_completed"
            )
        if action in MANDATORY_ACTIONS:
            raise AuditContractError(
                f"{action!r} is single-phase mandatory — caller must use "
                f"make_record() + its own session.add() + commit"
            )
        # Derived two-phase suffixes: doc.delete.attempted, doc.delete.completed,
        # doc.reassign.attempted, doc.reassign.completed. These are produced
        # ONLY by write_attempted/write_completed and must never be written
        # directly through write() — that would skip the mandatory fallback.
        for two_phase_base in TWO_PHASE_ACTIONS:
            if action == f"{two_phase_base}.attempted" or action == f"{two_phase_base}.completed":
                raise AuditContractError(
                    f"{action!r} is a derived two-phase action — "
                    f"call write_attempted/write_completed on {two_phase_base!r} instead"
                )
        try:
            await self._db_insert_best_effort(
                action, user_id, target_kind, target_id, payload,
            )
        except Exception as exc:
            emit_error(
                "audit.best_effort_failed", exc,
                action=action, target_id=target_id, user_id=user_id,
            )

    # ── Query (for GET /audit) ──────────────────────────────────────

    async def query(
        self,
        *,
        user_id: str | None = None,
        action: str | None = None,
        actions: list[str] | None = None,
        target_id: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Read audit rows with optional filters, newest-first, bounded.

        ``action`` and ``actions`` are mutually exclusive — use ``actions``
        when the caller wants multiple action types in one query (doctor
        team's destructive_7d view collapses 8 serial queries this way).
        Passing both raises ValueError so the caller surfaces the bug
        instead of silently AND'ing them together.
        """
        if action is not None and actions is not None:
            raise ValueError("query() accepts only one of action / actions, not both")
        async with self._session_factory() as session:
            stmt = select(AuditLogRecord).order_by(AuditLogRecord.ts.desc())
            if user_id is not None:
                stmt = stmt.where(AuditLogRecord.user_id == user_id)
            if action is not None:
                stmt = stmt.where(AuditLogRecord.action == action)
            if actions is not None:
                # Empty list → match nothing (avoids `IN ()` SQL error
                # AND respects "caller passed an empty filter on purpose").
                if not actions:
                    return []
                stmt = stmt.where(AuditLogRecord.action.in_(actions))
            if target_id is not None:
                stmt = stmt.where(AuditLogRecord.target_id == target_id)
            if since is not None:
                stmt = stmt.where(AuditLogRecord.ts >= since)
            stmt = stmt.limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "event_id": r.event_id,
                    "user_id": r.user_id,
                    "action": r.action,
                    "target_kind": r.target_kind,
                    "target_id": r.target_id,
                    "ts": r.ts.isoformat() if r.ts else None,
                    "payload_json": r.payload_json,
                }
                for r in rows
            ]

    async def count_by_action_user(
        self,
        *,
        actions: list[str],
        since: datetime | None = None,
    ) -> dict[str | None, dict[str, int]]:
        """DB-side GROUP BY (user_id, action) for doctor team view.

        Returns ``{user_id: {action: count, ...}, ...}``. User IDs not
        in the result have no rows in window; action keys not present
        for a user have a count of 0 (caller should `.get(..., 0)`).

        P1-6 review fix: doctor's pull-rows-then-`len` is inaccurate
        when row count exceeds the query's ``limit``. Composite index
        ``idx_audit_user_ts`` (T3-8) covers this query efficiently.
        """
        if not actions:
            return {}
        async with self._session_factory() as session:
            stmt = (
                select(
                    AuditLogRecord.user_id,
                    AuditLogRecord.action,
                    func.count(AuditLogRecord.event_id),
                )
                .where(AuditLogRecord.action.in_(actions))
                .group_by(AuditLogRecord.user_id, AuditLogRecord.action)
            )
            if since is not None:
                stmt = stmt.where(AuditLogRecord.ts >= since)
            rows = (await session.execute(stmt)).all()
        out: dict[str | None, dict[str, int]] = {}
        for user_id, action, count in rows:
            out.setdefault(user_id, {})[action] = count
        return out

    # ── Internal write helpers ──────────────────────────────────────

    async def _db_insert_best_effort(
        self, action, user_id, target_kind, target_id, payload,
    ):
        async with self._session_factory() as session:
            row = AuditLogRecord(
                user_id=user_id,
                action=action,
                target_kind=target_kind,
                target_id=target_id,
                ts=_now_naive(),
                payload_json=_serialize_payload(payload),
            )
            session.add(row)
            await session.commit()

    async def _db_insert_for_mandatory(
        self, action, user_id, target_kind, target_id, payload,
    ):
        """Same as best-effort but kept as a separate method so monkeypatch
        in tests can target exactly one path."""
        async with self._session_factory() as session:
            row = AuditLogRecord(
                user_id=user_id,
                action=action,
                target_kind=target_kind,
                target_id=target_id,
                ts=_now_naive(),
                payload_json=_serialize_payload(payload),
            )
            session.add(row)
            await session.commit()

    def _append_fallback_jsonl_synced(
        self, action, user_id, target_kind, target_id, payload,
    ):
        record = {
            "ts": _now_naive().isoformat(),
            "action": action,
            "user_id": user_id,
            "target_kind": target_kind,
            "target_id": target_id,
            "payload": payload or {},
            "via_fallback": True,
        }
        with open(self.fallback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
