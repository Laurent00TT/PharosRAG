"""User identity store.

Keeps it intentionally small: T1a only needs CRUD by user_id / username /
key_hash. T1b's APIKeyEnforcer will be the sole reader on the hot path
(get_by_key_hash per request); other methods are admin-CLI only.

Storage: separate ``users`` table in kb_metadata.db (so user identity
lives in the same SQLite tx domain as documents.owner_id and audit_log
— per P1-2 of spec review 3).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool

from kb.audit.log import AuditLog
from kb.auth.key_gen import generate_key

VALID_ROLES: frozenset[str] = frozenset({"member", "admin"})


class _Base(DeclarativeBase):
    pass


class _UserRow(_Base):
    __tablename__ = "users"
    user_id:     Mapped[str]              = mapped_column(String, primary_key=True)
    username:    Mapped[str]              = mapped_column(String, unique=True, nullable=False)
    key_prefix:  Mapped[str]              = mapped_column(String, unique=True, nullable=False)
    key_hash:    Mapped[str]              = mapped_column(String, unique=True, nullable=False)
    role:        Mapped[str]              = mapped_column(String, nullable=False)
    created_at:  Mapped[datetime]         = mapped_column(DateTime, nullable=False)
    disabled_at: Mapped[datetime | None]  = mapped_column(DateTime, nullable=True)


@dataclass(frozen=True)
class User:
    user_id: str
    username: str
    key_prefix: str
    key_hash: str
    role: str           # 'member' | 'admin'
    created_at: datetime
    disabled_at: datetime | None

    @property
    def is_disabled(self) -> bool:
        return self.disabled_at is not None


def _row_to_user(row: _UserRow) -> User:
    return User(
        user_id=row.user_id, username=row.username,
        key_prefix=row.key_prefix, key_hash=row.key_hash,
        role=row.role, created_at=row.created_at, disabled_at=row.disabled_at,
    )


class UsersStore:
    # T2 audit note (carry into T3): acting_user_id defaults to '_system'
    # for CLI bootstrap use. When admin-via-API endpoints land in T3+, every
    # call site MUST pass acting_user_id=current_user.user_id explicitly —
    # the default would silently misattribute API admin actions as system
    # actions. Consider making this kwarg required (no default) when those
    # endpoints land.
    def __init__(self, db_url: str) -> None:
        self._engine = create_async_engine(db_url, poolclass=NullPool)
        # WAL so admin CLI (writes) doesn't block server reads.
        @event.listens_for(self._engine.sync_engine, "connect")
        def _enable_wal(dbapi_conn, _record):  # pragma: no cover - trivial
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def aclose(self) -> None:
        await self._engine.dispose()

    # ── Writes ────────────────────────────────────────────────────────

    async def create_user(
        self,
        *,
        username: str,
        role: str,
        audit_log: AuditLog | None = None,
        acting_user_id: str = "_system",
    ) -> tuple[str, User]:
        """Return (plaintext_key, User). Plaintext shown ONCE to admin.

        If ``audit_log`` is provided, atomically writes a ``user.create``
        audit row in the SAME session (A-1 -- operation-is-atomic). A
        unique-constraint failure rolls back both the user attempt AND
        the audit row.
        """
        if role not in VALID_ROLES:
            raise ValueError(f"role must be 'member' or 'admin', got {role!r}")
        plaintext, key_prefix, key_hash = generate_key(username)
        user_row = _UserRow(
            user_id=uuid.uuid4().hex,
            username=username,
            key_prefix=key_prefix,
            key_hash=key_hash,
            role=role,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            disabled_at=None,
        )
        async with self._session_factory() as session:
            session.add(user_row)
            if audit_log is not None:
                session.add(audit_log.make_record(
                    "user.create",
                    user_id=acting_user_id,
                    target_kind="user",
                    target_id=user_row.user_id,
                    payload={
                        "username": username,
                        "role": role,
                        "key_prefix": key_prefix,
                    },
                ))
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ValueError(f"user {username!r} already exists") from exc
            await session.refresh(user_row)
        return plaintext, _row_to_user(user_row)

    async def disable_user(
        self,
        user_id: str,
        *,
        audit_log: AuditLog | None = None,
        acting_user_id: str = "_system",
    ) -> None:
        async with self._session_factory() as session:
            row = await session.get(_UserRow, user_id)
            if row is None:
                return
            row.disabled_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if audit_log is not None:
                session.add(audit_log.make_record(
                    "user.disable",
                    user_id=acting_user_id,
                    target_kind="user",
                    target_id=user_id,
                    payload={"username": row.username},
                ))
            await session.commit()

    async def set_role(
        self,
        user_id: str,
        role: str,
        *,
        audit_log: AuditLog | None = None,
        acting_user_id: str = "_system",
    ) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be 'member' or 'admin', got {role!r}")
        async with self._session_factory() as session:
            row = await session.get(_UserRow, user_id)
            if row is None:
                raise ValueError(f"user {user_id!r} not found")
            old_role = row.role
            row.role = role
            if audit_log is not None:
                session.add(audit_log.make_record(
                    "user.role_change",
                    user_id=acting_user_id,
                    target_kind="user",
                    target_id=user_id,
                    payload={"old_role": old_role, "new_role": role},
                ))
            await session.commit()

    async def reset_key(
        self,
        user_id: str,
        *,
        audit_log: AuditLog | None = None,
        acting_user_id: str = "_system",
    ) -> str:
        """Generate new key, invalidate old. Returns plaintext (shown once)."""
        async with self._session_factory() as session:
            row = await session.get(_UserRow, user_id)
            if row is None:
                raise ValueError(f"user {user_id!r} not found")
            plaintext, key_prefix, key_hash = generate_key(row.username)
            old_prefix = row.key_prefix
            row.key_prefix = key_prefix
            row.key_hash = key_hash
            if audit_log is not None:
                session.add(audit_log.make_record(
                    "user.key_reset",
                    user_id=acting_user_id,
                    target_kind="user",
                    target_id=user_id,
                    payload={"old_key_prefix": old_prefix, "new_key_prefix": key_prefix},
                ))
            await session.commit()
        return plaintext

    # ── Reads ─────────────────────────────────────────────────────────

    async def get_by_user_id(self, user_id: str) -> User | None:
        async with self._session_factory() as session:
            row = await session.get(_UserRow, user_id)
            return _row_to_user(row) if row is not None else None

    async def get_by_username(self, username: str) -> User | None:
        async with self._session_factory() as session:
            stmt = select(_UserRow).where(_UserRow.username == username)
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _row_to_user(row) if row is not None else None

    async def get_by_key_hash(self, key_hash: str) -> User | None:
        """Hot path for T1b APIKeyEnforcer. Indexed via UNIQUE constraint."""
        async with self._session_factory() as session:
            stmt = select(_UserRow).where(_UserRow.key_hash == key_hash)
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _row_to_user(row) if row is not None else None

    async def list_users(self, *, include_disabled: bool) -> list[User]:
        async with self._session_factory() as session:
            stmt = select(_UserRow).order_by(_UserRow.created_at)
            if not include_disabled:
                stmt = stmt.where(_UserRow.disabled_at.is_(None))
            rows = (await session.execute(stmt)).scalars().all()
            return [_row_to_user(r) for r in rows]
