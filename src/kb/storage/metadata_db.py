# src/kb/storage/metadata_db.py
import logging
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import Index, Integer, String, Date, DateTime, Text, select
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

DOC_STATUS_STAGING = "staging"
DOC_STATUS_ACTIVE = "active"
DOC_STATUS_DEPRECATED = "deprecated"
DOC_STATUS_FAILED = "failed"
DOC_STATUS_DELETED = "deleted"

OWNER_ID_SYSTEM = "_system"


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"
    doc_id: Mapped[str]               = mapped_column(String, primary_key=True)
    doc_name: Mapped[str]             = mapped_column(String)
    version: Mapped[Optional[str]]    = mapped_column(String, nullable=True)
    effective_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[Optional[date]]    = mapped_column(Date, nullable=True)
    supersedes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str]               = mapped_column(String, default=DOC_STATUS_ACTIVE)
    ingested_at: Mapped[datetime]     = mapped_column(DateTime, server_default=func.now())
    owner_id: Mapped[Optional[str]]   = mapped_column(String, nullable=True)  # T1a; T1b fills it
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # T5
    # Original-document store: sha256 + size of the retained source PDF.
    # NULL = no source kept (older docs, or STORE_SOURCE_DOCUMENTS=false).
    source_sha256: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source_bytes: Mapped[Optional[int]]  = mapped_column(Integer, nullable=True)


class AuditLogRecord(Base):
    """Spec §4.2 — audit trail. T2 wired writes; T3 adds composite
    indexes for GET /audit's two hot queries: by user (members own
    rows) and by action (admin filtering)."""
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("idx_audit_user_ts", "user_id", "ts"),
        Index("idx_audit_action_ts", "action", "ts"),
    )
    event_id:     Mapped[int]               = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:      Mapped[Optional[str]]     = mapped_column(String, nullable=True)  # NULL = system event
    action:       Mapped[str]               = mapped_column(String, nullable=False, index=True)
    target_kind:  Mapped[Optional[str]]     = mapped_column(String, nullable=True)
    target_id:    Mapped[Optional[str]]     = mapped_column(String, nullable=True)
    ts:           Mapped[datetime]          = mapped_column(DateTime, server_default=func.now(), index=True)
    payload_json: Mapped[str]               = mapped_column(Text, default="{}")


class MetadataDB:
    def __init__(self, db_url: str = "sqlite+aiosqlite:///./kb_metadata.db") -> None:
        self._engine = create_async_engine(db_url, poolclass=NullPool)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # T1a idempotent ALTER: add owner_id to existing documents.
            # SQLAlchemy create_all skips existing tables, so a pre-T1 DB
            # would keep its old documents schema without this. We use a
            # pragma probe instead of try/except so the migration is loud
            # if SQLite reshapes column metadata differently in the future.
            result = await conn.exec_driver_sql("PRAGMA table_info(documents)")
            existing_cols = {row[1] for row in result.fetchall()}
            if "owner_id" not in existing_cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN owner_id TEXT"
                )
            # T5: deleted_at idempotent ALTER (new column for soft-delete GC truth)
            if "deleted_at" not in existing_cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN deleted_at TIMESTAMP"
                )
            # Original-document store: sha256 + size of the retained source PDF.
            if "source_sha256" not in existing_cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN source_sha256 TEXT"
                )
            if "source_bytes" not in existing_cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN source_bytes INTEGER"
                )

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def close(self) -> None:
        await self.aclose()

    async def upsert_document(
        self,
        doc_id: str,
        doc_name: str,
        *,
        version: Optional[str],
        effective_date: Optional[date],
        expiry_date: Optional[date],
        supersedes: Optional[str],
        status: str = DOC_STATUS_ACTIVE,
        owner_id: str | None = None,
        source_sha256: str | None = None,
        source_bytes: int | None = None,
    ) -> None:
        async with self._session_factory() as session:
            existing = await session.get(DocumentRecord, doc_id)
            if existing:
                existing.doc_name = doc_name
                existing.version = version
                existing.effective_date = effective_date
                existing.expiry_date = expiry_date
                existing.supersedes = supersedes
                existing.status = status
                # Re-ingest path: keep existing owner_id unless caller passed
                # a non-None value (T3 ingest path always passes the current
                # owner; T1a backfill rows keep their None / _system).
                # This matters so admin-triggered re-ingest does not silently
                # transfer ownership from the original member.
                if owner_id is not None:
                    existing.owner_id = owner_id
                # Same rule for the source blob: only overwrite when a re-ingest
                # actually stored a new source; otherwise keep what's there.
                if source_sha256 is not None:
                    existing.source_sha256 = source_sha256
                    existing.source_bytes = source_bytes
            else:
                session.add(DocumentRecord(
                    doc_id=doc_id, doc_name=doc_name,
                    version=version, effective_date=effective_date,
                    expiry_date=expiry_date, supersedes=supersedes,
                    status=status, owner_id=owner_id,
                    source_sha256=source_sha256, source_bytes=source_bytes,
                ))
            await session.commit()

    @staticmethod
    def _record_to_dict(
        r: DocumentRecord,
        *,
        include_ingested_at: bool = False,
        include_deleted_at: bool = False,
    ) -> dict:
        """Shared row-to-dict converter — keep get_document() and
        list_documents() in lock-step when DocumentRecord gains fields."""
        d = {
            "doc_id": r.doc_id,
            "doc_name": r.doc_name,
            "version": r.version,
            "status": r.status,
            "effective_date": r.effective_date,
            "expiry_date": r.expiry_date,
            "owner_id": r.owner_id,
            "has_source": r.source_sha256 is not None,
            "source_sha256": r.source_sha256,
            "source_bytes": r.source_bytes,
        }
        if include_ingested_at:
            d["ingested_at"] = r.ingested_at
        if include_deleted_at:
            d["deleted_at"] = r.deleted_at
        return d

    async def get_document(
        self,
        doc_id: str,
        *,
        include_deleted_at: bool = False,
        include_ingested_at: bool = False,
    ) -> Optional[dict]:
        async with self._session_factory() as session:
            rec = await session.get(DocumentRecord, doc_id)
            if not rec:
                return None
            return self._record_to_dict(
                rec,
                include_ingested_at=include_ingested_at,
                include_deleted_at=include_deleted_at,
            )  # T3: owner_id required by require_owner_or_admin

    async def deprecate_document(self, doc_id: str) -> None:
        async with self._session_factory() as session:
            rec = await session.get(DocumentRecord, doc_id)
            if rec:
                rec.status = DOC_STATUS_DEPRECATED
                await session.commit()

    async def activate_document(self, doc_id: str) -> None:
        async with self._session_factory() as session:
            rec = await session.get(DocumentRecord, doc_id)
            if rec:
                rec.status = DOC_STATUS_ACTIVE
                await session.commit()

    async def mark_deleted(self, doc_id: str) -> None:
        async with self._session_factory() as session:
            rec = await session.get(DocumentRecord, doc_id)
            if rec:
                rec.status = DOC_STATUS_DELETED
                rec.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)  # T5: GC source of truth
                await session.commit()

    async def restore_document(self, doc_id: str) -> None:
        """T5: bring a soft-deleted doc back. Sets status='active' AND
        clears deleted_at so GC won't pick it up next cycle.

        Storage layer only — does NOT emit audit. Callers (T5-6
        POST /documents/{id}/restore handler) MUST write the audit row
        themselves via audit.write("doc.restore", ...). Mirrors the
        mark_deleted contract.
        """
        async with self._session_factory() as session:
            rec = await session.get(DocumentRecord, doc_id)
            if rec:
                rec.status = DOC_STATUS_ACTIVE
                rec.deleted_at = None
                await session.commit()

    async def mark_deleted_document(self, doc_id: str) -> None:
        await self.mark_deleted(doc_id)

    async def mark_failed_document(self, doc_id: str) -> None:
        async with self._session_factory() as session:
            rec = await session.get(DocumentRecord, doc_id)
            if rec:
                rec.status = DOC_STATUS_FAILED
                await session.commit()

    async def list_active_doc_ids(self) -> list[str]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentRecord.doc_id).where(DocumentRecord.status == DOC_STATUS_ACTIVE)
            )
            return [row[0] for row in result.all()]

    async def list_valid_doc_ids(self, as_of: date | None = None) -> list[str]:
        """Active documents that have not yet expired as of the given date."""
        today = as_of or date.today()
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentRecord.doc_id).where(
                    DocumentRecord.status == DOC_STATUS_ACTIVE
                ).where(
                    (DocumentRecord.expiry_date == None) |  # noqa: E711
                    (DocumentRecord.expiry_date > today)
                )
            )
            return [row[0] for row in result.all()]

    async def list_documents(
        self,
        *,
        status: str | None = None,
        owner_filter: str | None = None,
    ) -> list[dict]:
        """Generic document lister for the doctor team view.

        Filters:
          - status: exact match on documents.status (e.g. 'deleted',
            'deprecated', 'active'). None = no status filter.
            Unrecognized values silently return an empty list (no error).
          - owner_filter: 'null_or_system' returns rows whose owner_id
            is NULL or '_system' (the I-11 orphan bucket). 'non_null'
            returns the complement. None = no owner filter.

        Returned dicts include ``ingested_at`` and ``deleted_at`` — the
        latter is the T5 source of truth for GC retention (set by
        mark_deleted(), cleared by restore_document()). Pre-T5 rows
        whose deleted_at is NULL are looked up against audit_log by
        doctor's trash inventory as a best-effort fallback.
        """
        async with self._session_factory() as session:
            stmt = select(DocumentRecord)
            if status is not None:
                stmt = stmt.where(DocumentRecord.status == status)
            if owner_filter == "null_or_system":
                stmt = stmt.where(
                    (DocumentRecord.owner_id == None)  # noqa: E711
                    | (DocumentRecord.owner_id == OWNER_ID_SYSTEM)
                )
            elif owner_filter == "non_null":
                stmt = stmt.where(
                    DocumentRecord.owner_id != None  # noqa: E711
                ).where(
                    DocumentRecord.owner_id != OWNER_ID_SYSTEM
                )
            elif owner_filter is not None:
                raise ValueError(f"unknown owner_filter: {owner_filter!r}")
            rows = (await session.execute(stmt)).scalars().all()
            return [
                self._record_to_dict(
                    r, include_ingested_at=True, include_deleted_at=True,
                )
                for r in rows
            ]
