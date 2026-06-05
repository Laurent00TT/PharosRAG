# src/kb/feedback/db.py
import json
import logging
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class UsageFeedbackRecord(Base):
    __tablename__ = "usage_feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(Text, unique=True, index=True)
    query: Mapped[str] = mapped_column(Text)
    retrieved: Mapped[str] = mapped_column(Text)        # JSON: list of [doc_id, page_num]
    actually_used: Mapped[str] = mapped_column(Text)    # JSON: list of [doc_id, page_num]
    was_helpful: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str] = mapped_column(Text, default="", server_default="")
    expected_doc: Mapped[str] = mapped_column(Text, default="", server_default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class UsageFeedbackDB:
    def __init__(self, db_url: str) -> None:
        self._engine = create_async_engine(db_url, poolclass=NullPool)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await self._migrate_existing_table(conn)

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def close(self) -> None:
        await self.aclose()

    async def _migrate_existing_table(self, conn) -> None:
        """Add feedback-loop columns for SQLite DBs created by older versions."""
        result = await conn.execute(text("PRAGMA table_info(usage_feedback)"))
        columns = {row[1] for row in result.fetchall()}
        migrations = {
            "query_id": "ALTER TABLE usage_feedback ADD COLUMN query_id TEXT",
            "was_helpful": "ALTER TABLE usage_feedback ADD COLUMN was_helpful INTEGER",
            "comment": "ALTER TABLE usage_feedback ADD COLUMN comment TEXT NOT NULL DEFAULT ''",
            "expected_doc": "ALTER TABLE usage_feedback ADD COLUMN expected_doc TEXT NOT NULL DEFAULT ''",
        }
        for column, sql in migrations.items():
            if column not in columns:
                await conn.execute(text(sql))
        await conn.execute(text(
            "UPDATE usage_feedback "
            "SET query_id = 'legacy-' || id "
            "WHERE query_id IS NULL OR query_id = ''"
        ))
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_usage_feedback_query_id "
            "ON usage_feedback (query_id)"
        ))

    async def record(
        self,
        query: str,
        retrieved: list[tuple[str, int]],
        actually_used: list[tuple[str, int]],
    ) -> str:
        query_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(UsageFeedbackRecord(
                query_id=query_id,
                query=query,
                retrieved=json.dumps([[d, p] for d, p in retrieved]),
                actually_used=json.dumps([[d, p] for d, p in actually_used]),
            ))
            await session.commit()
        return query_id

    async def update_feedback(
        self,
        query_id: str,
        was_helpful: bool,
        comment: str = "",
        expected_doc: str = "",
    ) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                select(UsageFeedbackRecord).where(
                    UsageFeedbackRecord.query_id == query_id
                )
            )
            record = result.scalar_one_or_none()
            if record is None:
                return False
            record.was_helpful = 1 if was_helpful else 0
            record.comment = comment
            record.expected_doc = expected_doc
            await session.commit()
            return True

    async def list_recent(self, limit: int = 100) -> list[dict]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(UsageFeedbackRecord)
                .order_by(
                    UsageFeedbackRecord.timestamp.desc(),
                    UsageFeedbackRecord.id.desc(),
                )
                .limit(limit)
            )
            return [self._to_dict(r) for r in result.scalars().all()]

    async def list_recent_since(
        self,
        since: datetime,
        limit: int = 10000,
    ) -> list[dict]:
        """SQL-level filter — avoids pulling N rows into Python only to discard most."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(UsageFeedbackRecord)
                .where(UsageFeedbackRecord.timestamp > since)
                .order_by(
                    UsageFeedbackRecord.timestamp.desc(),
                    UsageFeedbackRecord.id.desc(),
                )
                .limit(limit)
            )
            return [self._to_dict(r) for r in result.scalars().all()]

    @staticmethod
    def _to_dict(r: UsageFeedbackRecord) -> dict:
        return {
            "query_id": r.query_id,
            "query": r.query,
            "retrieved": json.loads(r.retrieved),
            "actually_used": json.loads(r.actually_used),
            "was_helpful": (
                bool(r.was_helpful) if r.was_helpful is not None else None
            ),
            "comment": r.comment or "",
            "expected_doc": r.expected_doc or "",
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        }
