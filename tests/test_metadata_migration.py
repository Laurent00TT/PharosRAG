"""T1a — MetadataDB migration tests.

Verifies that adding owner_id to documents and creating audit_log are
idempotent + don't break existing 13-doc reads.
"""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from kb.storage.metadata_db import MetadataDB


async def test_metadata_db_init_creates_audit_log_table(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb.db"
    db = MetadataDB(db_url=db_url)
    await db.init()
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        tables = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
    assert "audit_log" in tables
    await engine.dispose()
    await db.aclose()


async def test_documents_table_has_owner_id_column(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb.db"
    db = MetadataDB(db_url=db_url)
    await db.init()
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("documents")}
        )
    assert "owner_id" in cols
    await engine.dispose()
    await db.aclose()


async def test_init_is_idempotent_on_existing_data(tmp_path):
    """Run init twice; preexisting docs survive + no schema error."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb.db"
    db = MetadataDB(db_url=db_url)
    await db.init()
    await db.upsert_document(
        doc_id="d1", doc_name="x.pdf",
        version=None, effective_date=None, expiry_date=None, supersedes=None,
    )

    # Second init: should not raise, doc should still be there
    await db.init()
    doc = await db.get_document("d1")
    assert doc is not None
    assert doc["doc_id"] == "d1"
    # owner_id should be NULL for pre-T1 data (I-9 + I-11)
    assert doc.get("owner_id") is None
    await db.aclose()


async def test_existing_db_without_owner_id_gets_column_added(tmp_path):
    """Simulate upgrade: pre-T1 DB has documents but no owner_id column.
    init() must ALTER TABLE to add it without losing data."""
    db_path = tmp_path / "kb.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    # Create a pre-T1-shaped DB manually (no owner_id, no audit_log)
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                doc_name TEXT,
                status TEXT DEFAULT 'active'
            )
        """))
        await conn.execute(text("INSERT INTO documents (doc_id, doc_name) VALUES ('legacy1', 'pre-t1.pdf')"))
    await engine.dispose()

    # Now run T1a migration via MetadataDB.init()
    db = MetadataDB(db_url=db_url)
    await db.init()

    engine2 = create_async_engine(db_url)
    async with engine2.begin() as conn:
        cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("documents")}
        )
        rows = (await conn.execute(text("SELECT doc_id, owner_id FROM documents"))).fetchall()
    await engine2.dispose()

    assert "owner_id" in cols
    assert ("legacy1", None) in [(r[0], r[1]) for r in rows]
    await db.aclose()
