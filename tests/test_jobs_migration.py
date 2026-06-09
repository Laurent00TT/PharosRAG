"""T1a — SQLiteJobStore migration tests.

Anti-regression for spec P1-2: jobs schema MUST live in ingestion_jobs.db,
NOT in kb_metadata.db. Verifies both new columns + backfill sentinel.
"""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from kb.jobs.sqlite_store import SQLiteJobStore


async def test_jobs_init_adds_owner_id_column(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/jobs.db"
    store = SQLiteJobStore(db_url=db_url)
    await store.init()
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("ingestion_jobs")}
        )
    assert "owner_id" in cols
    await engine.dispose()
    await store.close()


async def test_items_init_adds_claimed_at_column(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/jobs.db"
    store = SQLiteJobStore(db_url=db_url)
    await store.init()
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in inspect(sync_conn).get_columns("ingestion_job_items")}
        )
    assert "claimed_at" in cols
    await engine.dispose()
    await store.close()


async def test_init_backfills_existing_jobs_with_system_sentinel(tmp_path):
    """Pre-T1 jobs.db has rows but no owner_id. After migration, those
    rows must be filled with '_system' sentinel — code-level invariant
    (spec I-9 T1a phase). Schema-level NOT NULL would break the upgrade."""
    db_path = tmp_path / "jobs.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    # Pre-T1-shaped DB: ingestion_jobs without owner_id
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                total_items INTEGER DEFAULT 0,
                succeeded_items INTEGER DEFAULT 0,
                failed_items INTEGER DEFAULT 0,
                skipped_items INTEGER DEFAULT 0,
                cancel_requested INTEGER DEFAULT 0,
                config_json TEXT DEFAULT '{}'
            )
        """))
        await conn.execute(text(
            "INSERT INTO ingestion_jobs (job_id, status) VALUES ('legacy-job', 'queued')"
        ))
    await engine.dispose()

    # Run T1a migration
    store = SQLiteJobStore(db_url=db_url)
    await store.init()

    # Verify backfill
    engine2 = create_async_engine(db_url)
    async with engine2.begin() as conn:
        rows = (await conn.execute(text("SELECT job_id, owner_id FROM ingestion_jobs"))).fetchall()
    await engine2.dispose()
    assert ("legacy-job", "_system") in [(r[0], r[1]) for r in rows]
    await store.close()


async def test_init_is_idempotent(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/jobs.db"
    store = SQLiteJobStore(db_url=db_url)
    await store.init()
    await store.init()  # must not raise
    await store.close()


async def test_new_job_owner_id_default_is_null(tmp_path):
    """T1a phase: code-level write path still NULL because T1b not yet
    integrated. Spec I-9 explicitly allows NULL at schema level here."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/jobs.db"
    store = SQLiteJobStore(db_url=db_url)
    await store.init()
    job = await store.create_job(paths=["a.pdf"], config={})

    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        rows = (await conn.execute(text(
            "SELECT owner_id FROM ingestion_jobs WHERE job_id = :j"
        ), {"j": job.job_id})).fetchall()
    await engine.dispose()
    assert rows[0][0] is None  # NULL until T1b
    await store.close()
