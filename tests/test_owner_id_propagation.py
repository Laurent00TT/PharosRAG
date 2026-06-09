"""Owner_id propagation: dataclass exposure + pipeline writes (T3 Task 1)."""
import pytest


@pytest.mark.asyncio
async def test_ingestion_job_dataclass_exposes_owner_id():
    """IngestionJob dataclass must expose owner_id so worker can read it
    without ORM query (T2 carry-forward — required for B-2)."""
    from kb.jobs.models import IngestionJob
    job = IngestionJob(
        job_id="j-1", status="queued",
        total_items=1, succeeded_items=0, failed_items=0, skipped_items=0,
        cancel_requested=False, config={}, owner_id="u-alice",
    )
    assert job.owner_id == "u-alice"


@pytest.mark.asyncio
async def test_create_job_persists_owner_id(tmp_path):
    """SQLiteJobStore.create_job accepts owner_id kwarg and persists it."""
    from kb.jobs.sqlite_store import SQLiteJobStore
    store = SQLiteJobStore(
        db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db",
    )
    await store.init()
    try:
        job = await store.create_job(
            paths=["/fake.pdf"], config={}, owner_id="u-alice",
        )
        assert job.owner_id == "u-alice"
        # Round-trip via get_job
        fetched = await store.get_job(job.job_id)
        assert fetched.owner_id == "u-alice"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_create_job_owner_id_optional_for_backward_compat(tmp_path):
    """Existing call sites (worker.py, scripts) must keep working without
    the kwarg. owner_id defaults to None and the dataclass exposes None."""
    from kb.jobs.sqlite_store import SQLiteJobStore
    store = SQLiteJobStore(
        db_url=f"sqlite+aiosqlite:///{tmp_path}/jobs.db",
    )
    await store.init()
    try:
        job = await store.create_job(paths=["/fake.pdf"], config={})
        assert job.owner_id is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pipeline_writes_documents_owner_id(audit_setup, tmp_path, monkeypatch):
    """IngestionPipeline.ingest(..., owner_id=X) writes documents.owner_id=X.

    Skip full pipeline; this test verifies the contract by invoking
    pipeline._meta_db.upsert_document directly through a tiny shim — the
    full ingest path is covered by smoke in Task 9."""
    from kb.storage.metadata_db import MetadataDB
    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await meta.init()
    try:
        await meta.upsert_document(
            doc_id="doc-1", doc_name="manual.pdf",
            version=None, effective_date=None, expiry_date=None, supersedes=None,
            owner_id="u-alice",
        )
        doc = await meta.get_document("doc-1")
        assert doc["owner_id"] == "u-alice"
    finally:
        await meta.aclose()


@pytest.mark.asyncio
async def test_qdrant_upsert_writes_owner_id_to_payload(tmp_path, monkeypatch):
    """Three qdrant collections (text_chunks / vision_pages / desc_pages)
    all get owner_id in their payload so doctor reverse-lookup works.

    Reads payload back via QdrantClient.scroll (no dedicated fetch API
    exists; scroll with with_payload=True is the canonical read pattern
    in qdrant-client for inspection)."""
    from kb.storage.qdrant_store import QdrantStore
    from kb.config import Settings
    from kb.models import TextChunk

    settings = Settings()
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qd"))
    monkeypatch.setattr(settings, "qdrant_url", "")  # local file mode
    (tmp_path / "qd").mkdir(parents=True, exist_ok=True)
    qstore = QdrantStore(settings=settings)
    await qstore.ensure_collections()

    chunk = TextChunk(
        doc_id="d-1", doc_name="x.pdf", page_num=1,
        chunk_index=0, text="hello",
        heading_path=[], has_visual_on_page=False,
        page_type="text", parent_chunk_id=None, is_parent=False,
    )
    await qstore.upsert_text_chunk(
        chunk,
        dense=[0.1] * 1024,
        sparse_indices=[0, 5, 9],
        sparse_values=[0.5, 0.3, 0.2],
        owner_id="u-alice",
    )
    # Direct scroll read — QdrantClient.scroll returns (points, next_offset).
    # Synchronous client API (qdrant-client is sync); no await.
    points, _next = qstore.client.scroll(
        collection_name=qstore.text_collection,
        limit=10,
        with_payload=True,
    )
    assert len(points) == 1
    assert points[0].payload["owner_id"] == "u-alice"
    # Also verify the existing payload fields are still present
    assert points[0].payload["doc_id"] == "d-1"


@pytest.mark.asyncio
async def test_get_document_returns_owner_id(tmp_path):
    """T3 critical fix: get_document must surface owner_id so
    require_owner_or_admin can read it. Without this, all member
    requests hit the I-11 NULL-owner branch by accident."""
    from kb.storage.metadata_db import MetadataDB
    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await meta.init()
    try:
        await meta.upsert_document(
            doc_id="d-1", doc_name="x.pdf",
            version=None, effective_date=None,
            expiry_date=None, supersedes=None,
            owner_id="u-alice",
        )
        doc = await meta.get_document("d-1")
        assert doc["owner_id"] == "u-alice"
    finally:
        await meta.aclose()


@pytest.mark.asyncio
async def test_post_jobs_writes_current_user_as_owner(test_app_with_users):
    """POST /ingestion/jobs must persist current_user.user_id as the job
    owner. The handler MUST NOT accept owner_id from the request body
    (would let a member impersonate another user).

    Real request body schema (see ingestion_api.py:12): {paths, max_attempts}.
    Real router prefix is /ingestion (ingestion_api.py:9), so the full
    path is /ingestion/jobs."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    alice_id = setup["alice_user_id"]
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        r = client.post(
            "/ingestion/jobs",
            headers={"X-API-Key": alice_token},
            json={"paths": [path], "max_attempts": 3},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]   # handler returns _to_dict(job) — no nested "job" key
        store = client.app.state.job_store
        job = await store.get_job(job_id)
        assert job.owner_id == alice_id
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_post_jobs_ignores_owner_id_in_request_body(test_app_with_users):
    """Even if a client sends owner_id in the body, Pydantic's default
    extra="ignore" silently discards it and the handler uses
    current_user. B-1 invariant — no user impersonation."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    alice_id, bob_id = setup["alice_user_id"], setup["bob_user_id"]
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        r = client.post(
            "/ingestion/jobs",
            headers={"X-API-Key": alice_token},
            json={"paths": [path], "max_attempts": 3, "owner_id": bob_id},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        job = await client.app.state.job_store.get_job(job_id)
        assert job.owner_id == alice_id, "owner_id from body must be ignored"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_executor_binds_current_user_contextvar_from_job_owner(audit_setup, tmp_path):
    """IngestionJobExecutor.execute_item must call current_user.set
    from job.owner_id so audit/trace emits during pipeline execution
    record the owner. Reset in finally (B-2 — try/finally pairing,
    same pattern as T1b auth middleware)."""
    from kb.auth.users import UsersStore
    from kb.auth.context import current_user
    from kb.ingestion.job_executor import IngestionJobExecutor
    from kb.jobs.models import IngestionJob, IngestionJobItem
    from unittest.mock import AsyncMock

    db_url = f"sqlite+aiosqlite:///{tmp_path}/users.db"
    users = UsersStore(db_url=db_url)
    await users.init()
    _, alice = await users.create_user(username="alice", role="member")

    captured_user = []

    # Real executor takes a pipeline_factory (job_executor.py:14) that
    # produces an object with an `.ingest(path, owner_id=...)` method.
    # Build a minimal fake pipeline matching the contract used inside
    # execute_item.
    class _FakePipeline:
        async def ingest(self, path, *, owner_id=None):
            captured_user.append(current_user.get())
            return {"doc_id": "fake-doc", "skipped": False,
                    "pages_processed": 1, "chunks_stored": 0}
    def fake_factory(*, cancel_check=None):
        return _FakePipeline()

    store = AsyncMock()
    store.get_job = AsyncMock(return_value=IngestionJob(
        job_id="j-1", status="claimed", total_items=1,
        owner_id=alice.user_id,
    ))
    store.claim_next_item = AsyncMock()
    store.complete_item = AsyncMock()
    store.fail_item = AsyncMock()

    executor = IngestionJobExecutor(
        store,                              # positional store
        fake_factory,                       # positional pipeline_factory
        users_store=users,                  # T3 new kwarg
        # resource_manager, resources, retry_backoff_seconds, audit_log
        # all keep defaults (None / [] / 30 / None) — see job_executor.py:14
    )
    item = IngestionJobItem(item_id="i-1", job_id="j-1", path="x.pdf",
                            status="claimed", attempt=1)
    await executor.execute_item(item, worker_id="w-1")
    # Inside the pipeline call, current_user should be alice
    assert captured_user == [alice]
    # After return, current_user should be reset to None (B-2 try/finally)
    assert current_user.get() is None
    await users.aclose()
