# tests/test_storage.py
import pytest
import uuid
from qdrant_client.models import Filter, FieldCondition, MatchValue
from kb.storage.qdrant_store import QdrantStore
from kb.models import TextChunk

@pytest.fixture
def store(test_settings):
    return QdrantStore(settings=test_settings)

@pytest.mark.asyncio
async def test_ensure_collections_creates_all(store):
    await store.ensure_collections()
    names = [c.name for c in store.client.get_collections().collections]
    assert any("text_chunks" in n for n in names)
    assert any("vision_pages" in n for n in names)
    assert any("desc_pages" in n for n in names)

@pytest.mark.asyncio
async def test_upsert_text_chunk(store):
    await store.ensure_collections()
    dense = [0.1] * 1024
    sparse_indices = [1, 2, 3]
    sparse_values = [0.5, 0.3, 0.2]
    chunk = TextChunk(
        doc_id="test", doc_name="t.pdf", page_num=0, chunk_index=0,
        text="审批流程", heading_path=["Doc"], has_visual_on_page=False,
        page_type="text", parent_chunk_id="test_p0", is_parent=False
    )
    await store.upsert_text_chunk(chunk, dense, sparse_indices, sparse_values)
    results = store.client.scroll(
        collection_name=store.text_collection,
        scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value="test"))]),
        limit=10
    )
    assert len(results[0]) == 1

@pytest.mark.asyncio
async def test_delete_by_doc_id(store):
    await store.ensure_collections()
    dense = [0.0] * 1024
    chunk = TextChunk(
        doc_id="to_delete", doc_name="d.pdf", page_num=0, chunk_index=0,
        text="x", heading_path=[], has_visual_on_page=False,
        page_type="text", parent_chunk_id="to_delete_p0", is_parent=False
    )
    await store.upsert_text_chunk(chunk, dense, [1], [1.0])
    await store.delete_by_doc_id("to_delete")
    results = store.client.scroll(
        collection_name=store.text_collection,
        scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value="to_delete"))]),
        limit=10
    )
    assert len(results[0]) == 0

from kb.storage.metadata_db import MetadataDB
from kb.storage.image_store import ImageStore
from datetime import date

@pytest.mark.asyncio
async def test_metadata_upsert_and_get(test_settings, tmp_path):
    db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await db.init()
    await db.upsert_document(
        doc_id="doc1", doc_name="test.pdf",
        version="v1.0", effective_date=date(2026, 1, 1),
        expiry_date=None, supersedes=None,
    )
    record = await db.get_document("doc1")
    assert record["doc_name"] == "test.pdf"
    assert record["status"] == "active"

@pytest.mark.asyncio
async def test_metadata_active_only_filter(test_settings, tmp_path):
    db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/meta2.db")
    await db.init()
    await db.upsert_document("d1", "a.pdf", version=None,
                             effective_date=None, expiry_date=None, supersedes=None)
    await db.deprecate_document("d1")
    active = await db.list_active_doc_ids()
    assert "d1" not in active


@pytest.mark.asyncio
async def test_metadata_can_mark_deleted(tmp_path):
    from kb.storage.metadata_db import DOC_STATUS_DELETED

    db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/deleted.db")
    await db.init()
    await db.upsert_document(
        "d",
        "d.pdf",
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
    )
    await db.mark_deleted("d")

    rec = await db.get_document("d")
    assert rec["status"] == DOC_STATUS_DELETED


@pytest.mark.asyncio
async def test_metadata_can_mark_failed(tmp_path):
    from kb.storage.metadata_db import DOC_STATUS_FAILED

    db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/failed.db")
    await db.init()
    await db.upsert_document(
        "d",
        "d.pdf",
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
    )
    await db.mark_failed_document("d")

    rec = await db.get_document("d")
    assert rec["status"] == DOC_STATUS_FAILED


@pytest.mark.asyncio
async def test_metadata_can_activate_staging(tmp_path):
    from kb.storage.metadata_db import (
        DOC_STATUS_ACTIVE,
        DOC_STATUS_STAGING,
    )

    db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/staging.db")
    await db.init()
    await db.upsert_document(
        "d",
        "d.pdf",
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
        status=DOC_STATUS_STAGING,
    )
    await db.activate_document("d")

    rec = await db.get_document("d")
    assert rec["status"] == DOC_STATUS_ACTIVE


@pytest.mark.asyncio
async def test_image_store_save_and_path(test_settings, sample_image):
    store = ImageStore(settings=test_settings)
    url = await store.save(doc_id="img_test", page_num=0, image_bytes=sample_image)
    assert "img_test" in url
    assert url.endswith(".png")


@pytest.mark.asyncio
async def test_image_store_delete_doc_images(tmp_path):
    from kb.config import Settings

    settings = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "images"),
    )
    store = ImageStore(settings=settings)
    await store.save("doc1", 0, b"png")
    assert (tmp_path / "images" / "doc1" / "page_0.png").exists()

    result = await store.delete_doc_images("doc1")

    assert result["deleted"] is True
    assert not (tmp_path / "images" / "doc1").exists()


from datetime import date, timedelta

@pytest.mark.asyncio
async def test_list_valid_doc_ids_excludes_expired(tmp_path):
    db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/meta3.db")
    await db.init()
    yesterday = date.today() - timedelta(days=1)
    tomorrow = date.today() + timedelta(days=1)
    await db.upsert_document("active_ok", "a.pdf", version=None,
                             effective_date=None, expiry_date=tomorrow, supersedes=None)
    await db.upsert_document("expired", "b.pdf", version=None,
                             effective_date=None, expiry_date=yesterday, supersedes=None)
    await db.upsert_document("no_expiry", "c.pdf", version=None,
                             effective_date=None, expiry_date=None, supersedes=None)
    valid = await db.list_valid_doc_ids()
    assert "active_ok" in valid
    assert "no_expiry" in valid
    assert "expired" not in valid

@pytest.mark.asyncio
async def test_list_valid_doc_ids_excludes_deprecated(tmp_path):
    db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/meta4.db")
    await db.init()
    await db.upsert_document("dep", "x.pdf", version=None,
                             effective_date=None, expiry_date=None, supersedes=None)
    await db.deprecate_document("dep")
    valid = await db.list_valid_doc_ids()
    assert "dep" not in valid

@pytest.mark.asyncio
async def test_search_text_dense_returns_non_parent_only(store):
    await store.ensure_collections()
    parent = TextChunk(
        doc_id="srch", doc_name="s.pdf", page_num=0, chunk_index=0,
        text="CEO 需要签字", heading_path=["Doc"], has_visual_on_page=False,
        page_type="text", parent_chunk_id="srch_p0_c0", is_parent=True
    )
    child = TextChunk(
        doc_id="srch", doc_name="s.pdf", page_num=0, chunk_index=1,
        text="CEO 需要签字", heading_path=["Doc"], has_visual_on_page=False,
        page_type="text", parent_chunk_id="srch_p0_c0", is_parent=False
    )
    dense = [0.1] * 1024
    await store.upsert_text_chunk(parent, dense, [1], [1.0])
    await store.upsert_text_chunk(child, dense, [1], [1.0])
    results = store.search_text_dense(dense, limit=5)
    assert len(results) == 1
    assert results[0]["doc_id"] == "srch"
    assert results[0]["is_parent"] is False

@pytest.mark.asyncio
async def test_get_parent_chunk_returns_parent(store):
    await store.ensure_collections()
    parent = TextChunk(
        doc_id="par", doc_name="p.pdf", page_num=1, chunk_index=0,
        text="parent text", heading_path=[], has_visual_on_page=False,
        page_type="text", parent_chunk_id="par_p1_c0", is_parent=True
    )
    await store.upsert_text_chunk(parent, [0.2] * 1024, [2], [0.9])
    rec = store.get_parent_chunk("par", 1)
    assert rec is not None
    assert rec["text"] == "parent text"
    assert rec["is_parent"] is True


@pytest.mark.asyncio
async def test_get_vision_record_returns_payload(store):
    """get_vision_record 用于在纯文本页 desc_rec 缺失时回退取 image_url."""
    from kb.models import VisionPage
    await store.ensure_collections()
    page = VisionPage(
        doc_id="vis", doc_name="v.pdf", page_num=2,
        image_url="local://vis/page_2.png", heading_path=["Doc", "Ch"],
        vision_tier="qwen3vl",
    )
    # v1.1 single-vector Qwen3-VL embedding: list[float] (4096-dim)
    embedding = [0.1] * 4096
    await store.upsert_vision_page(page, embedding)
    rec = store.get_vision_record("vis", 2)
    assert rec is not None
    assert rec["image_url"] == "local://vis/page_2.png"
    assert rec["vision_tier"] == "qwen3vl"


@pytest.mark.asyncio
async def test_vision_page_payload_includes_page_type(mocker, test_settings):
    from kb.models import VisionPage
    from kb.storage.qdrant_store import QdrantStore

    store = QdrantStore(settings=test_settings)
    mock_upsert = mocker.patch.object(store.client, "upsert")
    page = VisionPage(
        doc_id="d",
        doc_name="d.pdf",
        page_num=0,
        image_url="local://x",
        heading_path=[],
        vision_tier="qwen3vl",
        page_type="table",
    )

    await store.upsert_vision_page(page, [0.1] * 4096)

    payload = mock_upsert.call_args.kwargs["points"][0].payload
    assert payload["page_type"] == "table"


@pytest.mark.asyncio
async def test_get_vision_record_returns_none_when_missing(store):
    await store.ensure_collections()
    assert store.get_vision_record("nonexistent", 0) is None


@pytest.mark.asyncio
async def test_list_documents_filter_by_status(tmp_path):
    from kb.storage.metadata_db import MetadataDB
    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        for did, status in [
            ("a", "active"), ("d1", "deleted"), ("d2", "deleted"), ("p", "deprecated"),
        ]:
            await m.upsert_document(
                doc_id=did, doc_name=f"{did}.pdf", version=None,
                effective_date=None, expiry_date=None, supersedes=None,
                owner_id="u-x",
            )
            if status != "active":
                # status starts as active per upsert_document; flip if needed
                async with m._engine.begin() as conn:
                    await conn.exec_driver_sql(
                        "UPDATE documents SET status=? WHERE doc_id=?", (status, did)
                    )
        deleted = await m.list_documents(status="deleted")
        assert {d["doc_id"] for d in deleted} == {"d1", "d2"}
        # Each row dict carries owner_id (T3-1 fix) and doc_name
        assert all("owner_id" in d and "doc_name" in d for d in deleted)
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_list_documents_orphan_owner_filter(tmp_path):
    """Doctor 'orphan owners' panel — finds NULL + _system owner docs."""
    from kb.storage.metadata_db import MetadataDB
    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        for did, owner in [
            ("a1", "u-alice"), ("legacy1", None), ("legacy2", "_system"), ("b1", "u-bob"),
        ]:
            await m.upsert_document(
                doc_id=did, doc_name=f"{did}.pdf", version=None,
                effective_date=None, expiry_date=None, supersedes=None,
                owner_id=owner,
            )
        orphans = await m.list_documents(owner_filter="null_or_system")
        assert {d["doc_id"] for d in orphans} == {"legacy1", "legacy2"}
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_list_documents_non_null_owner_filter(tmp_path):
    """The non_null branch returns the complement of null_or_system —
    excludes both NULL and '_system' owners. Used as a future hook (not
    by T4 doctor itself) but worth covering since it's its own SQL path."""
    from kb.storage.metadata_db import MetadataDB
    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        for did, owner in [
            ("a1", "u-alice"), ("b1", "u-bob"),
            ("legacy1", None), ("legacy2", "_system"),
        ]:
            await m.upsert_document(
                doc_id=did, doc_name=f"{did}.pdf", version=None,
                effective_date=None, expiry_date=None, supersedes=None,
                owner_id=owner,
            )
        members = await m.list_documents(owner_filter="non_null")
        assert {d["doc_id"] for d in members} == {"a1", "b1"}
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_list_documents_unknown_owner_filter_raises(tmp_path):
    """Unknown owner_filter values raise ValueError (fail-loud rather
    than silently returning the wrong thing)."""
    import pytest as _pytest
    from kb.storage.metadata_db import MetadataDB
    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        with _pytest.raises(ValueError, match="unknown owner_filter"):
            await m.list_documents(owner_filter="banana")
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_mark_deleted_writes_deleted_at(tmp_path):
    """T5: mark_deleted() must set deleted_at = now() so GC has truth."""
    from datetime import datetime, timezone, timedelta
    from kb.storage.metadata_db import MetadataDB

    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        await m.upsert_document(
            doc_id="d", doc_name="d.pdf", version=None,
            effective_date=None, expiry_date=None, supersedes=None,
            owner_id="u-x",
        )
        await m.mark_deleted("d")
        doc = await m.get_document(doc_id="d", include_deleted_at=True)
        assert doc is not None
        assert doc["status"] == "deleted"
        assert doc["deleted_at"] is not None
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        assert (now - doc["deleted_at"]) < timedelta(seconds=5)
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_restore_document_clears_deleted_at_and_reactivates(tmp_path):
    """T5: restore_document() flips status back to active and clears
    deleted_at so GC won't pick it up later."""
    from kb.storage.metadata_db import MetadataDB

    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        await m.upsert_document(
            doc_id="d", doc_name="d.pdf", version=None,
            effective_date=None, expiry_date=None, supersedes=None,
            owner_id="u-x",
        )
        await m.mark_deleted("d")
        await m.restore_document("d")
        doc = await m.get_document(doc_id="d")
        assert doc is not None
        assert doc["status"] == "active"
        doc_full = await m.get_document(doc_id="d", include_deleted_at=True)
        assert doc_full["deleted_at"] is None
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_migration_adds_deleted_at_column_idempotent(tmp_path):
    """Pre-T5 DB has documents table without deleted_at — migration adds it
    and is safe to re-run (idempotent ALTER pattern from T1a)."""
    from kb.storage.metadata_db import MetadataDB

    db_url = f"sqlite+aiosqlite:///{tmp_path}/m.db"
    from sqlalchemy.ext.asyncio import create_async_engine
    e = create_async_engine(db_url, poolclass=__import__("sqlalchemy.pool", fromlist=["NullPool"]).NullPool)
    async with e.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE documents ("
            "  doc_id TEXT PRIMARY KEY,"
            "  doc_name TEXT,"
            "  version TEXT,"
            "  effective_date DATE,"
            "  expiry_date DATE,"
            "  supersedes TEXT,"
            "  status TEXT DEFAULT 'active',"
            "  ingested_at DATETIME,"
            "  owner_id TEXT"
            ")"
        )
        await conn.exec_driver_sql(
            "INSERT INTO documents (doc_id, doc_name, status) VALUES "
            "('old1', 'old1.pdf', 'deleted')"
        )
    await e.dispose()

    m = MetadataDB(db_url=db_url)
    await m.init()
    try:
        await m.init()  # idempotent: second init() must not fail
        doc = await m.get_document(doc_id="old1", include_deleted_at=True)
        assert doc is not None
        assert doc["status"] == "deleted"
        assert doc["deleted_at"] is None  # pre-T5 row: NULL; GC must skip with warning
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_get_document_default_hides_deleted_at(tmp_path):
    """Backward compat: get_document() default shape must NOT include
    deleted_at (existing callers don't expect it)."""
    from kb.storage.metadata_db import MetadataDB

    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        await m.upsert_document(
            doc_id="d", doc_name="d.pdf", version=None,
            effective_date=None, expiry_date=None, supersedes=None,
            owner_id="u-x",
        )
        doc = await m.get_document(doc_id="d")
        assert "deleted_at" not in doc
        doc_full = await m.get_document(doc_id="d", include_deleted_at=True)
        assert "deleted_at" in doc_full
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_list_documents_includes_deleted_at_for_trash(tmp_path):
    """T5: list_documents(status='deleted') must include deleted_at so
    doctor_team's trash inventory can use it directly (closes T4 P1-5
    carry-forward)."""
    from kb.storage.metadata_db import MetadataDB

    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        await m.upsert_document(
            doc_id="d", doc_name="d.pdf", version=None,
            effective_date=None, expiry_date=None, supersedes=None,
            owner_id="u-x",
        )
        await m.mark_deleted("d")
        rows = await m.list_documents(status="deleted")
        assert len(rows) == 1
        assert rows[0]["deleted_at"] is not None
    finally:
        await m.aclose()


@pytest.mark.asyncio
async def test_list_documents_combined_status_and_owner_filter(tmp_path):
    """T4 carry-forward: list_documents supports BOTH filters together."""
    from kb.storage.metadata_db import MetadataDB

    m = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await m.init()
    try:
        for did, owner in [
            ("a1", "u-alice"), ("a2", "u-alice"),
            ("legacy", None),
        ]:
            await m.upsert_document(
                doc_id=did, doc_name=f"{did}.pdf",
                version=None, effective_date=None,
                expiry_date=None, supersedes=None, owner_id=owner,
            )
        await m.mark_deleted("a2")
        await m.mark_deleted("legacy")
        rows = await m.list_documents(status="deleted", owner_filter="null_or_system")
        assert {d["doc_id"] for d in rows} == {"legacy"}
    finally:
        await m.aclose()
