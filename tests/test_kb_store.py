from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_kb_store_cleanup_doc_calls_all_backends():
    from kb.storage.kb_store import KnowledgeBaseStore

    qdrant = MagicMock()
    qdrant.delete_by_doc_id = AsyncMock()
    meta_db = MagicMock()
    image_store = MagicMock()
    image_store.delete_doc_images = AsyncMock(return_value={"deleted": True})
    store = KnowledgeBaseStore(qdrant=qdrant, meta_db=meta_db, image_store=image_store)

    result = await store.cleanup_doc_assets("doc1")

    qdrant.delete_by_doc_id.assert_awaited_once_with("doc1")
    image_store.delete_doc_images.assert_awaited_once_with("doc1")
    assert result["vectors_deleted"] is True
    assert result["images"]["deleted"] is True


@pytest.mark.asyncio
async def test_kb_store_cleanup_doc_reports_partial_failures():
    from kb.storage.kb_store import KnowledgeBaseStore

    qdrant = MagicMock()
    qdrant.delete_by_doc_id = AsyncMock(side_effect=RuntimeError("qdrant down"))
    meta_db = MagicMock()
    image_store = MagicMock()
    image_store.delete_doc_images = AsyncMock(return_value={"deleted": False})
    store = KnowledgeBaseStore(qdrant=qdrant, meta_db=meta_db, image_store=image_store)

    result = await store.cleanup_doc_assets("doc1")

    assert result["vectors_deleted"] is False
    assert "qdrant down" in result["vector_error"]
    assert result["images"]["deleted"] is False


@pytest.mark.asyncio
async def test_kb_store_cleanup_staging_doc_reuses_asset_cleanup():
    from kb.storage.kb_store import KnowledgeBaseStore

    qdrant = MagicMock()
    qdrant.delete_by_doc_id = AsyncMock()
    meta_db = MagicMock()
    image_store = MagicMock()
    image_store.delete_doc_images = AsyncMock(return_value={"deleted": True})
    store = KnowledgeBaseStore(qdrant=qdrant, meta_db=meta_db, image_store=image_store)

    result = await store.cleanup_staging_doc("doc1")

    qdrant.delete_by_doc_id.assert_awaited_once_with("doc1")
    assert result["doc_id"] == "doc1"
    assert result["vectors_deleted"] is True
