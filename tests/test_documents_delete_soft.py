"""T5-4: DELETE handler is pure soft delete — Qdrant/images/nav untouched."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def api_client(tmp_path):
    """Real meta_db + mocked kb_store/nav_store so we can ASSERT the
    physical-cleanup methods are NEVER called by the DELETE handler."""
    from kb.auth.users import UsersStore
    from kb.config import Settings
    from kb.storage.metadata_db import MetadataDB

    settings = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))

    async def _seed():
        store = UsersStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
        await store.init()
        admin_pt, admin = await store.create_user(username="admin0", role="admin")
        alice_pt, alice = await store.create_user(username="alice", role="member")
        await store.aclose()
        return admin_pt, alice_pt, alice.user_id

    admin_token, alice_token, alice_uid = asyncio.run(_seed())
    meta_db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    asyncio.run(meta_db.init())

    mock_kb_store = MagicMock()
    mock_kb_store.cleanup_doc_assets = AsyncMock()
    mock_nav_store = MagicMock()
    mock_nav_store.delete_entries_for_doc = AsyncMock()

    mock_engine = MagicMock()
    mock_qdrant = MagicMock(); mock_qdrant.ensure_collections = AsyncMock()
    mock_feedback_db = MagicMock(); mock_feedback_db.init = AsyncMock()

    import kb.tool_server.app as app_module
    with patch.object(app_module, "settings", settings), \
         patch("kb.tool_server.app.QdrantStore", return_value=mock_qdrant), \
         patch("kb.tool_server.app.MetadataDB", return_value=meta_db), \
         patch("kb.tool_server.app.UsageFeedbackDB", return_value=mock_feedback_db), \
         patch("kb.tool_server.app.SearchEngine", return_value=mock_engine), \
         patch("kb.tool_server.app.KnowledgeBaseStore", return_value=mock_kb_store):
        from fastapi.testclient import TestClient
        with TestClient(app_module.app) as client:
            app_module.app.state.nav_store = mock_nav_store
            yield {
                "client": client,
                "alice_token": alice_token,
                "alice_uid": alice_uid,
                "meta_db": meta_db,
                "kb_store_mock": mock_kb_store,
                "nav_store_mock": mock_nav_store,
            }


def test_delete_marks_status_and_writes_deleted_at(api_client):
    """The new DELETE handler must call mark_deleted (which writes
    status='deleted' AND deleted_at=now per Task 3)."""
    ctx = api_client
    asyncio.run(ctx["meta_db"].upsert_document(
        doc_id="d", doc_name="d.pdf", version=None,
        effective_date=None, expiry_date=None, supersedes=None,
        owner_id=ctx["alice_uid"],
    ))
    resp = ctx["client"].delete(
        "/documents/d", headers={"X-API-Key": ctx["alice_token"]},
    )
    assert resp.status_code == 200
    doc = asyncio.run(ctx["meta_db"].get_document("d", include_deleted_at=True))
    assert doc["status"] == "deleted"
    assert doc["deleted_at"] is not None


def test_delete_does_NOT_touch_qdrant_or_images(api_client):
    """C-8: physical cleanup is GC's job. DELETE must not call
    cleanup_doc_assets — otherwise restore (Task 6) is useless."""
    ctx = api_client
    asyncio.run(ctx["meta_db"].upsert_document(
        doc_id="d", doc_name="d.pdf", version=None,
        effective_date=None, expiry_date=None, supersedes=None,
        owner_id=ctx["alice_uid"],
    ))
    ctx["client"].delete("/documents/d", headers={"X-API-Key": ctx["alice_token"]})
    ctx["kb_store_mock"].cleanup_doc_assets.assert_not_called()


def test_delete_does_NOT_touch_nav(api_client):
    """C-8 cont: nav entries must also stay — restored doc needs them."""
    ctx = api_client
    asyncio.run(ctx["meta_db"].upsert_document(
        doc_id="d", doc_name="d.pdf", version=None,
        effective_date=None, expiry_date=None, supersedes=None,
        owner_id=ctx["alice_uid"],
    ))
    ctx["client"].delete("/documents/d", headers={"X-API-Key": ctx["alice_token"]})
    ctx["nav_store_mock"].delete_entries_for_doc.assert_not_called()


def test_delete_still_invalidates_search_cache(api_client):
    """Cache must still drop — search results that mentioned the doc
    have to drop now, even though the underlying vectors stay (read
    path hides deleted docs via I-6 active gate)."""
    import kb.search.cache as cache_mod
    ctx = api_client
    asyncio.run(ctx["meta_db"].upsert_document(
        doc_id="d", doc_name="d.pdf", version=None,
        effective_date=None, expiry_date=None, supersedes=None,
        owner_id=ctx["alice_uid"],
    ))
    with patch.object(cache_mod, "invalidate_all") as inv:
        ctx["client"].delete("/documents/d", headers={"X-API-Key": ctx["alice_token"]})
        inv.assert_called_once()


def test_delete_still_writes_two_phase_audit(api_client):
    """T2 two-phase contract intact. Both .attempted and .completed
    rows must land in audit_log for the deleted doc."""
    ctx = api_client
    asyncio.run(ctx["meta_db"].upsert_document(
        doc_id="d", doc_name="d.pdf", version=None,
        effective_date=None, expiry_date=None, supersedes=None,
        owner_id=ctx["alice_uid"],
    ))
    ctx["client"].delete("/documents/d", headers={"X-API-Key": ctx["alice_token"]})
    import kb.tool_server.app as app_module
    audit = app_module.app.state.audit_log
    attempted = asyncio.run(audit.query(action="doc.delete.attempted", target_id="d"))
    completed = asyncio.run(audit.query(action="doc.delete.completed", target_id="d"))
    assert len(attempted) == 1
    assert len(completed) == 1
