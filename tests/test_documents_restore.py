"""POST /documents/{id}/restore (T5-6)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def api_client(tmp_path):
    """TestClient with realistic auth seeding + real meta_db.

    Returns dict with: client, admin_token, alice_token, bob_token,
    alice_uid, bob_uid, meta_db.
    """
    from kb.auth.users import UsersStore
    from kb.config import Settings

    settings = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))

    async def _seed():
        store = UsersStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
        await store.init()
        admin_pt, admin = await store.create_user(username="admin0", role="admin")
        alice_pt, alice = await store.create_user(username="alice", role="member")
        bob_pt, bob = await store.create_user(username="bob", role="member")
        await store.aclose()
        return admin_pt, alice_pt, bob_pt, alice.user_id, bob.user_id

    admin_token, alice_token, bob_token, alice_uid, bob_uid = asyncio.run(_seed())

    mock_engine = MagicMock()
    mock_qdrant = MagicMock(); mock_qdrant.ensure_collections = AsyncMock()
    mock_feedback_db = MagicMock(); mock_feedback_db.init = AsyncMock()

    from kb.storage.metadata_db import MetadataDB
    meta_db = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    asyncio.run(meta_db.init())

    import kb.tool_server.app as app_module
    with patch.object(app_module, "settings", settings), \
         patch("kb.tool_server.app.QdrantStore", return_value=mock_qdrant), \
         patch("kb.tool_server.app.MetadataDB", return_value=meta_db), \
         patch("kb.tool_server.app.UsageFeedbackDB", return_value=mock_feedback_db), \
         patch("kb.tool_server.app.SearchEngine", return_value=mock_engine):
        from fastapi.testclient import TestClient
        with TestClient(app_module.app) as client:
            yield {
                "client": client,
                "admin_token": admin_token,
                "alice_token": alice_token,
                "bob_token": bob_token,
                "alice_uid": alice_uid,
                "bob_uid": bob_uid,
                "meta_db": meta_db,
            }


async def _seed_deleted_doc(meta_db, doc_id, *, owner_id):
    await meta_db.upsert_document(
        doc_id=doc_id, doc_name=f"{doc_id}.pdf", version=None,
        effective_date=None, expiry_date=None, supersedes=None,
        owner_id=owner_id,
    )
    await meta_db.mark_deleted(doc_id)


def test_restore_returns_200_and_status_active(api_client):
    """Owner can restore their own soft-deleted doc."""
    ctx = api_client
    asyncio.run(_seed_deleted_doc(ctx["meta_db"], "d", owner_id=ctx["alice_uid"]))
    resp = ctx["client"].post(
        "/documents/d/restore",
        headers={"X-API-Key": ctx["alice_token"]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"doc_id": "d", "status": "active"}


def test_restore_clears_deleted_at_and_status(api_client):
    """status flips to active AND deleted_at goes back to NULL."""
    ctx = api_client
    asyncio.run(_seed_deleted_doc(ctx["meta_db"], "d", owner_id=ctx["alice_uid"]))
    ctx["client"].post(
        "/documents/d/restore",
        headers={"X-API-Key": ctx["alice_token"]},
    )
    doc = asyncio.run(ctx["meta_db"].get_document("d", include_deleted_at=True))
    assert doc is not None
    assert doc["status"] == "active"
    assert doc["deleted_at"] is None


def test_restore_blocked_during_maintenance(api_client):
    """maintenance ON → restore returns 503 (C-2)."""
    ctx = api_client
    asyncio.run(_seed_deleted_doc(ctx["meta_db"], "d", owner_id=ctx["alice_uid"]))
    on_resp = ctx["client"].post(
        "/admin/maintenance_mode", json={"on": True},
        headers={"X-API-Key": ctx["admin_token"]},
    )
    assert on_resp.status_code == 200
    try:
        resp = ctx["client"].post(
            "/documents/d/restore",
            headers={"X-API-Key": ctx["alice_token"]},
        )
        assert resp.status_code == 503
    finally:
        ctx["client"].post(
            "/admin/maintenance_mode", json={"on": False},
            headers={"X-API-Key": ctx["admin_token"]},
        )


def test_member_cannot_restore_others_doc(api_client):
    """T3 ownership rule: bob can't restore alice's doc → 403."""
    ctx = api_client
    asyncio.run(_seed_deleted_doc(ctx["meta_db"], "d", owner_id=ctx["alice_uid"]))
    resp = ctx["client"].post(
        "/documents/d/restore",
        headers={"X-API-Key": ctx["bob_token"]},
    )
    assert resp.status_code == 403


def test_admin_can_restore_null_owner_doc(api_client):
    """I-11: pre-T3 NULL-owner doc → admin only restore."""
    ctx = api_client
    asyncio.run(_seed_deleted_doc(ctx["meta_db"], "legacy", owner_id=None))
    # Member blocked
    member_resp = ctx["client"].post(
        "/documents/legacy/restore",
        headers={"X-API-Key": ctx["alice_token"]},
    )
    assert member_resp.status_code == 403
    # Admin succeeds
    admin_resp = ctx["client"].post(
        "/documents/legacy/restore",
        headers={"X-API-Key": ctx["admin_token"]},
    )
    assert admin_resp.status_code == 200


def test_restore_404_when_doc_missing(api_client):
    """Restoring a non-existent doc → 404 (not 400 / 403)."""
    ctx = api_client
    resp = ctx["client"].post(
        "/documents/nope/restore",
        headers={"X-API-Key": ctx["alice_token"]},
    )
    assert resp.status_code == 404


def test_restore_invalidates_search_cache(api_client):
    """Symmetric to test_delete_still_invalidates_search_cache — the
    restored doc must show up in subsequent searches, which requires
    dropping the cache that may have returned empty results during the
    deleted period."""
    import kb.search.cache as cache_mod
    ctx = api_client
    asyncio.run(_seed_deleted_doc(ctx["meta_db"], "d", owner_id=ctx["alice_uid"]))
    with patch.object(cache_mod, "invalidate_all") as inv:
        resp = ctx["client"].post(
            "/documents/d/restore",
            headers={"X-API-Key": ctx["alice_token"]},
        )
        assert resp.status_code == 200
        inv.assert_called_once()


def test_restore_400_when_doc_not_in_deleted_state(api_client):
    """Restoring an already-active doc → 400 (not the right inverse of delete)."""
    ctx = api_client
    asyncio.run(ctx["meta_db"].upsert_document(
        doc_id="active", doc_name="active.pdf", version=None,
        effective_date=None, expiry_date=None, supersedes=None,
        owner_id=ctx["alice_uid"],
    ))
    resp = ctx["client"].post(
        "/documents/active/restore",
        headers={"X-API-Key": ctx["alice_token"]},
    )
    assert resp.status_code == 400
