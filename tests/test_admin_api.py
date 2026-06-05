"""Admin endpoint — POST/GET /admin/maintenance_mode (T5-2)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def admin_client(tmp_path):
    """FastAPI TestClient wired with a real MaintenanceState + admin user.
    Other stores are mocked so we only exercise the admin path."""
    from kb.admin.maintenance import MaintenanceState
    from kb.auth.users import UsersStore
    from kb.config import Settings

    settings = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))

    async def _seed():
        store = UsersStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
        await store.init()
        admin_pt, _ = await store.create_user(username="admin0", role="admin")
        member_pt, _ = await store.create_user(username="bob", role="member")
        await store.aclose()
        return admin_pt, member_pt

    admin_token, member_token = asyncio.run(_seed())

    mock_engine = MagicMock()
    mock_meta_db = MagicMock(); mock_meta_db.init = AsyncMock()
    mock_qdrant = MagicMock(); mock_qdrant.ensure_collections = AsyncMock()
    mock_feedback_db = MagicMock(); mock_feedback_db.init = AsyncMock()

    import kb.tool_server.app as app_module
    with patch.object(app_module, "settings", settings), \
         patch("kb.tool_server.app.QdrantStore", return_value=mock_qdrant), \
         patch("kb.tool_server.app.MetadataDB", return_value=mock_meta_db), \
         patch("kb.tool_server.app.UsageFeedbackDB", return_value=mock_feedback_db), \
         patch("kb.tool_server.app.SearchEngine", return_value=mock_engine):
        from fastapi.testclient import TestClient
        with TestClient(app_module.app) as client:
            yield client, admin_token, member_token


def test_get_maintenance_mode_default_off(admin_client):
    client, admin_token, _ = admin_client
    resp = client.get("/admin/maintenance_mode", headers={"X-API-Key": admin_token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["on"] is False
    assert body["set_at"] is None
    assert body["set_by_user_id"] is None


def test_post_maintenance_mode_requires_admin(admin_client):
    """C-3 boundary: member cannot flip the flag (403)."""
    client, _, member_token = admin_client
    resp = client.post("/admin/maintenance_mode",
                       json={"on": True},
                       headers={"X-API-Key": member_token})
    assert resp.status_code == 403


def test_post_maintenance_mode_on_sets_flag(admin_client):
    client, admin_token, _ = admin_client
    resp = client.post("/admin/maintenance_mode",
                       json={"on": True},
                       headers={"X-API-Key": admin_token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["on"] is True
    # drained:True because no jobs were claimed in this test (queue empty)
    assert body["drained"] is True

    # GET reflects the new state
    gresp = client.get("/admin/maintenance_mode", headers={"X-API-Key": admin_token})
    assert gresp.json()["on"] is True
    assert gresp.json()["set_by_user_id"]  # whoever the admin's user_id is


def test_post_maintenance_mode_off_clears_flag(admin_client):
    client, admin_token, _ = admin_client
    client.post("/admin/maintenance_mode", json={"on": True}, headers={"X-API-Key": admin_token})
    resp = client.post("/admin/maintenance_mode",
                       json={"on": False},
                       headers={"X-API-Key": admin_token})
    assert resp.status_code == 200
    assert resp.json()["on"] is False
    assert resp.json().get("set_at") is not None  # off action also recorded


def test_post_maintenance_on_drains_active_workers(admin_client, monkeypatch):
    """C-4: when admin flips ON, the endpoint waits up to 5min for
    active claims to drain. Simulate via monkeypatched _count_active_claims_via_app
    that returns 2 → 1 → 0 across calls."""
    import kb.tool_server.admin_api as admin_module

    counts = iter([2, 1, 0])
    async def fake_count(request):
        # Default to 0 once the iter is exhausted to avoid StopIteration
        # if the test polls more times than expected on a slow CI runner.
        return next(counts, 0)

    with patch.object(admin_module, "_count_active_claims_via_app", new=fake_count):
        # Speed up the poll to keep the test fast (defaults to 1s).
        monkeypatch.setattr(admin_module, "_DRAIN_POLL_SECONDS", 0.05)
        client, admin_token, _ = admin_client
        resp = client.post("/admin/maintenance_mode",
                           json={"on": True},
                           headers={"X-API-Key": admin_token})
        assert resp.status_code == 200
        body = resp.json()
        assert body["on"] is True
        assert body["drained"] is True


def test_post_maintenance_on_drain_timeout_reports_false(admin_client, monkeypatch):
    """C-4: stuck worker → drained:false within 5min; flag still ON
    so admin / script can decide what to do (force backup or fix worker)."""
    import kb.tool_server.admin_api as admin_module

    # Shorten the drain timeout for the test (default 300s).
    monkeypatch.setattr(admin_module, "_DRAIN_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(admin_module, "_DRAIN_POLL_SECONDS", 0.1)

    # count_active_claims always returns 1 — never drains.
    async def fake_count(request):
        return 1
    with patch.object(admin_module, "_count_active_claims_via_app", new=fake_count):
        client, admin_token, _ = admin_client
        resp = client.post("/admin/maintenance_mode",
                           json={"on": True},
                           headers={"X-API-Key": admin_token})
        assert resp.status_code == 200
        body = resp.json()
        assert body["on"] is True
        assert body["drained"] is False
        assert body["still_active"] == 1
        assert body["count_failed"] is False  # counting succeeded; the worker is genuinely still active


def test_get_maintenance_mode_is_open_during_maintenance(admin_client):
    """C-3: GET /admin/maintenance_mode must always work (scripts probe
    this while flag is on). It must NOT be blocked by require_no_maintenance."""
    client, admin_token, _ = admin_client
    client.post("/admin/maintenance_mode", json={"on": True}, headers={"X-API-Key": admin_token})
    # GET still works while maintenance is on
    resp = client.get("/admin/maintenance_mode", headers={"X-API-Key": admin_token})
    assert resp.status_code == 200
    assert resp.json()["on"] is True
