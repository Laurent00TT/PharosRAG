import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def api_client(tmp_path):
    from kb.auth.users import UsersStore
    from kb.config import Settings
    from kb.models import SearchResponse

    settings = Settings(
        qdrant_path=str(tmp_path),
        image_storage_path=str(tmp_path),
    )

    # T1b: pre-populate the users DB at the path the lifespan will open so
    # the bootstrap refuse-start check (users-table-empty) passes.
    # asyncio.run() is safe here because sync fixtures have no running loop.
    async def _setup_users():
        store = UsersStore(
            db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
        )
        await store.init()
        admin_pt, _ = await store.create_user(username="admin0", role="admin")
        await store.aclose()
        return admin_pt

    admin_token = asyncio.run(_setup_users())

    mock_engine = MagicMock()
    mock_engine.search = AsyncMock(
        return_value=(SearchResponse(results=[], total=0), False)
    )
    mock_meta_db = MagicMock()
    mock_meta_db.init = AsyncMock()
    mock_meta_db.get_document = AsyncMock(return_value=None)
    mock_meta_db.list_active_doc_ids = AsyncMock(return_value=[])
    mock_qdrant = MagicMock()
    mock_qdrant.ensure_collections = AsyncMock()
    mock_feedback_db = MagicMock()
    mock_feedback_db.init = AsyncMock()

    import kb.tool_server.app as app_module
    with patch.object(app_module, "settings", settings), \
         patch("kb.tool_server.app.QdrantStore", return_value=mock_qdrant), \
         patch("kb.tool_server.app.MetadataDB", return_value=mock_meta_db), \
         patch("kb.tool_server.app.UsageFeedbackDB", return_value=mock_feedback_db), \
         patch("kb.tool_server.app.SearchEngine", return_value=mock_engine):
        from fastapi.testclient import TestClient
        # Pass admin token as default header so every request is authenticated.
        with TestClient(app_module.app, headers={"X-API-Key": admin_token}) as client:
            yield client


def test_create_ingestion_job_returns_job_id(api_client, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    resp = api_client.post("/ingestion/jobs", json={"paths": [str(pdf)]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert body["total_items"] == 1
    assert body["job_id"]


def test_create_ingestion_job_rejects_missing_path(api_client, tmp_path):
    missing = tmp_path / "missing.pdf"

    resp = api_client.post("/ingestion/jobs", json={"paths": [str(missing)]})

    assert resp.status_code == 400


def test_create_ingestion_job_rejects_unsupported_suffix(api_client, tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("not a pdf", encoding="utf-8")

    resp = api_client.post("/ingestion/jobs", json={"paths": [str(txt)]})

    assert resp.status_code == 400


def test_get_ingestion_job_returns_items(api_client, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    created = api_client.post("/ingestion/jobs", json={"paths": [str(pdf)]}).json()

    resp = api_client.get(f"/ingestion/jobs/{created['job_id']}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["job"]["job_id"] == created["job_id"]
    assert body["items"][0]["path"] == str(pdf)


def test_ingestion_job_events_support_after_id(api_client, tmp_path):
    from datetime import datetime

    import kb.tool_server.app as app_module
    from kb.jobs.models import IngestionJobEvent

    app_module.app.state.job_store.list_events = AsyncMock(
        return_value=[
            IngestionJobEvent(
                event_id=2,
                job_id="job1",
                item_id=None,
                ts=datetime(2026, 5, 10),
                level="info",
                event="second",
                message="two",
            )
        ]
    )

    resp = api_client.get(
        "/ingestion/jobs/job1/events",
        params={"after_id": 1},
    )

    assert [event["event"] for event in resp.json()["events"]] == ["second"]
    app_module.app.state.job_store.list_events.assert_awaited_once_with(
        "job1",
        after_id=1,
    )


def test_cancel_ingestion_job_sets_cancel_flag(api_client, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    created = api_client.post("/ingestion/jobs", json={"paths": [str(pdf)]}).json()

    resp = api_client.post(f"/ingestion/jobs/{created['job_id']}/cancel")

    assert resp.status_code == 200
    assert resp.json()["job"]["cancel_requested"] is True


def test_retry_ingestion_job_calls_store(api_client, tmp_path):
    """T3 follow-up (P1a review): retry handler now does
    get_job → 404 → require_owner_or_admin → retry. Create the job for
    real (so get_job sees it) instead of full-mocking the store —
    matches the cancel-test pattern in the same file."""
    import kb.tool_server.app as app_module

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    created = api_client.post("/ingestion/jobs", json={"paths": [str(pdf)]}).json()

    # Stub only retry_failed_items so we don't need a real failed item
    app_module.app.state.job_store.retry_failed_items = AsyncMock(return_value=2)

    resp = api_client.post(f"/ingestion/jobs/{created['job_id']}/retry")

    assert resp.status_code == 200
    assert resp.json()["retried_items"] == 2


def test_ingestion_allowed_roots_blocks_outside_path(api_client, tmp_path, monkeypatch):
    """When ingestion_allowed_roots is configured, a path outside every root
    must be rejected with 403 — prevents LLM-driven arbitrary-file ingestion."""
    import kb.tool_server.app as app_module
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(app_module.settings, "ingestion_allowed_roots", str(allowed))

    outside_pdf = tmp_path / "outside.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4 fake")

    resp = api_client.post(
        "/ingestion/jobs",
        json={"paths": [str(outside_pdf)]},
    )
    assert resp.status_code == 403
    assert "outside" in resp.json()["detail"].lower()


def test_ingestion_allowed_roots_accepts_inside_path(api_client, tmp_path, monkeypatch):
    """A path under the allowed root passes."""
    import kb.tool_server.app as app_module
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(app_module.settings, "ingestion_allowed_roots", str(allowed))

    inside_pdf = allowed / "good.pdf"
    inside_pdf.write_bytes(b"%PDF-1.4 fake")

    resp = api_client.post(
        "/ingestion/jobs",
        json={"paths": [str(inside_pdf)]},
    )
    assert resp.status_code == 200


def test_ingestion_allowed_roots_empty_disables_guard(api_client, tmp_path, monkeypatch):
    """Empty allowed_roots config means guard is off — historical behavior."""
    import kb.tool_server.app as app_module
    monkeypatch.setattr(app_module.settings, "ingestion_allowed_roots", "")

    pdf = tmp_path / "anywhere.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    resp = api_client.post(
        "/ingestion/jobs",
        json={"paths": [str(pdf)]},
    )
    assert resp.status_code == 200
