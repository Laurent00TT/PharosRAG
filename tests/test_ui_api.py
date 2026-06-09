import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb.models import SearchResponse, SearchResult
from kb.nav.models import NavEntry


def _nav_entry(doc_id: str = "doc-1", entry_id: str = "entry-approval") -> NavEntry:
    return NavEntry(
        entry_id=entry_id,
        label="Approval Flow",
        normalized_label="approval flow",
        entry_type="section",
        source="parser_heading",
        doc_id=doc_id,
        doc_name="manual.pdf",
        page_start=2,
        page_end=4,
        order_index=1,
        parent_entry_id=None,
        resource_uris=[f"kb://documents/{doc_id}/ranges/2-4"],
        source_ref_ids=[],
    )


@pytest.fixture
def ui_api_client(tmp_path):
    from fastapi.testclient import TestClient

    from kb.auth.users import UsersStore
    from kb.config import Settings
    from kb.storage.metadata_db import MetadataDB
    from kb.tool_server import app as app_module

    qd = tmp_path / "qd"
    qd.mkdir()
    settings = Settings(
        qdrant_path=str(qd),
        image_storage_path=str(tmp_path / "images"),
        nav_db_path=str(tmp_path / "nav_index.db"),
    )
    db_url = f"sqlite+aiosqlite:///{qd}/kb_metadata.db"

    async def _seed():
        meta = MetadataDB(db_url=db_url)
        await meta.init()
        await meta.upsert_document(
            doc_id="doc-1",
            doc_name="manual.pdf",
            version="v1",
            effective_date=None,
            expiry_date=None,
            supersedes=None,
            owner_id="placeholder",
        )
        await meta.upsert_document(
            doc_id="old-doc",
            doc_name="old.pdf",
            version=None,
            effective_date=None,
            expiry_date=None,
            supersedes=None,
            status="deleted",
        )
        store = UsersStore(db_url=db_url)
        await store.init()
        admin_token, admin = await store.create_user(username="admin0", role="admin")
        member_token, member = await store.create_user(username="alice", role="member")
        async with meta._engine.begin() as conn:
            await conn.exec_driver_sql(
                f"UPDATE documents SET owner_id='{member.user_id}' WHERE doc_id='doc-1'"
            )
        await store.aclose()
        return meta, admin_token, member_token, member.user_id

    meta_db, admin_token, member_token, member_user_id = asyncio.run(_seed())

    qdrant = MagicMock()
    qdrant.ensure_collections = AsyncMock()
    qdrant.get_parent_chunk = MagicMock(
        return_value={
            "doc_name": "manual.pdf",
            "text": "Approval must be completed within three working days.",
            "heading_path": ["Approval", "Flow"],
        }
    )
    qdrant.get_desc_record = MagicMock(
        return_value={
            "description": "Generated visual hint, not citable.",
            "figure_caption": "Approval table",
            "figure_index": "fig-1",
            "image_url": "local://doc-1/page-2.png",
            "page_type": "mixed",
        }
    )
    qdrant.get_vision_record = MagicMock(return_value=None)

    engine = MagicMock()
    engine.search = AsyncMock(
        return_value=(
            SearchResponse(
                results=[
                    SearchResult(
                        doc_id="doc-1",
                        doc_name="manual.pdf",
                        page_num=2,
                        heading_path=["Approval", "Flow"],
                        text_chunk="This text must not leak into hybrid pointer hits.",
                        vision_description="Hint text must not leak either.",
                        figure_index=None,
                        figure_caption=None,
                        image_url="",
                        page_type="text",
                        doc_version="v1",
                        effective_date=None,
                        score=0.91,
                        match_channels=["dense"],
                    )
                ],
                total=1,
            ),
            False,
        )
    )
    feedback_db = MagicMock()
    feedback_db.init = AsyncMock()
    feedback_db.update_feedback = AsyncMock(return_value=True)

    with patch.object(app_module, "settings", settings), \
         patch("kb.tool_server.app.QdrantStore", return_value=qdrant), \
         patch("kb.tool_server.app.MetadataDB", return_value=meta_db), \
         patch("kb.tool_server.app.UsageFeedbackDB", return_value=feedback_db), \
         patch("kb.tool_server.app.SearchEngine", return_value=engine):
        with TestClient(app_module.app) as client:
            asyncio.run(client.app.state.nav_store.upsert_entries([_nav_entry()]))
            yield {
                "client": client,
                "admin_token": admin_token,
                "member_token": member_token,
                "member_user_id": member_user_id,
                "settings": settings,
                "feedback_db": feedback_db,
            }


def test_ui_me_returns_authenticated_user(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].get(
        "/ui/api/me",
        headers={"Authorization": f"Bearer {ctx['member_token']}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "alice"
    assert body["role"] == "member"


def test_ui_documents_include_nav_coverage(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].get(
        "/ui/api/documents",
        headers={"X-API-Key": ctx["member_token"]},
    )
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert len(docs) == 1
    doc = docs[0]
    expected = {
        "doc_id": "doc-1",
        "doc_name": "manual.pdf",
        "version": "v1",
        "status": "active",
        "owner_id": ctx["member_user_id"],
        "nav_indexed": True,
        "nav_entry_count": 1,
        "resource_uri": "kb://documents/doc-1",
    }
    for key, value in expected.items():
        assert doc[key] == value, f"mismatch on {key}: {doc.get(key)!r} vs {value!r}"
    # ingested_at is populated by the metadata DB on insert; just verify shape.
    assert doc["ingested_at"] is None or isinstance(doc["ingested_at"], str)


def test_ui_status_and_documents_ignore_deleted_doc_nav_entries(ui_api_client):
    ctx = ui_api_client
    asyncio.run(ctx["client"].app.state.nav_store.upsert_entries([
        _nav_entry(doc_id="old-doc", entry_id="entry-old"),
    ]))

    status_resp = ctx["client"].get(
        "/ui/api/status",
        headers={"X-API-Key": ctx["member_token"]},
    )
    docs_resp = ctx["client"].get(
        "/ui/api/documents",
        headers={"X-API-Key": ctx["member_token"]},
    )

    assert status_resp.status_code == 200
    assert status_resp.json()["nav_entry_count"] == 1
    assert status_resp.json()["nav_doc_count"] == 1
    assert docs_resp.status_code == 200
    assert docs_resp.json()["documents"][0]["nav_entry_count"] == 1


def test_ui_toc_returns_pointer_only_entries(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].get(
        "/ui/api/documents/doc-1/toc",
        headers={"X-API-Key": ctx["member_token"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["contains_generated_content"] is False
    assert body["entries"][0]["resource_uris"] == ["kb://documents/doc-1/ranges/2-4"]
    assert "summary" not in body["entries"][0]
    assert "content_markdown" not in body["entries"][0]


def test_ui_hybrid_search_is_pointer_only(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].post(
        "/ui/api/hybrid_search",
        json={"query": "approval", "top_k": 5},
        headers={"X-API-Key": ctx["member_token"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["contains_generated_content"] is False
    assert body["evidence_hits"][0] == {
        "doc_id": "doc-1",
        "doc_name": "manual.pdf",
        "page_num": 2,
        "resource_uri": "kb://documents/doc-1/pages/2",
        "score": 0.91,
        "heading_path": ["Approval", "Flow"],
        "match_channels": ["dense"],
        "page_type": "text",
        "section": "Approval / Flow",
        "text_preview": "Approval must be completed within three working days.",
    }
    assert body["nav_hits"][0]["label"] == "Approval Flow"
    assert body["suggested_fetches"][0]["resource_uri"] == "kb://documents/doc-1/ranges/2-4"
    assert "Hint text must not leak" not in str(body)


def test_ui_page_preview_partitions_evidence_and_hints(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].get(
        "/ui/api/documents/doc-1/pages/2",
        headers={"X-API-Key": ctx["member_token"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["evidence"]["text"].startswith("Approval must")
    assert body["evidence"]["figure_caption"] == "Approval table"
    assert body["hints"]["generated_description"] == "Generated visual hint, not citable."
    assert "generated_description" not in body["evidence"]


def test_ui_page_preview_hides_non_active_doc(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].get(
        "/ui/api/documents/old-doc/pages/0",
        headers={"X-API-Key": ctx["member_token"]},
    )
    assert resp.status_code == 404


def test_ui_range_preview_returns_bounded_active_doc_pages(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].post(
        "/ui/api/documents/doc-1/range_preview",
        headers={"X-API-Key": ctx["member_token"]},
        json={"page_start": 2, "page_end": 4},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["page_start"] == 2
    assert body["page_end"] == 4
    assert [page["page_num"] for page in body["pages"]] == [2, 3, 4]


def test_ui_feedback_reuses_existing_payload_validation(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].post(
        "/ui/api/feedback",
        headers={"X-API-Key": ctx["member_token"]},
        json={"query_id": "q-1", "was_helpful": False},
    )
    assert resp.status_code == 200
    ctx["feedback_db"].update_feedback.assert_awaited_once_with(
        query_id="q-1",
        was_helpful=False,
        comment="",
        expected_doc="",
    )


def test_ui_upload_creates_owned_ingestion_job(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].post(
        "/ui/api/upload",
        headers={"X-API-Key": ctx["member_token"]},
        files={"file": ("policy.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job"]["status"] == "queued"
    assert body["job"]["owner_id"] == ctx["member_user_id"]
    stored_path = Path(body["stored_path"])
    assert stored_path.exists()
    # _safe_upload_dest nests each upload under a per-upload <uuid>/ dir, so
    # ui_uploads is the GRANDPARENT (parent is the uuid hex dir).
    assert stored_path.parent.parent.name == "ui_uploads"
    assert len(stored_path.parent.name) == 32  # uuid4().hex
    assert stored_path.suffix == ".pdf"


def test_ui_upload_rejects_renamed_non_pdf(ui_api_client):
    ctx = ui_api_client
    resp = ctx["client"].post(
        "/ui/api/upload",
        headers={"X-API-Key": ctx["member_token"]},
        files={"file": ("not-a-pdf.pdf", b"plain text", "application/pdf")},
    )

    assert resp.status_code == 400
    assert "valid PDF" in resp.json()["detail"]
