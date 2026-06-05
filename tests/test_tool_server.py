# tests/test_tool_server.py
import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Observability ──────────────────────────────────────────────────────────────

def test_log_search_emits_json(caplog):
    from kb.tool_server.observability import log_search
    with caplog.at_level(logging.INFO, logger="kb.observability"):
        log_search(
            query="审批流程",
            latency_ms=123.4,
            channels_used=["dense", "sparse"],
            result_count=3,
            top_score=0.91,
            hyde_triggered=False,
            cache_hit=True,
        )
    assert len(caplog.records) == 1
    entry = json.loads(caplog.records[0].message)
    assert entry["query"] == "审批流程"
    assert entry["result_count"] == 3
    assert entry["cache_hit"] is True
    assert "timestamp" in entry


def test_log_search_truncates_long_query(caplog):
    from kb.tool_server.observability import log_search
    with caplog.at_level(logging.INFO, logger="kb.observability"):
        log_search(
            query="x" * 300, latency_ms=10.0,
            channels_used=[], result_count=0, top_score=0.0,
            hyde_triggered=False, cache_hit=False,
        )
    entry = json.loads(caplog.records[0].message)
    assert len(entry["query"]) <= 200


# ── FastAPI endpoints ──────────────────────────────────────────────────────────

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
    # engine.search now returns (SearchResponse, cache_hit)
    mock_engine.search = AsyncMock(
        return_value=(SearchResponse(results=[], total=0), False)
    )
    mock_meta_db = MagicMock()
    mock_meta_db.init = AsyncMock()
    mock_meta_db.get_document = AsyncMock(return_value={
        "doc_id": "abc", "doc_name": "manual.pdf", "version": "v1",
        "status": "active", "effective_date": None, "expiry_date": None,
        # T5-4: new handler calls get_document(include_deleted_at=True) at the end
        # of DELETE to return deleted_at in the response body.
        "deleted_at": "2026-05-20T00:00:00",
    })
    mock_meta_db.list_active_doc_ids = AsyncMock(return_value=["abc"])
    mock_meta_db.mark_deleted = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.ensure_collections = AsyncMock()
    mock_qdrant.delete_by_doc_id = AsyncMock()
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
            # Expose the token on the client so tests that want to assert
            # the happy-path (e.g. correct token → 200) can access it
            # without re-deriving from the fixture's local scope.
            client.admin_token = admin_token  # type: ignore[attr-defined]
            yield client


def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_docs_rejects_missing_api_key(api_client):
    # T1b: auth is user-token based; invalid/missing token → 401/403.
    # api_client has a default X-API-Key header (valid admin token); override
    # with an obviously-invalid value to simulate a request with no valid token.
    resp = api_client.post(
        "/search_docs",
        json={"query": "approval"},
        headers={"X-API-Key": "kb_invalid_token_no_such_user"},
    )
    assert resp.status_code in (401, 403)


def test_search_docs_accepts_correct_api_key(api_client):
    # T1b: explicit happy-path assertion that a valid admin token in the
    # Authorization: Bearer header is accepted (the default fixture header
    # is X-API-Key, so this also exercises the Bearer fallback path).
    resp = api_client.post(
        "/search_docs",
        json={"query": "approval"},
        headers={"Authorization": f"Bearer {api_client.admin_token}"},
    )
    assert resp.status_code == 200


def test_search_docs_rejects_wrong_api_key(api_client):
    # T1b: invalid token → 401/403.
    # Override the default X-API-Key with an invalid value to ensure the
    # middleware rejects it rather than falling through to the default token.
    resp = api_client.post(
        "/search_docs",
        json={"query": "approval"},
        headers={"X-API-Key": "kb_wrong_token_not_in_db"},
    )
    assert resp.status_code in (401, 403)


def test_health_does_not_require_api_key(api_client):
    # /health is on the middleware allowlist and does not require auth
    resp = api_client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_verify_api_key_rejects_when_no_current_user():
    from fastapi import HTTPException
    from kb.auth.context import current_user
    from kb.tool_server.security import verify_api_key

    # Ensure current_user is not set (simulate middleware bypass)
    token = current_user.set(None)
    try:
        with pytest.raises(HTTPException) as exc:
            await verify_api_key()
        assert exc.value.status_code == 401
    finally:
        current_user.reset(token)


def test_search_docs_returns_200(api_client):
    resp = api_client.post("/search_docs", json={"query": "审批流程", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert "total" in body


def test_search_rejects_invalid_top_k(api_client):
    resp = api_client.post("/search_docs", json={"query": "q", "top_k": 0})
    assert resp.status_code == 422


def test_search_rejects_excessive_top_k(api_client):
    resp = api_client.post("/search_docs", json={"query": "q", "top_k": 99999})
    assert resp.status_code == 422


def test_search_rejects_unknown_filter(api_client):
    resp = api_client.post(
        "/search_docs",
        json={"query": "q", "doc_filter": {"unknown": "x"}},
    )
    assert resp.status_code == 422


def test_search_accepts_known_filter_and_passes_plain_dict(api_client):
    import kb.tool_server.app as app_module

    resp = api_client.post(
        "/search_docs",
        json={
            "query": "q",
            "doc_filter": {"doc_name": "manual.pdf", "page_type": "text"},
        },
    )
    assert resp.status_code == 200
    assert app_module.engine.search.await_args.kwargs["doc_filter"] == {
        "doc_name": "manual.pdf",
        "page_type": "text",
    }


def test_get_document_returns_doc(api_client):
    resp = api_client.get("/documents/abc")
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == "abc"


def test_get_document_404_when_missing(api_client):
    import kb.tool_server.app as app_module
    app_module.meta_db.get_document = AsyncMock(return_value=None)
    resp = api_client.get("/documents/nonexistent")
    assert resp.status_code == 404


def test_delete_document_returns_summary(api_client):
    """T5-4: DELETE is pure soft-delete. Response shape is now
    {doc_id, status, deleted_at} — no cleanup/nav stats (those belong
    to GC, see T5-7 / tests/test_gc_deleted_docs.py)."""
    import kb.tool_server.app as app_module

    resp = api_client.delete("/documents/abc")

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "abc"
    assert body["status"] == "deleted"
    assert "deleted_at" in body
    app_module.meta_db.mark_deleted.assert_awaited_once_with("abc")
    # T5-7 TODO: tests/test_gc_deleted_docs.py will assert
    # kb_store.cleanup_doc_assets and qdrant.delete_by_doc_id ARE called
    # during GC. Physical cleanup deliberately removed from this handler
    # (T5-4 refactor) — see documents_api.py docstring.


def test_delete_document_404_when_missing(api_client):
    import kb.tool_server.app as app_module

    app_module.meta_db.get_document = AsyncMock(return_value=None)
    resp = api_client.delete("/documents/missing")

    assert resp.status_code == 404


def test_list_docs_returns_list(api_client):
    resp = api_client.get("/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_ask_endpoint_returns_answer(api_client):
    """The /ask endpoint runs KBAgent and returns answer + citations + retrieved."""
    from kb.agent.citation import Citation

    fake_agent = MagicMock()
    fake_agent.answer = AsyncMock(return_value={
        "answer": "The CEO must approve [doc1:0].",
        "citations": [Citation(doc_id="doc1", page_num=0)],
        "citations_hallucinated": [],
        "retrieved": [("doc1", 0)],
        "query_id": "qid-1",
        "confidence": {"status": "grounded", "reasons": []},
    })

    with patch("kb.tool_server.ask_api.KBAgent", return_value=fake_agent), \
         patch("kb.adapters.openai_compat.OpenAICompatAdapter", return_value=MagicMock()):
        resp = api_client.post("/ask", json={
            "question": "Who approves >1M?",
            "adapter": "openai_compat",
            "openai_compat_base_url": "http://localhost:1234/v1",
            "openai_compat_api_key": "k",
            "openai_compat_model": "qwen3-32b",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert "approve" in body["answer"]
    assert body["query_id"] == "qid-1"
    assert body["confidence"] == {"status": "grounded", "reasons": []}
    assert body["citations_hallucinated"] == []
    assert body["citations"] == [{"doc_id": "doc1", "page_num": 0}]
    assert body["retrieved"] == [["doc1", 0]]


def test_ask_endpoint_closes_request_adapter(api_client):
    fake_adapter = MagicMock()
    fake_adapter.aclose = AsyncMock()
    fake_agent = MagicMock()
    fake_agent.answer = AsyncMock(return_value={
        "answer": "answer", "citations": [], "retrieved": [],
    })

    with patch("kb.adapters.factory.make_chat_adapter", return_value=fake_adapter), \
         patch("kb.tool_server.ask_api.KBAgent", return_value=fake_agent):
        resp = api_client.post("/ask", json={
            "question": "q",
            "adapter": "openai_compat",
            "openai_compat_base_url": "http://localhost:1234/v1",
            "openai_compat_api_key": "k",
            "openai_compat_model": "m",
        })

    assert resp.status_code == 200
    fake_adapter.aclose.assert_awaited_once()


def test_ask_rejects_excessive_iterations(api_client):
    fake_agent = MagicMock()
    fake_agent.answer = AsyncMock(return_value={
        "answer": "answer", "citations": [], "retrieved": [],
    })
    with patch("kb.tool_server.ask_api.KBAgent", return_value=fake_agent), \
         patch("kb.adapters.openai_compat.OpenAICompatAdapter", return_value=MagicMock()):
        resp = api_client.post(
            "/ask",
            json={"question": "q", "max_iterations": 999},
        )
    assert resp.status_code == 422


def test_ask_endpoint_rejects_unknown_adapter(api_client):
    resp = api_client.post("/ask", json={
        "question": "anything?",
        "adapter": "copilot",   # 现在 copilot 是 schema 层面的非法值
    })
    assert resp.status_code == 422
    assert "adapter" in str(resp.json()["detail"]).lower()


def test_ask_endpoint_openai_compat_dispatch(api_client):
    """Valid openai_compat config should construct an OpenAICompatAdapter
    and run the agent without error."""
    from kb.agent.citation import Citation

    fake_agent = MagicMock()
    fake_agent.answer = AsyncMock(return_value={
        "answer": "answer", "citations": [], "retrieved": [],
    })
    with patch("kb.tool_server.ask_api.KBAgent", return_value=fake_agent), \
         patch("kb.adapters.openai_compat.OpenAICompatAdapter",
               return_value=MagicMock()) as mock_cls:
        resp = api_client.post("/ask", json={
            "question": "q",
            "adapter": "openai_compat",
            "openai_compat_base_url": "https://api.x.com/v1",
            "openai_compat_api_key": "k",
            "openai_compat_model": "m",
        })
    assert resp.status_code == 200
    mock_cls.assert_called_once()


def test_ask_endpoint_openai_compat_falls_back_to_settings(api_client):
    """When openai_compat_* fields are omitted, _build_adapter should use Settings agent_* values."""
    from kb.agent.citation import Citation

    fake_agent = MagicMock()
    fake_agent.answer = AsyncMock(return_value={
        "answer": "answer", "citations": [], "retrieved": [],
    })
    with patch("kb.tool_server.ask_api.KBAgent", return_value=fake_agent), \
         patch("kb.adapters.openai_compat.OpenAICompatAdapter",
               return_value=MagicMock()) as mock_cls:
        resp = api_client.post("/ask", json={
            "question": "q",
            "adapter": "openai_compat",
            # 不传 openai_compat_base_url / api_key / model — 让 settings 提供缺省
        })
    assert resp.status_code == 200
    # 验证 OpenAICompatAdapter 用 settings 的缺省值构造
    mock_cls.assert_called_once()
    kwargs = mock_cls.call_args.kwargs
    # api_client fixture 用的是默认 Settings，所以 agent_url / agent_api_key / agent_model 走默认
    assert kwargs["base_url"] == "http://localhost:1235/v1"
    assert kwargs["api_key"] == "lm-studio"
    assert kwargs["model"] == "qwen3.6-27b"
    assert kwargs["supports_vision"] is True


def test_feedback_endpoint_records_user_feedback(api_client):
    import kb.tool_server.app as app_module

    app_module.app.state.feedback_db.update_feedback = AsyncMock(return_value=True)

    resp = api_client.post("/feedback", json={
        "query_id": "qid-1",
        "was_helpful": False,
        "comment": "wrong section",
        "expected_doc": "manual.pdf",
    })

    assert resp.status_code == 200
    assert resp.json()["updated"] is True
    app_module.app.state.feedback_db.update_feedback.assert_awaited_once_with(
        query_id="qid-1",
        was_helpful=False,
        comment="wrong section",
        expected_doc="manual.pdf",
    )
