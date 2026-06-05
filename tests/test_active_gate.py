"""I-6 active gate: read paths must not surface deleted/deprecated docs.

The invariant: any document whose ``status`` is not ``"active"`` is
hidden from every read path — GET /documents/{id}, kb_list MCP tool,
kb_get_toc MCP tool, SearchEngine.search (default flags), and the
nav search engine. The row itself stays in the DB so the audit trail
and any restore tooling can still see it.

This file verifies the gate end-to-end through the real FastAPI +
FastMCP lifespan (``test_app_with_users`` fixture). Per-component
unit tests live in ``test_mcp_nav_tools.py`` (kb_get_toc),
``test_search.py`` (SearchEngine), and ``test_mcp_resources.py``
(_fetch_page_payload); this file is the cross-cutting safety net so
a regression in any one path is caught by at least one assertion.
"""
from __future__ import annotations

import pytest

from tests._mcp_helpers import call_mcp_tool


async def _call_mcp_tool(client, token, tool_name, arguments):
    """Thin wrapper around the shared helper — adapts ``client.app``.

    Mirrors the wrapper in ``tests/test_mcp_nav_permissions.py`` so the
    call shape is identical across the active-gate, nav-permissions,
    and auth-middleware suites."""
    return await call_mcp_tool(client.app, token, tool_name, arguments)


@pytest.mark.asyncio
async def test_get_document_returns_404_for_deleted(test_app_with_users):
    """DELETE /documents/{id} sets status='deleted'; the subsequent GET
    must return 404. alice owns doc-alice-1 so the DELETE itself is
    authorized (no 403 confounding the test)."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    del_resp = client.delete(
        "/documents/doc-alice-1", headers={"X-API-Key": alice_token},
    )
    assert del_resp.status_code == 200, del_resp.text
    r = client.get("/documents/doc-alice-1", headers={"X-API-Key": alice_token})
    assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_kb_list_excludes_deleted_docs(test_app_with_users):
    """kb_list MCP tool iterates ``meta_db.list_active_doc_ids()`` so a
    soft-deleted doc disappears from its result on the next call.

    Sequence:
      1. call kb_list → returns N docs including doc-alice-1
      2. DELETE /documents/doc-alice-1 (alice owns → 200)
      3. call kb_list again → doc-alice-1 NOT in the doc set
    """
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]

    resp1 = await _call_mcp_tool(client, alice_token, "kb_list", {})
    structured1 = resp1["result"]["structuredContent"]
    docs_before = {d["doc_id"] for d in structured1["documents"]}
    assert "doc-alice-1" in docs_before

    del_resp = client.delete(
        "/documents/doc-alice-1", headers={"X-API-Key": alice_token},
    )
    assert del_resp.status_code == 200, del_resp.text

    resp2 = await _call_mcp_tool(client, alice_token, "kb_list", {})
    structured2 = resp2["result"]["structuredContent"]
    docs_after = {d["doc_id"] for d in structured2["documents"]}
    assert "doc-alice-1" not in docs_after
    assert len(docs_after) == len(docs_before) - 1


@pytest.mark.asyncio
async def test_kb_get_toc_marks_deleted_doc_not_active(test_app_with_users):
    """kb_get_toc must surface a ``doc_not_active`` marker for deleted
    docs even if nav entries linger (nav cleanup is best-effort in the
    DELETE handler — see documents_api.py). Pre-T3 kb_get_toc only
    checked ``nav_store is None``; T3 adds the explicit status gate."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]

    del_resp = client.delete(
        "/documents/doc-alice-1", headers={"X-API-Key": alice_token},
    )
    assert del_resp.status_code == 200, del_resp.text

    resp = await _call_mcp_tool(
        client, alice_token, "kb_get_toc", {"doc_id": "doc-alice-1"},
    )
    structured = resp["result"]["structuredContent"]
    assert structured["entries"] == []
    assert structured.get("reason") == "doc_not_active"


@pytest.mark.asyncio
async def test_search_engine_excludes_deprecated_by_default(test_app_with_users):
    """``SearchEngine.search`` defaults to ``include_deprecated=False``.

    The retriever short-circuits when ``list_valid_doc_ids()`` is empty
    (retriever.py:50-67) so after deleting the only seeded doc the call
    returns an empty result set without ever touching the embedding
    models — keeps this test fast and free of GPU/IO dependencies.

    Engine.search signature returns ``tuple[SearchResponse, bool]``;
    SearchResponse.results is the list of SearchResult dataclasses
    (kb/models.py:82)."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    engine = client.app.state.engine
    assert engine is not None, "lifespan must bind app.state.engine"

    del_resp = client.delete(
        "/documents/doc-alice-1", headers={"X-API-Key": alice_token},
    )
    assert del_resp.status_code == 200, del_resp.text

    response, _cache_hit = await engine.search(query="anything", top_k=20)
    # Vacuous-truth: with the seeded doc deleted the retriever returns
    # no hits at all. The point of the assertion is that IF anything
    # ever leaks past the gate it would surface as a result_doc_id
    # that equals the deleted doc — which we forbid.
    assert all(r.doc_id != "doc-alice-1" for r in response.results)


def test_search_docs_rejects_include_deprecated_for_members(test_app_with_users):
    """Security review P0 #1: include_deprecated=True bypasses the
    active gate by setting doc_ids=None in the retriever. A member who
    sets that flag could otherwise pull soft-deleted vectors during
    the T5 retention window. Endpoint must 403 for non-admin callers."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]

    resp = client.post(
        "/search_docs",
        headers={"X-API-Key": alice_token},
        json={"query": "secret leaked content", "include_deprecated": True},
    )
    assert resp.status_code == 403
    assert "include_deprecated" in resp.json()["detail"]


def test_search_docs_allows_include_deprecated_for_admin(test_app_with_users):
    """Admin keeps the forensic capability — same payload, different role,
    must NOT 403 (engine may still return an empty result set in this
    minimal fixture, but the gate must not refuse the call)."""
    setup = test_app_with_users
    client, admin_token = setup["client"], setup["admin_token"]

    resp = client.post(
        "/search_docs",
        headers={"X-API-Key": admin_token},
        json={"query": "anything", "include_deprecated": True},
    )
    assert resp.status_code == 200


def test_search_docs_default_path_unchanged_for_member(test_app_with_users):
    """include_deprecated=False (the default) must keep working for
    members — the security gate fires only on True."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]

    resp = client.post(
        "/search_docs",
        headers={"X-API-Key": alice_token},
        json={"query": "anything"},  # include_deprecated defaults to False
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_get_document_404_for_deprecated(test_app_with_users):
    """The active-gate covers *any* non-active status, not just deleted.
    Mark the doc deprecated directly via MetadataDB (skips the DELETE
    handler so we exercise the GET filter on the deprecated branch in
    isolation)."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    meta_db = client.app.state.meta_db
    await meta_db.deprecate_document("doc-alice-1")
    r = client.get("/documents/doc-alice-1", headers={"X-API-Key": alice_token})
    assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_delete_invalidates_search_cache(test_app_with_users):
    """T3 follow-up (P1b review): a doc that becomes non-active must not
    keep surfacing in cached search responses for 5 minutes.

    Reproducer: prime the cache with a query, soft-delete the doc, run
    the same query again — the second call must NOT return the deleted
    doc. Pre-fix the cache hit short-circuited the retriever before its
    active-doc filter ran."""
    from kb.models import SearchResult, SearchResponse
    from kb.search.cache import _cache, cache_key, set_cached

    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]

    # Plant a cache entry mentioning doc-alice-1. We bypass the real
    # SearchEngine to keep the test independent of embeddings.
    key = cache_key(
        query="approval", doc_filter={},
        include_deprecated=False, top_k=5,
        hyde=False, multi_query_enabled=False, multi_query_n=1,
    )
    set_cached(key, SearchResponse(
        results=[SearchResult(
            doc_id="doc-alice-1", doc_name="alice-manual.pdf",
            page_num=1, heading_path=[],
            text_chunk="approval flow",
            vision_description=None,
            figure_index=None, figure_caption=None,
            image_url="",
            page_type="text",
            doc_version=None, effective_date=None,
            score=0.9, match_channels=["dense"],
        )],
        total=1,
    ))
    assert _cache.get(key) is not None, "fixture: cache primed"

    # Soft-delete the doc via the real DELETE handler. Per T3-P1b fix, the
    # handler now invalidates the search cache after mark_deleted.
    r = client.delete("/documents/doc-alice-1", headers={"X-API-Key": alice_token})
    assert r.status_code == 200, r.text
    assert _cache.get(key) is None, (
        "search cache must drop the entry after mark_deleted — "
        "without this, a deleted doc would surface in cache hits for "
        "up to the TTL window"
    )


@pytest.mark.asyncio
async def test_pipeline_deprecate_set_filters_by_owner(tmp_path):
    """T3 P0 review fix: ingestion's same-name auto-deprecate must scope
    to the SAME owner. Without this filter, Bob uploading a file with the
    same name as Alice's would silently deprecate Alice's content —
    violating "大锅饭只覆盖 read，不覆盖 write".

    Reproduces the predicate from src/kb/ingestion/pipeline.py around the
    ``old_doc_ids`` collection (post-fix). If pipeline's predicate diverges
    from this test's logic in the future, both must update together — the
    test is the contract.
    """
    from kb.storage.metadata_db import MetadataDB

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await meta.init()
    try:
        # Seed: alice's manual.pdf, bob's manual.pdf (same name, different owner),
        # alice's other.pdf (different name, same owner)
        for doc_id, name, owner in [
            ("alice-manual", "manual.pdf", "u-alice"),
            ("bob-manual",   "manual.pdf", "u-bob"),
            ("alice-other",  "other.pdf",  "u-alice"),
        ]:
            await meta.upsert_document(
                doc_id=doc_id, doc_name=name,
                version=None, effective_date=None,
                expiry_date=None, supersedes=None,
                owner_id=owner,
            )

        async def deprecate_set(new_doc_id: str, new_doc_name: str, new_owner: str | None):
            """Mimic the pipeline.ingest predicate (post-T3-P0 fix)."""
            all_active = await meta.list_active_doc_ids()
            matches: list[str] = []
            for old_id in all_active:
                rec = await meta.get_document(old_id)
                if (
                    rec
                    and rec["doc_name"] == new_doc_name
                    and old_id != new_doc_id
                    and rec.get("owner_id") == new_owner
                ):
                    matches.append(old_id)
            return matches

        # Alice re-uploads manual.pdf → only her own old version is deprecated
        assert await deprecate_set("alice-manual-v2", "manual.pdf", "u-alice") == ["alice-manual"]
        # Bob re-uploads manual.pdf → only his own old version is deprecated
        assert await deprecate_set("bob-manual-v2",   "manual.pdf", "u-bob")   == ["bob-manual"]
        # Carol (a new owner) uploads manual.pdf → NO existing doc is touched
        assert await deprecate_set("carol-manual",    "manual.pdf", "u-carol") == []
        # CLI ingest (owner_id=None) → matches no real-user-owned docs
        assert await deprecate_set("cli-manual",      "manual.pdf", None)      == []
    finally:
        await meta.aclose()


@pytest.mark.asyncio
async def test_nav_write_tools_refuse_non_active_doc(test_app_with_users, monkeypatch):
    """T3 follow-up (P2b review): the 3 MCP nav write tools must refuse
    mutation on docs whose status != active, even when the caller is the
    legitimate owner. Otherwise a deleted/deprecated doc could grow new
    aliases or have its nav rebuilt, re-surfacing orphan content via
    stale URIs.

    Tests kb_rebuild_nav_index against a deleted alice-owned doc — the
    owner-passes-permission-check path would have proceeded pre-fix.
    """
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    # The Settings singleton is loaded at module import — env vars set
    # later in the test don't take effect. Patch the live state.settings
    # directly so the _write_tools_enabled gate lets us reach the new
    # P2b active-gate check.
    from kb.tool_server.mcp_state import state as mcp_state
    monkeypatch.setattr(mcp_state.settings, "mcp_enable_write_tools", True)

    # Delete doc-alice-1 first (alice owns it, so the DELETE goes through)
    r = client.delete("/documents/doc-alice-1", headers={"X-API-Key": alice_token})
    assert r.status_code == 200, r.text
    # Now alice tries to rebuild nav on her own (just-deleted) doc.
    # Pre-fix: owner check passes → rebuild runs against deleted-doc nav rows.
    # Post-fix: doc.status != active → reason=doc_not_active, no rebuild.
    resp = await call_mcp_tool(
        client.app, alice_token,
        "kb_rebuild_nav_index", {"doc_id": "doc-alice-1"},
    )
    structured = resp["result"]["structuredContent"]
    assert structured.get("reason") == "doc_not_active", structured
    assert structured.get("started") is False, structured
