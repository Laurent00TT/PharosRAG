import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp.server.fastmcp import FastMCP

from kb.tool_server.mcp_resources import (
    UNTRUSTED_SOURCE_HEADER,
    _page_looks_continued,
    parse_kb_uri,
    register_document_resources,
)


def _fake_qdrant(*, text_rec=None, desc_rec=None, vision_rec=None):
    """Build a fake QdrantStore with the three page getters wired to fixed
    records — matches what enrich_results / _fetch_page_payload uses."""
    qdrant = MagicMock()
    qdrant.get_parent_chunk = MagicMock(return_value=text_rec)
    qdrant.get_desc_record = MagicMock(return_value=desc_rec)
    qdrant.get_vision_record = MagicMock(return_value=vision_rec)
    return qdrant


class _FakeImageStore:
    def __init__(self, path: Path):
        self._path = path

    async def get_url(self, doc_id, page_num):
        return str(self._path)


def test_parse_documents_uri():
    parsed = parse_kb_uri("kb://documents")
    assert parsed.kind == "documents"


def test_parse_page_uri():
    parsed = parse_kb_uri("kb://documents/doc-1/pages/7")
    assert parsed.kind == "page"
    assert parsed.doc_id == "doc-1"
    assert parsed.page_num == 7


def test_parse_page_image_uri():
    parsed = parse_kb_uri("kb://documents/doc-1/pages/7/image")
    assert parsed.kind == "page_image"
    assert parsed.doc_id == "doc-1"
    assert parsed.page_num == 7


def test_parse_rejects_unknown_uri():
    with pytest.raises(ValueError):
        parse_kb_uri("file:///secret.txt")


def _make_state(*, docs_by_id: dict | None = None, meta_db: object | None = None):
    """Build a fake state matching MCPState's read surface for the resource handlers."""
    if meta_db is None and docs_by_id is not None:
        meta_db = SimpleNamespace(
            list_active_doc_ids=AsyncMock(return_value=list(docs_by_id.keys())),
            get_document=AsyncMock(side_effect=lambda doc_id: docs_by_id.get(doc_id)),
        )
    return SimpleNamespace(meta_db=meta_db)


def _get_handler(mcp: FastMCP, uri_template: str):
    """Pull the handler async function out of FastMCP by its URI template.

    FastMCP 1.27.x stores resources in two places:
      - _resource_manager._templates: keyed by uri_template string (for URIs with {placeholders})
      - _resource_manager._resources: keyed by AnyUrl (for concrete URIs)
    """
    rm = mcp._resource_manager
    for tmpl in rm._templates.values():
        if tmpl.uri_template == uri_template:
            return tmpl.fn
    for res in rm._resources.values():
        if str(res.uri) == uri_template or str(res.uri).rstrip("/") == uri_template:
            return res.fn
    raise KeyError(uri_template)


async def test_documents_resource_returns_doc_list():
    mcp = FastMCP("test")
    state = _make_state(docs_by_id={"a": {"doc_id": "a", "doc_name": "x.pdf"}})
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents")
    body = await fn()
    parsed = json.loads(body)
    assert parsed["documents"] == [{"doc_id": "a", "doc_name": "x.pdf"}]


async def test_document_resource_raises_when_doc_not_found():
    mcp = FastMCP("test")
    state = _make_state(docs_by_id={})  # no docs
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}")
    with pytest.raises(ValueError, match="Document not found"):
        await fn(doc_id="ghost")


async def test_document_resource_raises_when_meta_db_unbound():
    mcp = FastMCP("test")
    state = _make_state(meta_db=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}")
    with pytest.raises(RuntimeError, match="not initialized"):
        await fn(doc_id="any")


async def test_document_resource_returns_doc_not_active_for_deleted():
    """Review P2 #7: a stale URI for a soft-deleted doc must not leak
    metadata via kb://documents/{id} — return a doc_not_active envelope
    matching the page-resource behavior. Prevents agents with cached
    URIs from rediscovering doc names / versions after delete."""
    mcp = FastMCP("test")
    state = _make_state(docs_by_id={
        "deleted-doc": {"doc_id": "deleted-doc", "doc_name": "secret.pdf",
                        "status": "deleted"},
    })
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}")
    body = await fn(doc_id="deleted-doc")
    parsed = json.loads(body)
    assert parsed == {
        "doc_id": "deleted-doc",
        "status": "deleted",
        "reason": "doc_not_active",
    }
    # doc_name (the sensitive metadata) MUST not appear in the response.
    assert "doc_name" not in parsed
    assert "secret.pdf" not in body


async def test_document_resource_returns_doc_not_active_for_deprecated():
    """Same gate must catch 'deprecated' status (superseded older
    versions)."""
    mcp = FastMCP("test")
    state = _make_state(docs_by_id={
        "old-version": {"doc_id": "old-version", "doc_name": "x.pdf",
                        "status": "deprecated"},
    })
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}")
    body = await fn(doc_id="old-version")
    parsed = json.loads(body)
    assert parsed["reason"] == "doc_not_active"
    assert parsed["status"] == "deprecated"


async def test_document_page_resource_rejects_deprecated_doc():
    """Defense-in-depth: even if a stale URI points at a doc that was
    deprecated (and qdrant cleanup partially failed), the page resource
    must NOT return its content. NavSearch already filters by active,
    but URIs constructed by cached agents bypass nav."""
    mcp = FastMCP("test")
    deprecated_doc = {"doc_id": "old", "status": "deprecated"}
    meta_db = SimpleNamespace(
        get_document=AsyncMock(return_value=deprecated_doc),
    )
    # Use a qdrant that would WRONGLY return content if not gated upstream
    qdrant = _fake_qdrant(text_rec={"text": "stale content from old version"})
    state = SimpleNamespace(qdrant=qdrant, meta_db=meta_db, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="old", page_num="1")
    parsed = json.loads(body)
    assert parsed["status"] == "doc_not_active"
    assert parsed["doc_status"] == "deprecated"
    # Stale content must NOT leak
    assert "text" not in parsed or not parsed.get("text")
    # qdrant getters must NOT have been hit (gate runs first)
    qdrant.get_parent_chunk.assert_not_called()


async def test_document_page_resource_proceeds_when_meta_db_unavailable():
    """If meta_db is not bound (e.g. unit test, standalone embedding),
    the gate is skipped — preserves backward compat with the older
    no-meta-db handlers used by the in-process smoke."""
    mcp = FastMCP("test")
    qdrant = _fake_qdrant(text_rec={"text": "real content"})
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="any", page_num="0")
    parsed = json.loads(body)
    assert parsed["status"] == "ok"
    assert parsed["text"] == "real content"


async def test_document_page_resource_continues_when_meta_db_raises():
    """Transient meta_db errors must not block legit fetches. The gate
    fails open: if we can't determine active status, proceed to qdrant
    (NavSearch active filter is still the primary defense)."""
    mcp = FastMCP("test")
    meta_db = SimpleNamespace(get_document=AsyncMock(side_effect=RuntimeError("db hiccup")))
    qdrant = _fake_qdrant(text_rec={"text": "real content"})
    state = SimpleNamespace(qdrant=qdrant, meta_db=meta_db, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="any", page_num="0")
    parsed = json.loads(body)
    assert parsed["status"] == "ok"
    assert parsed["text"] == "real content"


async def test_document_page_resource_returns_real_text_when_chunk_exists():
    mcp = FastMCP("test")
    qdrant = _fake_qdrant(
        text_rec={
            "doc_name": "manual.pdf",
            "text": "Section 4.2 covers approval workflows.",
            "heading_path": ["Approvals", "Standard"],
        },
        desc_rec={"image_url": "/tmp/images/doc-1/page_7.png", "page_type": "mixed"},
    )
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="doc-1", page_num="7")
    parsed = json.loads(body)
    assert parsed["status"] == "ok"
    assert parsed["doc_id"] == "doc-1"
    assert parsed["page_num"] == 7
    assert parsed["safety"] == UNTRUSTED_SOURCE_HEADER
    assert parsed["text"] == "Section 4.2 covers approval workflows."
    assert parsed["heading_path"] == ["Approvals", "Standard"]
    assert parsed["image_resource_uri"] == "kb://documents/doc-1/pages/7/image"
    # Anti-regression: must NOT return the old placeholder marker
    assert parsed.get("phase") != "phase_1_metadata_only"
    assert parsed.get("status") != "placeholder"


async def test_document_page_resource_reports_page_not_found_when_no_chunks():
    mcp = FastMCP("test")
    qdrant = _fake_qdrant()  # all three getters return None
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="doc-1", page_num="99")
    parsed = json.loads(body)
    assert parsed["status"] == "page_not_found"
    assert parsed["safety"] == UNTRUSTED_SOURCE_HEADER


async def test_document_page_resource_reports_qdrant_not_initialized():
    mcp = FastMCP("test")
    state = SimpleNamespace(qdrant=None, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="doc-1", page_num="1")
    parsed = json.loads(body)
    assert parsed["status"] == "qdrant_not_initialized"


async def test_document_page_resource_truncates_oversized_text():
    mcp = FastMCP("test")
    qdrant = _fake_qdrant(text_rec={"text": "x" * 30_000})
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="doc-1", page_num="1")
    parsed = json.loads(body)
    assert parsed["text_truncated"] is True
    assert len(parsed["text"]) == 20_000


# ── #1 heuristic: _page_looks_continued ────────────────────────────────


def test_looks_continued_mid_sentence_is_continued():
    # The reported scenario: page ends mid-clause on a bare word.
    assert _page_looks_continued(
        "Net revenue rose, driven by the semiconductor segment, which"
    ) is True


def test_looks_continued_latin_sentence_end_is_complete():
    assert _page_looks_continued("Revenue grew 12% year over year.") is False


def test_looks_continued_cjk_sentence_end_is_complete():
    assert _page_looks_continued("净利润同比增长百分之十二。") is False


def test_looks_continued_cjk_comma_is_continued():
    assert _page_looks_continued("受半导体板块拉动，") is True


def test_looks_continued_trailing_colon_is_continued():
    # "as follows:" — the list almost certainly spills onto the next page.
    assert _page_looks_continued("The approval thresholds are as follows:") is True


def test_looks_continued_ellipsis_is_continued():
    # review P3: a trailing ellipsis (unicode … / ⋯ or ASCII "...") leans
    # "continued" under the eager calibration — it is not a full stop.
    assert _page_looks_continued("The terms continue…") is True
    assert _page_looks_continued("The terms continue⋯") is True
    assert _page_looks_continued("The terms continue...") is True


def test_looks_continued_single_period_stays_complete():
    # ...but a single period is still a full stop.
    assert _page_looks_continued("Done for the year.") is False


def test_looks_continued_peels_trailing_quote_then_complete():
    # Sentence-final mark wrapped in a closing quote still reads as complete.
    assert _page_looks_continued('The CFO said "we are on track."') is False


def test_looks_continued_blank_or_image_page_is_not_continued():
    assert _page_looks_continued("   \n  ") is False
    assert _page_looks_continued("") is False


# ── #1: page-adjacency navigation pointers ─────────────────────────────


async def test_document_page_resource_includes_navigation_pointers():
    """An interior page advertises both neighbours + a continuation flag so
    a consuming agent can move to prev/next without inferring the URI scheme."""
    mcp = FastMCP("test")
    qdrant = _fake_qdrant(text_rec={"text": "body that ends mid clause and"})
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="doc-1", page_num="7")
    nav = json.loads(body)["navigation"]
    assert nav["prev_page_uri"] == "kb://documents/doc-1/pages/6"
    assert nav["next_page_uri"] == "kb://documents/doc-1/pages/8"
    assert "looks_continued" in nav
    assert isinstance(nav["continuation_hint"], str) and nav["continuation_hint"]


async def test_document_page_resource_suppresses_prev_at_first_page():
    """Page 0 is the floor (0-based storage): no previous page to point at,
    but next is still optimistically advertised."""
    mcp = FastMCP("test")
    qdrant = _fake_qdrant(text_rec={"text": "first page."})
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    body = await fn(doc_id="doc-1", page_num="0")
    nav = json.loads(body)["navigation"]
    assert nav["prev_page_uri"] is None
    assert nav["next_page_uri"] == "kb://documents/doc-1/pages/1"


async def test_truncated_page_decouples_truncation_from_continuation():
    """review P2: the inline 20k cap (text_truncated) must NOT masquerade as a
    cross-page signal. looks_continued is judged on the FULL text, and the
    truncation case gets its own same-page hint pointing at the image/source,
    not next_page_uri."""
    mcp = FastMCP("test")
    # >20k chars, but the full page text ends on a full stop → self-contained.
    full = ("word " * 5000) + "end."
    qdrant = _fake_qdrant(text_rec={"text": full})
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    parsed = json.loads(await fn(doc_id="doc-1", page_num="2"))
    assert parsed["text_truncated"] is True
    # decoupled: full text ends on '.', so NOT flagged as cross-page continued
    assert parsed["navigation"]["looks_continued"] is False
    # truncation carries its own hint, pointing at the SAME-page full content
    hint = parsed["text_truncation_hint"]
    assert hint and ("image" in hint.lower() or "source" in hint.lower())


async def test_document_page_image_resource_returns_path_and_data_url(tmp_path):
    mcp = FastMCP("test")
    img_path = tmp_path / "page.png"
    img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # plausible PNG header + payload
    img_path.write_bytes(img_bytes)
    state = SimpleNamespace(
        qdrant=None, meta_db=None, image_store=_FakeImageStore(img_path),
    )
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}/image")
    body = await fn(doc_id="doc-1", page_num="7")
    parsed = json.loads(body)
    assert parsed["status"] == "ok"
    assert parsed["file_path"] == str(img_path)
    assert parsed["size_bytes"] == len(img_bytes)
    assert parsed["mime_type"] == "image/png"
    expected_b64 = base64.b64encode(img_bytes).decode("ascii")
    assert parsed["data_url"] == "data:image/png;base64," + expected_b64


async def test_document_page_image_resource_reports_image_not_found(tmp_path):
    mcp = FastMCP("test")
    state = SimpleNamespace(
        qdrant=None, meta_db=None,
        image_store=_FakeImageStore(tmp_path / "missing.png"),
    )
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}/image")
    body = await fn(doc_id="doc-1", page_num="7")
    parsed = json.loads(body)
    assert parsed["status"] == "image_not_found"
    assert "expected_path" in parsed


async def test_document_page_image_resource_skips_inline_for_oversize(tmp_path, monkeypatch):
    """Files over the inline cap return file_path but data_url is None."""
    mcp = FastMCP("test")
    big_path = tmp_path / "big.png"
    big_path.write_bytes(b"\x00" * 1024)  # 1KB, will be "big" after monkeypatch

    # Drop the cap to something tiny so we don't have to write a 5MB file.
    from kb.tool_server import mcp_resources as mr
    monkeypatch.setattr(mr, "_MAX_IMAGE_INLINE_BYTES", 100)

    state = SimpleNamespace(
        qdrant=None, meta_db=None, image_store=_FakeImageStore(big_path),
    )
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}/image")
    body = await fn(doc_id="doc-1", page_num="7")
    parsed = json.loads(body)
    assert parsed["status"] == "ok"
    assert parsed["data_url"] is None
    assert parsed["file_path"] == str(big_path)


# ── Task 4: parser tests for range / nav / reject ──────────────────────


def test_parse_page_range_uri():
    parsed = parse_kb_uri("kb://documents/doc/ranges/3-5")
    assert parsed.kind == "range"
    assert parsed.doc_id == "doc"
    assert parsed.page_start == 3
    assert parsed.page_end == 5


def test_parse_nav_entry_uri():
    parsed = parse_kb_uri("kb://nav/entries/entry-1")
    assert parsed.kind == "nav_entry"
    assert parsed.entry_id == "entry-1"


def test_reject_file_uri():
    with pytest.raises(ValueError):
        parse_kb_uri("file:///secret.txt")


# ── Task 4: range resource handler returns untrusted wrapper + pages ──


async def test_document_range_resource_returns_untrusted_wrapper_and_pages():
    mcp = FastMCP("test")
    qdrant = _fake_qdrant(
        text_rec={"text": "page body", "heading_path": ["H"]},
    )
    state = SimpleNamespace(qdrant=qdrant, meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/ranges/{range_spec}")
    body = await fn(doc_id="doc-1", range_spec="3-5")
    data = json.loads(body)
    assert data["safety"] == UNTRUSTED_SOURCE_HEADER
    assert data["doc_id"] == "doc-1"
    assert data["page_start"] == 3
    assert data["page_end"] == 5
    assert len(data["pages"]) == 3
    # Each page is now a full payload from _fetch_page_payload, not a stub.
    assert data["pages"][0]["status"] == "ok"
    assert data["pages"][0]["text"] == "page body"
    assert data["pages"][0]["image_resource_uri"] == "kb://documents/doc-1/pages/3/image"
    assert data["pages"][-1]["page_num"] == 5


async def test_document_range_resource_rejects_oversize_range():
    mcp = FastMCP("test")
    state = SimpleNamespace(qdrant=_fake_qdrant(), meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/ranges/{range_spec}")
    with pytest.raises(ValueError, match="range too large"):
        await fn(doc_id="doc-1", range_spec="0-99")  # 100 pages > 50 cap


async def test_document_range_resource_rejects_reversed_range():
    mcp = FastMCP("test")
    state = SimpleNamespace(qdrant=_fake_qdrant(), meta_db=None, image_store=None)
    register_document_resources(mcp, state)
    fn = _get_handler(mcp, "kb://documents/{doc_id}/ranges/{range_spec}")
    with pytest.raises(ValueError, match="page_start must be"):
        await fn(doc_id="doc-1", range_spec="5-3")


# ── Task 12: nav resources (kb://nav, kb://nav/entries/{id}, kb://documents/{id}/toc) ──


async def test_nav_root_resource_returns_pointer_list(tmp_path):
    from kb.nav.models import NavEntry
    from kb.nav.store import NavIndexStore
    from kb.tool_server.mcp_resources import register_nav_resources

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([NavEntry(
        entry_id="e1", label="Approval", normalized_label="approval",
        entry_type="section", source="parser_heading", doc_id="doc",
        doc_name="m.pdf", page_start=1, page_end=2, order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/ranges/1-2"],
        source_ref_ids=[],
    )])

    captured = {}

    class _FakeMCP:
        def resource(self, pattern):
            def deco(fn):
                captured[pattern] = fn
                return fn
            return deco

    state = SimpleNamespace(nav_store=store)
    register_nav_resources(_FakeMCP(), state)
    body = await captured["kb://nav"]()
    data = json.loads(body)
    assert data["safety"] == UNTRUSTED_SOURCE_HEADER
    assert data["total"] == 1
    assert data["entries"][0]["entry_id"] == "e1"
    # critical: no content_markdown / summary fields leak
    entry_keys = set(data["entries"][0].keys())
    assert not (entry_keys & {"content_markdown", "summary", "body"})
    await store.close()


async def test_document_toc_resource_returns_doc_entries(tmp_path):
    from kb.nav.models import NavEntry
    from kb.nav.store import NavIndexStore
    from kb.tool_server.mcp_resources import register_nav_resources

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([
        NavEntry(entry_id="e1", label="A", normalized_label="a", entry_type="section",
                 source="parser_heading", doc_id="doc-1", doc_name="m.pdf",
                 page_start=1, page_end=2, order_index=1, parent_entry_id=None,
                 resource_uris=["kb://documents/doc-1/ranges/1-2"], source_ref_ids=[]),
        NavEntry(entry_id="e2", label="B", normalized_label="b", entry_type="section",
                 source="parser_heading", doc_id="doc-2", doc_name="n.pdf",
                 page_start=3, page_end=4, order_index=1, parent_entry_id=None,
                 resource_uris=["kb://documents/doc-2/ranges/3-4"], source_ref_ids=[]),
    ])

    captured = {}

    class _FakeMCP:
        def resource(self, pattern):
            def deco(fn):
                captured[pattern] = fn
                return fn
            return deco

    state = SimpleNamespace(nav_store=store)
    register_nav_resources(_FakeMCP(), state)
    body = await captured["kb://documents/{doc_id}/toc"]("doc-1")
    data = json.loads(body)
    assert data["doc_id"] == "doc-1"
    assert len(data["entries"]) == 1
    assert data["entries"][0]["entry_id"] == "e1"
    await store.close()


async def test_nav_entry_resource_raises_on_missing(tmp_path):
    from kb.nav.store import NavIndexStore
    from kb.tool_server.mcp_resources import register_nav_resources

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    captured = {}

    class _FakeMCP:
        def resource(self, pattern):
            def deco(fn):
                captured[pattern] = fn
                return fn
            return deco

    register_nav_resources(_FakeMCP(), SimpleNamespace(nav_store=store))
    with pytest.raises(ValueError, match="Nav entry not found"):
        await captured["kb://nav/entries/{entry_id}"]("ghost")
    await store.close()
