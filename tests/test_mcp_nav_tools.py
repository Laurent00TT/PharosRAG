from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from kb.auth.context import current_user
from kb.auth.users import User
from kb.nav.models import NavEntry, NavHit
from kb.tool_server.mcp_tools import (
    nav_hits_payload,
    register_nav_tools,
    register_nav_write_tools,
)


def _set_admin_user():
    """Bind a synthetic admin user to ``current_user`` for the duration of
    a write-tool unit test. T3-6 adds ``require_owner_or_admin`` inside
    every write-tool ``_do()`` body; admin role bypasses the ownership
    branch, so we get full behavioral coverage of the write path without
    having to seed a doc + match user_id in every test."""
    return current_user.set(User(
        user_id="u-admin",
        username="admin",
        role="admin",
        key_prefix="kb_admin_xxx",
        key_hash="0" * 64,
        created_at=None,
        disabled_at=None,
    ))


def _doc_with_owner(doc_id: str = "d", owner_id: str = "u-admin") -> dict:
    return {"doc_id": doc_id, "owner_id": owner_id, "status": "active"}


class _FakeMCP:
    """Minimal MCP stub that records tools registered via @mcp.tool()."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _settings(enable_write: bool = False, hybrid_nav_enabled: bool = True):
    return SimpleNamespace(
        mcp_tool_result_limit=20,
        mcp_enable_write_tools=enable_write,
        hybrid_nav_enabled=hybrid_nav_enabled,
    )


def test_nav_hits_payload_is_pointer_only():
    hit = NavHit(
        entry_id="e1",
        label="Approval",
        entry_type="section",
        score=1.0,
        doc_id="doc",
        doc_name="manual.pdf",
        page_start=3,
        page_end=5,
        resource_uris=["kb://documents/doc/ranges/3-5"],
        source_refs=[],
        neighbor_entry_ids=[],
        match_channels=["nav_title"],
    )
    payload = nav_hits_payload([hit])
    assert payload["structuredContent"]["contains_generated_content"] is False
    assert "pages 3-5" in payload["content"]
    # `asdict(hit)` produces a dict with the NavHit fields; verify no summary leaks
    nav_hit_dict = payload["structuredContent"]["nav_hits"][0]
    forbidden = {"summary", "content_markdown", "body", "abstract"}
    leaked = forbidden & set(nav_hit_dict.keys())
    assert not leaked, f"Generated content fields leaked: {leaked}"


def test_nav_hits_payload_empty_list():
    payload = nav_hits_payload([])
    assert payload["structuredContent"]["nav_hits"] == []
    assert "No navigation entries found" in payload["content"]


async def test_register_nav_tools_attaches_all_read_tools():
    fake = _FakeMCP()
    register_nav_tools(fake, state=None)  # state not dereferenced at register time
    assert set(fake.tools.keys()) == {
        "kb_search_nav",
        "kb_hybrid_search",
        "kb_expand_nav_entry",
        "kb_list_documents",
        "kb_get_document_text",
        "kb_get_document_source",
        "kb_get_toc",
        "kb_get_status",
    }


async def test_register_nav_write_tools_attaches_three_tools():
    fake = _FakeMCP()
    register_nav_write_tools(fake, state=None)
    assert set(fake.tools.keys()) == {
        "kb_add_nav_alias",
        "kb_hide_nav_entry",
        "kb_rebuild_nav_index",
    }


async def test_kb_search_nav_returns_gentle_error_when_engine_missing():
    state = SimpleNamespace(
        nav_search=None,
        hybrid_navigator=None,
        nav_store=None,
        meta_db=None,
        engine=None,
        settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_search_nav"](query="any", top_k=5)
    assert "not initialized" in result.content[0].text.lower()
    assert result.structuredContent["nav_hits"] == []


async def test_kb_hybrid_search_respects_hybrid_nav_enabled_flag():
    """When settings.hybrid_nav_enabled is false, the tool stays registered
    (so agent discovery is stable) but refuses the call with a structured
    reason — operator can flip the flag without redeploying."""
    state = SimpleNamespace(
        nav_search=None,
        hybrid_navigator=object(),  # would be used if not gated
        nav_store=None,
        meta_db=None,
        engine=None,
        settings=_settings(hybrid_nav_enabled=False),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_hybrid_search"](query="any", top_k=5)
    assert result.structuredContent["reason"] == "hybrid_nav_disabled"


# ── kb_expand_nav_entry ──────────────────────────────────────────────────────


def _make_entry(
    entry_id: str,
    label: str = "Section",
    doc_id: str = "d",
    parent_entry_id: str | None = None,
    order_index: int = 1,
) -> NavEntry:
    return NavEntry(
        entry_id=entry_id,
        label=label,
        normalized_label=label.lower(),
        entry_type="section",
        source="parser_heading",
        doc_id=doc_id,
        doc_name="manual.pdf",
        page_start=1,
        page_end=1,
        order_index=order_index,
        parent_entry_id=parent_entry_id,
        resource_uris=[f"kb://documents/{doc_id}/pages/{order_index}"],
        source_ref_ids=[],
    )


class _InMemoryNavStore:
    def __init__(self, entries: list[NavEntry] | None = None):
        self._entries = entries or []
        self._hidden: set[str] = set()

    async def list_entries(self, doc_id: str | None = None) -> list[NavEntry]:
        if doc_id is None:
            return list(self._entries)
        return [e for e in self._entries if e.doc_id == doc_id]

    async def get_entry(self, entry_id: str) -> NavEntry | None:
        """T3-6: write tools resolve entry → doc via this method."""
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    async def list_hidden_entry_ids(self) -> set[str]:
        return set(self._hidden)

    async def hide_entry(self, entry_id: str, reason: str) -> None:
        self._hidden.add(entry_id)

    async def add_alias(self, entry_id: str, alias: str) -> None:
        for e in self._entries:
            if e.entry_id == entry_id:
                if alias not in e.aliases:
                    e.aliases.append(alias)
                return
        raise ValueError(f"Nav entry not found: {entry_id}")


async def test_kb_expand_nav_entry_returns_gentle_error_when_nav_missing():
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=None,
        meta_db=None, engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_expand_nav_entry"](entry_id="ghost")
    assert "not initialized" in result.content[0].text.lower()
    assert result.structuredContent["neighbors"] == []


async def test_kb_expand_nav_entry_returns_parent_children_siblings():
    parent = _make_entry("p", label="Parent", order_index=1)
    target = _make_entry("t", label="Target", parent_entry_id="p", order_index=2)
    sib = _make_entry("s", label="Sibling", parent_entry_id="p", order_index=3)
    child = _make_entry("c", label="Child", parent_entry_id="t", order_index=4)
    store = _InMemoryNavStore([parent, target, sib, child])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=None, engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_expand_nav_entry"](entry_id="t")
    neighbors = result.structuredContent["neighbors"]
    relations = {(n["relation"], n["entry_id"]) for n in neighbors}
    assert ("parent", "p") in relations
    assert ("child", "c") in relations
    assert ("sibling", "s") in relations
    assert ("sibling", "t") not in relations  # excludes self
    assert result.structuredContent["contains_generated_content"] is False


async def test_kb_expand_nav_entry_handles_unknown_entry():
    store = _InMemoryNavStore([_make_entry("a")])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=None, engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_expand_nav_entry"](entry_id="ghost")
    assert "not found" in result.content[0].text.lower()
    assert result.structuredContent["neighbors"] == []


# ── kb_list_documents ────────────────────────────────────────────────────────


class _FakeMetaDB:
    def __init__(self, docs: dict[str, dict]):
        self._docs = docs

    async def list_active_doc_ids(self) -> list[str]:
        return list(self._docs.keys())

    async def get_document(self, doc_id: str) -> dict | None:
        return self._docs.get(doc_id)


async def test_kb_list_documents_gentle_error_when_meta_db_missing():
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=None,
        meta_db=None, engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_list_documents"]()
    assert "not initialized" in result.content[0].text.lower()
    assert result.structuredContent["documents"] == []


async def test_kb_list_documents_marks_nav_indexed_docs():
    meta = _FakeMetaDB({
        "d1": {"doc_id": "d1", "doc_name": "first.pdf", "version": "v1"},
        "d2": {"doc_id": "d2", "doc_name": "second.pdf", "version": "v1"},
    })
    store = _InMemoryNavStore([_make_entry("e1", doc_id="d1")])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=meta, engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_list_documents"]()
    docs = {d["doc_id"]: d for d in result.structuredContent["documents"]}
    assert docs["d1"]["nav_indexed"] is True
    assert docs["d2"]["nav_indexed"] is False
    assert docs["d1"]["resource_uri"] == "kb://documents/d1"
    assert result.structuredContent["total"] == 2


# ── kb_get_toc ───────────────────────────────────────────────────────────────


async def test_kb_get_toc_gentle_error_when_nav_missing():
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=None,
        meta_db=None, engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_get_toc"](doc_id="d1")
    assert "not initialized" in result.content[0].text.lower()
    assert result.structuredContent["entries"] == []


async def test_kb_get_toc_returns_ordered_entries():
    e1 = _make_entry("e1", label="Chapter 1", order_index=1)
    e2 = _make_entry("e2", label="Chapter 2", order_index=2)
    store = _InMemoryNavStore([e1, e2])
    # T3: kb_get_toc now requires meta_db for the I-6 active gate.
    # Return an active doc so the gate passes and the entry assertions
    # still exercise the original code path.
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=SimpleNamespace(get_document=AsyncMock(return_value=_doc_with_owner("d"))),
        engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_get_toc"](doc_id="d")
    entries = result.structuredContent["entries"]
    assert [e["entry_id"] for e in entries] == ["e1", "e2"]
    assert entries[0]["label"] == "Chapter 1"
    assert result.structuredContent["contains_generated_content"] is False


async def test_kb_get_toc_handles_doc_with_no_entries():
    store = _InMemoryNavStore([])
    # T3: meta_db returns an active doc — exercise the "doc active but no
    # nav entries" path (separate from the "doc inactive" path tested in
    # test_kb_get_toc_returns_doc_not_active_for_deleted below).
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=SimpleNamespace(get_document=AsyncMock(return_value=_doc_with_owner("missing"))),
        engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_get_toc"](doc_id="missing")
    assert "no nav entries" in result.content[0].text.lower()
    assert result.structuredContent["entries"] == []


async def test_kb_get_toc_returns_doc_not_active_for_deleted():
    """T3 I-6: a doc with status != 'active' must return doc_not_active
    even if nav entries still exist (best-effort nav cleanup may leave
    orphan rows on the deprecate path)."""
    e1 = _make_entry("e1", label="Chapter 1", order_index=1)
    store = _InMemoryNavStore([e1])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=SimpleNamespace(get_document=AsyncMock(
            return_value={"doc_id": "d", "status": "deleted"},
        )),
        engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_get_toc"](doc_id="d")
    assert result.structuredContent["entries"] == []
    assert result.structuredContent["reason"] == "doc_not_active"


async def test_kb_get_toc_returns_doc_not_active_for_missing_doc():
    """T3 I-6: when meta_db has no row for doc_id, surface doc_not_active
    rather than silently returning an empty entries list (clearer signal
    to the agent)."""
    store = _InMemoryNavStore([])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=SimpleNamespace(get_document=AsyncMock(return_value=None)),
        engine=None, settings=_settings(),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_get_toc"](doc_id="ghost")
    assert result.structuredContent["entries"] == []
    assert result.structuredContent["reason"] == "doc_not_active"


# ── kb_get_status ────────────────────────────────────────────────────────────


async def test_kb_get_status_reports_nothing_initialized():
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=None,
        meta_db=None, engine=None, settings=None,
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_get_status"]()
    sc = result.structuredContent
    assert sc["evidence_ready"] is False
    assert sc["nav_ready"] is False
    assert sc["hybrid_ready"] is False
    assert sc["meta_db_ready"] is False
    assert sc["mcp_write_tools_enabled"] is False


async def test_kb_get_status_reports_ready_state_with_counts():
    meta = _FakeMetaDB({"d1": {"doc_id": "d1", "doc_name": "first.pdf"}})
    store = _InMemoryNavStore([
        _make_entry("e1", doc_id="d1"),
        _make_entry("e2", doc_id="d1"),
    ])
    state = SimpleNamespace(
        nav_search=object(),     # truthy: nav ready
        hybrid_navigator=object(),
        nav_store=store,
        meta_db=meta,
        engine=object(),
        settings=_settings(enable_write=True),
    )
    fake = _FakeMCP()
    register_nav_tools(fake, state)
    result = await fake.tools["kb_get_status"]()
    sc = result.structuredContent
    assert sc["evidence_ready"] is True
    assert sc["nav_ready"] is True
    assert sc["hybrid_ready"] is True
    assert sc["meta_db_ready"] is True
    assert sc["document_count"] == 1
    assert sc["nav_entry_count"] == 2
    assert sc["nav_doc_count"] == 1
    assert sc["mcp_write_tools_enabled"] is True
    assert "enabled" in result.content[0].text


# ── kb_add_nav_alias (write, gated) ──────────────────────────────────────────


async def test_kb_add_nav_alias_disabled_by_default():
    store = _InMemoryNavStore([_make_entry("e1")])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=None, engine=None, settings=_settings(enable_write=False),
        maintenance=None,  # I-1: explicit testing-exempt (not oversight)
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_add_nav_alias"](entry_id="e1", alias="extra")
    assert result.structuredContent["saved"] is False
    assert result.structuredContent["reason"] == "write_tools_disabled"


async def test_kb_add_nav_alias_writes_when_enabled():
    entry = _make_entry("e1", label="Approval")
    store = _InMemoryNavStore([entry])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=SimpleNamespace(get_document=AsyncMock(return_value=_doc_with_owner(entry.doc_id))),
        engine=None, settings=_settings(enable_write=True),
        audit_log=None,
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    token = _set_admin_user()
    try:
        result = await fake.tools["kb_add_nav_alias"](entry_id="e1", alias="approve")
    finally:
        current_user.reset(token)
    assert result.structuredContent["saved"] is True
    assert result.structuredContent["entry_id"] == "e1"
    assert "approve" in entry.aliases


async def test_kb_add_nav_alias_handles_missing_entry():
    store = _InMemoryNavStore([])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=SimpleNamespace(get_document=AsyncMock(return_value=_doc_with_owner())),
        engine=None, settings=_settings(enable_write=True),
        audit_log=None,
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    token = _set_admin_user()
    try:
        result = await fake.tools["kb_add_nav_alias"](entry_id="ghost", alias="any")
    finally:
        current_user.reset(token)
    assert result.structuredContent["saved"] is False
    assert result.structuredContent["reason"] == "entry_not_found"


# ── kb_hide_nav_entry (write, gated) ─────────────────────────────────────────


async def test_kb_hide_nav_entry_disabled_by_default():
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=_InMemoryNavStore([]),
        meta_db=None, engine=None, settings=_settings(enable_write=False),
        maintenance=None,  # I-1: explicit testing-exempt (not oversight)
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_hide_nav_entry"](entry_id="e1", reason="noise")
    assert result.structuredContent["hidden"] is False
    assert result.structuredContent["reason"] == "write_tools_disabled"


async def test_kb_hide_nav_entry_writes_when_enabled():
    entry = _make_entry("e1")
    store = _InMemoryNavStore([entry])
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=store,
        meta_db=SimpleNamespace(get_document=AsyncMock(return_value=_doc_with_owner(entry.doc_id))),
        engine=None, settings=_settings(enable_write=True),
        audit_log=None,
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    token = _set_admin_user()
    try:
        result = await fake.tools["kb_hide_nav_entry"](entry_id="e1", reason="duplicate")
    finally:
        current_user.reset(token)
    assert result.structuredContent["hidden"] is True
    assert "e1" in await store.list_hidden_entry_ids()


# ── kb_rebuild_nav_index (write, stub) ───────────────────────────────────────


async def test_kb_rebuild_nav_index_disabled_by_default():
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=None,
        meta_db=None, engine=None, settings=_settings(enable_write=False),
        maintenance=None,  # I-1: explicit testing-exempt (not oversight)
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_rebuild_nav_index"](doc_id="d1")
    assert result.structuredContent["started"] is False
    assert result.structuredContent["reason"] == "write_tools_disabled"


async def test_kb_rebuild_nav_index_returns_services_not_initialized_when_unbound():
    """When the tool is enabled but the supporting services aren't bound,
    we return a structured error rather than crashing."""
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None, nav_store=None, nav_builder=None,
        qdrant=None, meta_db=None, engine=None,
        settings=_settings(enable_write=True),
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_rebuild_nav_index"](doc_id="d1")
    assert result.structuredContent["started"] is False
    assert result.structuredContent["reason"] == "services_not_initialized"


async def test_kb_rebuild_nav_index_reports_doc_not_found():
    from unittest.mock import AsyncMock, MagicMock
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None,
        nav_store=MagicMock(), nav_builder=MagicMock(), qdrant=MagicMock(),
        meta_db=SimpleNamespace(get_document=AsyncMock(return_value=None)),
        engine=None, settings=_settings(enable_write=True),
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_rebuild_nav_index"](doc_id="ghost")
    assert result.structuredContent["started"] is False
    assert result.structuredContent["reason"] == "doc_not_found"


async def test_kb_rebuild_nav_index_rebuilds_from_qdrant(tmp_path):
    """Happy path: services bound, doc exists, qdrant has chunks → nav rebuild
    runs end-to-end and reports entries_built."""
    from unittest.mock import MagicMock

    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()

    # Fake qdrant that returns 2 chunks across 2 pages
    fake_qdrant = MagicMock()
    fake_qdrant.text_collection = "text_chunks"
    fake_qdrant.client = MagicMock()
    fake_qdrant.client.scroll = MagicMock(return_value=(
        [
            SimpleNamespace(payload={"page_num": 0, "text": "Page 0", "heading_path": ["A"]}),
            SimpleNamespace(payload={"page_num": 1, "text": "Page 1", "heading_path": ["B"]}),
        ],
        None,
    ))

    # T3-6: require_owner_or_admin reads doc["owner_id"]; admin user bypasses
    # the ownership check, so the doc fixture's owner_id is intentionally
    # left absent (admin role takes precedence regardless of owner_id value).
    # T3 follow-up (P2b): the new active-gate in kb_rebuild_nav_index reads
    # doc["status"] — must be "active" on the happy-path mock or rebuild
    # short-circuits with reason="doc_not_active".
    state = SimpleNamespace(
        nav_search=None, hybrid_navigator=None,
        nav_store=store, nav_builder=NavIndexBuilder(), qdrant=fake_qdrant,
        meta_db=SimpleNamespace(get_document=AsyncMock(
            return_value={
                "doc_id": "d1", "doc_name": "manual.pdf",
                "owner_id": "u-admin", "status": "active",
            }
        )),
        engine=None, settings=_settings(enable_write=True),
        audit_log=None,
    )
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    token = _set_admin_user()
    try:
        result = await fake.tools["kb_rebuild_nav_index"](doc_id="d1")
    finally:
        current_user.reset(token)
    assert result.structuredContent["started"] is True
    assert result.structuredContent["completed"] is True
    assert result.structuredContent["pages_processed"] == 2
    assert result.structuredContent["entries_built"] >= 3  # doc root + 2 pages
    # Nav store actually got the entries
    entries = await store.list_entries("d1")
    assert len(entries) >= 3
    await store.close()


# ── T5: maintenance gate runtime coverage (review I-2) ──────────────


def _state_with_maintenance_on():
    """Build a SimpleNamespace state with maintenance.is_on()=True.
    Follows the _FakeMCP pattern used throughout this file — tools are
    registered via register_nav_write_tools(fake, state) and invoked as
    fake.tools["name"](**args).  maintenance=None is intentionally NOT
    used here; we want the gate to fire."""
    maintenance = SimpleNamespace(is_on=AsyncMock(return_value=True))
    return SimpleNamespace(
        engine=None,
        meta_db=None,
        nav_store=None,
        nav_search=None,
        nav_builder=None,
        hybrid_navigator=None,
        settings=_settings(enable_write=True),
        qdrant=None,
        image_store=None,
        audit_log=None,
        maintenance=maintenance,
    )


async def test_kb_add_nav_alias_returns_maintenance_mode_when_on():
    """T5 I-2 runtime: write tool returns reason='maintenance_mode' when
    maintenance.is_on()=True, BEFORE evaluating _write_tools_enabled.
    (settings.mcp_enable_write_tools=True here — proves gate fires first.)"""
    state = _state_with_maintenance_on()
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_add_nav_alias"](entry_id="e1", alias="x")
    assert result.structuredContent["saved"] is False
    assert result.structuredContent["reason"] == "maintenance_mode"


async def test_kb_hide_nav_entry_returns_maintenance_mode_when_on():
    """T5 I-2 runtime: same gate semantics for hide."""
    state = _state_with_maintenance_on()
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_hide_nav_entry"](entry_id="e1", reason="test")
    assert result.structuredContent["hidden"] is False
    assert result.structuredContent["reason"] == "maintenance_mode"


async def test_kb_rebuild_nav_index_returns_maintenance_mode_when_on():
    """T5 I-2 runtime: same gate semantics for rebuild."""
    state = _state_with_maintenance_on()
    fake = _FakeMCP()
    register_nav_write_tools(fake, state)
    result = await fake.tools["kb_rebuild_nav_index"](doc_id="d1")
    assert result.structuredContent["started"] is False
    assert result.structuredContent["reason"] == "maintenance_mode"
