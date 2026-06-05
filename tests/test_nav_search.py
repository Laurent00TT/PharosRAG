from __future__ import annotations

from kb.nav.models import NavEntry
from kb.nav.search import NavSearchEngine
from kb.nav.store import NavIndexStore


async def test_nav_search_returns_pointers_only(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([NavEntry(
        entry_id="e1",
        label="Approval Flow",
        normalized_label="approval flow",
        entry_type="section",
        source="parser_heading",
        doc_id="doc",
        doc_name="manual.pdf",
        page_start=3,
        page_end=5,
        order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/ranges/3-5"],
        source_ref_ids=[],
    )])
    hits = await NavSearchEngine(store).search("approval")
    assert hits[0].resource_uris == ["kb://documents/doc/ranges/3-5"]
    assert hits[0].contains_generated_content is False
    # NavHit is a dataclass — content_markdown shouldn't be a field
    fields = type(hits[0]).__dataclass_fields__
    assert "content_markdown" not in fields
    await store.close()


async def test_nav_search_returns_empty_on_no_match(tmp_path):
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
    hits = await NavSearchEngine(store).search("nonexistent")
    assert hits == []
    await store.close()


async def test_nav_search_uses_aliases(tmp_path):
    """Match should hit aliases too, not just label."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([NavEntry(
        entry_id="e1", label="审批流程", normalized_label="审批流程",
        entry_type="section", source="parser_heading", doc_id="doc",
        doc_name="m.pdf", page_start=1, page_end=2, order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/ranges/1-2"],
        source_ref_ids=[],
        aliases=["approval flow"],
    )])
    hits = await NavSearchEngine(store).search("approval")
    assert len(hits) == 1
    assert hits[0].label == "审批流程"
    await store.close()


async def test_nav_search_ranks_by_term_count(tmp_path):
    """Entry whose label contains MORE query terms scores higher."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([
        NavEntry(entry_id="e1", label="Approval", normalized_label="approval",
                 entry_type="section", source="parser_heading", doc_id="d",
                 doc_name="m.pdf", page_start=1, page_end=1, order_index=1,
                 parent_entry_id=None,
                 resource_uris=["kb://documents/d/ranges/1-1"],
                 source_ref_ids=[]),
        NavEntry(entry_id="e2", label="Approval Workflow Steps", normalized_label="approval workflow steps",
                 entry_type="section", source="parser_heading", doc_id="d",
                 doc_name="m.pdf", page_start=2, page_end=2, order_index=2,
                 parent_entry_id=None,
                 resource_uris=["kb://documents/d/ranges/2-2"],
                 source_ref_ids=[]),
    ])
    hits = await NavSearchEngine(store).search("approval workflow")
    assert hits[0].entry_id == "e2"  # 2 matches > 1 match
    assert hits[1].entry_id == "e1"
    await store.close()


async def test_nav_search_respects_top_k(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([
        NavEntry(entry_id=f"e{i}", label="approval", normalized_label="approval",
                 entry_type="section", source="parser_heading", doc_id="d",
                 doc_name="m.pdf", page_start=i, page_end=i, order_index=i,
                 parent_entry_id=None,
                 resource_uris=[f"kb://documents/d/ranges/{i}-{i}"],
                 source_ref_ids=[])
        for i in range(1, 6)  # 5 entries
    ])
    hits = await NavSearchEngine(store).search("approval", top_k=2)
    assert len(hits) == 2
    await store.close()


async def test_nav_search_excludes_hidden_entries(tmp_path):
    """kb_hide_nav_entry should remove an entry from search results."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([NavEntry(
        entry_id="e1", label="Approval", normalized_label="approval",
        entry_type="section", source="parser_heading", doc_id="d",
        doc_name="m.pdf", page_start=1, page_end=2, order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/d/ranges/1-2"], source_ref_ids=[],
    )])
    # Sanity: visible before hide
    hits = await NavSearchEngine(store).search("approval")
    assert len(hits) == 1
    # Hide it
    await store.hide_entry("e1", reason="false positive")
    hits = await NavSearchEngine(store).search("approval")
    assert hits == []
    await store.close()


async def test_nav_search_filters_entries_for_inactive_docs(tmp_path):
    """When an active_doc_ids_provider is supplied, entries whose doc_id
    isn't in the active set must NOT appear in search hits — this is the
    double-safety against stale nav entries from deprecated docs that
    slipped past pipeline-level cleanup."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([
        NavEntry(
            entry_id="active", label="Approval", normalized_label="approval",
            entry_type="section", source="parser_heading", doc_id="active_doc",
            doc_name="active.pdf", page_start=1, page_end=1, order_index=1,
            parent_entry_id=None,
            resource_uris=["kb://documents/active_doc/ranges/1-1"],
            source_ref_ids=[],
        ),
        NavEntry(
            entry_id="stale", label="Approval", normalized_label="approval",
            entry_type="section", source="parser_heading", doc_id="old_doc",
            doc_name="old.pdf", page_start=1, page_end=1, order_index=1,
            parent_entry_id=None,
            resource_uris=["kb://documents/old_doc/ranges/1-1"],
            source_ref_ids=[],
        ),
    ])

    async def only_active_doc():
        return {"active_doc"}

    engine = NavSearchEngine(store, active_doc_ids_provider=only_active_doc)
    hits = await engine.search("approval")
    assert [h.entry_id for h in hits] == ["active"]
    await store.close()


async def test_nav_search_falls_back_when_active_provider_raises(tmp_path):
    """If the provider raises (metadata DB hiccup), search must still
    return results — never block on metadata trouble."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([NavEntry(
        entry_id="e1", label="Approval", normalized_label="approval",
        entry_type="section", source="parser_heading", doc_id="d",
        doc_name="m.pdf", page_start=1, page_end=1, order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/d/ranges/1-1"], source_ref_ids=[],
    )])

    async def broken_provider():
        raise RuntimeError("meta_db down")

    engine = NavSearchEngine(store, active_doc_ids_provider=broken_provider)
    hits = await engine.search("approval")
    assert len(hits) == 1  # fallback to no filtering, not an empty result
    await store.close()


async def test_nav_search_works_with_stub_store_without_hidden_method():
    """Backward-compat: stores lacking list_hidden_entry_ids still work."""
    from kb.nav.models import NavHit  # noqa: F401  (NavHit constructed by engine)

    class _StubStore:
        async def list_entries(self, doc_id: str | None = None):
            return [NavEntry(
                entry_id="e1", label="Approval", normalized_label="approval",
                entry_type="section", source="parser_heading", doc_id="d",
                doc_name="m.pdf", page_start=1, page_end=1, order_index=1,
                parent_entry_id=None,
                resource_uris=["kb://documents/d/ranges/1-1"],
                source_ref_ids=[],
            )]

    hits = await NavSearchEngine(_StubStore()).search("approval")
    assert len(hits) == 1
    assert hits[0].label == "Approval"
