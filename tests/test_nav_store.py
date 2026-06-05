from __future__ import annotations

from kb.nav.models import NavEdge, NavEntry
from kb.nav.store import NavIndexStore


async def test_nav_store_upsert_and_list(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    entry = NavEntry(
        entry_id="entry-1",
        label="Approval",
        normalized_label="approval",
        entry_type="section",
        source="parser_heading",
        doc_id="doc",
        doc_name="manual.pdf",
        page_start=1,
        page_end=3,
        order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/ranges/1-3"],
        source_ref_ids=["ref-1"],
    )
    await store.upsert_entries([entry])
    entries = await store.list_entries("doc")
    assert entries[0].label == "Approval"
    assert entries[0].resource_uris == ["kb://documents/doc/ranges/1-3"]
    await store.close()


async def test_nav_store_list_returns_empty_when_doc_unknown(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        entries = await store.list_entries("ghost")
        assert entries == []
    finally:
        await store.close()


async def test_nav_store_list_orders_by_order_index(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        entries = [
            NavEntry(
                entry_id=f"e-{idx}",
                label=f"Section {idx}",
                normalized_label=f"section {idx}",
                entry_type="section",
                source="parser_heading",
                doc_id="doc",
                doc_name="manual.pdf",
                page_start=idx + 1,
                page_end=idx + 1,
                order_index=idx,
                parent_entry_id=None,
                resource_uris=[f"kb://documents/doc/pages/{idx + 1}"],
                source_ref_ids=[],
            )
            # Insertion order intentionally scrambled (2, 0, 1) to prove
            # the ORDER BY clause is doing the sorting, not insertion order.
            for idx in (2, 0, 1)
        ]
        await store.upsert_entries(entries)
        listed = await store.list_entries("doc")
        assert [e.order_index for e in listed] == [0, 1, 2]
        assert [e.entry_id for e in listed] == ["e-0", "e-1", "e-2"]
    finally:
        await store.close()


async def test_nav_store_upsert_edges_round_trips(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        edges = [
            NavEdge(from_entry_id="a", to_entry_id="b",
                    edge_type="parent_child", weight=1.0, source="builder"),
            NavEdge(from_entry_id="b", to_entry_id="c",
                    edge_type="next", weight=1.0, source="builder"),
        ]
        await store.upsert_edges(edges)
        listed = await store.list_edges()
        assert len(listed) == 2
        assert {(e.from_entry_id, e.to_entry_id, e.edge_type) for e in listed} == {
            ("a", "b", "parent_child"),
            ("b", "c", "next"),
        }
    finally:
        await store.close()


async def test_nav_store_upsert_edges_is_idempotent(tmp_path):
    """Re-running upsert with the same composite key must not duplicate."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        edge = NavEdge(from_entry_id="a", to_entry_id="b",
                       edge_type="parent_child", weight=1.0, source="builder")
        await store.upsert_edges([edge])
        await store.upsert_edges([edge])
        listed = await store.list_edges()
        assert len(listed) == 1
    finally:
        await store.close()


async def test_nav_store_upsert_edges_empty_is_noop(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        await store.upsert_edges([])
        assert await store.list_edges() == []
    finally:
        await store.close()


async def test_store_add_alias_appends_idempotent(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        await store.upsert_entries([NavEntry(
            entry_id="e1", label="X", normalized_label="x", entry_type="section",
            source="parser_heading", doc_id="d", doc_name="m.pdf",
            page_start=1, page_end=1, order_index=1, parent_entry_id=None,
            resource_uris=["kb://documents/d/ranges/1-1"], source_ref_ids=[],
        )])
        await store.add_alias("e1", "xx")
        await store.add_alias("e1", "xx")  # idempotent
        await store.add_alias("e1", "yy")
        entries = await store.list_entries("d")
        assert entries[0].aliases == ["xx", "yy"]
    finally:
        await store.close()


async def test_store_add_alias_raises_on_missing(tmp_path):
    import pytest
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        with pytest.raises(ValueError, match="not found"):
            await store.add_alias("ghost", "anything")
    finally:
        await store.close()


async def test_store_hide_and_list_hidden(tmp_path):
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        await store.hide_entry("e1", "noise")
        await store.hide_entry("e2", "duplicate")
        hidden = await store.list_hidden_entry_ids()
        assert hidden == {"e1", "e2"}
    finally:
        await store.close()


async def test_store_hide_entry_is_idempotent(tmp_path):
    """Hiding the same entry twice is fine (INSERT OR REPLACE)."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        await store.hide_entry("e1", "first reason")
        await store.hide_entry("e1", "second reason")
        hidden = await store.list_hidden_entry_ids()
        assert hidden == {"e1"}
    finally:
        await store.close()


def _entry(entry_id: str, doc_id: str, order_index: int = 0) -> NavEntry:
    return NavEntry(
        entry_id=entry_id,
        label=entry_id,
        normalized_label=entry_id,
        entry_type="section",
        source="parser_heading",
        doc_id=doc_id,
        doc_name=f"{doc_id}.pdf",
        page_start=1,
        page_end=1,
        order_index=order_index,
        parent_entry_id=None,
        resource_uris=[f"kb://documents/{doc_id}/pages/1"],
        source_ref_ids=[],
    )


async def test_delete_entries_for_doc_removes_entries_edges_and_hidden(tmp_path):
    """Full lifecycle: insert entries+edges+hide one for doc-a, plus an
    independent entry for doc-b; delete doc-a, verify doc-a's rows are
    gone (entries, both endpoints of edges that touched doc-a, and the
    hidden flag) while doc-b is untouched."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        a1 = _entry("a-1", "doc-a", 1)
        a2 = _entry("a-2", "doc-a", 2)
        b1 = _entry("b-1", "doc-b", 1)
        await store.upsert_entries([a1, a2, b1])
        edges = [
            # within doc-a
            NavEdge("a-1", "a-2", "next", 1.0, "builder"),
            # cross-doc edge — must also disappear because one endpoint is a-2
            NavEdge("a-2", "b-1", "related", 0.5, "builder"),
            # purely within doc-b
            NavEdge("b-1", "b-1", "self", 1.0, "builder"),
        ]
        await store.upsert_edges(edges)
        await store.hide_entry("a-1", "noise")
        await store.hide_entry("b-1", "noise")

        deleted = await store.delete_entries_for_doc("doc-a")
        assert deleted == 2

        remaining_entries = await store.list_entries()
        assert {e.entry_id for e in remaining_entries} == {"b-1"}

        remaining_edges = await store.list_edges()
        # Only the b-1↔b-1 self-edge survives — the other two had an a-* endpoint.
        assert {(e.from_entry_id, e.to_entry_id, e.edge_type) for e in remaining_edges} == {
            ("b-1", "b-1", "self"),
        }

        hidden = await store.list_hidden_entry_ids()
        assert hidden == {"b-1"}
    finally:
        await store.close()


async def test_delete_entries_for_doc_returns_zero_for_unknown_doc(tmp_path):
    """Defensive: deleting an unknown doc returns 0 with no exception."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        deleted = await store.delete_entries_for_doc("ghost")
        assert deleted == 0
    finally:
        await store.close()


async def test_replace_for_doc_atomically_overwrites(tmp_path):
    """Insert old entries+edges, call replace_for_doc with different
    entries, verify only the new entries+edges remain."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        old_entries = [_entry("old-1", "doc", 1), _entry("old-2", "doc", 2)]
        old_edges = [NavEdge("old-1", "old-2", "next", 1.0, "builder")]
        await store.upsert_entries(old_entries)
        await store.upsert_edges(old_edges)

        new_entries = [_entry("new-1", "doc", 1)]
        new_edges = [NavEdge("new-1", "new-1", "self", 1.0, "builder")]
        count = await store.replace_for_doc("doc", new_entries, new_edges)
        assert count == 1

        remaining = await store.list_entries("doc")
        assert {e.entry_id for e in remaining} == {"new-1"}

        remaining_edges = await store.list_edges()
        assert {(e.from_entry_id, e.to_entry_id, e.edge_type) for e in remaining_edges} == {
            ("new-1", "new-1", "self"),
        }
    finally:
        await store.close()


async def test_list_doc_ids_returns_distinct_set(tmp_path):
    """Insert entries from 3 different doc_ids (one doc has multiple
    entries), verify list_doc_ids returns exactly those 3 unique ids."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        entries = [
            _entry("a-1", "doc-a", 1),
            _entry("a-2", "doc-a", 2),
            _entry("b-1", "doc-b", 1),
            _entry("c-1", "doc-c", 1),
        ]
        await store.upsert_entries(entries)
        assert await store.list_doc_ids() == {"doc-a", "doc-b", "doc-c"}
    finally:
        await store.close()


async def test_replace_for_doc_handles_empty_entries(tmp_path):
    """Edge case: replace_for_doc with empty entries list deletes the
    old data and inserts nothing — net effect is a pure delete."""
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        await store.upsert_entries([_entry("old-1", "doc", 1)])
        await store.upsert_edges([NavEdge("old-1", "old-1", "self", 1.0, "builder")])

        count = await store.replace_for_doc("doc", [], [])
        assert count == 0

        assert await store.list_entries("doc") == []
        assert await store.list_edges() == []
    finally:
        await store.close()


async def test_replace_for_doc_rolls_back_delete_when_upsert_fails(tmp_path):
    """Critical correctness guarantee: if the upsert half of replace_for_doc
    fails, the delete must roll back too — so old entries stay rather than
    leaving the nav empty for that doc. Earlier implementation ran delete +
    upsert in separate transactions, which left empty nav on partial fail.
    """
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        await store.upsert_entries([_entry("old-1", "doc", 1)])

        # Force a failure mid-replace by monkey-patching _upsert_entries_on
        # to raise. The delete should NOT persist.
        async def boom(conn, entries):  # noqa: ARG001
            raise RuntimeError("simulated DB hiccup mid-upsert")

        orig = store._upsert_entries_on
        store._upsert_entries_on = boom
        try:
            import pytest as _pytest
            with _pytest.raises(RuntimeError, match="simulated DB hiccup"):
                await store.replace_for_doc(
                    "doc", [_entry("new-1", "doc", 1)], [],
                )
        finally:
            store._upsert_entries_on = orig

        # Old entry must still be there — transaction rolled back.
        entries = await store.list_entries("doc")
        assert len(entries) == 1
        assert entries[0].entry_id == "old-1"
    finally:
        await store.close()
