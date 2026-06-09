from __future__ import annotations


def _make_page(doc_id: str = "d", page_num: int = 1, heading: list[str] | None = None):
    from kb.models import ParsedPage
    return ParsedPage(
        doc_id=doc_id,
        doc_name="m.pdf",
        page_num=page_num,
        structured_text="t",
        heading_path=heading if heading is not None else ["A"],
        page_image=b"",
        image_blocks=[],
        tables=[],
        metadata={},
    )


async def test_auto_build_returns_zero_when_disabled(tmp_path):
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        count = await auto_build_nav_for_doc(
            nav_store=store,
            nav_builder=NavIndexBuilder(),
            pages=[_make_page()],
            enabled=False,
        )
        assert count == 0
        entries = await store.list_entries()
        assert entries == []
    finally:
        await store.close()


async def test_auto_build_returns_zero_when_store_missing():
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.builder import NavIndexBuilder

    count = await auto_build_nav_for_doc(
        nav_store=None,
        nav_builder=NavIndexBuilder(),
        pages=[_make_page()],
        enabled=True,
    )
    assert count == 0


async def test_auto_build_returns_zero_when_builder_missing(tmp_path):
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        count = await auto_build_nav_for_doc(
            nav_store=store,
            nav_builder=None,
            pages=[_make_page()],
            enabled=True,
        )
        assert count == 0
    finally:
        await store.close()


async def test_auto_build_returns_zero_on_empty_pages(tmp_path):
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        count = await auto_build_nav_for_doc(
            nav_store=store,
            nav_builder=NavIndexBuilder(),
            pages=[],
            enabled=True,
        )
        assert count == 0
    finally:
        await store.close()


async def test_auto_build_populates_store_with_entries_and_edges(tmp_path):
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        pages = [
            _make_page(page_num=1, heading=["Intro"]),
            _make_page(page_num=2, heading=["Body"]),
        ]
        count = await auto_build_nav_for_doc(
            nav_store=store,
            nav_builder=NavIndexBuilder(),
            pages=pages,
            enabled=True,
        )
        # Builder emits 1 document + 2 section + 2 page entries: each heading
        # now materializes a section the page nests under, and ["Intro"]/["Body"]
        # are two distinct top-level sections.
        assert count == 5
        entries = await store.list_entries("d")
        assert len(entries) == 5
        # Builder emits a multi-level tree: doc → section → page. For this
        # 2-page / 2-section doc that is 4 parent_child edges (doc→Intro, Intro→p1,
        # doc→Body, Body→p2) + next/previous between the 2 pages = 6 edges total.
        # The EXACT count guards against a regression back to the flat doc→page tree
        # (which `>= 3` would have silently allowed).
        edges = await store.list_edges()
        assert len(edges) == 6
        assert len([e for e in edges if e.edge_type == "parent_child"]) == 4
        assert any(e.edge_type == "next" for e in edges)
        # no page hangs directly off the doc root anymore — each nests under its section
        doc_entry = next(e for e in entries if e.entry_type == "document")
        assert all(
            p.parent_entry_id != doc_entry.entry_id
            for p in entries if p.entry_type == "page"
        )
    finally:
        await store.close()


async def test_auto_build_is_idempotent(tmp_path):
    """Re-running auto-build for the same doc should overwrite, not duplicate."""
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        pages = [_make_page(page_num=1), _make_page(page_num=2)]
        builder = NavIndexBuilder()
        first = await auto_build_nav_for_doc(
            nav_store=store, nav_builder=builder, pages=pages, enabled=True,
        )
        second = await auto_build_nav_for_doc(
            nav_store=store, nav_builder=builder, pages=pages, enabled=True,
        )
        assert first == second
        entries = await store.list_entries("d")
        assert len(entries) == first  # no duplicates
    finally:
        await store.close()


async def test_auto_build_swallows_builder_exceptions(tmp_path):
    """Build failures must NOT raise — they only emit error trace."""
    from kb.nav.auto_build import auto_build_nav_for_doc

    class _BrokenBuilder:
        def build_from_pages(self, pages):
            raise RuntimeError("synthetic builder failure")

    class _Store:
        async def upsert_entries(self, e): pass  # noqa: E704
        async def upsert_edges(self, e): pass  # noqa: E704

    count = await auto_build_nav_for_doc(
        nav_store=_Store(),
        nav_builder=_BrokenBuilder(),
        pages=[_make_page()],
        enabled=True,
    )
    assert count == 0


async def test_auto_build_swallows_store_exceptions(tmp_path):
    """If the store fails on upsert, helper must not raise."""
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.builder import NavIndexBuilder

    class _BrokenStore:
        async def replace_for_doc(self, doc_id, entries, edges):
            raise RuntimeError("synthetic store failure")

    count = await auto_build_nav_for_doc(
        nav_store=_BrokenStore(),
        nav_builder=NavIndexBuilder(),
        pages=[_make_page()],
        enabled=True,
    )
    assert count == 0


async def test_auto_build_replaces_existing_entries_for_doc(tmp_path):
    """Re-running auto_build for the same doc_id with different pages
    must REPLACE old entries, not accumulate. Regression test for the
    nav-doc-lifecycle bug: prior implementation called upsert_entries,
    so stale v1 headings survived across a v2 ingest of the same doc.
    """
    from kb.models import ParsedPage
    from kb.nav.auto_build import auto_build_nav_for_doc
    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    try:
        # v1: 3 pages with headings A, B, C
        v1_pages = [
            ParsedPage(
                doc_id="doc", doc_name="x.pdf", page_num=i,
                structured_text="t", heading_path=[h],
                page_image=b"", image_blocks=[], tables=[], metadata={},
            )
            for i, h in enumerate(["A", "B", "C"], start=1)
        ]
        await auto_build_nav_for_doc(
            nav_store=store, nav_builder=NavIndexBuilder(),
            pages=v1_pages, enabled=True,
        )
        v1_entries = await store.list_entries("doc")
        v1_labels = {e.label for e in v1_entries}
        assert {"A", "B", "C"}.issubset(v1_labels)

        # v2: same doc_id, different headings X, Y (only 2 pages)
        v2_pages = [
            ParsedPage(
                doc_id="doc", doc_name="x.pdf", page_num=i,
                structured_text="t", heading_path=[h],
                page_image=b"", image_blocks=[], tables=[], metadata={},
            )
            for i, h in enumerate(["X", "Y"], start=1)
        ]
        await auto_build_nav_for_doc(
            nav_store=store, nav_builder=NavIndexBuilder(),
            pages=v2_pages, enabled=True,
        )
        v2_entries = await store.list_entries("doc")
        v2_labels = {e.label for e in v2_entries}

        # v1 labels MUST be gone — this is the regression assertion
        assert "A" not in v2_labels, f"stale v1 label leaked: A in {v2_labels}"
        assert "B" not in v2_labels
        assert "C" not in v2_labels
        # v2 labels must be present
        assert "X" in v2_labels and "Y" in v2_labels
    finally:
        await store.close()
