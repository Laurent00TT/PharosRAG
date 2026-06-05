from __future__ import annotations

from kb.nav.hybrid import nav_hit_to_fetch_target
from kb.nav.models import NavHit


def _make_hit(page_start: int = 3, page_end: int = 5) -> NavHit:
    return NavHit(
        entry_id="e1",
        label="Approval",
        entry_type="section",
        score=1.0,
        doc_id="doc",
        doc_name="manual.pdf",
        page_start=page_start,
        page_end=page_end,
        resource_uris=[f"kb://documents/doc/ranges/{page_start}-{page_end}"],
        source_refs=[],
        neighbor_entry_ids=[],
        match_channels=["nav_title"],
    )


def test_nav_hit_to_fetch_target():
    target = nav_hit_to_fetch_target(_make_hit())
    assert target.resource_uri == "kb://documents/doc/ranges/3-5"
    assert target.page_start == 3
    assert target.page_end == 5


def test_nav_hit_to_fetch_target_widens_with_window_pages():
    """window_pages=3 should add one page on each side of the hit."""
    target = nav_hit_to_fetch_target(_make_hit(page_start=3, page_end=5), window_pages=3)
    assert target.page_start == 2
    assert target.page_end == 6
    assert target.resource_uri == "kb://documents/doc/ranges/2-6"


def test_nav_hit_to_fetch_target_clamps_page_start_at_zero():
    """Wide windows around an early page must not produce negative starts —
    storage is 0-based and any negative page_num would be unfetchable."""
    target = nav_hit_to_fetch_target(_make_hit(page_start=0, page_end=0), window_pages=5)
    assert target.page_start == 0  # clamped
    assert target.page_end == 2


async def test_hybrid_navigator_combines_evidence_and_nav(tmp_path):
    from types import SimpleNamespace
    from kb.models import SearchResponse, SearchResult
    from kb.nav.hybrid import HybridNavigator
    from kb.nav.models import NavEntry
    from kb.nav.search import NavSearchEngine
    from kb.nav.store import NavIndexStore

    # nav side: real store
    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([NavEntry(
        entry_id="e1", label="Approval Flow", normalized_label="approval flow",
        entry_type="section", source="parser_heading", doc_id="doc",
        doc_name="m.pdf", page_start=3, page_end=5, order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/ranges/3-5"],
        source_ref_ids=[],
    )])
    nav_search = NavSearchEngine(store)

    # evidence side: stub engine
    class _StubEngine:
        async def search(self, *, query, top_k, doc_filter, hyde, multi_query, include_deprecated):
            sr = SearchResult(
                doc_id="doc", doc_name="m.pdf", page_num=4, heading_path=[],
                text_chunk="text", vision_description=None, figure_index=None,
                figure_caption=None, image_url="", page_type="text",
                doc_version=None, effective_date=None, score=0.9,
                match_channels=["dense"],
            )
            return SearchResponse(results=[sr], total=1), False

    nav = HybridNavigator(
        search_engine=_StubEngine(),
        nav_search=nav_search,
        settings=SimpleNamespace(),
    )
    response = await nav.search("approval", top_k=5)
    assert response.routing_reason == "evidence_plus_navigation"
    assert len(response.evidence_hits) == 1
    assert response.evidence_hits[0].resource_uri == "kb://documents/doc/pages/4"
    assert len(response.nav_hits) == 1
    assert len(response.suggested_fetches) == 1
    assert response.suggested_fetches[0].resource_uri == "kb://documents/doc/ranges/3-5"
    await store.close()


async def test_hybrid_navigator_emits_trace_when_enabled(tmp_path):
    """hybrid_trace_enabled=True should produce one 'hybrid.search' trace
    event per search call with hit counts + latency."""
    import json
    import logging
    from types import SimpleNamespace
    from kb.models import SearchResponse
    from kb.nav.hybrid import HybridNavigator
    from kb.nav.search import NavSearchEngine
    from kb.nav.store import NavIndexStore
    from kb.observability.trace import _logger as _trace_logger

    captured: list[dict] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            try:
                captured.append(json.loads(record.getMessage()))
            except json.JSONDecodeError:
                pass

    class _StubEngine:
        async def search(self, **kwargs):
            return SearchResponse(results=[], total=0), False

    handler = _Capture()
    _trace_logger.addHandler(handler)
    _trace_logger.setLevel(logging.INFO)
    try:
        store = NavIndexStore(str(tmp_path / "nav.db"))
        await store.init()
        nav = HybridNavigator(
            search_engine=_StubEngine(),
            nav_search=NavSearchEngine(store),
            settings=SimpleNamespace(hybrid_trace_enabled=True, nav_fetch_window_pages=1),
        )
        await nav.search("q", top_k=3)
        await store.close()
    finally:
        _trace_logger.removeHandler(handler)

    events = [e for e in captured if e.get("event") == "hybrid.search"]
    assert len(events) == 1
    assert events[0]["top_k"] == 3
    assert events[0]["evidence_hit_count"] == 0
    assert events[0]["window_pages"] == 1
    assert "latency_ms" in events[0]


async def test_hybrid_navigator_suppresses_trace_when_disabled(tmp_path):
    """hybrid_trace_enabled=False should suppress the trace event so noisy
    production servers don't fill the trace log."""
    import json
    import logging
    from types import SimpleNamespace
    from kb.models import SearchResponse
    from kb.nav.hybrid import HybridNavigator
    from kb.nav.search import NavSearchEngine
    from kb.nav.store import NavIndexStore
    from kb.observability.trace import _logger as _trace_logger

    captured: list[dict] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            try:
                captured.append(json.loads(record.getMessage()))
            except json.JSONDecodeError:
                pass

    handler = _Capture()
    _trace_logger.addHandler(handler)
    _trace_logger.setLevel(logging.INFO)
    try:
        store = NavIndexStore(str(tmp_path / "nav.db"))
        await store.init()

        class _StubEngine:
            async def search(self, **kwargs):
                return SearchResponse(results=[], total=0), False

        nav = HybridNavigator(
            search_engine=_StubEngine(),
            nav_search=NavSearchEngine(store),
            settings=SimpleNamespace(hybrid_trace_enabled=False, nav_fetch_window_pages=1),
        )
        await nav.search("q")
        await store.close()
    finally:
        _trace_logger.removeHandler(handler)

    assert not [e for e in captured if e.get("event") == "hybrid.search"]


async def test_hybrid_navigator_widens_suggested_fetches_via_window_pages(tmp_path):
    """nav_fetch_window_pages flows through to suggested_fetches: hits on
    page 4 with window=3 should produce a fetch range covering 3-5."""
    from types import SimpleNamespace
    from kb.models import SearchResponse
    from kb.nav.hybrid import HybridNavigator
    from kb.nav.models import NavEntry
    from kb.nav.search import NavSearchEngine
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()
    await store.upsert_entries([NavEntry(
        entry_id="e1", label="Approval", normalized_label="approval",
        entry_type="page", source="parser_heading", doc_id="doc",
        doc_name="m.pdf", page_start=4, page_end=4, order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/pages/4"],
        source_ref_ids=[],
    )])

    class _StubEngine:
        async def search(self, **kwargs):
            return SearchResponse(results=[], total=0), False

    nav = HybridNavigator(
        search_engine=_StubEngine(),
        nav_search=NavSearchEngine(store),
        settings=SimpleNamespace(hybrid_trace_enabled=False, nav_fetch_window_pages=3),
    )
    response = await nav.search("approval")
    assert response.suggested_fetches[0].page_start == 3
    assert response.suggested_fetches[0].page_end == 5
    await store.close()


async def test_hybrid_navigator_empty_nav_still_returns_evidence(tmp_path):
    """If nav index is empty, hybrid response still has evidence_hits but
    nav_hits and suggested_fetches are empty lists (not None)."""
    from types import SimpleNamespace
    from kb.models import SearchResponse
    from kb.nav.hybrid import HybridNavigator
    from kb.nav.search import NavSearchEngine
    from kb.nav.store import NavIndexStore

    store = NavIndexStore(str(tmp_path / "nav.db"))
    await store.init()  # empty
    nav_search = NavSearchEngine(store)

    class _StubEngine:
        async def search(self, **kwargs):
            return SearchResponse(results=[], total=0), False

    nav = HybridNavigator(
        search_engine=_StubEngine(),
        nav_search=nav_search,
        settings=SimpleNamespace(),
    )
    response = await nav.search("anything")
    assert response.evidence_hits == []
    assert response.nav_hits == []
    assert response.suggested_fetches == []
    await store.close()
