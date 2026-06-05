from __future__ import annotations

import time

from kb.nav.models import EvidenceHit, FetchTarget, HybridNavigatorResponse, NavHit
from kb.nav.resources import page_range_uri, page_uri
from kb.observability import emit


def nav_hit_to_fetch_target(
    hit: NavHit,
    *,
    reason: str = "nav_hit",
    window_pages: int = 1,
) -> FetchTarget:
    """Build a FetchTarget from a nav hit.

    ``window_pages`` (>= 1) widens the suggested fetch range symmetrically
    around the hit. ``window_pages=1`` (default) preserves the hit's exact
    range; ``window_pages=3`` adds one page on each side. Page floor is
    clamped at 0 since the storage layer is 0-based.
    """
    half = max(0, (window_pages - 1) // 2)
    page_start = max(0, hit.page_start - half)
    page_end = hit.page_end + half
    return FetchTarget(
        reason=reason,
        resource_uri=page_range_uri(hit.doc_id, page_start, page_end),
        doc_id=hit.doc_id,
        page_start=page_start,
        page_end=page_end,
    )


class HybridNavigator:
    def __init__(self, *, search_engine, nav_search, settings) -> None:
        self._search_engine = search_engine
        self._nav_search = nav_search
        self._settings = settings

    async def search(self, query: str, top_k: int = 5) -> HybridNavigatorResponse:
        started = time.monotonic()
        evidence_response, _cache_hit = await self._search_engine.search(
            query=query,
            top_k=top_k,
            doc_filter={},
            hyde=False,
            multi_query=False,
            include_deprecated=False,
        )
        evidence_hits = [
            EvidenceHit(
                doc_id=hit.doc_id,
                doc_name=hit.doc_name,
                page_num=hit.page_num,
                resource_uri=page_uri(hit.doc_id, hit.page_num),
                score=hit.score,
                heading_path=list(hit.heading_path or []),
                match_channels=list(hit.match_channels or []),
                page_type=hit.page_type or "",
            )
            for hit in evidence_response.results
        ]
        nav_hits = await self._nav_search.search(query=query, top_k=top_k)
        window_pages = max(1, int(getattr(self._settings, "nav_fetch_window_pages", 1)))
        suggested_fetches = [
            nav_hit_to_fetch_target(hit, window_pages=window_pages) for hit in nav_hits
        ]
        # Observability: emit a per-call trace when enabled so the doctor
        # and ops dashboards can see routing distribution + latency without
        # turning on full debug logging. Gated by hybrid_trace_enabled so
        # operators can drop the noise in production.
        if getattr(self._settings, "hybrid_trace_enabled", True):
            emit(
                "hybrid.search",
                query_preview=query[:120],
                top_k=top_k,
                evidence_hit_count=len(evidence_hits),
                nav_hit_count=len(nav_hits),
                suggested_fetch_count=len(suggested_fetches),
                window_pages=window_pages,
                latency_ms=round((time.monotonic() - started) * 1000, 1),
            )
        return HybridNavigatorResponse(
            query=query,
            evidence_hits=evidence_hits,
            nav_hits=nav_hits,
            suggested_fetches=suggested_fetches,
            routing_reason="evidence_plus_navigation",
        )
