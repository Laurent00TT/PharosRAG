"""Auto-build navigation index for a single document.

Bridges the ingestion pipeline and the navigation layer: when a doc
finishes parsing, the pipeline hands its pages to this helper, which
runs NavIndexBuilder.build_from_pages() and persists the resulting
entries + edges into NavIndexStore.

Design constraints:
  - Safe to call with disabled=False, None stores, or empty pages →
    returns 0 entries persisted, no side effects.
  - Failures emit an error trace but MUST NOT raise — auto-build is
    best-effort and should never block ingestion success. A broken
    nav index does not invalidate the parsed evidence.
"""
from __future__ import annotations

from kb.models import ParsedPage
from kb.nav.builder import NavIndexBuilder
from kb.nav.store import NavIndexStore
from kb.observability import emit, emit_error


async def auto_build_nav_for_doc(
    *,
    nav_store: NavIndexStore | None,
    nav_builder: NavIndexBuilder | None,
    pages: list[ParsedPage],
    enabled: bool = True,
) -> int:
    """Build nav index for one doc and persist. Returns entry count.

    Returns 0 (and skips persistence) when:
      - ``enabled`` is False (caller disabled auto-build via config)
      - ``nav_store`` or ``nav_builder`` is None (lifespan never wired
        the nav layer — e.g. nav_enabled=False)
      - ``pages`` is empty (parser returned nothing; nothing to index)

    Failures inside builder/store emit ``nav.build`` error trace via
    emit_error and return 0. Never re-raises.
    """
    if not enabled or nav_store is None or nav_builder is None or not pages:
        return 0
    try:
        result = nav_builder.build_from_pages(pages)
        doc_id = pages[0].doc_id
        # replace_for_doc = delete-then-upsert. Guarantees stale entries
        # from a previous build of the same doc_id are gone before the
        # new entries land — critical for re-ingestion where headings /
        # page count may have changed.
        await nav_store.replace_for_doc(doc_id, result.entries, result.edges)
        emit(
            "nav.build",
            doc_id=doc_id,
            entry_count=len(result.entries),
            edge_count=len(result.edges),
            replaced=True,
        )
        return len(result.entries)
    except Exception as exc:  # noqa: BLE001 — auto-build must never raise
        emit_error(
            "nav.build",
            exc,
            doc_id=pages[0].doc_id if pages else "",
            page_count=len(pages),
        )
        return 0
