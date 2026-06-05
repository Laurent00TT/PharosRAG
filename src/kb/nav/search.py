from __future__ import annotations

from collections.abc import Awaitable, Callable

from kb.nav.models import NavHit
from kb.nav.normalize import normalize_label

ActiveDocIdsProvider = Callable[[], Awaitable[list[str] | set[str]]]


class NavSearchEngine:
    def __init__(
        self,
        store,
        *,
        active_doc_ids_provider: ActiveDocIdsProvider | None = None,
    ) -> None:
        """``active_doc_ids_provider`` is an async callable returning the
        set of doc_ids that should be considered live. When supplied, any
        nav entry whose doc_id is outside that set is filtered out before
        scoring — double-safety alongside the pipeline-level nav cleanup,
        so a missed delete on a deprecate path can't leak a stale pointer
        to the agent.

        When ``None`` (e.g. unit tests), no filtering is applied — preserves
        the pre-existing behavior and keeps stub-store tests passing.
        """
        self._store = store
        self._active_doc_ids_provider = active_doc_ids_provider

    async def search(self, query: str, top_k: int = 5) -> list[NavHit]:
        terms = [term for term in normalize_label(query).split(" ") if term]
        entries = await self._store.list_entries()
        # Filter out soft-hidden entries via kb_hide_nav_entry. The hasattr
        # check keeps backward-compat with stub stores in tests that don't
        # implement list_hidden_entry_ids.
        hidden_ids: set[str] = set()
        if hasattr(self._store, "list_hidden_entry_ids"):
            hidden_ids = await self._store.list_hidden_entry_ids()
        # Active-doc filter (double-safety against stale nav entries from
        # deprecated docs). Provider returning None or raising falls back
        # to "no filter" — never block a search on metadata trouble.
        active_doc_ids: set[str] | None = None
        if self._active_doc_ids_provider is not None:
            try:
                raw = await self._active_doc_ids_provider()
                active_doc_ids = set(raw) if raw is not None else None
            except Exception:
                active_doc_ids = None
        hits: list[NavHit] = []
        for entry in entries:
            if entry.entry_id in hidden_ids:
                continue
            if active_doc_ids is not None and entry.doc_id not in active_doc_ids:
                continue
            haystack = " ".join([entry.normalized_label, *[normalize_label(a) for a in entry.aliases]])
            score = sum(1 for term in terms if term in haystack)
            if score <= 0:
                continue
            hits.append(NavHit(
                entry_id=entry.entry_id,
                label=entry.label,
                entry_type=entry.entry_type,
                score=float(score),
                doc_id=entry.doc_id,
                doc_name=entry.doc_name,
                page_start=entry.page_start,
                page_end=entry.page_end,
                resource_uris=entry.resource_uris,
                source_refs=[],
                neighbor_entry_ids=[],
                match_channels=["nav_title"],
                contains_generated_content=False,
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
