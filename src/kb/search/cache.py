# src/kb/search/cache.py
import hashlib
import json

from cachetools import TTLCache

from kb.models import SearchResponse

_cache: TTLCache = TTLCache(maxsize=256, ttl=300)


def cache_key(
    query: str,
    doc_filter: dict,
    include_deprecated: bool = False,
    top_k: int = 5,
    hyde: bool = False,
    multi_query_enabled: bool = False,
    multi_query_n: int = 1,
    epoch: int = 0,
) -> str:
    """Build a deterministic cache key for a search request.

    All fields that change retrieval results MUST be included. multi_query_*
    were missing originally — toggling Settings would silently leak old
    cached responses with the new behavior.

    Review P1 #6: ``epoch`` is the cross-process cache invalidation
    counter (kb.search.cache_epoch). Worker bumps it after ingest /
    deprecate; tool_server reads it and folds it into the key here.
    A bump changes the hash → all old entries become stale automatically
    on the next request. ``epoch=0`` is the no-op default so callers
    without a CacheEpochStore still get correct (just single-process)
    behavior.
    """
    raw = json.dumps(
        {
            "q": query,
            "f": doc_filter,
            "d": include_deprecated,
            "k": top_k,
            "h": hyde,
            "mq": multi_query_enabled,
            "mqn": multi_query_n,
            "e": epoch,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(key: str) -> SearchResponse | None:
    return _cache.get(key)


def set_cached(key: str, response: SearchResponse) -> None:
    _cache[key] = response


def invalidate_all() -> None:
    """Drop every entry in the search cache.

    T3 follow-up (P1b review): cache hits short-circuit ``SearchEngine.search``
    before the retriever applies its active-doc filter. Without invalidation,
    a doc that becomes non-active (deleted / deprecated) still surfaces in
    the cached response for up to 5 minutes (the TTLCache window).

    Call this from every write path that flips a doc's status:
      - ``meta_db.mark_deleted`` (via the DELETE handler post-op)
      - ``meta_db.deprecate_document`` (via the ingest deprecate-set path)
      - successful new ingest (a new active row can change a search's result set)

    In-process scope only — a worker process invalidating its own cache
    does NOT touch the tool_server process's cache (single-process small-team
    deployment is the assumption; multi-process needs an external cache like
    Redis with pub/sub, deliberately out of scope for the small-team stage).
    """
    _cache.clear()
