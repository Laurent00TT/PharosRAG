"""Cross-process cache invalidation epoch counter (review P1 #6)."""
import pytest


@pytest.mark.asyncio
async def test_epoch_starts_at_zero(tmp_path):
    """Fresh DB → init seeds row 1 with epoch=0."""
    from kb.search.cache_epoch import CacheEpochStore
    store = CacheEpochStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    await store.init()
    try:
        assert await store.get_epoch() == 0
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_bump_increments_and_returns_new_value(tmp_path):
    """bump() returns the new value; subsequent get_epoch() agrees."""
    from kb.search.cache_epoch import CacheEpochStore
    store = CacheEpochStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    await store.init()
    try:
        first = await store.bump()
        assert first == 1
        second = await store.bump()
        assert second == 2
        assert await store.get_epoch() == 2
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_two_instances_share_state_cross_process(tmp_path):
    """Core P1 #6 invariant: two CacheEpochStore instances pointing at
    the same DB (simulating tool_server + worker processes) must see
    each other's bumps. NullPool guarantees no in-process connection
    cache masks the change."""
    from kb.search.cache_epoch import CacheEpochStore
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    server = CacheEpochStore(db_url=db_url)
    worker = CacheEpochStore(db_url=db_url)
    await server.init()
    await worker.init()  # idempotent — should not overwrite
    try:
        assert await server.get_epoch() == 0
        assert await worker.get_epoch() == 0
        # Worker simulates an ingest: bumps after the doc is written.
        await worker.bump()
        # tool_server (which is processing a search RIGHT NOW) reads the
        # SAME row and sees the bump immediately — its cached responses
        # become invalid because cache_key now includes the new epoch.
        assert await server.get_epoch() == 1
    finally:
        await server.aclose()
        await worker.aclose()


@pytest.mark.asyncio
async def test_init_is_idempotent(tmp_path):
    """init() can be called multiple times from concurrent processes
    without overwriting the existing epoch. The INSERT OR IGNORE means
    only the first init seeds row 1; subsequent inits no-op."""
    from kb.search.cache_epoch import CacheEpochStore
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    a = CacheEpochStore(db_url=db_url)
    await a.init()
    await a.bump()
    assert await a.get_epoch() == 1
    # Second init from same or different instance — must NOT reset epoch.
    b = CacheEpochStore(db_url=db_url)
    await b.init()
    try:
        assert await b.get_epoch() == 1
    finally:
        await a.aclose()
        await b.aclose()


def test_cache_key_changes_when_epoch_changes():
    """cache_key MUST produce a different hash for different epoch
    values — otherwise the cross-process invalidation does nothing.
    All other inputs held constant."""
    from kb.search.cache import cache_key
    k0 = cache_key(query="hello", doc_filter={}, epoch=0)
    k1 = cache_key(query="hello", doc_filter={}, epoch=1)
    k2 = cache_key(query="hello", doc_filter={}, epoch=2)
    assert k0 != k1
    assert k1 != k2
    assert k0 != k2
    # Same epoch + same args → same key (cache hit path still works)
    assert cache_key(query="hello", doc_filter={}, epoch=1) == k1


def test_cache_key_default_epoch_is_zero():
    """Back-compat: callers without an epoch (tests / single-process
    deployments) get the same key they always got — epoch=0 default."""
    from kb.search.cache import cache_key
    explicit = cache_key(query="q", doc_filter={}, epoch=0)
    implicit = cache_key(query="q", doc_filter={})
    assert explicit == implicit
