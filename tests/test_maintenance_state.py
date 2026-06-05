"""MaintenanceState — SQLite-backed cross-process flag (T5 / C-1)."""
from datetime import datetime

import pytest


@pytest.fixture
async def state(tmp_path):
    from kb.admin.maintenance import MaintenanceState
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    s = MaintenanceState(db_url=db_url)
    await s.init()
    yield s
    await s.aclose()


@pytest.mark.asyncio
async def test_default_is_off(state):
    """A fresh DB starts with maintenance OFF (seed row id=1, on=0)."""
    assert (await state.is_on()) is False
    snap = await state.get_state()
    assert snap["on"] is False
    assert snap["set_at"] is None
    assert snap["set_by_user_id"] is None


@pytest.mark.asyncio
async def test_set_on_persists_and_reads_back(state):
    """set_on() flips the flag; is_on() agrees; get_state() carries who/when."""
    await state.set_on(user_id="u-admin")
    assert (await state.is_on()) is True
    snap = await state.get_state()
    assert snap["on"] is True
    assert snap["set_by_user_id"] == "u-admin"
    assert isinstance(snap["set_at"], datetime)


@pytest.mark.asyncio
async def test_set_off_clears(state):
    """set_off flips on→False but PRESERVES the audit trail: set_at +
    set_by_user_id record who/when did the LAST toggle, even when that
    toggle was 'off'. Operators must be able to attribute the unlock."""
    await state.set_on(user_id="u-admin")
    await state.set_off(user_id="u-admin")
    assert (await state.is_on()) is False
    snap = await state.get_state()
    assert snap["on"] is False
    # Audit trail preserved
    assert snap["set_by_user_id"] == "u-admin"
    assert isinstance(snap["set_at"], datetime)


@pytest.mark.asyncio
async def test_set_on_is_idempotent(state):
    """Two consecutive set_on calls don't break or double-write."""
    await state.set_on(user_id="u-admin-a")
    await state.set_on(user_id="u-admin-b")
    assert (await state.is_on()) is True
    snap = await state.get_state()
    assert snap["set_by_user_id"] == "u-admin-b"


@pytest.mark.asyncio
async def test_two_instances_on_same_db_see_each_other(tmp_path):
    """Cross-process semantics: two MaintenanceState instances pointing at
    the same DB must read each other's writes (no in-process cache).
    This is the core invariant — worker process sees server flag flip."""
    from kb.admin.maintenance import MaintenanceState
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    server = MaintenanceState(db_url=db_url)
    worker = MaintenanceState(db_url=db_url)
    await server.init()
    await worker.init()
    try:
        assert (await worker.is_on()) is False
        await server.set_on(user_id="u-admin")
        assert (await worker.is_on()) is True
        await server.set_off(user_id="u-admin")
        assert (await worker.is_on()) is False
    finally:
        await server.aclose()
        await worker.aclose()


@pytest.mark.asyncio
async def test_set_on_without_init_raises_runtime_error(tmp_path):
    """Defensive guard: _write asserts rowcount, so calling set_on on
    an uninitialised state raises rather than silently no-op'ing."""
    from kb.admin.maintenance import MaintenanceState
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    s = MaintenanceState(db_url=db_url)
    # NOTE: deliberately NOT calling s.init()
    try:
        # MaintenanceState __init__ doesn't create the table either, so
        # the first .execute() inside _write will fail. We need to seed
        # the schema (but not the row) to isolate the rowcount check.
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool
        from kb.admin.maintenance import _Base
        eng = create_async_engine(db_url, poolclass=NullPool)
        async with eng.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
        await eng.dispose()
        # Schema exists, row id=1 is NOT seeded — set_on must raise.
        with pytest.raises(RuntimeError, match="row id=1 missing"):
            await s.set_on(user_id="u-admin")
    finally:
        await s.aclose()
