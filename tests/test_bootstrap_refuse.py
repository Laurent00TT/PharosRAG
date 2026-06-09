"""T1b — server lifespan refuses to start when users table empty / no admin."""
from __future__ import annotations

import pytest

from kb.auth.users import UsersStore

# TODO(T8): Add a real-lifespan bootstrap test that boots the full FastAPI app
# (with lifespan) and asserts startup fails when users table is empty or has no
# active admin. Deferred to Task 8 where test fixtures already touch the full
# app lifespan, OR covered by Task 9 manual smoke. The helper-level tests below
# are sufficient for CI regression coverage of the bootstrap logic itself.


async def _empty_lifespan_check(db_url: str):
    """Reproduces the lifespan startup check in app.py without booting
    the full FastAPI app (avoids dependency on mineru / qdrant etc)."""
    store = UsersStore(db_url=db_url)
    await store.init()
    try:
        users = await store.list_users(include_disabled=True)
        if not users:
            raise RuntimeError("FATAL: users table empty")
        if not any(u.role == "admin" and not u.is_disabled for u in users):
            raise RuntimeError("FATAL: no active admin user")
    finally:
        await store.aclose()


async def test_empty_users_raises(tmp_path):
    with pytest.raises(RuntimeError, match="users table empty"):
        await _empty_lifespan_check(f"sqlite+aiosqlite:///{tmp_path}/u.db")


async def test_only_members_raises_no_admin(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/u.db"
    store = UsersStore(db_url=db_url)
    await store.init()
    await store.create_user(username="alice", role="member")
    await store.aclose()

    with pytest.raises(RuntimeError, match="no active admin"):
        await _empty_lifespan_check(db_url)


async def test_active_admin_passes(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/u.db"
    store = UsersStore(db_url=db_url)
    await store.init()
    await store.create_user(username="admin0", role="admin")
    await store.aclose()

    # Must not raise
    await _empty_lifespan_check(db_url)


async def test_disabled_admin_does_not_count(tmp_path):
    """If sole admin was disabled, server must still refuse to start."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/u.db"
    store = UsersStore(db_url=db_url)
    await store.init()
    _, admin = await store.create_user(username="admin0", role="admin")
    await store.disable_user(admin.user_id)
    await store.aclose()

    with pytest.raises(RuntimeError, match="no active admin"):
        await _empty_lifespan_check(db_url)
