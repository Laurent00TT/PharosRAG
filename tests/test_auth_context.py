"""T1b — ContextVar semantics for current_user."""
from __future__ import annotations

import asyncio
import pytest

from kb.auth.context import current_user, get_current_user
from kb.auth.users import User


def _make_user(name: str) -> User:
    from datetime import datetime, timezone
    return User(
        user_id=f"id-{name}", username=name, key_prefix=f"kb_{name}_xx",
        key_hash="0" * 64, role="member",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        disabled_at=None,
    )


def test_get_current_user_raises_when_unset():
    """If middleware was bypassed, code path must fail loud, not return None."""
    with pytest.raises(RuntimeError, match="bypassed"):
        get_current_user()


def test_set_and_get_current_user():
    token = current_user.set(_make_user("alice"))
    try:
        assert get_current_user().username == "alice"
    finally:
        current_user.reset(token)


def test_reset_restores_outer_value():
    """Crucial for streaming: nested set/reset must return to outer ctx."""
    outer_token = current_user.set(_make_user("alice"))
    try:
        inner_token = current_user.set(_make_user("bob"))
        try:
            assert get_current_user().username == "bob"
        finally:
            current_user.reset(inner_token)
        # After inner reset, outer alice is restored
        assert get_current_user().username == "alice"
    finally:
        current_user.reset(outer_token)


async def test_contextvar_isolated_across_tasks():
    """Each asyncio Task gets its own context copy. alice in task A must
    not leak into task B (anti-regression for ContextVar misuse)."""
    results: list[str] = []

    async def as_user(name: str) -> None:
        token = current_user.set(_make_user(name))
        try:
            await asyncio.sleep(0.01)  # yield
            results.append(get_current_user().username)
        finally:
            current_user.reset(token)

    await asyncio.gather(as_user("alice"), as_user("bob"), as_user("carol"))
    assert sorted(results) == ["alice", "bob", "carol"]


async def test_contextvar_leaks_without_reset():
    """Sanity: without reset, set value leaks across the same Task. This
    is exactly why middleware MUST use try/finally reset (I-8)."""
    current_user.set(_make_user("alice"))
    # No reset → next caller in the same Task still sees alice
    assert get_current_user().username == "alice"
    # Cleanup for test isolation (other tests assume fresh ContextVar)
    current_user.set(None)
