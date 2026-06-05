"""Auth permissions: require_role + require_owner_or_admin (T3 Task 3)."""
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from kb.auth.context import current_user
from kb.auth.permissions import require_owner_or_admin, require_role
from kb.auth.users import User


def _user(role="member", user_id="u-alice", username="alice"):
    # ``key_hash`` is required by the User dataclass; created_at=None is
    # tolerated because the permission helpers only read .user_id / .role.
    return User(
        user_id=user_id, username=username, role=role,
        key_prefix="kb_alice_xxx", key_hash="0" * 64,
        created_at=None, disabled_at=None,
    )


@pytest.fixture(autouse=True)
def _reset_current_user():
    """ContextVar isolation: every test that runs in this file sets
    current_user; without an explicit reset between tests the same
    Task can leak the previous test's identity (see
    test_contextvar_leaks_without_reset in test_auth_context.py)."""
    yield
    current_user.set(None)


# ── require_role factory ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_require_role_allows_matching():
    current_user.set(_user(role="admin"))
    dep = require_role("admin")
    # FastAPI deps return None on success; call directly
    result = await dep()
    assert result is None


@pytest.mark.asyncio
async def test_require_role_admin_passes_member_check():
    """admin can do anything member can — role hierarchy."""
    current_user.set(_user(role="admin"))
    dep = require_role("member")
    await dep()  # no raise


@pytest.mark.asyncio
async def test_require_role_member_blocked_from_admin():
    current_user.set(_user(role="member"))
    dep = require_role("admin")
    with pytest.raises(HTTPException) as ei:
        await dep()
    assert ei.value.status_code == 403


# ── require_owner_or_admin ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_passes_regardless_of_owner():
    audit = AsyncMock()
    admin = _user(role="admin", user_id="u-admin", username="admin")
    target = {"owner_id": "u-someone-else"}
    # Should not raise
    await require_owner_or_admin(
        target, admin, audit_log=audit,
        target_kind="document", target_id="d-1",
    )
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_owner_member_allowed():
    audit = AsyncMock()
    alice = _user(role="member", user_id="u-alice")
    target = {"owner_id": "u-alice"}
    await require_owner_or_admin(
        target, alice, audit_log=audit,
        target_kind="document", target_id="d-1",
    )
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_non_owner_member_403_emits_authz_denied():
    """B-3: 403 path must emit authz.denied (best-effort, I-13)."""
    audit = AsyncMock()
    bob = _user(role="member", user_id="u-bob", username="bob")
    target = {"owner_id": "u-alice"}
    with pytest.raises(HTTPException) as ei:
        await require_owner_or_admin(
            target, bob, audit_log=audit,
            target_kind="document", target_id="d-1",
        )
    assert ei.value.status_code == 403
    audit.write.assert_awaited_once()
    args, kwargs = audit.write.call_args
    assert args[0] == "authz.denied"
    assert kwargs["user_id"] == "u-bob"
    assert kwargs["target_kind"] == "document"
    assert kwargs["target_id"] == "d-1"


@pytest.mark.asyncio
async def test_null_owner_member_403():
    """B-4 / I-11: NULL owner_id (pre-T3 backfill) → member 403."""
    audit = AsyncMock()
    bob = _user(role="member", user_id="u-bob")
    target = {"owner_id": None}
    with pytest.raises(HTTPException) as ei:
        await require_owner_or_admin(
            target, bob, audit_log=audit,
            target_kind="document", target_id="d-old",
        )
    assert ei.value.status_code == 403
    audit.write.assert_awaited_once_with(
        "authz.denied",
        user_id="u-bob", target_kind="document", target_id="d-old",
        payload={"reason": "null_owner_admin_only", "attempted_role": "member"},
    )


@pytest.mark.asyncio
async def test_null_owner_admin_allowed():
    """B-4 / I-11: NULL owner_id → admin can write."""
    audit = AsyncMock()
    admin = _user(role="admin")
    target = {"owner_id": None}
    await require_owner_or_admin(
        target, admin, audit_log=audit,
        target_kind="document", target_id="d-old",
    )
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_system_owner_member_403():
    """T1a backfill sentinel '_system' is treated identical to NULL —
    member 403, admin allowed."""
    audit = AsyncMock()
    bob = _user(role="member", user_id="u-bob")
    target = {"owner_id": "_system"}
    with pytest.raises(HTTPException) as ei:
        await require_owner_or_admin(
            target, bob, audit_log=audit,
            target_kind="job", target_id="j-old",
        )
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_audit_write_failure_swallowed():
    """authz.denied is best-effort (A-3) — if audit DB blips, still 403."""
    audit = AsyncMock()
    audit.write = AsyncMock(side_effect=RuntimeError("audit dead"))
    bob = _user(role="member", user_id="u-bob")
    target = {"owner_id": "u-alice"}
    with pytest.raises(HTTPException) as ei:
        await require_owner_or_admin(
            target, bob, audit_log=audit,
            target_kind="document", target_id="d-1",
        )
    assert ei.value.status_code == 403   # still 403, not 500
