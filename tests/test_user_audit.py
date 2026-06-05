"""user.* single-phase mandatory audits -- same-session atomicity (A-1)."""
import json

import pytest

from kb.audit.log import AuditLog
from kb.auth.users import UsersStore


@pytest.fixture
async def stack(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    users = UsersStore(db_url=db_url)
    audit = AuditLog(db_url=db_url, fallback_path=tmp_path / "fallback.jsonl")
    try:
        await users.init()
        await audit.init()
        yield users, audit, tmp_path
    finally:
        await audit.aclose()
        await users.aclose()


@pytest.mark.asyncio
async def test_create_user_writes_audit(stack):
    users, audit, _ = stack
    _, alice = await users.create_user(
        username="alice", role="member",
        audit_log=audit, acting_user_id="_system",
    )
    rows = await audit.query(action="user.create", limit=10)
    assert len(rows) == 1
    assert rows[0]["user_id"] == "_system"
    assert rows[0]["target_kind"] == "user"
    assert rows[0]["target_id"] == alice.user_id
    payload = json.loads(rows[0]["payload_json"])
    assert payload["username"] == "alice"
    assert payload["role"] == "member"


@pytest.mark.asyncio
async def test_create_user_atomic_rollback_on_dup_username(stack):
    """Same-session atomicity: a unique-constraint violation on a second
    `create alice` must roll back both the user row attempt AND the audit
    row (so we don't end up with an orphan user.create audit pointing at
    a user that doesn't exist)."""
    users, audit, _ = stack
    await users.create_user(
        username="alice", role="member",
        audit_log=audit, acting_user_id="_system",
    )
    with pytest.raises(ValueError, match="already exists"):
        await users.create_user(
            username="alice", role="member",
            audit_log=audit, acting_user_id="_system",
        )
    rows = await audit.query(action="user.create", limit=10)
    assert len(rows) == 1, "exactly one audit row, not a second from the failed attempt"


@pytest.mark.asyncio
async def test_disable_user_writes_audit_and_is_atomic(stack):
    users, audit, _ = stack
    _, alice = await users.create_user(
        username="alice", role="member",
        audit_log=audit, acting_user_id="_system",
    )
    await users.disable_user(
        alice.user_id, audit_log=audit, acting_user_id="_system",
    )
    rows = await audit.query(action="user.disable", limit=10)
    assert len(rows) == 1
    assert rows[0]["target_id"] == alice.user_id


@pytest.mark.asyncio
async def test_set_role_writes_old_and_new_in_payload(stack):
    users, audit, _ = stack
    _, alice = await users.create_user(
        username="alice", role="member",
        audit_log=audit, acting_user_id="_system",
    )
    await users.set_role(
        alice.user_id, "admin", audit_log=audit, acting_user_id="_system",
    )
    rows = await audit.query(action="user.role_change", limit=10)
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload_json"])
    assert payload == {"old_role": "member", "new_role": "admin"}


@pytest.mark.asyncio
async def test_reset_key_audit_does_not_leak_secret(stack):
    users, audit, _ = stack
    _, alice = await users.create_user(
        username="alice", role="member",
        audit_log=audit, acting_user_id="_system",
    )
    new_plaintext = await users.reset_key(
        alice.user_id, audit_log=audit, acting_user_id="_system",
    )
    rows = await audit.query(action="user.key_reset", limit=10)
    payload_json = rows[0]["payload_json"]
    assert new_plaintext not in payload_json, "full secret must never enter audit"
    payload = json.loads(payload_json)
    assert "new_key_prefix" in payload
    assert "old_key_prefix" in payload


@pytest.mark.asyncio
async def test_create_user_without_audit_log_still_works(stack):
    """audit_log kwarg is optional (existing tests / one-shot scripts)."""
    users, _, _ = stack
    _, alice = await users.create_user(username="alice", role="member")
    assert alice.username == "alice"
