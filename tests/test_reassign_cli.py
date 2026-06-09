"""manage_users.py reassign + disable --reassign-to (T5-9)."""
import asyncio
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
async def seeded(tmp_path, monkeypatch):
    """Two users + two docs owned by alice."""
    from kb.audit.log import AuditLog
    from kb.auth.users import UsersStore
    from kb.storage.metadata_db import MetadataDB

    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    monkeypatch.setenv("KB_USERS_DB_URL", db_url)

    users = UsersStore(db_url=db_url)
    await users.init()
    meta = MetadataDB(db_url=db_url)
    await meta.init()
    audit = AuditLog(db_url=db_url, fallback_path=tmp_path / "audit.jsonl")
    await audit.init()

    _, alice = await users.create_user(username="alice", role="member", audit_log=audit, acting_user_id="_system")
    _, bob = await users.create_user(username="bob",   role="member", audit_log=audit, acting_user_id="_system")
    for did in ("d1", "d2"):
        await meta.upsert_document(
            doc_id=did, doc_name=f"{did}.pdf", version=None,
            effective_date=None, expiry_date=None, supersedes=None,
            owner_id=alice.user_id,
        )

    yield {"users": users, "meta": meta, "audit": audit, "alice": alice, "bob": bob, "tmp": tmp_path}
    await audit.aclose()
    await meta.aclose()
    await users.aclose()


@pytest.mark.asyncio
async def test_reassign_updates_documents_owner_id(seeded):
    """After reassign, both docs point to bob — alice keeps everything else."""
    import scripts.manage_users as mu
    args = type("X", (), dict(from_username="alice", to_username="bob"))()
    rc = await mu.cmd_reassign(args)
    assert rc == 0
    for did in ("d1", "d2"):
        doc = await seeded["meta"].get_document(did)
        assert doc["owner_id"] == seeded["bob"].user_id


@pytest.mark.asyncio
async def test_reassign_writes_two_phase_audit_per_doc(seeded):
    """For each doc: one doc.reassign.attempted, one doc.reassign.completed."""
    import scripts.manage_users as mu
    args = type("X", (), dict(from_username="alice", to_username="bob"))()
    await mu.cmd_reassign(args)
    attempted = await seeded["audit"].query(action="doc.reassign.attempted", limit=10)
    completed = await seeded["audit"].query(action="doc.reassign.completed", limit=10)
    assert {r["target_id"] for r in attempted} == {"d1", "d2"}
    assert {r["target_id"] for r in completed} == {"d1", "d2"}


@pytest.mark.asyncio
async def test_reassign_does_not_modify_historical_audit_rows(seeded):
    """Original ingest / delete audit rows MUST keep their original user_id —
    reassign only adds NEW reassign rows, never updates old user_id."""
    await seeded["audit"].write(
        "doc.ingest",
        user_id=seeded["alice"].user_id,
        target_kind="document",
        target_id="d1",
    )
    import scripts.manage_users as mu
    args = type("X", (), dict(from_username="alice", to_username="bob"))()
    await mu.cmd_reassign(args)
    # Old row still points to alice (NOT bob)
    rows = await seeded["audit"].query(action="doc.ingest", target_id="d1", limit=10)
    assert rows[0]["user_id"] == seeded["alice"].user_id


@pytest.mark.asyncio
async def test_disable_with_reassign_to_chains_both(seeded):
    """disable --reassign-to bob: alice is disabled AND her docs are bob's."""
    import scripts.manage_users as mu
    args = type("X", (), dict(username="alice", reassign_to="bob"))()
    rc = await mu.cmd_disable(args)
    assert rc == 0
    # alice is disabled
    u = await seeded["users"].get_by_username("alice")
    assert u.disabled_at is not None
    # alice's docs are now bob's
    for did in ("d1", "d2"):
        doc = await seeded["meta"].get_document(did)
        assert doc["owner_id"] == seeded["bob"].user_id
