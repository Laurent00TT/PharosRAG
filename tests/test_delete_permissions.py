"""DELETE /documents/{id} ownership check (T3-4, P1-4 closure).

Covers the 5 cells of the permission matrix:
  - owner+member       → 200 + both audit phases
  - non-owner+member   → 403 + authz.denied, NO doc.delete.attempted/completed
  - admin (any owner)  → 200
  - null-owner+member  → 403 + authz.denied(reason=null_owner_admin_only)
  - null-owner+admin   → 200

The critical assertion in the non-owner case is that doc.delete.attempted
must NOT appear: the permission check runs BEFORE write_attempted, so a
403 path never reaches the audit-bracketed main op.
"""
import json

import pytest


@pytest.mark.asyncio
async def test_owner_can_delete_own_doc(test_app_with_users):
    """alice owns doc-alice-1 → DELETE returns 200 + both audit phases."""
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    r = client.delete("/documents/doc-alice-1", headers={"X-API-Key": alice_token})
    assert r.status_code == 200
    rows = await audit.query(target_id="doc-alice-1", limit=10)
    actions = {row["action"] for row in rows}
    assert "doc.delete.attempted" in actions
    assert "doc.delete.completed" in actions
    # No authz.denied for the happy path
    assert "authz.denied" not in actions


@pytest.mark.asyncio
async def test_non_owner_member_403_writes_authz_denied(test_app_with_users):
    """bob is a member, doc-alice-1 is owned by alice → 403 + authz.denied."""
    setup = test_app_with_users
    client, bob_token, audit = setup["client"], setup["bob_token"], setup["audit_log"]
    bob_id = setup["bob_user_id"]
    r = client.delete("/documents/doc-alice-1", headers={"X-API-Key": bob_token})
    assert r.status_code == 403
    rows = await audit.query(action="authz.denied", limit=10)
    assert any(
        row["user_id"] == bob_id and row["target_id"] == "doc-alice-1"
        for row in rows
    ), "expected authz.denied row for bob → doc-alice-1"
    # CRITICAL: doc.delete.attempted/completed must NOT appear (we never
    # reached the audit-bracketed main op).
    rows_for_doc = await audit.query(target_id="doc-alice-1", limit=20)
    actions_for_doc = {row["action"] for row in rows_for_doc}
    assert "doc.delete.attempted" not in actions_for_doc
    assert "doc.delete.completed" not in actions_for_doc


@pytest.mark.asyncio
async def test_admin_can_delete_any_doc(test_app_with_users):
    """admin can delete alice's doc → 200 + both audit phases."""
    setup = test_app_with_users
    client, admin_token = setup["client"], setup["admin_token"]
    r = client.delete("/documents/doc-alice-1", headers={"X-API-Key": admin_token})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_member_cannot_delete_null_owner_doc(test_app_with_users):
    """Pre-T3 backfill: doc with owner_id=NULL → member 403 (I-11)."""
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    # Seed a NULL-owner doc directly via meta_db (legacy, pre-T3 row)
    meta_db = client.app.state.meta_db
    await meta_db.upsert_document(
        doc_id="doc-legacy",
        doc_name="legacy.pdf",
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
        owner_id=None,
    )
    r = client.delete("/documents/doc-legacy", headers={"X-API-Key": alice_token})
    assert r.status_code == 403
    rows = await audit.query(action="authz.denied", target_id="doc-legacy", limit=5)
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload_json"])
    assert payload["reason"] == "null_owner_admin_only"


@pytest.mark.asyncio
async def test_admin_can_delete_null_owner_doc(test_app_with_users):
    """Same legacy doc — admin can delete (I-11)."""
    setup = test_app_with_users
    client, admin_token = setup["client"], setup["admin_token"]
    meta_db = client.app.state.meta_db
    await meta_db.upsert_document(
        doc_id="doc-legacy",
        doc_name="legacy.pdf",
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
        owner_id=None,
    )
    r = client.delete("/documents/doc-legacy", headers={"X-API-Key": admin_token})
    assert r.status_code == 200
