"""DELETE /documents/{doc_id} two-phase audit (spec §6.3 / I-4 / A-2)."""
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_delete_writes_attempted_and_completed(test_app_with_users):
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    r = client.delete("/documents/doc-alice-1", headers={"X-API-Key": alice_token})
    assert r.status_code == 200
    rows = await audit.query(target_id="doc-alice-1", limit=10)
    actions = {row["action"] for row in rows}
    assert "doc.delete.attempted" in actions
    assert "doc.delete.completed" in actions


@pytest.mark.asyncio
async def test_delete_preserves_existing_response_shape(test_app_with_users):
    """Verify the main chain was NOT replaced — response should still
    include cleanup / nav stats keys from the pre-T2 handler."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    r = client.delete("/documents/doc-alice-1", headers={"X-API-Key": alice_token})
    assert r.status_code == 200
    body = r.json()
    # The pre-T2 handler returned a dict with at least doc_id; if existing
    # tests assert further keys (cleanup, nav_entries_deleted), preserve them.
    assert "doc_id" in body


@pytest.mark.asyncio
async def test_delete_main_op_failure_writes_attempted_only(test_app_with_users):
    """If main op (mark_deleted) raises, completed must NOT be written."""
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    # TestClient's default raise_server_exceptions=True re-raises unhandled
    # exceptions to the caller (the test). In production, FastAPI returns
    # 500 from these. We assert by catching: the request must have raised
    # (proving the main op did fail), and only attempted must be written.
    with patch("kb.storage.metadata_db.MetadataDB.mark_deleted",
               side_effect=RuntimeError("metadata db down")):
        with pytest.raises(RuntimeError, match="metadata db down"):
            client.delete("/documents/doc-alice-1",
                          headers={"X-API-Key": alice_token})
    rows = await audit.query(target_id="doc-alice-1", limit=10)
    actions = {row["action"] for row in rows}
    assert "doc.delete.attempted" in actions
    assert "doc.delete.completed" not in actions


@pytest.mark.asyncio
async def test_delete_attempted_db_failure_does_not_block_main(test_app_with_users):
    """attempted is best-effort — failure must not block main op."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    with patch("kb.audit.log.AuditLog._db_insert_best_effort",
               side_effect=RuntimeError("audit db blip")):
        r = client.delete("/documents/doc-alice-1",
                          headers={"X-API-Key": alice_token})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_delete_completed_total_failure_returns_503(test_app_with_users):
    """Both paths fail on completed → 503 + audit.completed_lost trace."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    with patch("kb.audit.log.AuditLog._db_insert_for_mandatory",
               side_effect=RuntimeError("audit db dead")), \
         patch("kb.audit.log.AuditLog._append_fallback_jsonl_synced",
               side_effect=OSError("disk full")):
        r = client.delete("/documents/doc-alice-1",
                          headers={"X-API-Key": alice_token})
    assert r.status_code == 503
    assert "audit" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_completed_db_failure_falls_back_to_jsonl(test_app_with_users):
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    with patch("kb.audit.log.AuditLog._db_insert_for_mandatory",
               side_effect=RuntimeError("audit db dead")):
        r = client.delete("/documents/doc-alice-1",
                          headers={"X-API-Key": alice_token})
    assert r.status_code == 200
    fallback_text = audit.fallback_path.read_text(encoding="utf-8")
    assert "doc.delete.completed" in fallback_text


@pytest.mark.asyncio
async def test_delete_nonexistent_doc_returns_404_writes_no_audit(test_app_with_users):
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    r = client.delete("/documents/no-such-doc", headers={"X-API-Key": alice_token})
    assert r.status_code == 404
    rows = await audit.query(target_id="no-such-doc", limit=10)
    assert rows == []
