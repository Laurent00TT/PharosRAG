"""Best-effort audits: doc.ingest, job.cancel, job.failed (spec §4.5 / A-3)."""
import inspect
import json
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_pipeline_init_accepts_audit_log(audit_setup):
    """IngestionPipeline.__init__ must accept ``audit_log`` so worker.py's
    pipeline_factory can inject one."""
    from kb.ingestion.pipeline import IngestionPipeline
    sig = inspect.signature(IngestionPipeline.__init__)
    assert "audit_log" in sig.parameters
    assert sig.parameters["audit_log"].default is None


@pytest.mark.asyncio
async def test_executor_init_accepts_audit_log(audit_setup):
    from kb.ingestion.job_executor import IngestionJobExecutor
    sig = inspect.signature(IngestionJobExecutor.__init__)
    assert "audit_log" in sig.parameters
    assert sig.parameters["audit_log"].default is None


@pytest.mark.asyncio
async def test_doc_ingest_audit_shape(audit_setup):
    """doc.ingest payload contract: doc_id as target_id, doc_name + pages
    in payload, user_id may be None (T2) or owner (T3+)."""
    audit = audit_setup
    await audit.write(
        "doc.ingest",
        user_id=None,
        target_kind="document",
        target_id="doc-test",
        payload={"doc_name": "x.pdf", "page_count": 3},
    )
    rows = await audit.query(action="doc.ingest", limit=10)
    assert len(rows) == 1
    assert rows[0]["target_id"] == "doc-test"
    payload = json.loads(rows[0]["payload_json"])
    assert payload == {"doc_name": "x.pdf", "page_count": 3}


@pytest.mark.asyncio
async def test_job_cancel_writes_audit(test_app_with_users):
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    alice_id = setup["alice_user_id"]
    store = client.app.state.job_store
    # Real seeding via current SQLiteJobStore API. T3-5 requires owner_id
    # match for member cancel; this test exercises the audit-write path,
    # not the permission path, so we make alice the owner.
    job = await store.create_job(
        paths=["/fake/path.pdf"], config={}, owner_id=alice_id,
    )
    r = client.post(f"/ingestion/jobs/{job.job_id}/cancel",
                    headers={"X-API-Key": alice_token})
    assert r.status_code == 200
    rows = await audit.query(action="job.cancel", limit=10)
    assert len(rows) == 1
    assert rows[0]["target_id"] == job.job_id


@pytest.mark.asyncio
async def test_job_failed_terminal_audit_shape(audit_setup):
    """job.failed is written by job_executor on terminal (non-retryable)
    failure. Shape contract verified here; wiring is in Step 6.5."""
    audit = audit_setup
    await audit.write(
        "job.failed",
        user_id=None,
        target_kind="job",
        target_id="job-alice-1",
        payload={
            "item_id": "item-1",
            "error_type": "ValueError",
            "error_msg": "broken pdf",
        },
    )
    rows = await audit.query(action="job.failed", limit=10)
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload_json"])
    assert payload["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_job_cancel_audit_failure_does_not_break_cancel(test_app_with_users):
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    alice_id = setup["alice_user_id"]
    store = client.app.state.job_store
    # T3-5: alice owns the job so we exercise the audit-failure path
    # rather than the permission-denial path.
    job = await store.create_job(
        paths=["/fake/path.pdf"], config={}, owner_id=alice_id,
    )
    with patch("kb.audit.log.AuditLog._db_insert_best_effort",
               side_effect=RuntimeError("audit dead")):
        r = client.post(f"/ingestion/jobs/{job.job_id}/cancel",
                        headers={"X-API-Key": alice_token})
    # Cancel itself must still succeed (A-3 best-effort)
    assert r.status_code == 200
