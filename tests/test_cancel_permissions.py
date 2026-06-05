"""POST /jobs/{id}/cancel ownership check (T3-5).

Covers the 4 cells of the cancel-permission matrix:
  - owner+member        → 200 + job.cancel audit row (target_id=job_id)
  - non-owner+member    → 403 + authz.denied (target_id=job_id)
  - admin (any owner)   → 200
  - _system owner+member → 403 (I-11: pre-T3 backfill rows are admin-only)

Critical: the cancel handler must run the permission check BEFORE
``store.request_cancel`` so a denied member never mutates job state.
``test_non_owner_member_cancel_403`` does not directly assert "no
cancel_requested mutation" (that lives in the handler reorder rather
than a separate test); the spec's plain-English contract is the only
guarantee, and is enforced by reading the handler source rather than
by a runtime probe.
"""
import pytest


@pytest.mark.asyncio
async def test_owner_can_cancel_own_job(test_app_with_users):
    """alice owns job → POST /cancel returns 200 + job.cancel audit."""
    setup = test_app_with_users
    client, alice_token, audit = (
        setup["client"], setup["alice_token"], setup["audit_log"]
    )
    alice_id = setup["alice_user_id"]
    store = client.app.state.job_store
    job = await store.create_job(
        paths=["/fake.pdf"], config={}, owner_id=alice_id,
    )
    r = client.post(
        f"/ingestion/jobs/{job.job_id}/cancel",
        headers={"X-API-Key": alice_token},
    )
    assert r.status_code == 200
    rows = await audit.query(action="job.cancel", target_id=job.job_id, limit=5)
    assert len(rows) == 1
    assert rows[0]["target_id"] == job.job_id


@pytest.mark.asyncio
async def test_non_owner_member_cancel_403(test_app_with_users):
    """alice's job, bob (member) tries to cancel → 403 + authz.denied row."""
    setup = test_app_with_users
    client, bob_token, audit = (
        setup["client"], setup["bob_token"], setup["audit_log"]
    )
    alice_id, bob_id = setup["alice_user_id"], setup["bob_user_id"]
    store = client.app.state.job_store
    job = await store.create_job(
        paths=["/fake.pdf"], config={}, owner_id=alice_id,
    )
    r = client.post(
        f"/ingestion/jobs/{job.job_id}/cancel",
        headers={"X-API-Key": bob_token},
    )
    assert r.status_code == 403
    rows = await audit.query(action="authz.denied", target_id=job.job_id, limit=5)
    assert len(rows) == 1
    assert rows[0]["user_id"] == bob_id
    assert rows[0]["target_id"] == job.job_id


@pytest.mark.asyncio
async def test_admin_can_cancel_any_job(test_app_with_users):
    """admin can cancel alice's job (role hierarchy bypass)."""
    setup = test_app_with_users
    client, admin_token = setup["client"], setup["admin_token"]
    alice_id = setup["alice_user_id"]
    store = client.app.state.job_store
    job = await store.create_job(
        paths=["/fake.pdf"], config={}, owner_id=alice_id,
    )
    r = client.post(
        f"/ingestion/jobs/{job.job_id}/cancel",
        headers={"X-API-Key": admin_token},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_member_cannot_cancel_system_owned_legacy_job(test_app_with_users):
    """T1a backfill ingestion_jobs.owner_id='_system' → admin-only (I-11)."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    store = client.app.state.job_store
    job = await store.create_job(
        paths=["/fake.pdf"], config={}, owner_id="_system",
    )
    r = client.post(
        f"/ingestion/jobs/{job.job_id}/cancel",
        headers={"X-API-Key": alice_token},
    )
    assert r.status_code == 403


# ── retry permission (T3 follow-up, P1a review) ──────────────────────


@pytest.mark.asyncio
async def test_owner_can_retry_own_job(test_app_with_users):
    """T3 follow-up: retry mirrors cancel — owner allowed, denied member
    must NOT resurrect the job."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    alice_id = setup["alice_user_id"]
    store = client.app.state.job_store
    job = await store.create_job(
        paths=["/fake.pdf"], config={}, owner_id=alice_id,
    )
    r = client.post(
        f"/ingestion/jobs/{job.job_id}/retry",
        headers={"X-API-Key": alice_token},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_non_owner_member_retry_403_does_not_requeue(test_app_with_users):
    """P1a fix: a non-owner member's retry must 403 AND must not mutate
    the failed items back to queued. Verify by inspecting item rows
    directly after the denial."""
    setup = test_app_with_users
    client, bob_token = setup["client"], setup["bob_token"]
    alice_id, bob_id = setup["alice_user_id"], setup["bob_user_id"]
    audit = setup["audit_log"]
    store = client.app.state.job_store
    job = await store.create_job(
        paths=["/fake.pdf"], config={}, owner_id=alice_id,
    )
    # Drive the single item into FAILED state without going through the
    # full pipeline. JobError is required for fail_item; construct a
    # minimal one matching the dataclass.
    items = await store.list_job_items(job.job_id)
    from kb.jobs.errors import JobError
    await store.fail_item(
        items[0].item_id,
        JobError(error_type="ForcedFail", message="seeded", retryable=False),
        None,
    )
    # Bob tries to retry alice's failed job → 403
    r = client.post(
        f"/ingestion/jobs/{job.job_id}/retry",
        headers={"X-API-Key": bob_token},
    )
    assert r.status_code == 403
    # Verify the failed item stayed FAILED (was not resurrected to QUEUED)
    items_after = await store.list_job_items(job.job_id)
    from kb.jobs.sqlite_store import ITEM_FAILED
    assert items_after[0].status == ITEM_FAILED, (
        "denied retry must not requeue: item still FAILED"
    )
    # authz.denied audit row records bob's attempt against the job
    rows = await audit.query(action="authz.denied", target_id=job.job_id, limit=5)
    assert any(r["user_id"] == bob_id for r in rows)


@pytest.mark.asyncio
async def test_admin_can_retry_any_job(test_app_with_users):
    """admin override — can retry anyone's job, mirrors cancel admin path."""
    setup = test_app_with_users
    client, admin_token = setup["client"], setup["admin_token"]
    alice_id = setup["alice_user_id"]
    store = client.app.state.job_store
    job = await store.create_job(
        paths=["/fake.pdf"], config={}, owner_id=alice_id,
    )
    r = client.post(
        f"/ingestion/jobs/{job.job_id}/retry",
        headers={"X-API-Key": admin_token},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_retry_404_for_missing_job(test_app_with_users):
    """Missing job → 404 BEFORE permission check (mirrors cancel)."""
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    r = client.post(
        "/ingestion/jobs/no-such-job/retry",
        headers={"X-API-Key": alice_token},
    )
    assert r.status_code == 404
