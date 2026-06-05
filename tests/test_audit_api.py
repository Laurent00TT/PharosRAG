"""GET /audit endpoint — scoping with REAL user_ids, filters, pagination."""
import asyncio
from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_member_sees_only_own_audits(test_app_with_users):
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    alice_id, bob_id = setup["alice_user_id"], setup["bob_user_id"]
    await audit.write("doc.ingest", user_id=alice_id,
                      target_kind="document", target_id="d-1")
    await audit.write("doc.ingest", user_id=bob_id,
                      target_kind="document", target_id="d-2")
    r = client.get("/audit", headers={"X-API-Key": alice_token})
    assert r.status_code == 200
    rows = r.json()["rows"]
    # Note: rows include the user.create audits from fixture-seeded users.
    # Filter to the rows we just wrote.
    own = [row for row in rows if row["target_id"] in {"d-1", "d-2"}]
    assert all(row["user_id"] == alice_id for row in own)


@pytest.mark.asyncio
async def test_admin_sees_all_audits(test_app_with_users):
    setup = test_app_with_users
    client, admin_token, audit = setup["client"], setup["admin_token"], setup["audit_log"]
    alice_id, bob_id = setup["alice_user_id"], setup["bob_user_id"]
    await audit.write("doc.ingest", user_id=alice_id,
                      target_kind="document", target_id="d-1")
    await audit.write("doc.ingest", user_id=bob_id,
                      target_kind="document", target_id="d-2")
    r = client.get("/audit", headers={"X-API-Key": admin_token})
    rows = r.json()["rows"]
    user_ids = {row["user_id"] for row in rows}
    assert alice_id in user_ids
    assert bob_id in user_ids


@pytest.mark.asyncio
async def test_admin_can_filter_by_specific_user(test_app_with_users):
    setup = test_app_with_users
    client, admin_token, audit = setup["client"], setup["admin_token"], setup["audit_log"]
    alice_id, bob_id = setup["alice_user_id"], setup["bob_user_id"]
    await audit.write("doc.ingest", user_id=alice_id,
                      target_kind="document", target_id="d-1")
    await audit.write("doc.ingest", user_id=bob_id,
                      target_kind="document", target_id="d-2")
    r = client.get(f"/audit?user_id={bob_id}", headers={"X-API-Key": admin_token})
    rows = r.json()["rows"]
    assert all(row["user_id"] == bob_id for row in rows)


@pytest.mark.asyncio
async def test_member_cannot_filter_to_other_users_audit(test_app_with_users):
    """?user_id=bob is silently ignored for members — they see only own."""
    setup = test_app_with_users
    client, alice_token, audit = setup["client"], setup["alice_token"], setup["audit_log"]
    bob_id = setup["bob_user_id"]
    await audit.write("doc.ingest", user_id=bob_id,
                      target_kind="document", target_id="d-1")
    r = client.get(f"/audit?user_id={bob_id}", headers={"X-API-Key": alice_token})
    rows = r.json()["rows"]
    # No row should have bob's user_id — query is forced back to alice
    assert not any(row["user_id"] == bob_id for row in rows)


@pytest.mark.asyncio
async def test_filter_by_action(test_app_with_users):
    setup = test_app_with_users
    client, admin_token, audit = setup["client"], setup["admin_token"], setup["audit_log"]
    alice_id = setup["alice_user_id"]
    await audit.write("doc.ingest", user_id=alice_id,
                      target_kind="document", target_id="d-1")
    await audit.write("job.cancel", user_id=alice_id,
                      target_kind="job", target_id="j-1")
    r = client.get("/audit?action=job.cancel", headers={"X-API-Key": admin_token})
    rows = r.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["action"] == "job.cancel"


@pytest.mark.asyncio
async def test_limit_default_and_cap(test_app_with_users):
    setup = test_app_with_users
    client, admin_token, audit = setup["client"], setup["admin_token"], setup["audit_log"]
    alice_id = setup["alice_user_id"]
    for i in range(80):
        await audit.write("doc.ingest", user_id=alice_id,
                          target_kind="document", target_id=f"d-{i}")
    # Default limit = 50
    r = client.get("/audit?action=doc.ingest", headers={"X-API-Key": admin_token})
    assert len(r.json()["rows"]) == 50
    # Above max gets clamped to audit_max_limit (default 500)
    r = client.get("/audit?action=doc.ingest&limit=10000",
                   headers={"X-API-Key": admin_token})
    assert len(r.json()["rows"]) <= 500
    assert len(r.json()["rows"]) == 80  # all of them, since 80 < 500


@pytest.mark.asyncio
async def test_filter_by_since_utc(test_app_with_users):
    """since must be parsed as UTC, not local time (P1-2)."""
    setup = test_app_with_users
    client, admin_token, audit = setup["client"], setup["admin_token"], setup["audit_log"]
    alice_id = setup["alice_user_id"]
    await audit.write("doc.ingest", user_id=alice_id,
                      target_kind="document", target_id="d-old")
    await asyncio.sleep(0.05)
    cutoff = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    await asyncio.sleep(0.05)
    await audit.write("doc.ingest", user_id=alice_id,
                      target_kind="document", target_id="d-new")
    r = client.get(f"/audit?since={cutoff}", headers={"X-API-Key": admin_token})
    ids = {row["target_id"] for row in r.json()["rows"]}
    assert "d-new" in ids
    assert "d-old" not in ids


@pytest.mark.asyncio
async def test_unauthenticated_returns_401(test_app_with_users):
    client = test_app_with_users["client"]
    r = client.get("/audit")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_disabled_user_returns_401(test_app_with_users):
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    alice_id = setup["alice_user_id"]
    store = client.app.state.users_store
    await store.disable_user(alice_id)
    r = client.get("/audit", headers={"X-API-Key": alice_token})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_audit_endpoint_returns_well_formed_rows(test_app_with_users):
    setup = test_app_with_users
    client, admin_token, audit = setup["client"], setup["admin_token"], setup["audit_log"]
    alice_id = setup["alice_user_id"]
    await audit.write("doc.ingest", user_id=alice_id,
                      target_kind="document", target_id="d-1",
                      payload={"foo": "bar"})
    r = client.get("/audit", headers={"X-API-Key": admin_token})
    rows = r.json()["rows"]
    assert len(rows) >= 1
    row = rows[0]
    assert {"event_id", "user_id", "action", "target_kind",
            "target_id", "ts", "payload"} <= set(row.keys())
    # payload is parsed dict, not raw JSON string
    assert isinstance(row["payload"], dict)
