"""AuditLog unit tests — three semantics + fallback + strict write() guard."""
import json
from pathlib import Path

import pytest

from kb.audit.log import (
    MANDATORY_ACTIONS,
    TWO_PHASE_ACTIONS,
    AuditContractError,
    AuditLog,
    AuditWriteFailure,
)
from kb.storage.metadata_db import AuditLogRecord


@pytest.fixture
async def audit_log(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    log = AuditLog(
        db_url=db_url,
        fallback_path=tmp_path / "audit_fallback.jsonl",
    )
    await log.init()
    yield log
    await log.aclose()


# ── Action membership ────────────────────────────────────────────────

def test_mandatory_action_set_matches_spec():
    assert MANDATORY_ACTIONS == frozenset({
        "user.create", "user.disable", "user.role_change", "user.key_reset",
    })


def test_two_phase_action_set_matches_spec():
    assert TWO_PHASE_ACTIONS == frozenset({"doc.delete", "doc.reassign"})


def test_disjoint_action_sets():
    assert MANDATORY_ACTIONS.isdisjoint(TWO_PHASE_ACTIONS)


# ── init() creates audit_log table when DB is fresh ─────────────────

@pytest.mark.asyncio
async def test_init_creates_audit_log_table_on_fresh_db(tmp_path):
    """AuditLog.init() must not depend on MetadataDB.init() — CLI bootstrap
    path constructs AuditLog directly to write user.create audit."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    log = AuditLog(db_url=db_url, fallback_path=tmp_path / "fb.jsonl")
    await log.init()
    await log.write(
        "doc.ingest", user_id="u-x", target_kind="document", target_id="d-1",
    )
    rows = await log.query(limit=10)
    assert len(rows) == 1
    await log.aclose()


# ── write() strict guards ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_rejects_two_phase_action(audit_log):
    with pytest.raises(AuditContractError, match="two-phase"):
        await audit_log.write(
            "doc.delete", user_id="u-x",
            target_kind="document", target_id="d-1",
        )


@pytest.mark.asyncio
async def test_write_rejects_mandatory_action(audit_log):
    with pytest.raises(AuditContractError, match="single-phase mandatory"):
        await audit_log.write(
            "user.create", user_id="u-admin",
            target_kind="user", target_id="u-bob",
        )


@pytest.mark.asyncio
async def test_write_rejects_derived_two_phase_actions(audit_log):
    """Derived suffixes (e.g. ``doc.delete.completed``) must NOT slip
    through best-effort ``write()`` — that would bypass the mandatory
    fallback path and silently lose audit on DB failure."""
    derived = [
        "doc.delete.attempted",  "doc.delete.completed",
        "doc.reassign.attempted", "doc.reassign.completed",
    ]
    for action in derived:
        with pytest.raises(AuditContractError, match="derived two-phase"):
            await audit_log.write(
                action, user_id="u-x",
                target_kind="document", target_id="d-1",
            )


def test_audit_contract_error_survives_python_dash_O():
    """Contract guards use real ``raise``, not ``assert`` — they must
    survive ``python -O`` (which strips assert statements). Verifying
    by reading the source: search for any 'assert ' in the public
    methods. If this finds one, switch it to raise AuditContractError."""
    import inspect
    from kb.audit import log as log_mod
    src = inspect.getsource(log_mod)
    # Find lines with 'assert ' that are inside method bodies (not
    # in test docstrings, not comments — naively skip lines starting
    # with '#' or inside triple-quoted strings is hard; this check is
    # intentionally simple — if you legitimately need an assert inside
    # log.py for some non-contract reason, add an 'assert is fine'
    # comment on the line so this test can exclude it.)
    offending = [
        line.strip() for line in src.splitlines()
        if line.lstrip().startswith("assert ") and "is fine" not in line
    ]
    assert not offending, (
        f"log.py contains assert statements that would be stripped under "
        f"python -O — convert to `raise AuditContractError(...)`: {offending}"
    )


# ── Happy path: write() for best-effort actions ─────────────────────

@pytest.mark.asyncio
async def test_write_best_effort_inserts_row(audit_log):
    await audit_log.write(
        "doc.ingest",
        user_id="u-alice",
        target_kind="document",
        target_id="doc-1",
        payload={"doc_name": "manual.pdf"},
    )
    rows = await audit_log.query(limit=10)
    assert len(rows) == 1
    assert rows[0]["action"] == "doc.ingest"
    assert json.loads(rows[0]["payload_json"]) == {"doc_name": "manual.pdf"}


@pytest.mark.asyncio
async def test_write_best_effort_swallows_db_failure(audit_log, monkeypatch):
    async def boom(self, *a, **kw):
        raise RuntimeError("simulated db down")
    monkeypatch.setattr(AuditLog, "_db_insert_best_effort", boom)
    # No raise:
    await audit_log.write(
        "doc.ingest", user_id="u-x",
        target_kind="document", target_id="d-1",
    )


# ── make_record() (single-phase mandatory contract) ─────────────────

@pytest.mark.asyncio
async def test_make_record_returns_orm_record_for_mandatory_action(audit_log):
    rec = audit_log.make_record(
        "user.create",
        user_id="u-admin",
        target_kind="user",
        target_id="u-bob",
        payload={"username": "bob", "role": "member"},
    )
    assert isinstance(rec, AuditLogRecord)
    assert rec.action == "user.create"
    assert rec.user_id == "u-admin"
    assert rec.target_id == "u-bob"
    assert json.loads(rec.payload_json) == {"username": "bob", "role": "member"}


def test_make_record_rejects_non_mandatory_action(audit_log):
    with pytest.raises(AuditContractError, match="single-phase mandatory"):
        audit_log.make_record(
            "doc.ingest",
            user_id="u-x", target_kind="document", target_id="d-1",
        )
    with pytest.raises(AuditContractError, match="single-phase mandatory"):
        audit_log.make_record(
            "doc.delete",
            user_id="u-x", target_kind="document", target_id="d-1",
        )


# ── Two-phase write_attempted / write_completed ─────────────────────

@pytest.mark.asyncio
async def test_write_attempted_then_completed_two_phase(audit_log):
    await audit_log.write_attempted(
        "doc.delete",
        user_id="u-alice",
        target_kind="document",
        target_id="doc-1",
    )
    await audit_log.write_completed(
        "doc.delete",
        user_id="u-alice",
        target_kind="document",
        target_id="doc-1",
    )
    rows = await audit_log.query(limit=10)
    actions = {r["action"] for r in rows}
    assert "doc.delete.attempted" in actions
    assert "doc.delete.completed" in actions


@pytest.mark.asyncio
async def test_write_attempted_rejects_non_two_phase(audit_log):
    with pytest.raises(AuditContractError, match="not two-phase"):
        await audit_log.write_attempted(
            "user.create", user_id="u-x",
            target_kind="user", target_id="u-y",
        )


@pytest.mark.asyncio
async def test_write_completed_db_failure_falls_back_to_jsonl(audit_log, monkeypatch):
    async def boom(self, *a, **kw):
        raise RuntimeError("audit db dead")
    monkeypatch.setattr(AuditLog, "_db_insert_for_mandatory", boom)

    await audit_log.write_completed(
        "doc.delete",
        user_id="u-alice",
        target_kind="document",
        target_id="doc-1",
    )
    line = Path(audit_log.fallback_path).read_text(encoding="utf-8").splitlines()[-1]
    rec = json.loads(line)
    assert rec["action"] == "doc.delete.completed"
    assert rec["via_fallback"] is True


@pytest.mark.asyncio
async def test_write_completed_both_paths_failed_raises(audit_log, monkeypatch):
    async def boom_db(self, *a, **kw):
        raise RuntimeError("db dead")
    def boom_jsonl(self, *a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(AuditLog, "_db_insert_for_mandatory", boom_db)
    monkeypatch.setattr(AuditLog, "_append_fallback_jsonl_synced", boom_jsonl)

    with pytest.raises(AuditWriteFailure):
        await audit_log.write_completed(
            "doc.delete", user_id="u-x",
            target_kind="document", target_id="d-1",
        )


# ── Query API ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_filter_by_user_id_and_action(audit_log):
    await audit_log.write("doc.ingest", user_id="u-alice",
                          target_kind="document", target_id="d-1")
    await audit_log.write("job.cancel", user_id="u-bob",
                          target_kind="job", target_id="j-1")
    rows = await audit_log.query(user_id="u-alice", limit=10)
    assert len(rows) == 1
    assert rows[0]["user_id"] == "u-alice"

    rows = await audit_log.query(action="job.cancel", limit=10)
    assert len(rows) == 1
    assert rows[0]["target_id"] == "j-1"


@pytest.mark.asyncio
async def test_query_orders_newest_first_and_respects_limit(audit_log):
    for i in range(5):
        await audit_log.write(
            "doc.ingest", user_id="u-a",
            target_kind="document", target_id=f"d-{i}",
        )
    rows = await audit_log.query(limit=3)
    assert len(rows) == 3
    assert rows[0]["target_id"] == "d-4"


@pytest.mark.asyncio
async def test_query_filter_by_actions_in_list(audit_log):
    """T5-followup: doctor's destructive_7d collapsed 8 serial queries
    into one via `WHERE action IN (...)`. The actions kwarg must
    accept a list and return rows matching ANY of them, newest-first.

    Uses actions that route through write() cleanly (non-MANDATORY,
    non-TWO_PHASE) so the test stays isolated from the audit
    contract — the kwarg semantics are orthogonal to the contract."""
    await audit_log.write("doc.ingest",   user_id="u-a", target_kind="document", target_id="d1")
    await audit_log.write("job.cancel",   user_id="u-a", target_kind="job",      target_id="j1")
    await audit_log.write("authz.denied", user_id="u-a", target_kind="document", target_id="d2")
    await audit_log.write("doc.ingest",   user_id="u-b", target_kind="document", target_id="d3")
    rows = await audit_log.query(
        actions=["job.cancel", "authz.denied"],
        limit=10,
    )
    assert {r["action"] for r in rows} == {"job.cancel", "authz.denied"}
    # Newest-first ordering preserved across the IN clause.
    assert rows[0]["action"] == "authz.denied"


@pytest.mark.asyncio
async def test_query_actions_empty_list_returns_nothing(audit_log):
    """An explicitly empty actions list MUST return [] — not 'all rows'
    via vacuous filter. SQL `IN ()` is a syntax error in SQLite; we
    short-circuit in Python before constructing the query."""
    await audit_log.write("doc.ingest", user_id="u-a", target_kind="document", target_id="d1")
    rows = await audit_log.query(actions=[], limit=10)
    assert rows == []


@pytest.mark.asyncio
async def test_query_rejects_both_action_and_actions(audit_log):
    """Mutually exclusive — caller bug, not silent AND."""
    import pytest as _pytest
    with _pytest.raises(ValueError, match="action / actions"):
        await audit_log.query(action="doc.ingest", actions=["doc.ingest"])


@pytest.mark.asyncio
async def test_init_enables_wal_pragma(tmp_path):
    """AuditLog must put kb_metadata.db into WAL mode so concurrent
    writers (UsersStore + MetadataDB + AuditLog) don't serialize."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    log = AuditLog(db_url=db_url, fallback_path=tmp_path / "fb.jsonl")
    await log.init()
    async with log._engine.connect() as conn:
        result = await conn.exec_driver_sql("PRAGMA journal_mode")
        mode = result.scalar()
    assert mode.lower() == "wal"
    await log.aclose()


@pytest.mark.asyncio
async def test_count_by_action_user_groups_correctly(audit_log):
    """P1-6 review fix: doctor needs DB-side GROUP BY for accurate
    per-user counts; pull-rows-then-len undercounts past limit."""
    # Seed: alice 8 ingests, bob 2 ingests, alice 1 cancel
    for i in range(8):
        await audit_log.write("doc.ingest", user_id="u-alice",
                               target_kind="document", target_id=f"d-a{i}")
    for i in range(2):
        await audit_log.write("doc.ingest", user_id="u-bob",
                               target_kind="document", target_id=f"d-b{i}")
    await audit_log.write("job.cancel", user_id="u-alice",
                           target_kind="job", target_id="j-1")

    counts = await audit_log.count_by_action_user(
        actions=["doc.ingest", "job.cancel", "job.failed"],
        since=None,
    )
    # Shape: {user_id: {action: count}}
    assert counts["u-alice"]["doc.ingest"] == 8
    assert counts["u-alice"]["job.cancel"] == 1
    assert counts["u-bob"]["doc.ingest"] == 2
    # Missing combos default to 0
    assert counts["u-bob"].get("job.cancel", 0) == 0
    assert counts.get("u-carol", {}) == {}


@pytest.mark.asyncio
async def test_write_allows_null_user_id_for_system_events(audit_log):
    await audit_log.write(
        "auth.brute_force",
        user_id=None,
        target_kind="system",
        target_id=None,
        payload={"source_ip": "1.2.3.4", "count": 10},
    )
    rows = await audit_log.query(limit=10)
    assert rows[0]["user_id"] is None


@pytest.mark.asyncio
async def test_audit_log_has_composite_indexes(audit_log):
    """P2-6: GET /audit core queries are (user_id, ts) and (action, ts).
    Without composite indexes a 1000+-row audit_log scans linearly on
    each filter. Single-column indexes on action / ts (T1a) plus the
    new composites cover both patterns."""
    async with audit_log._engine.connect() as conn:
        result = await conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='audit_log'"
        )
        index_names = {row[0] for row in result.fetchall()}
    assert "idx_audit_user_ts" in index_names
    assert "idx_audit_action_ts" in index_names
