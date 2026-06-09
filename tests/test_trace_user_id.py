"""T1b — emit() must auto-inject user_id from ContextVar."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from kb.auth.context import current_user
from kb.auth.users import User
from kb.observability.trace import _logger as _trace_logger
from kb.observability.trace import emit


def _make_user(name: str) -> User:
    return User(
        user_id=f"id-{name}", username=name, key_prefix="x",
        key_hash="0" * 64, role="member",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        disabled_at=None,
    )


@pytest.fixture
def trace_records():
    records: list[dict] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            try:
                records.append(json.loads(record.getMessage()))
            except json.JSONDecodeError:
                pass

    handler = _Capture()
    _trace_logger.addHandler(handler)
    prev = _trace_logger.level
    _trace_logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        _trace_logger.removeHandler(handler)
        _trace_logger.setLevel(prev)


def test_emit_includes_user_id_when_set(trace_records):
    token = current_user.set(_make_user("alice"))
    try:
        emit("kb.search", query="hello")
    finally:
        current_user.reset(token)
    assert trace_records
    assert trace_records[-1]["user_id"] == "id-alice"


def test_emit_omits_user_id_when_unset(trace_records):
    # Before any middleware ran: anonymous (e.g. system startup events)
    current_user.set(None)
    emit("ingest.start", doc_id="d1")
    rec = trace_records[-1]
    # Either field absent or set to None / "" — pick whichever the
    # implementation chose. Most consumers (analyze_traces.py) treat
    # missing as anonymous, so absent is preferred.
    assert "user_id" not in rec or rec["user_id"] in (None, "")


def test_emit_user_id_changes_with_context(trace_records):
    """alice emit → bob emit → records must show different user_ids."""
    token_a = current_user.set(_make_user("alice"))
    try:
        emit("event.one")
    finally:
        current_user.reset(token_a)

    token_b = current_user.set(_make_user("bob"))
    try:
        emit("event.two")
    finally:
        current_user.reset(token_b)

    assert trace_records[-2]["user_id"] == "id-alice"
    assert trace_records[-1]["user_id"] == "id-bob"
