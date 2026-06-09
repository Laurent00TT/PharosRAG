"""Tests for MCP transport security (Origin validation) + audit trace."""
import json
import logging

import pytest

from kb.observability.trace import _logger as _trace_logger
from kb.tool_server.mcp_security import (
    OriginValidator,
    audit_tool_call,
    origin_is_allowed,
    parse_allowed_origins,
)

# NOTE (T1b Task 4): The old single-key APIKeyEnforcer (mcp_security.py) was
# deleted in T1b hard cutover. Tests that covered:
#   - test_api_key_enforcer_accepts_x_api_key_header
#   - test_api_key_enforcer_accepts_bearer_authorization
#   - test_api_key_enforcer_rejects_missing_key
#   - test_api_key_enforcer_rejects_wrong_key
#   - test_api_key_enforcer_fails_closed_when_key_unset
# were removed here. Task 7 (tests/test_auth_middleware.py) should add
# equivalent coverage for the new user-token APIKeyEnforcer in kb.auth.middleware.


def test_parse_allowed_origins_trims_values():
    assert parse_allowed_origins(" http://localhost, http://127.0.0.1/ ") == {
        "http://localhost",
        "http://127.0.0.1",
    }


def test_origin_is_allowed_for_loopback_and_configured_origin():
    allowed = "http://localhost,http://127.0.0.1"
    assert origin_is_allowed("http://localhost", allowed) is True
    assert origin_is_allowed("http://127.0.0.1", allowed) is True
    assert origin_is_allowed("http://evil.example", allowed) is False


def test_origin_is_allowed_rejects_unlisted_loopback_port():
    """Loopback hostname alone is no longer trusted — port must match an
    explicit allowlist entry. Prevents a hostile localhost service (e.g.,
    a malicious dev tool on port 31337) from driving MCP via browser fetch
    when the operator's allowlist only includes the default loopback origins.
    """
    allowed = "http://localhost,http://127.0.0.1"
    # Implicit "any loopback port" is no longer permitted
    assert origin_is_allowed("http://localhost:31337", allowed) is False
    assert origin_is_allowed("http://127.0.0.1:8080", allowed) is False
    # But explicitly allowed ports DO pass
    allowed_with_port = "http://localhost,http://localhost:8000,http://127.0.0.1"
    assert origin_is_allowed("http://localhost:8000", allowed_with_port) is True


def test_origin_is_allowed_empty_allowlist_only_passes_none_or_empty_origin():
    """With empty allowlist, only no-Origin requests (curl, MCP stdio) pass.
    Previously the implicit loopback fallback meant empty allowlist still
    accepted localhost — that was inconsistent with operator intent."""
    allowed = ""
    assert origin_is_allowed(None, allowed) is True
    assert origin_is_allowed("", allowed) is True
    assert origin_is_allowed("http://localhost", allowed) is False
    assert origin_is_allowed("http://evil.example", allowed) is False


@pytest.mark.asyncio
async def test_origin_validator_rejects_bad_origin():
    sent = []

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = OriginValidator(inner, allowed_origins=["http://localhost"])
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/",
        "headers": [(b"origin", b"http://evil.example")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    assert sent[0]["status"] == 403


# ── audit_tool_call ────────────────────────────────────────────────────────
# kb.observability.trace._logger has propagate=False, so pytest's caplog
# (which attaches to root) cannot see records. We attach a dedicated capture
# handler directly to that logger, matching the pattern used by
# tests/test_observability_integration.py.


@pytest.fixture
def trace_records():
    """Capture trace JSON lines emitted during the test."""
    records: list[dict] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            try:
                records.append(json.loads(record.getMessage()))
            except json.JSONDecodeError:
                pass

    handler = _Capture()
    _trace_logger.addHandler(handler)
    prev_level = _trace_logger.level
    _trace_logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        _trace_logger.removeHandler(handler)
        _trace_logger.setLevel(prev_level)


@pytest.mark.asyncio
async def test_audit_tool_call_emits_mcp_tool_event(trace_records):
    async def inner():
        return "result"

    result = await audit_tool_call(
        "kb_search",
        input_preview={"query": "test", "top_k": 5},
        is_write_tool=False,
        call=inner,
    )

    assert result == "result"
    mcp_events = [r for r in trace_records if r.get("event") == "mcp.tool"]
    assert len(mcp_events) == 1
    ev = mcp_events[0]
    assert ev["tool"] == "kb_search"
    assert ev["is_write_tool"] is False
    assert ev["success"] is True
    assert ev["latency_ms"] >= 0
    assert ev["input_preview"] == {"query": "test", "top_k": 5}


@pytest.mark.asyncio
async def test_audit_tool_call_emits_error_event_on_exception(trace_records):
    async def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        await audit_tool_call(
            "kb_search",
            input_preview={"query": "x"},
            is_write_tool=False,
            call=boom,
        )

    mcp_events = [r for r in trace_records if r.get("event") == "mcp.tool"]
    assert len(mcp_events) == 1
    ev = mcp_events[0]
    assert ev["error_type"] == "ValueError"
    assert "kaboom" in ev["error_msg"]
    assert ev["level"] == "ERROR"
