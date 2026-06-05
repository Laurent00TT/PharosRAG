# tests/test_anthropic_adapter.py
"""Unit tests for AnthropicNativeAdapter.

No real API calls — patches httpx.AsyncClient at the module level.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kb.adapters.anthropic_native import (
    AnthropicNativeAdapter,
    _to_anthropic_message,
    _openai_tool_to_anthropic,
    _parse_anthropic_response,
)
from kb.adapters.base import AdapterMessage, ToolCall


# ── _to_anthropic_message ───────────────────────────────────────────────────


def test_plain_text_message_passes_through():
    m = AdapterMessage(role="user", content="hello")
    out = _to_anthropic_message(m)
    assert out == {"role": "user", "content": "hello"}


def test_assistant_message_with_tool_calls():
    m = AdapterMessage(
        role="assistant",
        content="Let me search.",
        tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})],
    )
    out = _to_anthropic_message(m)
    assert out["role"] == "assistant"
    assert out["content"][0] == {"type": "text", "text": "Let me search."}
    assert out["content"][1] == {
        "type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"},
    }


def test_tool_result_translates_to_user_role_with_tool_result_block():
    m = AdapterMessage(role="tool", content='{"answer": 42}', tool_call_id="t1")
    out = _to_anthropic_message(m)
    assert out["role"] == "user"
    assert out["content"][0]["type"] == "tool_result"
    assert out["content"][0]["tool_use_id"] == "t1"


def test_image_url_data_uri_translates_to_base64_block():
    m = AdapterMessage(role="user", content=[
        {"type": "text", "text": "what's in this?"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgo="}},
    ])
    out = _to_anthropic_message(m)
    assert out["content"][0] == {"type": "text", "text": "what's in this?"}
    img = out["content"][1]
    assert img["type"] == "image"
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png"
    assert img["source"]["data"] == "iVBORw0KGgo="


def test_malformed_data_url_is_dropped_not_crashing(caplog):
    """A malformed data: URL (missing comma) must not crash the whole
    message conversion — the bad image block is dropped, the surrounding
    text/blocks still go through. Anti-regression for the original
    ValueError path that took out entire requests."""
    import logging
    m = AdapterMessage(role="user", content=[
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64iVBORw0KGgo="}},  # MISSING comma
        {"type": "text", "text": "after"},
    ])
    with caplog.at_level(logging.WARNING, logger="kb.adapters.anthropic_native"):
        out = _to_anthropic_message(m)
    # Bad image dropped; both text blocks survive
    texts = [b for b in out["content"] if b["type"] == "text"]
    images = [b for b in out["content"] if b["type"] == "image"]
    assert [b["text"] for b in texts] == ["before", "after"]
    assert images == []
    # Warning was emitted so operators can find it in logs
    assert any("malformed data URL" in r.message for r in caplog.records)


def test_data_url_with_empty_payload_is_dropped():
    """data:image/png;base64, (empty after comma) — should be treated
    as malformed rather than producing an empty-data image block."""
    m = AdapterMessage(role="user", content=[
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
    ])
    out = _to_anthropic_message(m)
    assert out["content"] == []


# ── _openai_tool_to_anthropic ────────────────────────────────────────────────


def test_openai_tool_schema_translates():
    openai_tool = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search docs",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    out = _openai_tool_to_anthropic(openai_tool)
    assert out == {
        "name": "search",
        "description": "search docs",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }


# ── _parse_anthropic_response ────────────────────────────────────────────────


def test_parse_text_only_response():
    resp = {
        "content": [{"type": "text", "text": "the answer is 42"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    out = _parse_anthropic_response(resp)
    assert out.content == "the answer is 42"
    assert out.tool_calls == []
    assert out.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert out.total_tokens == 15


def test_parse_tool_use_response():
    resp = {
        "content": [
            {"type": "text", "text": "Searching..."},
            {"type": "tool_use", "id": "abc", "name": "search", "input": {"q": "x"}},
        ],
        "usage": {"input_tokens": 20, "output_tokens": 30},
    }
    out = _parse_anthropic_response(resp)
    assert out.content == "Searching..."
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].id == "abc"
    assert out.tool_calls[0].name == "search"
    assert out.tool_calls[0].arguments == {"q": "x"}


# ── End-to-end chat with mocked httpx ────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_strips_system_messages_to_top_level():
    parser = AnthropicNativeAdapter(
        base_url="http://test", api_key="k", model="claude-3",
    )
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })

    captured_body: dict = {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, **kw):
            captured_body.update(kw["json"])
            captured_body["__url"] = url
            captured_body["__headers"] = kw["headers"]
            return fake_resp

    msgs = [
        AdapterMessage(role="system", content="you are helpful"),
        AdapterMessage(role="user", content="hi"),
    ]
    with patch("kb.adapters.anthropic_native.httpx.AsyncClient", _FakeClient):
        out = await parser.chat(msgs)

    assert out.content == "ok"
    assert captured_body["system"] == "you are helpful"
    assert captured_body["messages"] == [{"role": "user", "content": "hi"}]
    assert captured_body["__url"] == "http://test/v1/messages"
    headers = captured_body["__headers"]
    assert headers["x-api-key"] == "k"
    assert headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_session_id_sent_when_configured():
    parser = AnthropicNativeAdapter(
        base_url="http://test", api_key="k", model="m",
        session_id="ingest-batch-7",
    )
    captured_headers: dict = {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, **kw):
            captured_headers.update(kw["headers"])
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json = MagicMock(return_value={
                "content": [{"type": "text", "text": ""}],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })
            return r

    with patch("kb.adapters.anthropic_native.httpx.AsyncClient", _FakeClient):
        await parser.chat([AdapterMessage(role="user", content="x")])

    assert captured_headers["x-session-id"] == "ingest-batch-7"


# ── Factory wiring ───────────────────────────────────────────────────────────


def test_factory_returns_anthropic_when_configured():
    from kb.adapters.factory import make_chat_adapter
    adapter = make_chat_adapter(
        adapter_type="anthropic",
        url="http://127.0.0.1:3456",
        api_key="k",
        model="claude-sonnet-4-6",
        supports_vision=True,
        session_id="abc",
    )
    assert isinstance(adapter, AnthropicNativeAdapter)
    assert adapter._session_id == "abc"


def test_factory_raises_on_unknown_adapter():
    from kb.adapters.factory import make_chat_adapter
    with pytest.raises(ValueError, match="Unknown adapter_type"):
        make_chat_adapter(
            adapter_type="nope", url="x", api_key="x", model="x",
            supports_vision=False,
        )
