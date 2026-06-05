"""Integration tests for observability埋点 in business code paths."""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb.adapters.base import AdapterMessage, AdapterResponse, ToolCall
from kb.observability.trace import _logger


@pytest.fixture
def trace_records():
    """Capture all trace events emitted during a test."""
    records: list[dict] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            try:
                records.append(json.loads(record.getMessage()))
            except json.JSONDecodeError:
                pass

    handler = _Capture()
    _logger.addHandler(handler)
    prev_level = _logger.level
    _logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        _logger.removeHandler(handler)
        _logger.setLevel(prev_level)


def _events_named(records, name):
    return [r for r in records if r.get("event") == name]


# ── HyDE埋点 ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hyde_emits_call_event_with_thinking_metadata(trace_records):
    """hyde.call 事件包含 duration_ms / thinking_* / output_preview。"""
    from kb.search.hyde import hyde_expand

    fake_adapter = MagicMock()
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="hypothetical document about X",
        thinking_chars=0,
        content_chars=30,
        thinking_detected=False,
        total_tokens=8,
    ))
    result = await hyde_expand("query about X", adapter=fake_adapter)
    assert result == "hypothetical document about X"

    events = _events_named(trace_records, "hyde.call")
    assert len(events) == 1
    e = events[0]
    assert "duration_ms" in e
    assert e["thinking_detected"] is False
    assert e["fallback"] is False
    assert "output_preview" in e


@pytest.mark.asyncio
async def test_hyde_emits_error_event_on_failure(trace_records):
    """adapter.chat 抛异常时 emit_error,fallback=True。"""
    from kb.search.hyde import hyde_expand

    fake_adapter = MagicMock()
    fake_adapter.chat = AsyncMock(side_effect=RuntimeError("server down"))
    result = await hyde_expand("q", adapter=fake_adapter)
    assert result == "q"  # fallback

    events = _events_named(trace_records, "hyde.call")
    assert len(events) == 1
    assert events[0]["level"] == "ERROR"
    assert events[0]["fallback"] is True
    assert "RuntimeError" in events[0]["error_type"]


# ── Multi-Query埋点 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_query_emits_call_event_with_variants(trace_records):
    from kb.search.multi_query import expand_queries

    fake_adapter = MagicMock()
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="1. variant A\n2. variant B\n3. variant C",
        thinking_chars=0, content_chars=40, total_tokens=12,
    ))
    result = await expand_queries("original", n=3, adapter=fake_adapter)
    assert len(result) == 3

    events = _events_named(trace_records, "multi_query.call")
    assert len(events) == 1
    e = events[0]
    assert e["n_variants"] == 3
    assert e["fallback"] is False
    assert isinstance(e["variants"], list)


@pytest.mark.asyncio
async def test_multi_query_n_le_1_does_not_emit(trace_records):
    """n=1 短路 — 不调 LLM,不 emit。"""
    from kb.search.multi_query import expand_queries
    fake_adapter = MagicMock()
    result = await expand_queries("q", n=1, adapter=fake_adapter)
    assert result == ["q"]
    assert len(_events_named(trace_records, "multi_query.call")) == 0


# ── Agent埋点 ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_emits_start_iteration_final_events(trace_records):
    from kb.agent.agent import KBAgent

    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=False)
    # 一轮就给最终答案 (无 tool_call)
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="The answer is 42 [doc1:1].",
        thinking_chars=15, content_chars=27,
        total_tokens=12, thinking_detected=True,
    ))
    fake_engine = MagicMock()

    agent = KBAgent(adapter=fake_adapter, engine=fake_engine, max_iterations=3)
    result = await agent.answer("what is the answer?")
    assert "42" in result["answer"]

    starts = _events_named(trace_records, "agent.start")
    iters = _events_named(trace_records, "agent.iteration")
    finals = _events_named(trace_records, "agent.final")

    assert len(starts) == 1
    assert starts[0]["max_iterations"] == 3
    assert starts[0]["supports_vision"] is False

    assert len(iters) == 1
    assert iters[0]["round"] == 1
    assert iters[0]["thinking_detected"] is True
    assert iters[0]["thinking_chars"] == 15
    assert iters[0]["finish_reason"] == "final_answer"

    assert len(finals) == 1
    assert finals[0]["iterations"] == 1
    assert finals[0]["success"] is True
    # In this test path the agent never called search_docs, so retrieved_pages
    # is empty. The LLM still wrote `[doc1:1]` — citation validation correctly
    # flags this as hallucinated (would be a valid citation in a real run
    # where doc1:1 was actually retrieved).
    assert finals[0]["citations_count"] == 0
    assert finals[0]["citations_hallucinated_count"] == 1
    halluc_events = _events_named(trace_records, "agent.hallucinated_citation")
    assert len(halluc_events) == 1
    assert halluc_events[0]["count"] == 1


@pytest.mark.asyncio
async def test_agent_emits_tool_call_event(trace_records):
    """tool_call → search → final 两轮场景。"""
    from kb.agent.agent import KBAgent
    from kb.models import SearchResponse, SearchResult

    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=False)

    tool_call = ToolCall(id="c1", name="search_docs", arguments={"query": "x"})
    fake_adapter.chat = AsyncMock(side_effect=[
        AdapterResponse(content="", tool_calls=[tool_call], thinking_chars=0, content_chars=0),
        AdapterResponse(content="found [doc1:1]", thinking_chars=0, content_chars=15),
    ])

    fake_engine = MagicMock()
    fake_engine.search = AsyncMock(return_value=(
        SearchResponse(results=[], total=0), False,
    ))

    agent = KBAgent(adapter=fake_adapter, engine=fake_engine, max_iterations=3)
    await agent.answer("find x")

    tool_events = _events_named(trace_records, "agent.tool_call")
    assert len(tool_events) == 1
    e = tool_events[0]
    assert e["tool_name"] == "search_docs"
    assert e["recognized"] is True
    assert e["args_ok"] is True
    assert "query_preview" in e

    # 两轮 iteration
    iters = _events_named(trace_records, "agent.iteration")
    assert len(iters) == 2
    assert iters[0]["finish_reason"] == "tool_call"
    assert iters[1]["finish_reason"] == "final_answer"
