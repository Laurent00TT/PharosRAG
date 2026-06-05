"""Tests for observability infrastructure: trace, parse_thinking, preview."""
import json
import logging
import re

import pytest

from kb.observability import (
    emit, emit_error, get_trace_id, new_trace, parse_thinking, preview, timed,
)
from kb.observability.trace import _logger


# ── Test fixture: capture trace records via dedicated handler ─────────────────
# kb.trace logger has propagate=False, so pytest's caplog (root-attached) won't
# see records. We attach our own MemoryHandler.

@pytest.fixture
def trace_records():
    """Capture trace JSON lines emitted during the test."""
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    _logger.addHandler(handler)
    prev_level = _logger.level
    _logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        _logger.removeHandler(handler)
        _logger.setLevel(prev_level)


# ── trace_id contextvar ──────────────────────────────────────────────────────


def test_new_trace_creates_short_hex_id():
    tid = new_trace()
    assert len(tid) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", tid)
    assert get_trace_id() == tid


def test_get_trace_id_default_when_unset():
    from kb.observability.trace import _trace_id_var
    _trace_id_var.set("")
    assert get_trace_id() == "no-trace"


# ── emit() ───────────────────────────────────────────────────────────────────


def test_emit_writes_json_with_trace_id(trace_records):
    new_trace()
    emit("test.event", foo="bar", n=42)
    rec = json.loads(trace_records[-1])
    assert rec["event"] == "test.event"
    assert rec["foo"] == "bar"
    assert rec["n"] == 42
    assert rec["trace_id"] == get_trace_id()
    assert "ts" in rec


def test_emit_serializes_unknown_types_via_default_str(trace_records):
    """default=str 兜底 — pathlib/datetime 等可序列化为 repr。"""
    from pathlib import Path
    new_trace()
    emit("test.path", p=Path("/tmp/x"))
    rec = json.loads(trace_records[-1])
    assert "p" in rec  # serialized as str, no exception


def test_emit_never_raises_when_logger_breaks(monkeypatch):
    """logger.info 内部 raise 时,emit 必须吞掉。"""
    new_trace()
    monkeypatch.setattr(
        _logger, "info",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # 不应抛
    emit("dies.silently", x=1)


# ── emit_error() ─────────────────────────────────────────────────────────────


def test_emit_error_records_exception_info(trace_records):
    new_trace()
    try:
        raise ValueError("test bad")
    except ValueError as e:
        emit_error("op.failed", e, retried=False)
    rec = json.loads(trace_records[-1])
    assert rec["level"] == "ERROR"
    assert rec["error_type"] == "ValueError"
    assert "test bad" in rec["error_msg"]
    assert rec["retried"] is False


def test_emit_error_truncates_long_message(trace_records):
    new_trace()
    long_msg = "x" * 1000
    try:
        raise RuntimeError(long_msg)
    except RuntimeError as e:
        emit_error("op.failed", e)
    rec = json.loads(trace_records[-1])
    assert len(rec["error_msg"]) == 500  # truncated


# ── timed() ──────────────────────────────────────────────────────────────────


def test_timed_emits_duration_on_success(trace_records):
    new_trace()
    with timed("slow.thing", x=1):
        pass
    rec = json.loads(trace_records[-1])
    assert rec["event"] == "slow.thing"
    assert "duration_ms" in rec
    assert rec["x"] == 1
    assert rec["success"] is True


def test_timed_emits_failure_on_exception_and_reraises(trace_records):
    new_trace()
    with pytest.raises(RuntimeError):
        with timed("breaks"):
            raise RuntimeError("boom")
    rec = json.loads(trace_records[-1])
    assert rec["event"] == "breaks"
    assert rec["success"] is False


# ── parse_thinking() ─────────────────────────────────────────────────────────


def test_parse_thinking_track1_reasoning_content_field():
    """vLLM/SGLang style: reasoning_content separate from content."""
    resp = {
        "choices": [{"message": {
            "content": "answer",
            "reasoning_content": "let me think about this...",
        }}],
        "usage": {"completion_tokens": 50, "prompt_tokens": 20},
    }
    out = parse_thinking(resp)
    assert out["thinking_detected"] is True
    assert out["thinking_chars"] == len("let me think about this...")
    assert out["content_chars"] == len("answer")
    assert out["clean_content"] == "answer"
    assert out["total_tokens"] == 50


def test_parse_thinking_track2_think_tag_in_content():
    """Qwen3.x + llama.cpp style: <think>...</think> embedded in content."""
    resp = {
        "choices": [{"message": {"content": "<think>let me consider</think>final answer"}}],
        "usage": {"completion_tokens": 10},
    }
    out = parse_thinking(resp)
    assert out["thinking_detected"] is True
    assert out["thinking_chars"] == len("let me consider")
    assert out["clean_content"] == "final answer"


def test_parse_thinking_track2_with_leading_text():
    """Think tag not necessarily at start — should still strip cleanly."""
    resp = {
        "choices": [{"message": {"content": "Note: <think>x</think>actual answer here"}}],
        "usage": {"completion_tokens": 10},
    }
    out = parse_thinking(resp)
    assert out["thinking_detected"] is True
    # 去掉 <think>x</think> 后剩 "Note: actual answer here" (单空格)
    assert out["clean_content"] == "Note: actual answer here"
    assert "<think>" not in out["clean_content"]


def test_parse_thinking_track2_multiline_thinking():
    """re.DOTALL — <think>...</think> can span lines."""
    content = "<think>line 1\nline 2\nline 3</think>just the answer"
    resp = {
        "choices": [{"message": {"content": content}}],
        "usage": {"completion_tokens": 5},
    }
    out = parse_thinking(resp)
    assert out["thinking_detected"] is True
    assert "line 2" in out["clean_content"] or out["clean_content"] == "just the answer"
    # 关键: clean_content 不含 <think> 标签
    assert "<think>" not in out["clean_content"]


def test_parse_thinking_track3_no_thinking():
    resp = {
        "choices": [{"message": {"content": "direct answer"}}],
        "usage": {"completion_tokens": 5},
    }
    out = parse_thinking(resp)
    assert out["thinking_detected"] is False
    assert out["thinking_chars"] == 0
    assert out["clean_content"] == "direct answer"


def test_parse_thinking_handles_none_content():
    """tool_call-only response has content=None — must not crash."""
    resp = {
        "choices": [{"message": {"content": None}}],
        "usage": {"completion_tokens": 0},
    }
    out = parse_thinking(resp)
    assert out["thinking_detected"] is False
    assert out["clean_content"] == ""


def test_parse_thinking_handles_openai_sdk_object():
    """传入 openai SDK ChatCompletion-shaped object (有 .choices 属性) 也能解析。"""
    from unittest.mock import MagicMock
    msg = MagicMock(content="<think>hmm</think>ok")
    msg.reasoning_content = None  # 显式设
    resp = MagicMock(
        choices=[MagicMock(message=msg)],
        usage=MagicMock(completion_tokens=8, prompt_tokens=3),
    )
    out = parse_thinking(resp)
    assert out["thinking_detected"] is True
    assert out["clean_content"] == "ok"
    assert out["total_tokens"] == 8


def test_parse_thinking_missing_usage():
    """response 没有 usage 字段时 total_tokens=0 而非异常。"""
    resp = {"choices": [{"message": {"content": "hi"}}]}
    out = parse_thinking(resp)
    assert out["total_tokens"] == 0


# ── preview() ────────────────────────────────────────────────────────────────


def test_preview_truncates_long_string():
    s = "x" * 300
    p = preview(s, n=200)
    assert p.startswith("x" * 200)
    assert "100 chars" in p


def test_preview_passes_through_short():
    assert preview("hi") == "hi"


def test_preview_handles_empty_and_none():
    assert preview("") == ""
    assert preview(None) == ""


def test_preview_default_max_is_200():
    s = "y" * 250
    assert len(preview(s)) > 200  # contains truncation marker
    assert s[:200] in preview(s)


# ── analyze_traces.py: smoke tests on _p and load_traces ─────────────────────


def test_analyze_p_percentile():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analyze_traces",
        "scripts/analyze_traces.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._p([1, 2, 3, 4, 5], 0.95) == 5
    assert mod._p([], 0.5) == 0.0
    assert mod._p([10], 0.95) == 10


# ── Typed event emission ─────────────────────────────────────────────────────


def test_emit_accepts_typed_event(trace_records):
    """emit(SearchStartEvent(...)) produces same JSON as emit('search.start', ...)."""
    from kb.observability import SearchStartEvent
    new_trace()
    emit(SearchStartEvent(
        query="test",
        top_k=5,
        hyde=True,
        multi_query_enabled=True,
        multi_query_n=3,
        doc_filter={},
        include_deprecated=False,
    ))
    rec = json.loads(trace_records[-1])
    assert rec["event"] == "search.start"
    assert rec["query"] == "test"
    assert rec["top_k"] == 5
    assert rec["hyde"] is True
    assert rec["multi_query_n"] == 3
    # EVENT_NAME field must NOT leak into trace output
    assert "EVENT_NAME" not in rec


def test_emit_typed_and_legacy_produce_identical_output(trace_records):
    """Round-trip: same event via dataclass and via string + kwargs ⇒ same fields."""
    from kb.observability import AgentIterationEvent
    new_trace()
    emit(AgentIterationEvent(
        round=1, duration_ms=100.0,
        thinking_detected=True, thinking_chars=10,
        content_chars=20, total_tokens=8,
        tool_calls_count=0, finish_reason="final_answer",
    ))
    typed_rec = json.loads(trace_records[-1])

    new_trace()
    emit("agent.iteration",
         round=1, duration_ms=100.0,
         thinking_detected=True, thinking_chars=10,
         content_chars=20, total_tokens=8,
         tool_calls_count=0, finish_reason="final_answer",
    )
    legacy_rec = json.loads(trace_records[-1])

    # ts and trace_id will differ; everything else must match
    typed_payload = {k: v for k, v in typed_rec.items() if k not in ("ts", "trace_id")}
    legacy_payload = {k: v for k, v in legacy_rec.items() if k not in ("ts", "trace_id")}
    assert typed_payload == legacy_payload


def test_typed_event_has_correct_event_name():
    from kb.observability import (
        AgentFinalEvent,
        AgentIterationEvent,
        RetrieveVariantEvent,
        SearchEndEvent,
        SearchStartEvent,
    )
    assert SearchStartEvent.EVENT_NAME == "search.start"
    assert SearchEndEvent.EVENT_NAME == "search.end"
    assert RetrieveVariantEvent.EVENT_NAME == "retrieve.variant"
    assert AgentIterationEvent.EVENT_NAME == "agent.iteration"
    assert AgentFinalEvent.EVENT_NAME == "agent.final"


def test_analyze_load_traces_filters_by_since(tmp_path):
    """load_traces honors --since by skipping older events."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analyze_traces",
        "scripts/analyze_traces.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    log_file = tmp_path / "kb_trace.jsonl"
    log_file.write_text(
        '{"ts": "2026-01-01T00:00:00.000Z", "event": "old", "trace_id": "a"}\n'
        '{"ts": "2026-05-09T00:00:00.000Z", "event": "new", "trace_id": "b"}\n'
        '\n'  # blank line — should skip
        'not json at all\n'  # malformed — should skip
        '{"ts": "2026-05-09T01:00:00.000Z", "event": "newer", "trace_id": "c"}\n',
        encoding="utf-8",
    )

    from datetime import datetime, timezone
    since = datetime(2026, 3, 1, tzinfo=timezone.utc)
    events = mod.load_traces(tmp_path, since=since)
    names = [e["event"] for e in events]
    assert "old" not in names
    assert "new" in names
    assert "newer" in names
