"""Observability primitives — structured event tracing for offline analysis."""
from kb.observability.events import (
    AgentFinalEvent,
    AgentIterationEvent,
    RetrieveVariantEvent,
    SearchEndEvent,
    SearchStartEvent,
    TraceEvent,
)
from kb.observability.preview import preview
from kb.observability.token_parser import parse_thinking
from kb.observability.trace import (
    emit,
    emit_error,
    get_trace_id,
    new_trace,
    set_trace_id,
    setup_tracing,
    timed,
)

__all__ = [
    # Trace primitives
    "emit",
    "emit_error",
    "get_trace_id",
    "new_trace",
    "parse_thinking",
    "preview",
    "set_trace_id",
    "setup_tracing",
    "timed",
    # Typed events
    "TraceEvent",
    "SearchStartEvent",
    "SearchEndEvent",
    "RetrieveVariantEvent",
    "AgentIterationEvent",
    "AgentFinalEvent",
]
