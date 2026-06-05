"""Typed trace event schemas.

Goal: catch field-name typos at edit time. emit("agent.iteration", thinkign_chars=1)
silently dropped a field; emit(AgentIterationEvent(thinkign_chars=1)) is a syntax error.

Strategy:
- Define dataclass per high-frequency event (5 most-used).
- emit() accepts either a dataclass instance or the legacy string + kwargs form.
- Lower-frequency events keep the string form to limit refactor surface.

The dataclass field names MUST match the legacy emit() field names exactly so
analyze_traces.py and downstream consumers don't break.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class TraceEvent:
    """Base for typed events. Subclasses set EVENT_NAME and add typed fields."""
    EVENT_NAME: str = field(init=False, default="")

    def to_emit_args(self) -> tuple[str, dict]:
        """Return (event_name, field_dict) for emit()."""
        d = asdict(self)
        d.pop("EVENT_NAME", None)
        return self.EVENT_NAME, d


# ── Search pipeline ──────────────────────────────────────────────────────


@dataclass
class SearchStartEvent(TraceEvent):
    """Emitted at the top of SearchEngine.search()."""
    query: str                         # preview-truncated
    top_k: int
    hyde: bool
    multi_query_enabled: bool
    multi_query_n: int
    doc_filter: dict
    include_deprecated: bool
    EVENT_NAME: str = field(init=False, default="search.start")


@dataclass
class SearchEndEvent(TraceEvent):
    """Emitted on the success path of SearchEngine.search()."""
    total_duration_ms: float
    cache_hit: bool
    result_count: int
    top_score: float
    success: bool = True
    cold_start: bool = False
    EVENT_NAME: str = field(init=False, default="search.end")


# ── Retrieve pipeline ────────────────────────────────────────────────────


@dataclass
class RetrieveVariantEvent(TraceEvent):
    """Emitted once per retrieve.retrieve() call (one per query variant)."""
    embed_ms: float
    vision_query_ms: float
    vision_query_ok: bool
    search_ms: float
    dense_hits: int
    sparse_hits: int
    vision_hits: int
    desc_hits: int
    limit: int
    include_deprecated: bool
    EVENT_NAME: str = field(init=False, default="retrieve.variant")


# ── Agent pipeline ───────────────────────────────────────────────────────


@dataclass
class AgentIterationEvent(TraceEvent):
    """Emitted once per Agent loop iteration."""
    round: int
    duration_ms: float
    thinking_detected: bool
    thinking_chars: int
    content_chars: int
    total_tokens: int
    tool_calls_count: int
    finish_reason: Literal["tool_call", "final_answer"]
    EVENT_NAME: str = field(init=False, default="agent.iteration")


@dataclass
class AgentFinalEvent(TraceEvent):
    """Emitted when Agent produces its final answer (success or max-iter)."""
    iterations: int
    citations_count: int
    answer_chars: int
    retrieved_count: int
    success: bool
    answer_preview: str = ""
    citations_hallucinated_count: int = 0
    failure_reason: str = ""
    EVENT_NAME: str = field(init=False, default="agent.final")
