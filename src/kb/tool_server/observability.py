# src/kb/tool_server/observability.py
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

logger = logging.getLogger("kb.observability")


@dataclass
class _SearchLog:
    timestamp: str
    query: str
    latency_ms: float
    channels_used: list[str]
    result_count: int
    top_score: float
    hyde_triggered: bool
    cache_hit: bool


def log_search(
    *,
    query: str,
    latency_ms: float,
    channels_used: list[str],
    result_count: int,
    top_score: float,
    hyde_triggered: bool,
    cache_hit: bool,
) -> None:
    entry = _SearchLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=query[:200],
        latency_ms=round(latency_ms, 1),
        channels_used=channels_used,
        result_count=result_count,
        top_score=round(top_score, 4),
        hyde_triggered=hyde_triggered,
        cache_hit=cache_hit,
    )
    logger.info(json.dumps(asdict(entry)))
