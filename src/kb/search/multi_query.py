# src/kb/search/multi_query.py
"""Multi-Query Expansion — let an LLM produce query variants, retrieve in parallel, then RRF-merge.

Improves recall for fuzzy queries: when the user's phrasing is imprecise,
hitting the index with multiple paraphrases boosts the chance of a match.
"""
import logging
import re
import time

from kb.adapters.base import AdapterMessage, BaseLLMAdapter
from kb.observability import emit, emit_error

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT_VERSION = "multi-query-v1"


_SYSTEM = """You generate paraphrases of a search query to improve retrieval recall.
Output exactly N numbered variants, one per line. Each variant should approach
the query from a different angle:
- synonym substitution / terminology variants
- broadening or narrowing the scope
- extracting key entities for separate retrieval

Output format:
1. <variant>
2. <variant>
...

Output only the numbered list. No preamble, commentary, or extra text."""


_LINE_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$", re.MULTILINE)


async def expand_queries(
    query: str,
    n: int,
    adapter: BaseLLMAdapter,
) -> list[str]:
    """Generate n query variants. On failure or unparseable output, return [query] (fallback)."""
    if n <= 1:
        return [query]
    t0 = time.monotonic()
    try:
        response = await adapter.chat([
            AdapterMessage(role="system", content=_SYSTEM.replace("N", str(n))),
            AdapterMessage(role="user", content=query),
        ])
        variants = _parse_numbered_list(response.content)
        if not variants:
            logger.debug("Multi-Query parsing failed; falling back to original query")
            emit(
                "multi_query.call",
                duration_ms=round((time.monotonic() - t0) * 1000, 1),
                success=True,
                n_variants=1,
                fallback=True,
                fallback_reason="parse_failed",
                thinking_detected=response.thinking_detected,
                thinking_chars=response.thinking_chars,
                content_chars=response.content_chars,
                total_tokens=response.total_tokens,
                prompt_version=MULTI_QUERY_PROMPT_VERSION,
            )
            return [query]
        # Always include the original query as a baseline variant
        if query not in variants:
            variants.insert(0, query)
        result = variants[:n]
        emit(
            "multi_query.call",
            duration_ms=round((time.monotonic() - t0) * 1000, 1),
            success=True,
            n_variants=len(result),
            variants=result,
            fallback=False,
            thinking_detected=response.thinking_detected,
            thinking_chars=response.thinking_chars,
            content_chars=response.content_chars,
            total_tokens=response.total_tokens,
            prompt_version=MULTI_QUERY_PROMPT_VERSION,
        )
        return result
    except Exception as e:
        logger.warning("Multi-Query expansion failed (%s); falling back to original query", e)
        emit_error(
            "multi_query.call",
            e,
            duration_ms=round((time.monotonic() - t0) * 1000, 1),
            fallback=True,
            fallback_reason="exception",
            prompt_version=MULTI_QUERY_PROMPT_VERSION,
        )
        return [query]


def _parse_numbered_list(text: str) -> list[str]:
    """Extract items from text shaped like '1. xxx\n2. yyy'."""
    return [m.group(1).strip() for m in _LINE_RE.finditer(text)]
