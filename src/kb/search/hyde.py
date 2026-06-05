# src/kb/search/hyde.py
"""HyDE query expansion — let an LLM generate a hypothetical answer to use as the retrieval query.

The adapter is dependency-injected, so any BaseLLMAdapter (OllamaAdapter /
OpenAICompatAdapter) can drive expansion. On failure, falls back to the
original query so the search path stays alive.
"""
import logging
import time

from kb.adapters.base import AdapterMessage, BaseLLMAdapter
from kb.observability import emit, emit_error, preview

logger = logging.getLogger(__name__)

HYDE_PROMPT_VERSION = "hyde-v1"


_SYSTEM = (
    "You are a knowledge base expert. Generate a concise hypothetical document excerpt "
    "that would answer the given question. Output only the excerpt, no explanation."
)


async def hyde_expand(query: str, adapter: BaseLLMAdapter) -> str:
    """Generate HyDE text. On any failure return the original query."""
    t0 = time.monotonic()
    try:
        response = await adapter.chat([
            AdapterMessage(role="system", content=_SYSTEM),
            AdapterMessage(role="user", content=query),
        ])
        result = response.content or query
        emit(
            "hyde.call",
            duration_ms=round((time.monotonic() - t0) * 1000, 1),
            success=True,
            thinking_detected=response.thinking_detected,
            thinking_chars=response.thinking_chars,
            content_chars=response.content_chars,
            total_tokens=response.total_tokens,
            prompt_version=HYDE_PROMPT_VERSION,
            output_preview=preview(result),
            fallback=False,
        )
        return result
    except Exception as e:
        logger.warning("HyDE expansion failed (%s); falling back to original query", e)
        emit_error(
            "hyde.call",
            e,
            duration_ms=round((time.monotonic() - t0) * 1000, 1),
            fallback=True,
            prompt_version=HYDE_PROMPT_VERSION,
        )
        return query
