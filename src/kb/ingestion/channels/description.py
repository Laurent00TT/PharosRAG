# src/kb/ingestion/channels/description.py
"""Description channel (Channel 3) — uses a vision LLM adapter to produce
a semantic description of each page image.

Adapter is injected, so any multimodal LLM (OpenAI-compat / Ollama) can drive
description generation.
"""
import json
import logging
import time

from kb.adapters.base import (
    AdapterMessage, BaseLLMAdapter, ImagePart, build_multimodal_content,
)
from kb.models import DescPage
from kb.observability import emit, emit_error

logger = logging.getLogger(__name__)

DESCRIPTION_PROMPT_VERSION = "description-v1"


_PROMPT = """You are an enterprise document analysis expert. Analyze the page image and output JSON:
{
  "page_type": "flowchart" | "table" | "screenshot" | "mixed" | "text",
  "figure_index": "Figure X-X or null",
  "figure_caption": "caption text or null",
  "position_hint": "where on the page the figure sits",
  "description": "full business-semantic description (steps, relationships, roles)",
  "key_entities": ["entity1", "entity2"]
}
Output JSON only. Do not wrap in markdown code fences. Do not add commentary."""


def _strip_json_fence(s: str) -> str:
    """Strip ```json ... ``` markdown fences. Qwen3.6 wraps JSON in them
    despite system prompt instructions; safer to handle client-side."""
    s = s.strip()
    if not s.startswith("```"):
        return s
    # Drop opening fence line ("```json" or "```")
    nl = s.find("\n")
    if nl == -1:
        return s
    s = s[nl + 1:].rstrip()
    # Drop trailing closing fence
    if s.endswith("```"):
        s = s[:-3].rstrip()
    return s


class DescriptionChannel:
    """Vision-LLM-driven page description channel."""

    def __init__(self, adapter: BaseLLMAdapter) -> None:
        if not adapter.supports_vision():
            logger.warning(
                "DescriptionChannel got a non-vision adapter — "
                "description quality will rely on text-only input"
            )
        self._adapter = adapter

    def should_process(self, has_image_blocks: bool, text_char_ratio: float) -> bool:
        """Skip pure-text pages without any image blocks."""
        return has_image_blocks or text_char_ratio < 0.9

    async def aclose(self) -> None:
        await self._adapter.aclose()

    async def describe(
        self,
        doc_id: str,
        doc_name: str,
        page_num: int,
        image_bytes: bytes,
        heading_path: list[str],
        image_url: str,
    ) -> DescPage:
        """Send the page image to the vision LLM and return a structured DescPage."""
        heading = " / ".join(heading_path) or "(none)"
        messages = [
            AdapterMessage(role="system", content=_PROMPT),
            AdapterMessage(
                role="user",
                content=build_multimodal_content(
                    text=f"Analyze this page (heading: {heading}):",
                    images=[ImagePart(data=image_bytes, mime_type="image/png")],
                ),
            ),
        ]

        t0 = time.monotonic()
        json_parse_ok = False
        adapter_ok = False
        thinking_detected = False
        thinking_chars = 0
        raw_content = ""
        try:
            response = await self._adapter.chat(messages)
            adapter_ok = True
            thinking_detected = response.thinking_detected
            thinking_chars = response.thinking_chars
            raw_content = response.content
            data = json.loads(_strip_json_fence(response.content))
            json_parse_ok = True
        except Exception as e:
            logger.warning("DescriptionChannel adapter/parse failed (%s); empty description", e)
            emit_error(
                "describe.page",
                e,
                duration_ms=round((time.monotonic() - t0) * 1000, 1),
                doc_id=doc_id,
                page_num=page_num,
                adapter_ok=adapter_ok,
                json_parse_ok=False,
                prompt_version=DESCRIPTION_PROMPT_VERSION,
                image_size_bytes=len(image_bytes),
                raw_content_preview=raw_content[:300] if raw_content else "(empty)",
                raw_content_len=len(raw_content),
            )
            data = {}

        # Defensive coercion — LLM may emit JSON with wrong types
        # (e.g. key_entities as str instead of list, page_type as number)
        def _str_or(default: str | None, key: str) -> str | None:
            v = data.get(key)
            return v if isinstance(v, str) else default

        key_entities_raw = data.get("key_entities", [])
        key_entities = (
            [str(e) for e in key_entities_raw]
            if isinstance(key_entities_raw, list)
            else []
        )

        page_type_val = _str_or("mixed", "page_type")
        result = DescPage(
            doc_id=doc_id,
            doc_name=doc_name,
            page_num=page_num,
            page_type=page_type_val,
            figure_index=_str_or(None, "figure_index"),
            figure_caption=_str_or(None, "figure_caption"),
            position_hint=_str_or(None, "position_hint"),
            description=_str_or("", "description") or "",
            key_entities=key_entities,
            image_url=image_url,
            heading_path=heading_path,
        )
        # Success-path emit (failure path already emitted via emit_error above)
        if adapter_ok:
            emit(
                "describe.page",
                duration_ms=round((time.monotonic() - t0) * 1000, 1),
                success=True,
                doc_id=doc_id,
                page_num=page_num,
                page_type=page_type_val,
                json_parse_ok=json_parse_ok,
                prompt_version=DESCRIPTION_PROMPT_VERSION,
                description_chars=len(result.description),
                key_entities_count=len(result.key_entities),
                thinking_detected=thinking_detected,
                thinking_chars=thinking_chars,
                image_size_bytes=len(image_bytes),
            )
        return result
