# src/kb/adapters/anthropic_native.py
"""Anthropic Messages API client.

Compatible with:
  - Anthropic cloud  : https://api.anthropic.com
  - Local proxies wrapping the Anthropic SDK (e.g. a local claude-proxy
    style projects that re-expose anthropic-python over HTTP)

Wire-format differences from OpenAI Chat Completions:
  - `system` is a TOP-LEVEL string, not a message in the array
  - tool definitions: {name, description, input_schema} (vs OpenAI's nested
    `function` wrapper)
  - tool calls in response: content blocks with type=tool_use
  - tool results in next turn: content blocks with type=tool_result
  - images: {type: image, source: {type: base64, media_type, data}}
  - auth: x-api-key header (NOT Authorization Bearer)
  - required: anthropic-version header

The adapter handles all of the above so SearchEngine / Agent / Description
channel keep using AdapterMessage and don't see Anthropic-specific shapes.

Optional session_id (x-session-id) helps proxies reuse cached
system-prompt prefix tokens across calls in the same logical session.
"""
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

from kb.adapters.base import (
    AdapterMessage, AdapterResponse, BaseLLMAdapter, ToolCall,
)


class AnthropicNativeAdapter(BaseLLMAdapter):
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        supports_vision: bool = True,
        max_tokens: int = 4096,
        anthropic_version: str = "2023-06-01",
        session_id: str | None = None,
        timeout_s: float = 180.0,
        task_type: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._supports_vision = supports_vision
        self._max_tokens = max_tokens
        self._anthropic_version = anthropic_version
        self._session_id = session_id
        self._timeout_s = timeout_s
        # Phase 6.4 — semantic role for kb.llm.call attribution.
        self._task_type = task_type

    def supports_vision(self) -> bool:
        return self._supports_vision

    def _headers(self) -> dict:
        h = {
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
            "Content-Type": "application/json",
        }
        if self._session_id:
            # Local proxies use this to keep the system-prompt
            # prefix-cache hot across multi-turn calls in one logical session.
            h["x-session-id"] = self._session_id
        return h

    async def chat(
        self,
        messages: list[AdapterMessage],
        tools: list[dict] | None = None,
    ) -> AdapterResponse:
        # Anthropic places `system` at top level; pull system messages out and
        # concatenate (multiple system messages are flattened).
        system_text = ""
        non_system: list[AdapterMessage] = []
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_text += (("\n\n" if system_text else "") + m.content)
                # ignore non-string system content (rare)
            else:
                non_system.append(m)

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [_to_anthropic_message(m) for m in non_system],
            "temperature": 0.0,
        }
        if system_text:
            body["system"] = system_text
        if tools:
            body["tools"] = [_openai_tool_to_anthropic(t) for t in tools]

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(
                f"{self._base}/v1/messages",
                json=body,
                headers=self._headers(),
            )
            r.raise_for_status()
            resp = r.json()

        result = _parse_anthropic_response(resp)
        # Phase 6.4: per-call trace for cost/latency attribution.
        from kb.observability import emit
        emit(
            "kb.llm.call",
            adapter="anthropic",
            task_type=self._task_type,
            model=self._model,
            prompt_tokens=(result.usage or {}).get("prompt_tokens"),
            completion_tokens=(result.usage or {}).get("completion_tokens"),
            thinking_detected=result.thinking_detected,
            tools_offered=bool(tools),
            tool_calls=len(result.tool_calls),
        )
        return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _to_anthropic_message(m: AdapterMessage) -> dict:
    """Translate AdapterMessage → Anthropic message dict."""
    # Tool result (came from a previous tool call)
    if m.role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content if isinstance(m.content, str)
                           else json.dumps(m.content, ensure_ascii=False),
            }],
        }

    # Assistant message that issued tool calls (or both text + tool calls)
    if m.tool_calls:
        blocks: list[dict] = []
        if isinstance(m.content, str) and m.content:
            blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        return {"role": m.role, "content": blocks}

    # Plain text content
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}

    # Multipart content (OpenAI-shaped: text + image_url) → translate to
    # Anthropic content blocks.
    blocks_out: list[dict] = []
    for part in m.content:
        ptype = part.get("type")
        if ptype == "text":
            blocks_out.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                # data:image/png;base64,XXX → split into media_type + data.
                # Wrap in try/except: a malformed data URL (missing comma,
                # missing colon, etc.) used to raise ValueError and crash
                # the whole conversion. Now we warn and drop this block —
                # the rest of the message still gets through.
                try:
                    header, b64 = url.split(",", 1)
                    mime = header.split(":", 1)[1].split(";", 1)[0]
                    if not mime or not b64:
                        raise ValueError("empty mime or payload")
                except (ValueError, IndexError) as exc:
                    logger.warning(
                        "anthropic_native: dropping malformed data URL "
                        "(%s); URL prefix=%r", exc, url[:50],
                    )
                    continue
                blocks_out.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                })
            else:
                blocks_out.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
        # silently drop unknown parts (defensive)
    return {"role": m.role, "content": blocks_out}


def _openai_tool_to_anthropic(t: dict) -> dict:
    """Translate OpenAI tool schema → Anthropic tool schema.

    OpenAI : {"type":"function","function":{"name","description","parameters"}}
    Anthropic: {"name","description","input_schema"}
    """
    if "function" in t:
        f = t["function"]
        return {
            "name": f["name"],
            "description": f.get("description", ""),
            "input_schema": f.get("parameters", {}),
        }
    # Already Anthropic-shaped — pass through
    return t


def _parse_anthropic_response(resp: dict) -> AdapterResponse:
    """Translate Anthropic Messages response → AdapterResponse."""
    content_text = ""
    tool_calls: list[ToolCall] = []

    for block in resp.get("content", []):
        btype = block.get("type")
        if btype == "text":
            content_text += block.get("text", "")
        elif btype == "tool_use":
            tool_calls.append(ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input", {}),
            ))
        # 'thinking' blocks (extended thinking) are ignored — Anthropic returns
        # them only when explicitly requested via thinking={"type":"enabled"}
        # which we don't enable for HyDE/MQ/Description (short structured output).

    usage = resp.get("usage", {})
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)
    return AdapterResponse(
        content=content_text,
        tool_calls=tool_calls,
        usage={
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
        },
        content_chars=len(content_text),
        total_tokens=in_tokens + out_tokens,
    )
