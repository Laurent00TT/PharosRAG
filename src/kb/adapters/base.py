# src/kb/adapters/base.py
import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# Qwen3.x "disable thinking" payload — OpenAI-compat extra_body forwards it to the server.
# HyDE / Multi-Query / Description are short structured-output tasks where thinking
# must be off:
#   1. wastes tokens + adds 5-10x latency
#   2. short outputs can end up content="" with all the bytes in reasoning_content,
#      so downstream parsing of an empty string fails
DISABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AdapterMessage:
    role: str               # "system" | "user" | "assistant" | "tool"
    content: str | list[dict]
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class AdapterResponse:
    content: str                                # clean content (thinking already stripped)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None
    # Thinking-split metadata from parse_thinking; defaults are 0/False so older
    # call sites that ignore these fields still work.
    thinking_chars: int = 0
    content_chars: int = 0
    total_tokens: int = 0
    thinking_detected: bool = False


class BaseLLMAdapter(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[AdapterMessage],
        tools: list[dict] | None = None,
    ) -> AdapterResponse:
        ...

    @abstractmethod
    def supports_vision(self) -> bool:
        ...

    async def aclose(self) -> None:
        return None


# ── Shared helpers for OpenAI-shaped APIs (OpenAI-compat wire format) ────────


def to_openai_message(m: AdapterMessage) -> dict:
    """Convert AdapterMessage → OpenAI chat-completions message dict.
    Used by OpenAICompatAdapter (OpenAI chat-completions wire format).
    """
    msg: dict = {"role": m.role, "content": m.content}
    if m.tool_call_id:
        msg["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    return msg


def parse_openai_response(response) -> AdapterResponse:
    """Convert openai.ChatCompletion-shaped response → AdapterResponse.
    Shared by all OpenAI-compatible adapters.

    AdapterResponse.content has been processed by parse_thinking to strip any
    <think>...</think> tags, so downstream consumers (citation parser, multi_query
    parser) do not need to be aware of thinking at all.
    """
    from kb.observability import parse_thinking

    choice = response.choices[0].message
    tool_calls: list[ToolCall] = []
    if choice.tool_calls:
        for tc in choice.tool_calls:
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            ))

    parsed = parse_thinking(response)
    return AdapterResponse(
        content=parsed["clean_content"],
        tool_calls=tool_calls,
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
        thinking_chars=parsed["thinking_chars"],
        content_chars=parsed["content_chars"],
        total_tokens=parsed["total_tokens"],
        thinking_detected=parsed["thinking_detected"],
    )


# ── Multimodal content helpers ───────────────────────────────────────────────


@dataclass
class ImagePart:
    """Image part of a multimodal message.
    OpenAI / Ollama / any OpenAI-compat servers all accept the image_url + data-URL protocol.
    """
    data: bytes
    mime_type: str = "image/png"

    def to_openai_dict(self) -> dict:
        b64 = base64.b64encode(self.data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.mime_type};base64,{b64}"},
        }


def build_multimodal_content(
    text: str,
    images: list[ImagePart] | None = None,
) -> list[dict]:
    """Build an OpenAI-style multipart content list.
    The return value can be passed directly as AdapterMessage.content into chat()."""
    parts: list[dict] = [{"type": "text", "text": text}]
    if images:
        parts.extend(img.to_openai_dict() for img in images)
    return parts
