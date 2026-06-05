"""Parse thinking vs content from an LLM response.

Two formats are supported:
- Track 1 (vLLM / SGLang): a separate `reasoning_content` field carries thinking.
- Track 2 (Qwen3.x + llama.cpp): `<think>...</think>` tags embedded in content.

The returned clean_content has thinking stripped, so it can be passed directly
as AdapterResponse.content (multi_query / citation parser do not need to
distinguish thinking from real output).

We return chars rather than tokens — re-tokenizing client-side is expensive and
the tokenizer may not be available. total_tokens comes from the server's usage
field (which includes thinking), and analysis can derive the thinking share
from the chars ratio.
"""
import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_thinking(response) -> dict:
    """Extract thinking/content split.

    Args:
      response: dict OR openai SDK ChatCompletion-shaped object.

    Returns:
      {
        "thinking_chars": int,
        "content_chars": int,
        "clean_content": str,    # content with <think>...</think> removed (or unchanged if no tag)
        "total_tokens": int,     # from usage.completion_tokens (includes thinking)
        "thinking_detected": bool,
      }
    """
    msg, usage = _extract_message_usage(response)
    raw_content = msg.get("content", "") or ""
    if not isinstance(raw_content, str):
        raw_content = ""
    completion_tokens = usage.get("completion_tokens", 0) if usage else 0

    # Track 1: reasoning_content field (vLLM / SGLang)
    # Must be a non-empty str — MagicMock's auto-attributes would bypass a
    # None check, so we tighten it to an explicit isinstance(str) test.
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return {
            "thinking_chars": len(reasoning),
            "content_chars": len(raw_content),
            "clean_content": raw_content,
            "total_tokens": completion_tokens,
            "thinking_detected": True,
        }

    # Track 2: <think>...</think> tag (Qwen3.x + llama.cpp)
    m = _THINK_RE.search(raw_content)
    if m:
        thinking = m.group(1)
        clean = (raw_content[:m.start()] + raw_content[m.end():]).strip()
        return {
            "thinking_chars": len(thinking),
            "content_chars": len(clean),
            "clean_content": clean,
            "total_tokens": completion_tokens,
            "thinking_detected": True,
        }

    # Track 3: no thinking detected
    return {
        "thinking_chars": 0,
        "content_chars": len(raw_content),
        "clean_content": raw_content,
        "total_tokens": completion_tokens,
        "thinking_detected": False,
    }


def _extract_message_usage(response) -> tuple[dict, dict | None]:
    """Normalize OpenAI SDK object OR raw dict into (message_dict, usage_dict_or_None)."""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        msg_obj = choices[0].get("message", {}) if choices else {}
        usage = response.get("usage")
        if isinstance(usage, dict):
            return msg_obj, usage
        return msg_obj, None

    # openai SDK object — read via getattr
    choices = getattr(response, "choices", [])
    if not choices:
        return {}, None
    msg_obj = choices[0].message
    msg_dict = {
        "content": getattr(msg_obj, "content", "") or "",
        "reasoning_content": getattr(msg_obj, "reasoning_content", None),
    }
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return msg_dict, None
    usage_dict = {
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
    }
    return msg_dict, usage_dict
