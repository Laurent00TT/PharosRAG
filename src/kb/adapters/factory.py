# src/kb/adapters/factory.py
"""Adapter factory shared by description channel, HyDE/MQ, and Agent.

Three call sites used to duplicate the same `if adapter == "ollama" / elif
"openai_compat"` plumbing. Adding a new backend (Anthropic Messages via
a local proxy) made that drift painful — centralising here keeps backend selection
in one place.

`disable_thinking` only matters for openai_compat (Qwen3.6 short structured
tasks need thinking off). Anthropic and Ollama paths ignore it.
"""
from kb.adapters.base import BaseLLMAdapter, DISABLE_THINKING_EXTRA_BODY


def make_chat_adapter(
    *,
    adapter_type: str,
    url: str,
    api_key: str,
    model: str,
    supports_vision: bool,
    disable_thinking: bool = False,
    session_id: str | None = None,
    task_type: str | None = None,
) -> BaseLLMAdapter:
    """Build a chat-completion adapter from settings.

    Args:
      adapter_type: "openai_compat" | "ollama" | "anthropic"
      url:          base URL (e.g. http://localhost:1235/v1, http://127.0.0.1:3456)
      api_key:      auth key (sent as Bearer for openai_compat, x-api-key for anthropic,
                    ignored for ollama)
      model:        model name / id
      supports_vision: whether the backend can take image content blocks
      disable_thinking: openai_compat-only; injects extra_body to disable
                        Qwen3.x thinking (HyDE / MQ / Description need this)
      session_id:   anthropic-only; sent as x-session-id for prompt-cache
                    locality on such local proxies.
      task_type:    Phase 6.4 — semantic role of this adapter
                    ("query_rewrite" / "description" / "answer" / ...).
                    Emitted on every chat() call as kb.llm.call trace so
                    operators can attribute LLM cost/latency per task.
                    None = legacy untagged callsite (still emits, with
                    task_type=None in the event).

    Raises:
      ValueError on unknown adapter_type.
    """
    if adapter_type == "ollama":
        # Ollama has no thinking control / extra_body concept.
        from kb.adapters.ollama import OllamaAdapter
        return OllamaAdapter(model=model, task_type=task_type)

    if adapter_type == "openai_compat":
        from kb.adapters.openai_compat import OpenAICompatAdapter
        return OpenAICompatAdapter(
            base_url=url,
            api_key=api_key,
            model=model,
            supports_vision=supports_vision,
            extra_body=DISABLE_THINKING_EXTRA_BODY if disable_thinking else None,
            task_type=task_type,
        )

    if adapter_type == "anthropic":
        from kb.adapters.anthropic_native import AnthropicNativeAdapter
        return AnthropicNativeAdapter(
            base_url=url,
            api_key=api_key,
            model=model,
            supports_vision=supports_vision,
            session_id=session_id,
            task_type=task_type,
        )

    raise ValueError(
        f"Unknown adapter_type: {adapter_type!r}. "
        f"Must be one of: openai_compat, ollama, anthropic."
    )
