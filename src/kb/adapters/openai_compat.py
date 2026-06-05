# src/kb/adapters/openai_compat.py
import openai

from kb.adapters.base import (
    AdapterMessage, AdapterResponse, BaseLLMAdapter,
    parse_openai_response, to_openai_message,
)
from kb.observability import emit


class OpenAICompatAdapter(BaseLLMAdapter):
    """Adapter for any OpenAI-compatible HTTP API (vLLM, LM Studio, llama-server, ...).

    extra_body lets callers pass server-specific kwargs that aren't part of the
    OpenAI schema. Critical use case: Qwen3.6 thinking control —
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    Must be set for HyDE / Multi-Query / Description (so they get clean structured
    output); the Agent leaves it unset to keep thinking enabled.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        supports_vision: bool = False,
        extra_body: dict | None = None,
        task_type: str | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._supports_vision = supports_vision
        self._extra_body = extra_body or {}
        # Phase 6.4 — semantic role for cost/latency attribution in kb.llm.call.
        self._task_type = task_type

    def supports_vision(self) -> bool:
        return self._supports_vision

    async def aclose(self) -> None:
        await self._client.close()

    async def chat(
        self,
        messages: list[AdapterMessage],
        tools: list[dict] | None = None,
    ) -> AdapterResponse:
        kwargs: dict = {
            "model": self._model,
            "messages": [to_openai_message(m) for m in messages],
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = tools
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        response = await self._client.chat.completions.create(**kwargs)
        result = parse_openai_response(response)
        # Phase 6.4: per-call trace so operators can attribute LLM cost
        # and latency per semantic task. Emitted AFTER parse so any
        # adapter-level exception short-circuits without partial events.
        emit(
            "kb.llm.call",
            adapter="openai_compat",
            task_type=self._task_type,
            model=self._model,
            prompt_tokens=(result.usage or {}).get("prompt_tokens"),
            completion_tokens=(result.usage or {}).get("completion_tokens"),
            thinking_detected=result.thinking_detected,
            tools_offered=bool(tools),
            tool_calls=len(result.tool_calls),
        )
        return result
