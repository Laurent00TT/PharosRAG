# src/kb/adapters/ollama.py
import httpx

from kb.adapters.base import (
    AdapterMessage, AdapterResponse, BaseLLMAdapter, ToolCall,
)


# Prefix-match instead of a hard-coded tag list — extension-friendly.
# Covers Qwen2 / Qwen3 / Qwen3.6 vision variants.
_VISION_CAPABLE_PREFIXES = (
    "llava",
    "qwen2-vl",
    "qwen3-vl",
    "qwen3.6",         # the 3.6 family is natively multimodal
    "minicpm-v",
    "gemma2:vision",
    "gemma3:vision",
)


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for local Ollama server (Gemma 4 / Qwen3 / Llava)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma2",
        timeout: float = 120.0,
        task_type: str | None = None,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout)
        # Phase 6.4 — semantic role for kb.llm.call attribution.
        self._task_type = task_type

    def supports_vision(self) -> bool:
        m = self._model.lower()
        return any(m.startswith(p) for p in _VISION_CAPABLE_PREFIXES)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[AdapterMessage],
        tools: list[dict] | None = None,
    ) -> AdapterResponse:
        from kb.observability import parse_thinking

        ollama_messages = [self._to_ollama(m) for m in messages]
        body: dict = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if tools:
            body["tools"] = tools
        response = await self._client.post(f"{self._url}/api/chat", json=body)
        response.raise_for_status()
        data = response.json()
        msg = data.get("message", {})
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                import json
                args = json.loads(args)
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        # Run parse_thinking to strip <think> tags — keeps behavior in lockstep with OpenAICompatAdapter
        synthetic = {
            "choices": [{"message": {"content": msg.get("content", "")}}],
            "usage": {},
        }
        parsed = parse_thinking(synthetic)
        result = AdapterResponse(
            content=parsed["clean_content"],
            tool_calls=tool_calls,
            thinking_chars=parsed["thinking_chars"],
            content_chars=parsed["content_chars"],
            total_tokens=parsed["total_tokens"],
            thinking_detected=parsed["thinking_detected"],
        )
        # Phase 6.4: per-call trace for cost / latency attribution.
        # Ollama responses don't carry prompt/completion token counts in
        # the same shape as OpenAI; we emit None for those fields.
        from kb.observability import emit
        emit(
            "kb.llm.call",
            adapter="ollama",
            task_type=self._task_type,
            model=self._model,
            prompt_tokens=None,
            completion_tokens=None,
            thinking_detected=result.thinking_detected,
            tools_offered=bool(tools),
            tool_calls=len(result.tool_calls),
        )
        return result

    def _to_ollama(self, m: AdapterMessage) -> dict:
        # Ollama multimodal: content is a string and images is a separate list of base64 strings
        if isinstance(m.content, list):
            text_parts = [p["text"] for p in m.content if p.get("type") == "text"]
            image_urls = [p["image_url"]["url"] for p in m.content if p.get("type") == "image_url"]
            # Extract the base64 part from each data URL
            images_b64 = []
            for url in image_urls:
                if url.startswith("data:") and "base64," in url:
                    images_b64.append(url.split("base64,", 1)[1])
            msg: dict = {
                "role": m.role,
                "content": " ".join(text_parts),
            }
            if images_b64:
                msg["images"] = images_b64   # Ollama protocol field
        else:
            msg = {"role": m.role, "content": m.content}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        return msg
