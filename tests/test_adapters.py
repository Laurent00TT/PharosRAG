# tests/test_adapters.py
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb.adapters.base import (
    AdapterMessage, AdapterResponse, ToolCall, BaseLLMAdapter
)


def test_tool_call_dataclass():
    tc = ToolCall(id="call_1", name="search_docs", arguments={"query": "x"})
    assert tc.name == "search_docs"
    assert tc.arguments["query"] == "x"


def test_adapter_message_with_tool_calls():
    tc = ToolCall(id="c1", name="search_docs", arguments={"q": "a"})
    msg = AdapterMessage(role="assistant", content="", tool_calls=[tc])
    assert msg.role == "assistant"
    assert msg.tool_calls[0].name == "search_docs"


def test_adapter_response_defaults():
    resp = AdapterResponse(content="hello", tool_calls=[])
    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert resp.usage is None


def test_base_adapter_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        BaseLLMAdapter()  # cannot instantiate ABC


@pytest.mark.asyncio
async def test_base_adapter_default_aclose_noop():
    class MinimalAdapter(BaseLLMAdapter):
        async def chat(self, messages, tools=None):
            return AdapterResponse(content="")

        def supports_vision(self):
            return False

    await MinimalAdapter().aclose()


def test_to_openai_message_simple():
    from kb.adapters.base import to_openai_message
    msg = AdapterMessage(role="user", content="hello")
    assert to_openai_message(msg) == {"role": "user", "content": "hello"}


def test_to_openai_message_with_tool_calls():
    from kb.adapters.base import to_openai_message
    tc = ToolCall(id="c1", name="search_docs", arguments={"query": "x"})
    msg = AdapterMessage(role="assistant", content="", tool_calls=[tc])
    out = to_openai_message(msg)
    assert out["tool_calls"][0]["function"]["name"] == "search_docs"
    assert out["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'


def test_parse_openai_response_with_tool_call():
    from unittest.mock import MagicMock
    from kb.adapters.base import parse_openai_response
    fn = MagicMock()
    fn.name = "search_docs"
    fn.arguments = '{"query": "y"}'
    tc = MagicMock(id="call_1", function=fn)
    msg = MagicMock(content=None, tool_calls=[tc])
    resp = MagicMock(choices=[MagicMock(message=msg)])
    resp.usage = MagicMock(prompt_tokens=4, completion_tokens=2)
    parsed = parse_openai_response(resp)
    assert parsed.content == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "search_docs"
    assert parsed.tool_calls[0].arguments == {"query": "y"}
    assert parsed.usage["prompt_tokens"] == 4


from kb.adapters.ollama import OllamaAdapter


def test_ollama_supports_vision_for_known_models():
    a = OllamaAdapter(base_url="http://x:11434", model="llava")
    assert a.supports_vision() is True
    b = OllamaAdapter(base_url="http://x:11434", model="qwen2-vl")
    assert b.supports_vision() is True


def test_ollama_does_not_claim_vision_by_default():
    a = OllamaAdapter(base_url="http://x:11434", model="gemma2")
    assert a.supports_vision() is False


@pytest.mark.asyncio
async def test_ollama_chat_text_only():
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="gemma2")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "message": {"role": "assistant", "content": "hi there", "tool_calls": []},
    }
    fake_resp.raise_for_status = MagicMock()
    with patch.object(adapter._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        result = await adapter.chat([
            AdapterMessage(role="user", content="hello"),
        ])
    assert result.content == "hi there"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_ollama_adapter_closes_client():
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="gemma2")
    close = AsyncMock()
    adapter._client.aclose = close

    await adapter.aclose()

    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ollama_chat_with_tool_calls():
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="gemma2")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "search_docs", "arguments": {"query": "x"}},
            }],
        },
    }
    fake_resp.raise_for_status = MagicMock()
    with patch.object(adapter._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        result = await adapter.chat(
            [AdapterMessage(role="user", content="find x")],
            tools=[{"type": "function", "function": {"name": "search_docs"}}],
        )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_docs"
    assert result.tool_calls[0].arguments == {"query": "x"}


from kb.adapters.openai_compat import OpenAICompatAdapter


def test_openai_compat_explicit_vision_flag():
    a = OpenAICompatAdapter(base_url="https://api.x.com/v1",
                            api_key="k", model="gpt-4o-mini", supports_vision=True)
    assert a.supports_vision() is True

    b = OpenAICompatAdapter(base_url="https://api.x.com/v1",
                            api_key="k", model="text-only-model")
    assert b.supports_vision() is False


@pytest.mark.asyncio
async def test_openai_compat_chat():
    adapter = OpenAICompatAdapter(
        base_url="https://api.x.com/v1", api_key="k", model="m"
    )
    fake_msg = MagicMock(content="answer", tool_calls=None)
    fake_resp = MagicMock(
        choices=[MagicMock(message=fake_msg)],
        usage=MagicMock(prompt_tokens=5, completion_tokens=3),
    )
    with patch.object(adapter._client.chat.completions, "create",
                      new_callable=AsyncMock, return_value=fake_resp):
        result = await adapter.chat([AdapterMessage(role="user", content="q")])
    assert result.content == "answer"
    assert result.usage["prompt_tokens"] == 5


@pytest.mark.asyncio
async def test_openai_compat_adapter_closes_client():
    adapter = OpenAICompatAdapter(
        base_url="https://api.x.com/v1",
        api_key="k",
        model="m",
    )
    close = AsyncMock()
    adapter._client.close = close

    await adapter.aclose()

    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_clients_used_by_channels_are_closable(test_settings):
    from kb.ingestion.channels.vision import VisionChannel
    from kb.parser.mineru_local import MinerULocalParser
    from kb.search.vision_query import VisionQueryEmbedder

    # VisionChannel (v1.1) is clientless — it opens a fresh httpx client per
    # embed_async call (see channels/vision.py), so there is no persistent
    # client to leak. aclose() is a documented no-op kept for caller-API
    # compatibility; just verify it's awaitable and never raises.
    vision = VisionChannel(test_settings)
    await vision.aclose()

    query = VisionQueryEmbedder(test_settings)
    query_close = AsyncMock()
    query._client.aclose = query_close
    await query.aclose()
    query_close.assert_awaited_once()

    parser = MinerULocalParser(test_settings)
    parser_close = AsyncMock()
    parser._client.aclose = parser_close
    await parser.aclose()
    parser_close.assert_awaited_once()


from kb.adapters.base import ImagePart, build_multimodal_content


def test_image_part_to_openai_format():
    part = ImagePart(data=b"fake_png_bytes", mime_type="image/png")
    msg = part.to_openai_dict()
    assert msg["type"] == "image_url"
    assert msg["image_url"]["url"].startswith("data:image/png;base64,")
    # 解码验证
    b64 = msg["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(b64) == b"fake_png_bytes"


def test_build_multimodal_content_mixes_text_and_images():
    content = build_multimodal_content(
        text="See the chart below:",
        images=[
            ImagePart(data=b"img1_bytes", mime_type="image/png"),
            ImagePart(data=b"img2_bytes", mime_type="image/jpeg"),
        ],
    )
    assert isinstance(content, list)
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "See the chart below:"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "image_url"


@pytest.mark.asyncio
async def test_openai_compat_chat_with_multimodal_content():
    from kb.adapters.openai_compat import OpenAICompatAdapter
    adapter = OpenAICompatAdapter(
        base_url="https://x/v1", api_key="k", model="m", supports_vision=True,
    )
    fake_msg = MagicMock(content="The chart shows ...", tool_calls=None)
    fake_resp = MagicMock(
        choices=[MagicMock(message=fake_msg)],
        usage=MagicMock(prompt_tokens=12, completion_tokens=5),
    )
    captured_kwargs = {}

    async def capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_resp

    with patch.object(adapter._client.chat.completions, "create",
                      new=capture):
        result = await adapter.chat([
            AdapterMessage(role="user", content=[
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]),
        ])
    # 多模态消息应该原样传递
    sent_msgs = captured_kwargs["messages"]
    assert isinstance(sent_msgs[0]["content"], list)
    assert sent_msgs[0]["content"][1]["type"] == "image_url"
    assert result.content == "The chart shows ..."


@pytest.mark.asyncio
async def test_ollama_chat_with_multimodal_content():
    """Ollama: list-content gets converted to content+images protocol."""
    from kb.adapters.ollama import OllamaAdapter
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="llava")

    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "message": {"role": "assistant", "content": "I see a chart", "tool_calls": []},
    }
    fake_resp.raise_for_status = MagicMock()

    captured_body = {}
    async def capture_post(url, **kwargs):
        captured_body.update(kwargs.get("json", {}))
        return fake_resp

    with patch.object(adapter._client, "post", new=capture_post):
        result = await adapter.chat([
            AdapterMessage(role="user", content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,SGVsbG8="}},
            ]),
        ])

    # Verify Ollama-shape body
    sent_msg = captured_body["messages"][0]
    assert sent_msg["content"] == "What's in this image?"
    assert sent_msg["images"] == ["SGVsbG8="]
    assert result.content == "I see a chart"


@pytest.mark.asyncio
async def test_ollama_chat_with_text_only_list_content():
    """Ollama: list with text only (no images) — content gets joined, no images key."""
    from kb.adapters.ollama import OllamaAdapter
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="gemma2")

    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "message": {"role": "assistant", "content": "ok", "tool_calls": []},
    }
    fake_resp.raise_for_status = MagicMock()

    captured_body = {}
    async def capture_post(url, **kwargs):
        captured_body.update(kwargs.get("json", {}))
        return fake_resp

    with patch.object(adapter._client, "post", new=capture_post):
        await adapter.chat([
            AdapterMessage(role="user", content=[
                {"type": "text", "text": "Hello"},
            ]),
        ])

    sent_msg = captured_body["messages"][0]
    assert sent_msg["content"] == "Hello"
    assert "images" not in sent_msg   # 没有图片 = 不带 images 字段


# ── extra_body 透传 + thinking 控制 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_compat_passes_extra_body_to_server():
    """extra_body 应原样透传给 openai SDK(server 端处理 chat_template_kwargs)。"""
    from kb.adapters.base import DISABLE_THINKING_EXTRA_BODY
    adapter = OpenAICompatAdapter(
        base_url="https://x/v1", api_key="k", model="m",
        extra_body=DISABLE_THINKING_EXTRA_BODY,
    )
    fake_msg = MagicMock(content="answer", tool_calls=None)
    fake_resp = MagicMock(
        choices=[MagicMock(message=fake_msg)],
        usage=MagicMock(prompt_tokens=5, completion_tokens=3),
    )
    captured_kwargs = {}

    async def capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_resp

    with patch.object(adapter._client.chat.completions, "create", new=capture):
        await adapter.chat([AdapterMessage(role="user", content="q")])

    assert "extra_body" in captured_kwargs
    assert captured_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.asyncio
async def test_openai_compat_no_extra_body_when_unset():
    """没传 extra_body 时不应给 SDK 加这个 kwarg(保留 server 默认行为)。"""
    adapter = OpenAICompatAdapter(base_url="https://x/v1", api_key="k", model="m")
    fake_msg = MagicMock(content="answer", tool_calls=None)
    fake_resp = MagicMock(
        choices=[MagicMock(message=fake_msg)],
        usage=MagicMock(prompt_tokens=5, completion_tokens=3),
    )
    captured_kwargs = {}

    async def capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_resp

    with patch.object(adapter._client.chat.completions, "create", new=capture):
        await adapter.chat([AdapterMessage(role="user", content="q")])

    assert "extra_body" not in captured_kwargs


@pytest.mark.asyncio
async def test_openai_compat_response_strips_think_tags():
    """response 含 <think>...</think> 时 AdapterResponse.content 应是 clean_content。"""
    adapter = OpenAICompatAdapter(base_url="https://x/v1", api_key="k", model="m")
    fake_msg = MagicMock(content="<think>analyzing...</think>final answer", tool_calls=None)
    fake_resp = MagicMock(
        choices=[MagicMock(message=fake_msg)],
        usage=MagicMock(prompt_tokens=5, completion_tokens=10),
    )
    with patch.object(adapter._client.chat.completions, "create",
                      new_callable=AsyncMock, return_value=fake_resp):
        result = await adapter.chat([AdapterMessage(role="user", content="q")])

    assert result.content == "final answer"
    assert result.thinking_detected is True
    assert result.thinking_chars == len("analyzing...")
    assert result.content_chars == len("final answer")
    assert result.total_tokens == 10


@pytest.mark.asyncio
async def test_openai_compat_response_no_thinking():
    """普通响应 thinking_detected=False。"""
    adapter = OpenAICompatAdapter(base_url="https://x/v1", api_key="k", model="m")
    fake_msg = MagicMock(content="just an answer", tool_calls=None)
    fake_resp = MagicMock(
        choices=[MagicMock(message=fake_msg)],
        usage=MagicMock(prompt_tokens=3, completion_tokens=4),
    )
    with patch.object(adapter._client.chat.completions, "create",
                      new_callable=AsyncMock, return_value=fake_resp):
        result = await adapter.chat([AdapterMessage(role="user", content="q")])

    assert result.thinking_detected is False
    assert result.thinking_chars == 0
    assert result.content == "just an answer"


@pytest.mark.asyncio
async def test_ollama_chat_strips_think_tags():
    """Ollama adapter 也应通过 parse_thinking 剥离 <think> 标签。"""
    from kb.adapters.ollama import OllamaAdapter
    adapter = OllamaAdapter(base_url="http://localhost:11434", model="llava")

    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "<think>weighing options</think>chose A",
            "tool_calls": [],
        },
    }
    fake_resp.raise_for_status = MagicMock()
    with patch.object(adapter._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        result = await adapter.chat([AdapterMessage(role="user", content="pick")])
    assert result.content == "chose A"
    assert result.thinking_detected is True
    assert result.thinking_chars == len("weighing options")


# ── Phase 6.4 follow-up: description vision + agent tools coverage ──────


def test_description_adapter_propagates_supports_vision_flag():
    """6.4 execution list: 'description adapter 要求 vision 时配置生效.'
    The pipeline builds the description channel via make_chat_adapter
    with supports_vision=settings.description_supports_vision. Verify
    the openai_compat path actually propagates that flag through to the
    adapter instance (so the description channel can refuse to send
    images to a text-only model)."""
    from kb.adapters.factory import make_chat_adapter

    vision_adapter = make_chat_adapter(
        adapter_type="openai_compat",
        url="https://api.x.com/v1",
        api_key="k",
        model="vision-model",
        supports_vision=True,
        disable_thinking=True,
        task_type="description",
    )
    assert vision_adapter.supports_vision() is True
    assert vision_adapter._task_type == "description"
    # disable_thinking=True must inject the extra_body so structured
    # description outputs don't get trapped in reasoning_content.
    assert vision_adapter._extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    text_only = make_chat_adapter(
        adapter_type="openai_compat",
        url="https://api.x.com/v1",
        api_key="k",
        model="text-only-model",
        supports_vision=False,
        disable_thinking=True,
        task_type="description",
    )
    assert text_only.supports_vision() is False


@pytest.mark.asyncio
async def test_agent_adapter_preserves_tool_calls_in_response():
    """6.4 execution list: 'agent adapter 保留工具调用能力.'
    The agent task uses make_chat_adapter(..., disable_thinking=False)
    and expects tool_calls to round-trip through the adapter. Mock the
    underlying client to return a tool_call response and verify the
    AdapterResponse surfaces it cleanly with name + arguments intact."""
    from kb.adapters.factory import make_chat_adapter

    agent_adapter = make_chat_adapter(
        adapter_type="openai_compat",
        url="https://api.x.com/v1",
        api_key="k",
        model="gpt-4o-mini",
        supports_vision=True,
        disable_thinking=False,    # agent KEEPS thinking
        task_type="answer",
    )
    # disable_thinking=False → no extra_body interfering with tool path.
    assert agent_adapter._extra_body == {}
    assert agent_adapter._task_type == "answer"

    fake_msg = MagicMock()
    fake_msg.content = "calling tool..."
    fake_msg.reasoning_content = None
    fake_tc = MagicMock()
    fake_tc.id = "call_42"
    fake_tc.function.name = "search_docs"
    fake_tc.function.arguments = '{"query": "alpha"}'
    fake_msg.tool_calls = [fake_tc]

    fake_choice = MagicMock(); fake_choice.message = fake_msg
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    fake_resp.usage.prompt_tokens = 12
    fake_resp.usage.completion_tokens = 7

    with patch.object(agent_adapter._client.chat.completions, "create",
                      new=AsyncMock(return_value=fake_resp)):
        result = await agent_adapter.chat(
            [AdapterMessage(role="user", content="find alpha")],
            tools=[{"type": "function", "function": {"name": "search_docs"}}],
        )

    # Tool round-trip intact — name, arguments parsed back to dict.
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_42"
    assert result.tool_calls[0].name == "search_docs"
    assert result.tool_calls[0].arguments == {"query": "alpha"}
    # Content is preserved alongside tool_calls (mixed response).
    assert "calling tool" in result.content
