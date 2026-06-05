# tests/test_agent.py
from kb.agent.citation import Citation, parse_citations


def test_parse_citations_extracts_format():
    text = "The CEO must approve [abc123:3] amounts over 1M."
    cites = parse_citations(text)
    assert len(cites) == 1
    assert cites[0] == Citation(doc_id="abc123", page_num=3)


def test_parse_citations_multiple():
    text = "[doc1:0] is the first reference, [doc2:5] is the second."
    cites = parse_citations(text)
    assert len(cites) == 2
    assert cites[0].doc_id == "doc1"
    assert cites[1].page_num == 5


def test_parse_citations_no_match():
    text = "The CEO must approve."
    assert parse_citations(text) == []


def test_parse_citations_handles_alphanumeric_ids():
    text = "See [a1b2c3d4:12]"
    cites = parse_citations(text)
    assert cites[0].doc_id == "a1b2c3d4"


def test_parse_citations_ignores_malformed():
    # No colon, missing page, non-digit page
    text = "[doc1] [doc2:] [doc3:abc]"
    assert parse_citations(text) == []


from kb.agent.prompts import SYSTEM_PROMPT, SEARCH_DOCS_TOOL_DEF


def test_system_prompt_mentions_citation_format():
    assert "[doc_id:page_num]" in SYSTEM_PROMPT


def test_system_prompt_mentions_low_confidence_behavior():
    assert "low_confidence" in SYSTEM_PROMPT


def test_prompt_versions_are_exported():
    from kb.agent.prompts import AGENT_PROMPT_VERSION
    from kb.ingestion.channels.description import DESCRIPTION_PROMPT_VERSION
    from kb.search.hyde import HYDE_PROMPT_VERSION
    from kb.search.multi_query import MULTI_QUERY_PROMPT_VERSION

    assert AGENT_PROMPT_VERSION.startswith("agent-")
    assert DESCRIPTION_PROMPT_VERSION.startswith("description-")
    assert HYDE_PROMPT_VERSION.startswith("hyde-")
    assert MULTI_QUERY_PROMPT_VERSION.startswith("multi-query-")


def test_search_docs_tool_def_is_valid_openai_format():
    assert SEARCH_DOCS_TOOL_DEF["type"] == "function"
    fn = SEARCH_DOCS_TOOL_DEF["function"]
    assert fn["name"] == "search_docs"
    assert "query" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["query"]


import pytest
from unittest.mock import AsyncMock, MagicMock
from kb.adapters.base import AdapterMessage, AdapterResponse, ToolCall
from kb.agent.agent import KBAgent
from kb.models import SearchResponse, SearchResult


def _make_search_result(doc_id: str, page_num: int, text: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id, doc_name=f"{doc_id}.pdf", page_num=page_num,
        heading_path=[], text_chunk=text,
        vision_description=None, figure_index=None, figure_caption=None,
        image_url="", page_type="text", doc_version=None, effective_date=None,
        score=0.9, match_channels=["dense"],
    )


@pytest.mark.asyncio
async def test_agent_returns_direct_answer_no_tool_call():
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="The sky is blue.", tool_calls=[],
    ))
    engine = MagicMock()

    agent = KBAgent(adapter=adapter, engine=engine)
    result = await agent.answer("Why is the sky blue?")

    assert result["answer"] == "The sky is blue."
    assert result["citations"] == []
    assert result["retrieved"] == []
    adapter.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_calls_search_then_answers():
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    adapter.chat = AsyncMock(side_effect=[
        # First turn: tool call
        AdapterResponse(
            content="",
            tool_calls=[ToolCall(id="c1", name="search_docs",
                                 arguments={"query": "approval"})],
        ),
        # Second turn: final answer with citation
        AdapterResponse(
            content="CEO approval is required [doc1:0].", tool_calls=[],
        ),
    ])
    engine = MagicMock()
    engine.search = AsyncMock(return_value=(
        SearchResponse(
            results=[_make_search_result("doc1", 0, "CEO must sign for >1M")],
            total=1,
        ),
        False,
    ))

    agent = KBAgent(adapter=adapter, engine=engine)
    result = await agent.answer("Who approves >1M?")

    assert "CEO approval is required" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0].doc_id == "doc1"
    assert result["retrieved"] == [("doc1", 0)]
    assert adapter.chat.await_count == 2
    engine.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_records_feedback_when_db_provided(tmp_path):
    from kb.feedback.db import UsageFeedbackDB
    fb_db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/agent_fb.db")
    await fb_db.init()

    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    adapter.chat = AsyncMock(side_effect=[
        AdapterResponse(content="",
                        tool_calls=[ToolCall(id="c1", name="search_docs",
                                             arguments={"query": "x"})]),
        AdapterResponse(content="Answer is X [doc1:0].", tool_calls=[]),
    ])
    engine = MagicMock()
    engine.search = AsyncMock(return_value=(
        SearchResponse(
            results=[_make_search_result("doc1", 0, "X content")], total=1
        ), False
    ))

    agent = KBAgent(adapter=adapter, engine=engine, feedback_db=fb_db)
    result = await agent.answer("question?")

    rows = await fb_db.list_recent()
    assert len(rows) == 1
    assert result["query_id"] == rows[0]["query_id"]
    assert rows[0]["retrieved"] == [["doc1", 0]]
    assert rows[0]["actually_used"] == [["doc1", 0]]


@pytest.mark.asyncio
async def test_agent_max_iterations_returns_fallback():
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    # Always tool-calls, never final answer
    adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="search_docs", arguments={"query": "y"})],
    ))
    engine = MagicMock()
    engine.search = AsyncMock(return_value=(
        SearchResponse(results=[], total=0), False,
    ))

    agent = KBAgent(adapter=adapter, engine=engine, max_iterations=2)
    result = await agent.answer("won't terminate")

    assert "iteration limit" in result["answer"].lower()
    assert adapter.chat.await_count == 2


@pytest.mark.asyncio
async def test_agent_supports_vision_includes_image_url_and_description():
    """Vision-capable LLM should receive BOTH image_url and vision_description
    (image gives detail, description gives semantic summary)."""
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=True)
    captured_messages = []

    async def chat_capture(messages, tools=None):
        captured_messages.append([(m.role, m.content) for m in messages])
        if len(captured_messages) == 1:
            return AdapterResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_docs",
                                     arguments={"query": "flow"})],
            )
        return AdapterResponse(content="answer [doc1:0]", tool_calls=[])

    adapter.chat = chat_capture

    engine = MagicMock()
    sr = _make_search_result("doc1", 0, "text")
    sr.image_url = "https://blob/x.png"
    sr.vision_description = "flowchart description"
    engine.search = AsyncMock(return_value=(SearchResponse(results=[sr], total=1), False))

    agent = KBAgent(adapter=adapter, engine=engine)
    await agent.answer("show the flow")

    # Tool message (last in second-call's history) must include both fields
    tool_msg_content = captured_messages[1][-1][1]
    assert "https://blob/x.png" in tool_msg_content
    assert "flowchart description" in tool_msg_content


@pytest.mark.asyncio
async def test_agent_propagates_cold_start_message_to_llm():
    """When KB is empty, SearchResponse.message + suggested_upload_topics
    must reach the LLM so it can tell the user 'KB is empty, please upload docs'."""
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    captured_messages = []

    async def chat_capture(messages, tools=None):
        captured_messages.append([(m.role, m.content) for m in messages])
        if len(captured_messages) == 1:
            return AdapterResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_docs",
                                     arguments={"query": "anything"})],
            )
        return AdapterResponse(content="The KB is empty.", tool_calls=[])

    adapter.chat = chat_capture

    engine = MagicMock()
    empty_resp = SearchResponse(
        results=[], total=0,
        message="知识库中暂无相关内容，建议上传相关文档后重试。",
        suggested_upload_topics=["产品审批流程", "操作手册"],
    )
    engine.search = AsyncMock(return_value=(empty_resp, False))

    agent = KBAgent(adapter=adapter, engine=engine)
    await agent.answer("anything?")

    tool_msg_content = captured_messages[1][-1][1]
    assert "知识库中暂无相关内容" in tool_msg_content
    assert "产品审批流程" in tool_msg_content


@pytest.mark.asyncio
async def test_agent_propagates_low_confidence_to_llm():
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    captured_messages = []

    async def chat_capture(messages, tools=None):
        captured_messages.append([(m.role, m.content) for m in messages])
        if len(captured_messages) == 1:
            return AdapterResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_docs",
                                     arguments={"query": "anything"})],
            )
        return AdapterResponse(content="I am not sure.", tool_calls=[])

    adapter.chat = chat_capture
    engine = MagicMock()
    resp = SearchResponse(results=[], total=0)
    resp.low_confidence = True
    resp.confidence_reason = "top_score_below_threshold"
    engine.search = AsyncMock(return_value=(resp, False))

    agent = KBAgent(adapter=adapter, engine=engine)
    await agent.answer("anything?")

    tool_msg_content = captured_messages[1][-1][1]
    assert "low_confidence" in tool_msg_content
    assert "top_score_below_threshold" in tool_msg_content


@pytest.mark.asyncio
async def test_agent_marks_low_confidence_final_answer_uncertain():
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    adapter.chat = AsyncMock(side_effect=[
        AdapterResponse(
            content="",
            tool_calls=[ToolCall(id="c1", name="search_docs",
                                 arguments={"query": "policy"})],
        ),
        AdapterResponse(content="The policy allows it.", tool_calls=[]),
    ])
    engine = MagicMock()
    resp = SearchResponse(results=[], total=0)
    resp.low_confidence = True
    resp.confidence_reason = "top_score_below_threshold"
    engine.search = AsyncMock(return_value=(resp, False))

    agent = KBAgent(adapter=adapter, engine=engine)
    result = await agent.answer("is it allowed?")

    assert result["confidence"]["status"] == "low"
    assert "top_score_below_threshold" in result["confidence"]["reasons"]
    assert "知识库证据不足" in result["answer"]


import os


def _image_part_count(msg: AdapterMessage) -> int:
    return sum(1 for part in msg.content if part.get("type") == "image_url")


@pytest.mark.asyncio
async def test_agent_appends_multimodal_user_message_for_vision_adapter(tmp_path):
    """vision-capable adapter 应该在 search 后收到追加的多模态 user message，
    其中含有 image_url multipart 内容。"""
    # 创建一张假的 PNG 文件
    img_path = tmp_path / "page.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png")
    image_url = f"local://{img_path}"

    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=True)

    captured: list = []

    async def chat_capture(messages, tools=None):
        captured.append(list(messages))
        if len(captured) == 1:
            return AdapterResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_docs",
                                     arguments={"query": "flow"})],
            )
        return AdapterResponse(content="The flow has 5 steps [doc1:0].", tool_calls=[])

    adapter.chat = chat_capture

    engine = MagicMock()
    sr = _make_search_result("doc1", 0, "approval text")
    sr.image_url = str(img_path)
    sr.vision_description = "flow chart"
    engine.search = AsyncMock(return_value=(SearchResponse(results=[sr], total=1), False))

    agent = KBAgent(adapter=adapter, engine=engine)
    await agent.answer("show the flow")

    # 第二次 chat 调用时，messages 中应该有一个带多部分 content 的 user 消息
    second_call_messages = captured[1]
    multipart_msgs = [m for m in second_call_messages
                      if m.role == "user" and isinstance(m.content, list)]
    assert len(multipart_msgs) >= 1, "应有至少一条多模态 user message"

    # 该消息应当包含 image_url part
    found_image = any(
        any(p.get("type") == "image_url" for p in m.content)
        for m in multipart_msgs
    )
    assert found_image, "多模态 user message 应含 image_url part"


@pytest.mark.asyncio
async def test_agent_no_multimodal_for_non_vision_adapter():
    """non-vision adapter 不应该追加多模态 user message。"""
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)

    captured = []
    async def chat_capture(messages, tools=None):
        captured.append(list(messages))
        if len(captured) == 1:
            return AdapterResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_docs", arguments={"query": "x"})],
            )
        return AdapterResponse(content="answer [doc1:0]", tool_calls=[])
    adapter.chat = chat_capture

    engine = MagicMock()
    sr = _make_search_result("doc1", 0, "text")
    sr.image_url = "https://example.com/page.png"   # 非 local:// URL，agent 不会去 fetch
    engine.search = AsyncMock(return_value=(SearchResponse(results=[sr], total=1), False))

    agent = KBAgent(adapter=adapter, engine=engine)
    await agent.answer("q")

    second_call_messages = captured[1]
    multipart_msgs = [m for m in second_call_messages
                      if m.role == "user" and isinstance(m.content, list)]
    assert len(multipart_msgs) == 0, "non-vision adapter 不应接收多模态消息"


def test_agent_multimodal_budget_limits_image_count(tmp_path, test_settings):
    test_settings.agent_max_images_per_turn = 2
    paths = []
    for i in range(3):
        path = tmp_path / f"page{i}.png"
        path.write_bytes(b"image-bytes")
        paths.append(path)
    results = []
    for i, path in enumerate(paths):
        sr = _make_search_result(f"doc{i}", i, "text")
        sr.image_url = str(path)
        results.append(sr)
    adapter = MagicMock()
    engine = MagicMock()

    agent = KBAgent(adapter=adapter, engine=engine, settings=test_settings)
    msg = agent._build_multimodal_message(results)

    assert msg is not None
    assert _image_part_count(msg) == 2


def test_agent_multimodal_budget_limits_total_image_bytes(tmp_path, test_settings):
    test_settings.agent_max_images_per_turn = 3
    test_settings.agent_max_image_bytes_per_turn = 10
    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    small.write_bytes(b"12345")
    large.write_bytes(b"1234567890abc")
    first = _make_search_result("doc1", 0, "text")
    first.image_url = str(small)
    second = _make_search_result("doc2", 0, "text")
    second.image_url = str(large)
    adapter = MagicMock()
    engine = MagicMock()

    agent = KBAgent(adapter=adapter, engine=engine, settings=test_settings)
    msg = agent._build_multimodal_message([first, second])

    assert msg is not None
    assert _image_part_count(msg) == 1


def test_agent_multimodal_budget_skips_duplicate_pages(tmp_path, test_settings):
    img = tmp_path / "page.png"
    img.write_bytes(b"image-bytes")
    first = _make_search_result("doc1", 0, "first")
    first.image_url = str(img)
    duplicate = _make_search_result("doc1", 0, "duplicate")
    duplicate.image_url = str(img)
    adapter = MagicMock()
    engine = MagicMock()

    agent = KBAgent(adapter=adapter, engine=engine, settings=test_settings)
    msg = agent._build_multimodal_message([first, duplicate])
    second_msg = agent._build_multimodal_message([first])

    assert msg is not None
    assert _image_part_count(msg) == 1
    assert second_msg is None


def test_agent_multimodal_budget_emits_trace(tmp_path, test_settings):
    from unittest.mock import patch

    test_settings.agent_max_images_per_turn = 1
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.png"
    img1.write_bytes(b"one")
    img2.write_bytes(b"two")
    first = _make_search_result("doc1", 0, "first")
    first.image_url = str(img1)
    second = _make_search_result("doc2", 0, "second")
    second.image_url = str(img2)
    agent = KBAgent(adapter=MagicMock(), engine=MagicMock(), settings=test_settings)

    with patch("kb.agent.agent.emit") as emit_spy:
        agent._build_multimodal_message([first, second])

    budget_events = [
        call for call in emit_spy.call_args_list
        if call.args and call.args[0] == "agent.multimodal_budget"
    ]
    assert budget_events
    assert budget_events[-1].kwargs["skipped_by_count"] == 1


@pytest.mark.asyncio
async def test_agent_handles_missing_query_arg_gracefully():
    """LLM 调 search_docs 不传 query → 返回 error 给 LLM 重试,不崩."""
    adapter = MagicMock()
    adapter.supports_vision = MagicMock(return_value=False)
    captured = []

    async def chat_capture(messages, tools=None):
        captured.append(list(messages))
        if len(captured) == 1:
            return AdapterResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="search_docs", arguments={})],
            )
        return AdapterResponse(content="I cannot help [doc1:0]", tool_calls=[])

    adapter.chat = chat_capture
    engine = MagicMock()
    engine.search = AsyncMock()

    agent = KBAgent(adapter=adapter, engine=engine)
    result = await agent.answer("question")

    # search 不应被调用(因为 args 里没有 query)
    engine.search.assert_not_called()
    # 第二次 chat 时 LLM 看到了 error 消息
    second_call_messages = captured[1]
    tool_msg = next(m for m in second_call_messages if m.role == "tool")
    assert "missing" in tool_msg.content.lower() or "query" in tool_msg.content.lower()
    # 不应崩溃,应返回最终回答
    assert "answer" in result
