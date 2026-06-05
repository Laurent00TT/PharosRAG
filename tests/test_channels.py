# tests/test_channels.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from kb.ingestion.channels.text import TextChannel
from kb.ingestion.pipeline import IngestionPipeline
from kb.models import TextChunk


@pytest.mark.asyncio
async def test_ingestion_pipeline_closes_owned_clients():
    """Beyond the 3 channels, aclose() must also dispose the MetadataDB
    engine and close the Qdrant HTTP client. Worker-per-job creation
    pattern means missing any of these accumulates handles in proportion
    to ingest throughput."""
    pipeline = object.__new__(IngestionPipeline)
    pipeline._parser = MagicMock()
    pipeline._parser.aclose = AsyncMock()
    pipeline._vision_ch = MagicMock()
    pipeline._vision_ch.aclose = AsyncMock()
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.aclose = AsyncMock()
    pipeline._meta_db = MagicMock()
    pipeline._meta_db.aclose = AsyncMock()
    pipeline._qdrant = MagicMock()
    pipeline._qdrant.client = MagicMock()
    pipeline._qdrant.client.close = MagicMock()

    await pipeline.aclose()

    pipeline._parser.aclose.assert_awaited_once()
    pipeline._vision_ch.aclose.assert_awaited_once()
    pipeline._desc_ch.aclose.assert_awaited_once()
    pipeline._meta_db.aclose.assert_awaited_once()
    pipeline._qdrant.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_ingestion_pipeline_aclose_tolerates_partial_failure():
    """One stuck resource must not block teardown of the rest. Each close
    is independently wrapped so the meta_db engine still gets disposed
    even if the parser HTTP client raised."""
    pipeline = object.__new__(IngestionPipeline)
    pipeline._parser = MagicMock()
    pipeline._parser.aclose = AsyncMock(side_effect=RuntimeError("parser stuck"))
    pipeline._vision_ch = MagicMock()
    pipeline._vision_ch.aclose = AsyncMock()
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.aclose = AsyncMock()
    pipeline._meta_db = MagicMock()
    pipeline._meta_db.aclose = AsyncMock()
    pipeline._qdrant = MagicMock()
    pipeline._qdrant.client = MagicMock()
    pipeline._qdrant.client.close = MagicMock()

    await pipeline.aclose()  # must NOT raise

    pipeline._meta_db.aclose.assert_awaited_once()
    pipeline._qdrant.client.close.assert_called_once()


@pytest.fixture(scope="module")
def text_channel():
    return TextChannel()  # 首次运行会下载 bge-m3（约 2GB），之后缓存

def test_text_channel_returns_dense(text_channel):
    chunk = TextChunk(
        doc_id="t", doc_name="t.pdf", page_num=0, chunk_index=0,
        text="审批须在3个工作日内完成。",
        heading_path=[], has_visual_on_page=False,
        page_type="text", parent_chunk_id="t_p0", is_parent=False
    )
    dense, sparse_idx, sparse_val = text_channel.embed(chunk)
    assert len(dense) == 1024
    assert len(sparse_idx) > 0
    assert len(sparse_idx) == len(sparse_val)

def test_text_channel_dense_is_normalized(text_channel):
    import math
    chunk = TextChunk(
        doc_id="t", doc_name="t.pdf", page_num=0, chunk_index=0,
        text="test", heading_path=[], has_visual_on_page=False,
        page_type="text", parent_chunk_id="t_p0", is_parent=False
    )
    dense, _, _ = text_channel.embed(chunk)
    norm = math.sqrt(sum(x**2 for x in dense))
    assert abs(norm - 1.0) < 0.01


from unittest.mock import AsyncMock, MagicMock, patch
from kb.adapters.base import AdapterResponse
from kb.ingestion.channels.description import DescriptionChannel
from kb.models import DescPage


def test_description_should_process_skips_pure_text(test_settings):
    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=True)
    ch = DescriptionChannel(adapter=fake_adapter)
    assert ch.should_process(has_image_blocks=False, text_char_ratio=0.95) is False


def test_description_should_process_includes_visual(test_settings):
    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=True)
    ch = DescriptionChannel(adapter=fake_adapter)
    assert ch.should_process(has_image_blocks=True, text_char_ratio=0.3) is True


@pytest.mark.asyncio
async def test_description_uses_multimodal_message(test_settings, sample_image):
    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=True)
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content='{"page_type":"flowchart","figure_index":"图1-1","figure_caption":"审批",'
                '"position_hint":"中部","description":"5步审批流程","key_entities":["CEO"]}',
        tool_calls=[],
    ))
    ch = DescriptionChannel(adapter=fake_adapter)

    result = await ch.describe(
        doc_id="d", doc_name="d.pdf", page_num=0,
        image_bytes=sample_image,
        heading_path=["Doc"],
        image_url="local://d/page_0.png",
    )

    # 验证返回结构
    assert isinstance(result, DescPage)
    assert result.page_type == "flowchart"
    assert result.description == "5步审批流程"
    assert "CEO" in result.key_entities

    # 验证传给 adapter 的消息是多模态的
    call_args = fake_adapter.chat.await_args
    messages = call_args.args[0]   # 第一个位置参数
    user_msg = next(m for m in messages if m.role == "user")
    assert isinstance(user_msg.content, list)
    types = [p.get("type") for p in user_msg.content]
    assert "text" in types
    assert "image_url" in types


@pytest.mark.asyncio
async def test_description_falls_back_when_adapter_returns_invalid_json(test_settings, sample_image):
    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=True)
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="this is not json",
        tool_calls=[],
    ))
    ch = DescriptionChannel(adapter=fake_adapter)

    result = await ch.describe(
        doc_id="d", doc_name="d.pdf", page_num=0,
        image_bytes=sample_image,
        heading_path=[],
        image_url="local://d/page_0.png",
    )

    # 失败时仍应返回 DescPage（page_type='mixed'，description 空）
    assert isinstance(result, DescPage)
    assert result.page_type == "mixed"
    assert result.description == ""


# ── _strip_json_fence unit tests ─────────────────────────────────────────


def test_strip_json_fence_with_json_prefix():
    """Qwen3.6 most common: ```json ... ``` wrapper."""
    from kb.ingestion.channels.description import _strip_json_fence
    raw = '```json\n{"page_type": "text"}\n```'
    assert _strip_json_fence(raw) == '{"page_type": "text"}'


def test_strip_json_fence_without_lang_suffix():
    """Sometimes Qwen drops the json suffix."""
    from kb.ingestion.channels.description import _strip_json_fence
    raw = '```\n{"a": 1}\n```'
    assert _strip_json_fence(raw) == '{"a": 1}'


def test_strip_json_fence_no_fence_passthrough():
    """Plain JSON returns unchanged."""
    from kb.ingestion.channels.description import _strip_json_fence
    raw = '{"a": 1}'
    assert _strip_json_fence(raw) == raw


def test_strip_json_fence_with_leading_trailing_whitespace():
    """Whitespace around fence."""
    from kb.ingestion.channels.description import _strip_json_fence
    raw = '\n  ```json\n{"a": 1}\n```  \n'
    assert _strip_json_fence(raw) == '{"a": 1}'


def test_strip_json_fence_handles_no_closing_fence():
    """If LLM truncated, just drop the opening fence."""
    from kb.ingestion.channels.description import _strip_json_fence
    raw = '```json\n{"a": 1}'
    assert _strip_json_fence(raw) == '{"a": 1}'


def test_strip_json_fence_empty_string():
    """Empty input."""
    from kb.ingestion.channels.description import _strip_json_fence
    assert _strip_json_fence("") == ""


def test_strip_json_fence_only_opening_fence():
    """Edge: no content after opening fence."""
    from kb.ingestion.channels.description import _strip_json_fence
    # The function strips opening fence; what remains is empty or fence-only
    result = _strip_json_fence('```json\n')
    # We don't crash; remaining content may be empty or partial
    assert result == "" or result == "```json"


@pytest.mark.asyncio
async def test_description_strips_json_fence_before_parsing(test_settings, sample_image):
    """End-to-end: LLM wraps JSON in ```json fence; describe() succeeds."""
    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=True)
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content='```json\n{"page_type": "flowchart", "description": "approval flow"}\n```',
        tool_calls=[],
    ))
    ch = DescriptionChannel(adapter=fake_adapter)
    result = await ch.describe(
        doc_id="d", doc_name="d.pdf", page_num=0,
        image_bytes=sample_image,
        heading_path=[],
        image_url="local://d/page_0.png",
    )
    assert result.page_type == "flowchart"
    assert result.description == "approval flow"


@pytest.mark.asyncio
async def test_pipeline_processes_pdf(sample_pdf, test_settings):
    from kb.models import ParsedPage
    pipeline = IngestionPipeline(settings=test_settings)

    # Mock parser，避免依赖 mineru_server
    fake_pages = [
        ParsedPage(
            doc_id="dummy",   # 会被 ingest 内部覆盖
            doc_name="sample.pdf",
            page_num=0,
            structured_text="page 0 content",
            heading_path=[],
            page_image=b"\x89PNG\r\n\x1a\n",
            image_blocks=[],
            tables=[],
            metadata={},
        ),
    ]
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=fake_pages)
    # mock description adapter（避免连真实 LM Studio）
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)  # 跳过 description 流程

    result = await pipeline.ingest(sample_pdf)
    assert result["doc_id"] is not None
    assert result["pages_processed"] == 1
    assert result["chunks_stored"] >= 1


@pytest.mark.asyncio
async def test_pipeline_raises_jobcancelled_before_parse_when_cancel_check_true(
    sample_pdf, test_settings,
):
    """Cancel checkpoint #1: when cancel_check is asserted before parser
    runs, ingest() must raise JobCancelled and NOT call parser/qdrant."""
    from kb.jobs.cancel import JobCancelled

    cancel_count = {"calls": 0}

    async def always_cancel():
        cancel_count["calls"] += 1
        return True

    pipeline = IngestionPipeline(settings=test_settings, cancel_check=always_cancel)
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock()
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    with pytest.raises(JobCancelled, match="before parsing"):
        await pipeline.ingest(sample_pdf)

    # Parser must NOT have been called — early bail
    pipeline._parser.parse_async.assert_not_awaited()
    assert cancel_count["calls"] >= 1


@pytest.mark.asyncio
async def test_pipeline_cleans_partial_writes_when_cancelled_mid_page_loop(
    sample_pdf, test_settings,
):
    """Cancel checkpoint #2: when cancel triggers inside the page loop,
    partial qdrant points and image files must be cleaned up before the
    exception propagates — otherwise we leave orphan data for the
    cancelled doc_id."""
    from kb.jobs.cancel import JobCancelled
    from kb.models import ParsedPage

    # Trip cancel only on the SECOND check — so checkpoint #1 (before
    # parse) returns False and parse runs, then checkpoint #2 (top of
    # page 0) returns True. This exercises the partial-cleanup path.
    poll_count = {"n": 0}

    async def cancel_after_first():
        poll_count["n"] += 1
        return poll_count["n"] >= 2

    pipeline = IngestionPipeline(
        settings=test_settings, cancel_check=cancel_after_first,
    )
    fake_pages = [
        ParsedPage(
            doc_id="dummy", doc_name="sample.pdf", page_num=0,
            structured_text="page 0", heading_path=[],
            page_image=b"\x89PNG\r\n\x1a\n",
            image_blocks=[], tables=[], metadata={},
        ),
    ]
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=fake_pages)
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    # Spy on the cleanup methods so we can assert they ran.
    pipeline._qdrant = MagicMock()
    pipeline._qdrant.ensure_collections = AsyncMock()
    pipeline._qdrant.delete_by_doc_id = AsyncMock()
    pipeline._image_store = MagicMock()
    pipeline._image_store.delete_doc_images = AsyncMock(return_value={"deleted": True})

    with pytest.raises(JobCancelled, match="at page 0"):
        await pipeline.ingest(sample_pdf)

    pipeline._qdrant.delete_by_doc_id.assert_awaited_once()
    pipeline._image_store.delete_doc_images.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipeline_swallows_cancel_check_errors_and_continues(
    sample_pdf, test_settings,
):
    """Fail-open guarantee: if cancel_check itself raises (e.g. transient
    SQLite hiccup polling job state), ingestion must continue rather
    than fail. The lease-expiry path is the existing fallback for real
    cancels that get missed this way."""
    from kb.models import ParsedPage

    async def flaky_cancel():
        raise RuntimeError("meta_db transient hiccup")

    pipeline = IngestionPipeline(settings=test_settings, cancel_check=flaky_cancel)
    fake_pages = [
        ParsedPage(
            doc_id="dummy", doc_name="sample.pdf", page_num=0,
            structured_text="page 0", heading_path=[],
            page_image=b"\x89PNG\r\n\x1a\n",
            image_blocks=[], tables=[], metadata={},
        ),
    ]
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=fake_pages)
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    # Must NOT raise — pipeline treats hook failures as "not cancelled"
    result = await pipeline.ingest(sample_pdf)
    assert result["pages_processed"] == 1


@pytest.mark.asyncio
async def test_pipeline_skips_existing_unchanged_doc(sample_pdf, test_settings):
    from kb.models import ParsedPage
    pipeline = IngestionPipeline(settings=test_settings)
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=[])
    # mock description adapter（避免连真实 LM Studio）
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    r1 = await pipeline.ingest(sample_pdf)
    r2 = await pipeline.ingest(sample_pdf)
    assert r2["skipped"] is True


@pytest.mark.asyncio
async def test_pipeline_self_heals_missing_nav_on_unchanged_skip(
    sample_pdf, test_settings, tmp_path,
):
    """Anti-regression for the main-path nav gap. If an unchanged-hash
    skip finds an active doc in metadata but NO nav entries, it must
    synthesize pages from qdrant and rebuild nav transparently. Without
    this, users had to remember to run scripts/backfill_nav_index.py."""
    import hashlib
    from kb.models import ParsedPage
    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    # The pipeline derives doc_id from path bytes. ParsedPage.doc_id MUST
    # match — otherwise nav entries land under a different key than
    # metadata, and the test asserts pass/fail on the wrong row.
    expected_doc_id = hashlib.md5(sample_pdf.read_bytes()).hexdigest()

    nav_store = NavIndexStore(str(tmp_path / "nav.db"))
    await nav_store.init()
    nav_builder = NavIndexBuilder()
    pipeline = IngestionPipeline(
        settings=test_settings, nav_store=nav_store, nav_builder=nav_builder,
    )

    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=[
        ParsedPage(
            doc_id=expected_doc_id, doc_name="sample.pdf", page_num=0,
            structured_text="hello world", heading_path=["Intro"],
            page_image=b"\x89PNG\r\n\x1a\n",
            image_blocks=[], tables=[], metadata={},
        ),
    ])
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    # First ingest builds nav normally
    r1 = await pipeline.ingest(sample_pdf)
    assert r1["skipped"] is False
    assert r1["doc_id"] == expected_doc_id
    entries_before = await nav_store.list_entries(doc_id=expected_doc_id)
    assert len(entries_before) >= 1

    # Simulate "nav db wiped between sessions" — drop the entries
    await nav_store.delete_entries_for_doc(expected_doc_id)
    assert await nav_store.list_entries(doc_id=expected_doc_id) == []

    # Second ingest hits the unchanged-hash path; must self-heal
    r2 = await pipeline.ingest(sample_pdf)
    assert r2["skipped"] is True
    assert r2["nav_repaired"] >= 1, "self-heal must rebuild nav entries"
    entries_after = await nav_store.list_entries(doc_id=expected_doc_id)
    assert len(entries_after) >= 1, "nav must be repopulated after self-heal"
    await nav_store.close()


@pytest.mark.asyncio
async def test_pipeline_skip_does_not_re_heal_when_nav_already_present(
    sample_pdf, test_settings, tmp_path,
):
    """Self-heal must be idempotent — when nav is already populated for
    the doc, the skip path returns nav_repaired=0 (no wasted work)."""
    import hashlib
    from kb.models import ParsedPage
    from kb.nav.builder import NavIndexBuilder
    from kb.nav.store import NavIndexStore

    expected_doc_id = hashlib.md5(sample_pdf.read_bytes()).hexdigest()
    nav_store = NavIndexStore(str(tmp_path / "nav.db"))
    await nav_store.init()
    nav_builder = NavIndexBuilder()
    pipeline = IngestionPipeline(
        settings=test_settings, nav_store=nav_store, nav_builder=nav_builder,
    )

    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=[
        ParsedPage(
            doc_id=expected_doc_id, doc_name="sample.pdf", page_num=0,
            structured_text="hello world", heading_path=[],
            page_image=b"\x89PNG\r\n\x1a\n",
            image_blocks=[], tables=[], metadata={},
        ),
    ])
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    await pipeline.ingest(sample_pdf)  # initial — builds nav
    r2 = await pipeline.ingest(sample_pdf)  # skip, nav already healthy
    assert r2["skipped"] is True
    assert r2["nav_repaired"] == 0
    await nav_store.close()


@pytest.mark.asyncio
async def test_pipeline_sets_vision_page_type_from_tables(sample_pdf, test_settings):
    from kb.models import ParsedPage
    pipeline = IngestionPipeline(settings=test_settings)
    fake_pages = [
        ParsedPage(
            doc_id="dummy",
            doc_name="sample.pdf",
            page_num=0,
            structured_text="table page",
            heading_path=[],
            page_image=b"\x89PNG\r\n\x1a\n",
            image_blocks=[],
            tables=[{"rows": [["a"]]}],
            metadata={},
        ),
    ]
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=fake_pages)
    pipeline._qdrant = MagicMock()
    pipeline._qdrant.ensure_collections = AsyncMock()
    pipeline._qdrant.upsert_vision_page = AsyncMock()
    pipeline._meta_db = MagicMock()
    pipeline._meta_db.init = AsyncMock()
    pipeline._meta_db.get_document = AsyncMock(return_value=None)
    pipeline._meta_db.list_active_doc_ids = AsyncMock(return_value=[])
    pipeline._meta_db.upsert_document = AsyncMock()
    pipeline._image_store = MagicMock()
    pipeline._image_store.save = AsyncMock(return_value="local://dummy/page_0.png")
    pipeline._chunker = MagicMock()
    pipeline._chunker.chunk_page.return_value = []
    pipeline._vision_ch = MagicMock()
    pipeline._vision_ch.embed_async = AsyncMock(return_value=[[0.1] * 128])
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    await pipeline.ingest(sample_pdf)

    vision_page = pipeline._qdrant.upsert_vision_page.await_args.args[0]
    assert vision_page.page_type == "table"


@pytest.mark.asyncio
async def test_ingest_keeps_old_version_active_when_new_parse_fails(sample_pdf, test_settings):
    pipeline = IngestionPipeline(settings=test_settings)
    await pipeline._meta_db.init()
    await pipeline._meta_db.upsert_document(
        "old",
        sample_pdf.name,
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
    )
    pipeline._qdrant = MagicMock()
    pipeline._qdrant.ensure_collections = AsyncMock()
    pipeline._qdrant.delete_by_doc_id = AsyncMock()
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(side_effect=RuntimeError("parse down"))

    with pytest.raises(RuntimeError, match="parse down"):
        await pipeline.ingest(sample_pdf)

    old = await pipeline._meta_db.get_document("old")
    assert old["status"] == "active"
    pipeline._qdrant.delete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_keeps_new_version_active_when_old_cleanup_fails(sample_pdf, test_settings):
    from kb.models import ParsedPage

    pipeline = IngestionPipeline(settings=test_settings)
    await pipeline._meta_db.init()
    await pipeline._meta_db.upsert_document(
        "old",
        sample_pdf.name,
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
    )
    fake_pages = [
        ParsedPage(
            doc_id="dummy",
            doc_name=sample_pdf.name,
            page_num=0,
            structured_text="replacement",
            heading_path=[],
            page_image=b"\x89PNG\r\n\x1a\n",
            image_blocks=[],
            tables=[],
            metadata={},
        ),
    ]
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=fake_pages)
    pipeline._qdrant = MagicMock()
    pipeline._qdrant.ensure_collections = AsyncMock()
    pipeline._qdrant.delete_by_doc_id = AsyncMock(side_effect=RuntimeError("cleanup down"))
    pipeline._image_store = MagicMock()
    pipeline._image_store.save = AsyncMock(return_value="local://dummy/page_0.png")
    pipeline._chunker = MagicMock()
    pipeline._chunker.chunk_page.return_value = []
    pipeline._vision_ch = MagicMock()
    pipeline._vision_ch.embed_async = AsyncMock(return_value=None)
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    result = await pipeline.ingest(sample_pdf)

    old = await pipeline._meta_db.get_document("old")
    new = await pipeline._meta_db.get_document(result["doc_id"])
    assert old["status"] == "deprecated"
    assert new["status"] == "active"


def test_embed_query_returns_dense_and_sparse(text_channel):
    dense, s_idx, s_val = text_channel.embed_query("审批流程是什么")
    assert len(dense) == 1024
    assert len(s_idx) > 0
    assert len(s_idx) == len(s_val)
    assert all(isinstance(v, float) for v in dense)

def test_embed_query_dense_is_normalized(text_channel):
    import math
    dense, _, _ = text_channel.embed_query("approval process")
    norm = math.sqrt(sum(x**2 for x in dense))
    assert abs(norm - 1.0) < 0.01


from kb.ingestion.channels.vision import VisionChannel


class _FakeEmbedClient:
    """Stand-in for httpx.AsyncClient used by VisionChannel.embed_async.

    The v1.1 VisionChannel opens a fresh client inside embed_async (no
    persistent ._client to patch), so we patch the module-level
    httpx.AsyncClient symbol with this fake instead.
    """
    captured: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        _FakeEmbedClient.captured = {"url": url, "json": json}

        class _Resp:
            def raise_for_status(self_inner):
                return None

            def json(self_inner):
                return {"data": [{"embedding": [0.1] * 4096}]}

        return _Resp()


@pytest.mark.asyncio
async def test_vision_channel_embeds_via_embedding_server(test_settings):
    """VisionChannel.embed_async POSTs a chat-messages image payload to the
    multimodal embedding server /v1/embeddings and returns ONE dense vector
    (replaces the retired ColQwen multi-vector path)."""
    ch = VisionChannel(settings=test_settings)

    with patch("kb.ingestion.channels.vision.httpx.AsyncClient", _FakeEmbedClient):
        result = await ch.embed_async(b"fake_png_bytes")

    # single-vector (4096-dim) return, not a list-of-lists multi-vector
    assert isinstance(result, list)
    assert len(result) == 4096
    assert all(isinstance(v, float) for v in result)
    # routed to the embedding server's /v1/embeddings with a messages payload
    assert _FakeEmbedClient.captured["url"].endswith("/v1/embeddings")
    assert "messages" in _FakeEmbedClient.captured["json"]


@pytest.mark.asyncio
async def test_vision_channel_returns_none_on_server_failure(test_settings):
    """Embedding server unreachable → returns None (degrade, never raises)."""
    ch = VisionChannel(settings=test_settings)

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            raise Exception("connection refused")

    with patch("kb.ingestion.channels.vision.httpx.AsyncClient", _FailingClient):
        result = await ch.embed_async(b"fake_png_bytes")

    assert result is None


# ── Phase 6.5: description failure trace carries prompt_version ────────


@pytest.mark.asyncio
async def test_description_parse_failure_event_includes_prompt_version(
    test_settings, sample_image,
):
    """Phase 6.5 execution list: 'description parse failure event 包含
    prompt_version.' When the adapter returns content that fails JSON
    parsing, the emit_error trace MUST include prompt_version so the
    operator can correlate the failure with a specific prompt revision.
    """
    from kb.ingestion.channels.description import (
        DESCRIPTION_PROMPT_VERSION,
        DescriptionChannel,
    )

    fake_adapter = MagicMock()
    fake_adapter.supports_vision = MagicMock(return_value=True)
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="<not json at all>",
        tool_calls=[],
    ))
    ch = DescriptionChannel(adapter=fake_adapter)

    # Patch emit_error in the channel module so we can inspect the call
    # args (preferable to patching the underlying trace module because
    # it pins the assertion to the exact call site under test).
    import kb.ingestion.channels.description as desc_mod
    with patch.object(desc_mod, "emit_error") as mocked:
        await ch.describe(
            doc_id="d", doc_name="d.pdf", page_num=0,
            image_bytes=sample_image,
            heading_path=[],
            image_url="local://d/page_0.png",
        )

    # The error path runs through emit_error with prompt_version kwarg.
    mocked.assert_called_once()
    _, kwargs = mocked.call_args
    assert kwargs.get("prompt_version") == DESCRIPTION_PROMPT_VERSION
    # Also confirm failure-specific fields are present (sanity).
    assert kwargs.get("json_parse_ok") is False
    assert kwargs.get("doc_id") == "d"
