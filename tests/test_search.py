# tests/test_search.py
from kb.search.rrf import reciprocal_rank_fusion
from kb.search.constants import CHANNEL_TEXT_DENSE, CHANNEL_TEXT_SPARSE, ALL_CHANNELS


def test_constants_are_strings():
    assert isinstance(CHANNEL_TEXT_DENSE, str)
    assert len(ALL_CHANNELS) == 4


def test_rrf_single_list():
    hits = [("doc1", 0, 0.9), ("doc2", 1, 0.8), ("doc3", 2, 0.7)]
    result = reciprocal_rank_fusion([hits], channel_names=["dense"])
    ids = [(r[0], r[1]) for r in result]
    assert ids == [("doc1", 0), ("doc2", 1), ("doc3", 2)]


def test_rrf_two_lists_merge():
    list_a = [("doc1", 0, 0.9), ("doc2", 1, 0.8)]
    list_b = [("doc2", 1, 0.95), ("doc3", 2, 0.7)]
    result = reciprocal_rank_fusion([list_a, list_b], channel_names=["dense", "sparse"])
    ids_order = [(r[0], r[1]) for r in result]
    # doc2 appears in both lists -> should rank highest
    assert ids_order[0] == ("doc2", 1)


def test_rrf_channels_tracked():
    list_a = [("doc1", 0, 0.9)]
    list_b = [("doc1", 0, 0.8), ("doc2", 1, 0.7)]
    result = reciprocal_rank_fusion([list_a, list_b], channel_names=["dense", "sparse"])
    doc1_entry = next(r for r in result if r[0] == "doc1")
    assert "dense" in doc1_entry[3]
    assert "sparse" in doc1_entry[3]


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion([[], []], channel_names=["a", "b"])
    assert result == []


def test_rrf_k_parameter():
    hits = [("doc1", 0, 0.9)]
    r60 = reciprocal_rank_fusion([hits], channel_names=["dense"], k=60)
    r10 = reciprocal_rank_fusion([hits], channel_names=["dense"], k=10)
    # rank 0 with k=60: 1/(60+1)~0.0164, k=10: 1/(10+1)~0.0909
    assert r10[0][2] > r60[0][2]


from unittest.mock import MagicMock, patch
from kb.config import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path),
        image_storage_path=str(tmp_path),
    )


import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from kb.search.retriever import Retriever
from kb.search.vision_query import VisionQueryEmbedder
from kb.search.constants import CHANNEL_TEXT_DENSE, CHANNEL_TEXT_SPARSE, CHANNEL_DESC


@pytest.mark.asyncio
async def test_vision_query_embeds_via_embedding_server(tmp_path):
    """VisionQueryEmbedder POSTs an instruction-prefixed text query to the
    multimodal embedding server /v1/embeddings and returns ONE dense vector
    (replaces the retired ColQwen multi-vec query path)."""
    settings = Settings(
        qdrant_path=str(tmp_path),
        image_storage_path=str(tmp_path),
    )
    embedder = VisionQueryEmbedder(settings)
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={
        "data": [{"embedding": [0.2] * 4096}],
    })
    with patch.object(embedder._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp) as mock_post:
        result = await embedder.embed_query_async("approval flow")
    assert isinstance(result, list)
    assert len(result) == 4096
    # routed to the embedding server /v1/embeddings, not the retired colqwen server
    assert mock_post.call_args.args[0].endswith("/v1/embeddings")


@pytest.mark.asyncio
async def test_vision_query_returns_none_on_server_failure(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path),
        image_storage_path=str(tmp_path),
        multimodal_embedding_server_url="http://127.0.0.1:1",   # invalid port
    )
    embedder = VisionQueryEmbedder(settings)
    result = await embedder.embed_query_async("test")
    assert result is None


@pytest.fixture
def mock_retriever_deps(tmp_path):
    settings = _settings(tmp_path)
    qdrant = MagicMock()
    qdrant.search_text_dense.return_value = [
        {"doc_id": "d1", "page_num": 0, "score": 0.9}
    ]
    qdrant.search_text_sparse.return_value = [
        {"doc_id": "d1", "page_num": 0, "score": 0.8}
    ]
    qdrant.search_desc.return_value = []
    qdrant.search_vision.return_value = []
    meta_db = MagicMock()
    meta_db.list_valid_doc_ids = AsyncMock(return_value=["d1"])
    return settings, qdrant, meta_db


@pytest.mark.asyncio
async def test_retriever_returns_four_channels(mock_retriever_deps):
    settings, qdrant, meta_db = mock_retriever_deps
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.1] * 1024, [1], [1.0])
    r = Retriever(settings=settings, qdrant=qdrant, meta_db=meta_db,
                  text_channel=mock_text)
    dense_h, sparse_h, vision_h, desc_h = await r.retrieve("test query", 5, {})
    assert dense_h == [("d1", 0, 0.9)]
    assert sparse_h == [("d1", 0, 0.8)]
    assert vision_h == []   # DISABLED tier
    assert desc_h == []


@pytest.mark.asyncio
async def test_retriever_returns_empty_when_no_valid_docs(mock_retriever_deps):
    settings, qdrant, meta_db = mock_retriever_deps
    meta_db.list_valid_doc_ids = AsyncMock(return_value=[])
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    r = Retriever(settings=settings, qdrant=qdrant, meta_db=meta_db,
                  text_channel=mock_text)
    r._vision_q.embed_query_async = AsyncMock(return_value=None)

    dense_h, sparse_h, vision_h, desc_h = await r.retrieve("q", 5, {})

    assert dense_h == []
    assert sparse_h == []
    assert vision_h == []
    assert desc_h == []
    qdrant.search_text_dense.assert_not_called()
    qdrant.search_text_sparse.assert_not_called()
    qdrant.search_desc.assert_not_called()
    qdrant.search_vision.assert_not_called()


@pytest.mark.asyncio
async def test_retriever_closes_vision_query_embedder(mock_retriever_deps):
    settings, qdrant, meta_db = mock_retriever_deps
    r = Retriever(settings=settings, qdrant=qdrant, meta_db=meta_db,
                  text_channel=MagicMock())
    r._vision_q.aclose = AsyncMock()

    await r.aclose()

    r._vision_q.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_retriever_passes_doc_name_filter(mock_retriever_deps):
    settings, qdrant, meta_db = mock_retriever_deps
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    r = Retriever(settings=settings, qdrant=qdrant, meta_db=meta_db,
                  text_channel=mock_text)
    await r.retrieve("q", 5, {"doc_name": "manual.pdf"})
    call_kwargs = qdrant.search_text_dense.call_args
    # extra_must is the 4th positional arg or keyword
    extra_must = call_kwargs[0][3] if len(call_kwargs[0]) > 3 else call_kwargs[1].get("extra_must")
    assert extra_must is not None and len(extra_must) > 0


@pytest.mark.asyncio
async def test_retriever_passes_page_type_filter_to_vision(mock_retriever_deps):
    settings, qdrant, meta_db = mock_retriever_deps
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    r = Retriever(settings=settings, qdrant=qdrant, meta_db=meta_db,
                  text_channel=mock_text)
    r._vision_q.embed_query_async = AsyncMock(return_value=[[0.1] * 128])

    await r.retrieve("q", 5, {"page_type": "table"})

    call_kwargs = qdrant.search_vision.call_args
    extra_must = call_kwargs[0][3] if len(call_kwargs[0]) > 3 else call_kwargs[1].get("extra_must")
    keys = [getattr(condition, "key", None) for condition in extra_must]
    assert "page_type" in keys


@pytest.mark.asyncio
async def test_retriever_skips_lifecycle_filter_when_include_deprecated(mock_retriever_deps):
    settings, qdrant, meta_db = mock_retriever_deps
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    r = Retriever(settings=settings, qdrant=qdrant, meta_db=meta_db,
                  text_channel=mock_text)
    await r.retrieve("q", 5, {}, include_deprecated=True)
    # With include_deprecated=True, list_valid_doc_ids must NOT be called
    meta_db.list_valid_doc_ids.assert_not_awaited()
    # doc_ids kwarg passed to search_text_dense must be None
    call_kwargs = qdrant.search_text_dense.call_args
    doc_ids = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs[1].get("doc_ids")
    assert doc_ids is None


from kb.search.expander import enrich_results
from kb.models import SearchResult
from datetime import date as _date


@pytest.mark.asyncio
async def test_enrich_results_builds_search_result():
    qdrant = MagicMock()
    qdrant.get_parent_chunk = MagicMock(return_value={
        "doc_id": "d1", "doc_name": "manual.pdf", "page_num": 0,
        "text": "parent text content",
        "heading_path": ["Doc", "Ch1"],
    })
    qdrant.get_desc_record = MagicMock(return_value={
        "page_type": "flowchart",
        "description": "审批流程图描述",
        "figure_index": "图1-1",
        "figure_caption": "审批",
        "image_url": "local://d1/page_0.png",
    })
    qdrant.get_vision_record = MagicMock(return_value=None)
    meta_db = MagicMock()
    meta_db.get_document = AsyncMock(return_value={
        "doc_id": "d1", "doc_name": "manual.pdf",
        "version": "v1", "status": "active",
        "effective_date": _date(2026, 1, 1), "expiry_date": None,
    })
    fused = [("d1", 0, 0.85, ["dense", "vision"])]
    results = await enrich_results(fused, qdrant, meta_db)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SearchResult)
    assert r.text_chunk == "parent text content"
    assert r.vision_description == "审批流程图描述"
    assert r.doc_version == "v1"
    assert "dense" in r.match_channels


@pytest.mark.asyncio
async def test_enrich_results_skips_when_all_missing():
    qdrant = MagicMock()
    qdrant.get_parent_chunk = MagicMock(return_value=None)
    qdrant.get_desc_record = MagicMock(return_value=None)
    qdrant.get_vision_record = MagicMock(return_value=None)
    meta_db = MagicMock()
    meta_db.get_document = AsyncMock(return_value=None)
    results = await enrich_results([("x", 0, 0.5, ["dense"])], qdrant, meta_db)
    assert results == []


@pytest.mark.asyncio
async def test_enrich_results_skips_when_no_content(mocker):
    """Hit with metadata but no text and no description is skipped."""
    qdrant = MagicMock()
    qdrant.get_parent_chunk = MagicMock(return_value=None)  # no text
    qdrant.get_desc_record = MagicMock(return_value=None)   # no desc
    qdrant.get_vision_record = MagicMock(return_value=None)
    meta_db = MagicMock()
    meta_db.get_document = AsyncMock(return_value={
        "doc_id": "d2", "doc_name": "x.pdf", "version": None,
        "status": "active", "effective_date": None, "expiry_date": None,
    })
    results = await enrich_results([("d2", 0, 0.5, ["dense"])], qdrant, meta_db)
    # text_rec=None, desc_rec=None → no content → skip
    assert results == []


@pytest.mark.asyncio
async def test_enrich_falls_back_to_vision_record_for_image_url():
    """When desc_rec has no image_url (pure-text page DescriptionChannel skipped),
    expander should fall back to vision_rec.image_url so the multimodal path still works."""
    qdrant = MagicMock()
    qdrant.get_parent_chunk = MagicMock(return_value={
        "doc_id": "d1", "doc_name": "x.pdf", "page_num": 0,
        "text": "text", "heading_path": [],
    })
    qdrant.get_desc_record = MagicMock(return_value=None)  # 纯文字页,无 desc
    qdrant.get_vision_record = MagicMock(return_value={"image_url": "local://x/p0.png"})
    meta_db = MagicMock()
    meta_db.get_document = AsyncMock(return_value=None)

    fused = [("d1", 0, 0.5, ["dense"])]
    results = await enrich_results(fused, qdrant, meta_db)
    assert len(results) == 1
    assert results[0].image_url == "local://x/p0.png"


from kb.search.reranker import Reranker


def _make_result(text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        doc_id="d", doc_name="d.pdf", page_num=0,
        heading_path=[], text_chunk=text,
        vision_description=None, figure_index=None, figure_caption=None,
        image_url="", page_type="text", doc_version=None, effective_date=None,
        score=score, match_channels=["dense"],
    )


def test_filter_results_by_doc_filter_page_type():
    from kb.search.contracts import filter_results_by_contract
    a = _make_result("a")
    a.doc_id = "table-doc"
    a.page_type = "table"
    b = _make_result("b")
    b.doc_id = "text-doc"
    b.page_type = "text"

    out = filter_results_by_contract([a, b], {"page_type": "table"})

    assert [r.doc_id for r in out] == ["table-doc"]
    assert out[0].page_type == "table"


def test_filter_results_by_doc_name():
    from kb.search.contracts import filter_results_by_contract
    a = _make_result("a")
    a.doc_name = "manual.pdf"
    b = _make_result("b")
    b.doc_name = "other.pdf"

    out = filter_results_by_contract([a, b], {"doc_name": "manual.pdf"})

    assert len(out) == 1
    assert out[0].doc_name == "manual.pdf"


@pytest.mark.asyncio
async def test_search_engine_closes_owned_clients(tmp_path):
    from kb.search.engine import SearchEngine

    qdrant = MagicMock()
    meta_db = MagicMock()
    engine = SearchEngine(_settings(tmp_path), qdrant=qdrant, meta_db=meta_db)
    engine._retriever.aclose = AsyncMock()
    engine._llm_adapter = MagicMock()
    engine._llm_adapter.aclose = AsyncMock()

    await engine.aclose()

    engine._retriever.aclose.assert_awaited_once()
    engine._llm_adapter.aclose.assert_awaited_once()


def test_diversify_limits_same_doc_run():
    from kb.search.contracts import diversify_results
    results = []
    for i in range(5):
        result = _make_result(f"text {i}", score=1.0 - i * 0.1)
        result.doc_id = "same"
        results.append(result)
    other = _make_result("other", score=0.5)
    other.doc_id = "other"
    results.append(other)

    out = diversify_results(results, max_per_doc=2)

    assert sum(1 for r in out if r.doc_id == "same") == 2
    assert any(r.doc_id == "other" for r in out)


def test_reranker_returns_top_k(mocker):
    mocker.patch("kb.search.reranker.FlagReranker.__init__", return_value=None)
    mocker.patch(
        "kb.search.reranker.FlagReranker.compute_score",
        return_value=[0.3, 0.9, 0.1],
    )
    from FlagEmbedding import FlagReranker as _FR
    reranker = Reranker.__new__(Reranker)
    reranker._model = _FR.__new__(_FR)

    results = [_make_result("a"), _make_result("b"), _make_result("c")]
    top2 = reranker.rerank("query", results, top_k=2)
    assert len(top2) == 2
    assert top2[0].score == 0.9
    assert top2[0].text_chunk == "b"


def test_reranker_empty_input(mocker):
    mocker.patch("kb.search.reranker.FlagReranker.__init__", return_value=None)
    reranker = Reranker.__new__(Reranker)
    assert reranker.rerank("q", [], top_k=5) == []


def test_reranker_filters_empty_content(mocker):
    mocker.patch("kb.search.reranker.FlagReranker.__init__", return_value=None)
    mock_score = mocker.patch(
        "kb.search.reranker.FlagReranker.compute_score",
        return_value=[0.7],
    )
    reranker = Reranker.__new__(Reranker)
    from FlagEmbedding import FlagReranker as _FR
    reranker._model = _FR.__new__(_FR)

    results = [
        _make_result("real text", score=0.5),
        SearchResult(  # both content fields empty/None
            doc_id="empty", doc_name="x.pdf", page_num=0,
            heading_path=[], text_chunk="",
            vision_description=None, figure_index=None, figure_caption=None,
            image_url="", page_type="text", doc_version=None, effective_date=None,
            score=0.5, match_channels=["dense"],
        ),
    ]
    out = reranker.rerank("q", results, top_k=5)
    assert len(out) == 1
    assert out[0].text_chunk == "real text"
    # Should only score the non-empty candidate, so compute_score called with 1 pair
    assert len(mock_score.call_args[0][0]) == 1


def test_reranker_preserves_vision_only_hits_at_rrf_score(mocker):
    """Security review P0 #2 + P1 #3: cross-encoder gets ONLY text_chunk
    pairs (vision_description is LLM-generated hint, not citable text;
    feeding it to a text reranker is double-counting a noisy signal).
    Vision-only hits (no text_chunk but has image_url or description)
    must survive at their incoming RRF score — dropping them silently
    killed multimodal recall for chart/screenshot pages."""
    mocker.patch("kb.search.reranker.FlagReranker.__init__", return_value=None)
    mock_score = mocker.patch(
        "kb.search.reranker.FlagReranker.compute_score",
        return_value=[0.9],
    )
    reranker = Reranker.__new__(Reranker)
    from FlagEmbedding import FlagReranker as _FR
    reranker._model = _FR.__new__(_FR)

    # Text hit (will be rescored) + vision-only hit (will pass through).
    text_hit = _make_result("text content", score=0.5)
    vision_hit = SearchResult(
        doc_id="img", doc_name="chart.pdf", page_num=3,
        heading_path=[], text_chunk="",
        vision_description="A bar chart showing revenue by quarter",
        figure_index="Fig 2.1", figure_caption=None,
        image_url="local://img/page_3.png", page_type="flowchart",
        doc_version=None, effective_date=None,
        score=0.6, match_channels=["vision"],
    )
    out = reranker.rerank("q", [text_hit, vision_hit], top_k=5)
    # Both preserved; cross-encoder called with EXACTLY one pair (text only)
    assert len(out) == 2
    assert len(mock_score.call_args[0][0]) == 1
    assert mock_score.call_args[0][0][0][1] == "text content"
    # vision hit retains its RRF score; text hit gets the cross-encoder 0.9
    by_doc = {r.doc_id: r for r in out}
    assert by_doc["d"].score == pytest.approx(0.9)
    assert by_doc["img"].score == pytest.approx(0.6)
    # Sorted by score desc
    assert out[0].doc_id == "d"  # 0.9 > 0.6


def test_reranker_drops_completely_empty_hits(mocker):
    """Companion to the vision-only test: a SearchResult with no text,
    no description, no image, no caption is degenerate noise — drop it."""
    mocker.patch("kb.search.reranker.FlagReranker.__init__", return_value=None)
    mock_score = mocker.patch(
        "kb.search.reranker.FlagReranker.compute_score",
        return_value=[0.7],
    )
    reranker = Reranker.__new__(Reranker)
    from FlagEmbedding import FlagReranker as _FR
    reranker._model = _FR.__new__(_FR)

    text_hit = _make_result("real", score=0.5)
    empty_hit = SearchResult(
        doc_id="empty", doc_name="x.pdf", page_num=0,
        heading_path=[], text_chunk="",
        vision_description="", figure_index=None, figure_caption="",
        image_url="", page_type="text", doc_version=None, effective_date=None,
        score=0.99, match_channels=["dense"],   # high score but no content
    )
    out = reranker.rerank("q", [text_hit, empty_hit], top_k=5)
    assert len(out) == 1
    assert out[0].doc_id == "d"


def test_reranker_invalid_top_k(mocker):
    mocker.patch("kb.search.reranker.FlagReranker.__init__", return_value=None)
    reranker = Reranker.__new__(Reranker)
    assert reranker.rerank("q", [_make_result("a")], top_k=0) == []
    assert reranker.rerank("q", [_make_result("a")], top_k=-1) == []


def test_reranker_does_not_mutate_input(mocker):
    mocker.patch("kb.search.reranker.FlagReranker.__init__", return_value=None)
    mocker.patch(
        "kb.search.reranker.FlagReranker.compute_score",
        return_value=[0.99],
    )
    reranker = Reranker.__new__(Reranker)
    from FlagEmbedding import FlagReranker as _FR
    reranker._model = _FR.__new__(_FR)

    original = _make_result("text", score=0.123)
    results = [original]
    out = reranker.rerank("q", results, top_k=5)
    # Input must be untouched
    assert original.score == 0.123
    # Output should have the new score
    assert out[0].score == 0.99
    # Output should be a different object
    assert out[0] is not original


from kb.search.hyde import hyde_expand
from kb.adapters.base import AdapterResponse


@pytest.mark.asyncio
async def test_hyde_uses_adapter():
    fake_adapter = MagicMock()
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="审批需在3个工作日内完成，超1M需CEO签字。",
        tool_calls=[],
    ))
    result = await hyde_expand("审批流程是什么", adapter=fake_adapter)
    assert "审批" in result


@pytest.mark.asyncio
async def test_hyde_falls_back_on_adapter_error():
    fake_adapter = MagicMock()
    fake_adapter.chat = AsyncMock(side_effect=Exception("LLM down"))
    result = await hyde_expand("original query", adapter=fake_adapter)
    assert result == "original query"


from kb.search.cache import cache_key, get_cached, set_cached, _cache
from kb.models import SearchResponse


def test_cache_key_is_deterministic():
    k1 = cache_key("query", {"doc_name": "a.pdf"}, False)
    k2 = cache_key("query", {"doc_name": "a.pdf"}, False)
    assert k1 == k2


def test_cache_key_differs_on_different_include_deprecated():
    k1 = cache_key("q", {}, include_deprecated=False)
    k2 = cache_key("q", {}, include_deprecated=True)
    assert k1 != k2


def test_cache_key_differs_on_different_top_k():
    k1 = cache_key("q", {}, False, top_k=3, hyde=False)
    k2 = cache_key("q", {}, False, top_k=10, hyde=False)
    assert k1 != k2


def test_cache_key_differs_on_different_hyde():
    k1 = cache_key("q", {}, False, top_k=5, hyde=False)
    k2 = cache_key("q", {}, False, top_k=5, hyde=True)
    assert k1 != k2


def test_cache_miss_returns_none():
    assert get_cached("nonexistent_key_xyz") is None


def test_cache_set_and_get():
    _cache.clear()
    resp = SearchResponse(results=[], total=0)
    set_cached("mykey", resp)
    assert get_cached("mykey") is resp


def test_search_response_has_confidence_fields():
    resp = SearchResponse(results=[], total=0)
    assert resp.low_confidence is False
    assert resp.confidence_reason == ""


def test_search_engine_query_rewrite_adapter_uses_settings_and_disables_thinking(tmp_path):
    from kb.adapters.base import DISABLE_THINKING_EXTRA_BODY

    settings = _settings(tmp_path)
    settings.query_rewrite_url = "http://rewrite.local/v1"
    settings.query_rewrite_api_key = "rewrite-key"
    settings.query_rewrite_model = "rewrite-model"
    settings.query_rewrite_supports_vision = False
    engine = SearchEngine(settings=settings, qdrant=MagicMock(), meta_db=MagicMock())

    with _patch("kb.adapters.openai_compat.OpenAICompatAdapter") as adapter_cls:
        engine._get_llm_adapter()

    kwargs = adapter_cls.call_args.kwargs
    assert kwargs["base_url"] == "http://rewrite.local/v1"
    assert kwargs["api_key"] == "rewrite-key"
    assert kwargs["model"] == "rewrite-model"
    assert kwargs["supports_vision"] is False
    assert kwargs["extra_body"] == DISABLE_THINKING_EXTRA_BODY


from kb.search.engine import SearchEngine
from unittest.mock import patch as _patch


@pytest.mark.asyncio
async def test_search_engine_returns_search_response(tmp_path):
    settings = _settings(tmp_path)
    qdrant = MagicMock()
    meta_db = MagicMock()
    meta_db.list_valid_doc_ids = AsyncMock(return_value=[])
    meta_db.list_active_doc_ids = AsyncMock(return_value=[])
    meta_db.get_document = AsyncMock(return_value=None)
    qdrant.search_text_dense.return_value = []
    qdrant.search_text_sparse.return_value = []
    qdrant.search_desc.return_value = []

    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    engine._retriever._text = mock_text

    resp, cache_hit = await engine.search("test query", top_k=3)
    assert isinstance(resp, SearchResponse)
    assert cache_hit is False


@pytest.mark.asyncio
async def test_search_engine_cache_hit_on_second_call(tmp_path):
    from kb.search.cache import _cache, set_cached
    _cache.clear()
    settings = _settings(tmp_path)
    qdrant = MagicMock()
    meta_db = MagicMock()
    meta_db.list_valid_doc_ids = AsyncMock(return_value=[])
    meta_db.list_active_doc_ids = AsyncMock(return_value=[])
    meta_db.get_document = AsyncMock(return_value=None)
    qdrant.search_text_dense.return_value = []
    qdrant.search_text_sparse.return_value = []
    qdrant.search_desc.return_value = []

    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    engine._retriever._text = mock_text

    # First call: miss
    with _patch("kb.search.engine.get_cached", wraps=get_cached) as spy_get, \
         _patch("kb.search.engine.set_cached", wraps=set_cached) as spy_set:
        _, hit1 = await engine.search("cached query", top_k=3)
        _, hit2 = await engine.search("cached query", top_k=3)

    assert hit1 is False
    assert hit2 is True
    assert spy_set.call_count == 1   # only set once
    assert spy_get.call_count == 2   # checked both times


@pytest.mark.asyncio
async def test_search_engine_empty_kb_includes_suggestions(tmp_path):
    from kb.search.cache import _cache
    _cache.clear()
    settings = _settings(tmp_path)
    qdrant = MagicMock()
    meta_db = MagicMock()
    # Empty KB: no active doc IDs at all
    meta_db.list_valid_doc_ids = AsyncMock(return_value=[])
    meta_db.list_active_doc_ids = AsyncMock(return_value=[])
    meta_db.get_document = AsyncMock(return_value=None)
    qdrant.search_text_dense.return_value = []
    qdrant.search_text_sparse.return_value = []
    qdrant.search_desc.return_value = []

    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    engine._retriever._text = mock_text

    resp, _ = await engine.search("产品审批流程", top_k=3)
    assert resp.total == 0
    assert resp.message != ""
    assert isinstance(resp.suggested_upload_topics, list)
    assert len(resp.suggested_upload_topics) > 0


@pytest.mark.asyncio
async def test_search_engine_non_empty_kb_no_suggestions(tmp_path):
    from kb.search.cache import _cache
    _cache.clear()
    settings = _settings(tmp_path)
    qdrant = MagicMock()
    meta_db = MagicMock()
    # Non-empty KB with no matching results
    meta_db.list_valid_doc_ids = AsyncMock(return_value=["doc1"])
    meta_db.list_active_doc_ids = AsyncMock(return_value=["doc1"])
    meta_db.get_document = AsyncMock(return_value=None)
    qdrant.search_text_dense.return_value = []
    qdrant.search_text_sparse.return_value = []
    qdrant.search_desc.return_value = []

    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    engine._retriever._text = mock_text

    resp, _ = await engine.search("anything", top_k=3)
    assert resp.total == 0
    # KB has docs, just no matches: should NOT trigger cold-start UX
    assert resp.message == ""
    assert resp.suggested_upload_topics == []


@pytest.mark.asyncio
async def test_search_engine_applies_final_filter_after_rerank(tmp_path):
    from kb.search.cache import _cache
    _cache.clear()
    settings = _settings(tmp_path)
    settings.hyde_enabled = False
    settings.multi_query_enabled = False
    qdrant = MagicMock()
    meta_db = MagicMock()
    meta_db.list_active_doc_ids = AsyncMock(return_value=["doc1"])

    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    engine._retriever.retrieve = AsyncMock(return_value=(
        [("table-doc", 0, 0.9), ("text-doc", 0, 0.8)],
        [],
        [],
        [],
    ))
    text_result = _make_result("text")
    text_result.doc_id = "text-doc"
    text_result.page_type = "text"
    table_result = _make_result("table")
    table_result.doc_id = "table-doc"
    table_result.page_type = "table"
    engine._reranker = MagicMock()
    engine._reranker.rerank.return_value = [text_result, table_result]

    with _patch("kb.search.engine.enrich_results", new=AsyncMock(return_value=[
        text_result,
        table_result,
    ])):
        resp, cache_hit = await engine.search(
            "q",
            top_k=5,
            doc_filter={"page_type": "table"},
        )

    assert cache_hit is False
    assert [r.doc_id for r in resp.results] == ["table-doc"]
    assert resp.total == 1


@pytest.mark.asyncio
async def test_search_engine_loads_reranker_inside_worker_thread(tmp_path, monkeypatch):
    from kb.search.cache import _cache
    _cache.clear()
    settings = _settings(tmp_path)
    settings.hyde_enabled = False
    settings.multi_query_enabled = False
    qdrant = MagicMock()
    meta_db = MagicMock()
    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    engine._retriever.retrieve = AsyncMock(return_value=([("d", 0, 0.9)], [], [], []))
    result = _make_result("text")
    call_order = []

    class FakeReranker:
        def rerank(self, query, candidates, top_k):
            return [result]

    def get_reranker():
        call_order.append("get_reranker")
        return FakeReranker()

    async def fake_to_thread(func, *args):
        assert call_order == []
        return func(*args)

    engine._get_reranker = get_reranker
    monkeypatch.setattr("kb.search.engine.asyncio.to_thread", fake_to_thread)

    with _patch("kb.search.engine.enrich_results", new=AsyncMock(return_value=[result])):
        await engine.search("q", top_k=5)

    assert call_order == ["get_reranker"]


@pytest.mark.asyncio
async def test_search_engine_marks_low_confidence_when_top_score_is_low(tmp_path):
    from kb.search.cache import _cache
    _cache.clear()
    settings = _settings(tmp_path)
    settings.hyde_enabled = False
    settings.multi_query_enabled = False
    settings.search_low_confidence_threshold = 0.5
    qdrant = MagicMock()
    meta_db = MagicMock()
    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    engine._retriever.retrieve = AsyncMock(return_value=([("d", 0, 0.2)], [], [], []))
    low = _make_result("low", score=0.1)
    engine._reranker = MagicMock()
    engine._reranker.rerank.return_value = [low]

    with _patch("kb.search.engine.enrich_results", new=AsyncMock(return_value=[low])):
        resp, _ = await engine.search("q", top_k=5)

    assert resp.low_confidence is True
    assert resp.confidence_reason == "top_score_below_threshold"


@pytest.mark.asyncio
async def test_search_engine_can_disable_query_planner(tmp_path):
    from kb.search.cache import _cache
    _cache.clear()
    settings = _settings(tmp_path)
    settings.search_planner_enabled = False
    settings.hyde_enabled = False
    settings.multi_query_enabled = False
    qdrant = MagicMock()
    meta_db = MagicMock()
    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    engine._retriever.retrieve = AsyncMock(return_value=([("d", 0, 0.2)], [], [], []))
    result = _make_result("text")
    engine._reranker = MagicMock()
    engine._reranker.rerank.return_value = [result]

    with _patch("kb.search.engine.enrich_results", new=AsyncMock(return_value=[result])), \
         _patch("kb.search.engine.emit") as emit_spy:
        await engine.search("这个页面图表长什么样", top_k=5)

    assert not any(call.args and call.args[0] == "search.plan"
                   for call in emit_spy.call_args_list)


@pytest.mark.asyncio
async def test_cold_start_fields_co_populated(tmp_path):
    """Invariant: message and suggested_upload_topics are populated together.
    No state should produce one without the other."""
    from kb.search.cache import _cache
    _cache.clear()
    settings = _settings(tmp_path)
    qdrant = MagicMock()
    meta_db = MagicMock()
    meta_db.list_valid_doc_ids = AsyncMock(return_value=[])
    meta_db.list_active_doc_ids = AsyncMock(return_value=[])
    meta_db.get_document = AsyncMock(return_value=None)
    qdrant.search_text_dense.return_value = []
    qdrant.search_text_sparse.return_value = []
    qdrant.search_desc.return_value = []

    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)
    mock_text = MagicMock()
    mock_text.embed_query.return_value = ([0.0] * 1024, [], [])
    engine._retriever._text = mock_text

    resp, _ = await engine.search("q", top_k=3)
    has_msg = bool(resp.message)
    has_topics = bool(resp.suggested_upload_topics)
    assert has_msg == has_topics, "message and suggested_upload_topics must be co-populated"


from kb.search.multi_query import expand_queries


@pytest.mark.asyncio
async def test_multi_query_returns_n_variants():
    fake_adapter = MagicMock()
    fake_adapter.chat = AsyncMock(return_value=AdapterResponse(
        content="""
1. 审批流程的具体步骤
2. CEO 何时需要签字
3. 大额款项的审批权限
""",
        tool_calls=[],
    ))
    variants = await expand_queries("审批", n=3, adapter=fake_adapter)
    # 期望至少 3 个变体（含原 query）
    assert len(variants) >= 3
    # 至少应有原始 query
    assert any("审批" in v for v in variants)


@pytest.mark.asyncio
async def test_multi_query_falls_back_to_single():
    fake_adapter = MagicMock()
    fake_adapter.chat = AsyncMock(side_effect=Exception("LLM down"))
    variants = await expand_queries("test", n=3, adapter=fake_adapter)
    assert variants == ["test"]   # 失败时返回单元素列表


@pytest.mark.asyncio
async def test_multi_query_n_eq_1_returns_original():
    fake_adapter = MagicMock()
    variants = await expand_queries("test", n=1, adapter=fake_adapter)
    assert variants == ["test"]
    fake_adapter.chat.assert_not_called()  # n=1 不应调 LLM
