# tests/test_integration.py
"""端到端集成测试 — 主进程 + mineru_server + 多模态 embedding/rerank/sparse server + LLM。

由于真实服务不一定可达，所有服务调用都被 mock。
此测试验证编排逻辑：parser → chunker → channels → qdrant → search → reranker → 返回结果。
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from kb.config import Settings
from kb.ingestion.pipeline import IngestionPipeline
from kb.models import ParsedPage, SearchResponse
from kb.search.cache import _cache
from kb.search.engine import SearchEngine
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore


@pytest.fixture
def integration_settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        image_storage_path=str(tmp_path / "images"),
        # 服务 URL 用 mock，不会真调
        mineru_server_url="http://test-mineru:8001",
        description_url="http://test-llm:1234/v1",
        agent_url="http://test-llm:1234/v1",
        hyde_enabled=False,           # 简化测试，不调 LLM
        multi_query_enabled=False,
        # 默认 parser_backend=mineru_api，但测试里 parse 已被 mock，
        # 显式选 local 路径避免 factory 因缺 token 而失败。
        parser_backend="mineru_local",
        ingest_use_gpu_mineru=False,  # 不 spawn 真 mineru worker（parse 已被 mock）
    )


@pytest.fixture
def fake_pdf(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 fake content for testing")
    return p


@pytest.mark.asyncio
async def test_ingest_then_search_e2e(integration_settings, fake_pdf):
    _cache.clear()

    # ── Ingest ──────────────────────────────────────────────────────────────
    pipeline = IngestionPipeline(settings=integration_settings)

    # Mock parser（避免依赖真实 mineru_server）
    fake_pages = [
        ParsedPage(
            doc_id="dummy", doc_name="fake.pdf", page_num=0,
            structured_text="approval process: CEO must sign for amounts >1M",
            heading_path=["Doc", "Section 1"],
            page_image=b"fake_png_data",
            image_blocks=[], tables=[],
            metadata={},
        ),
    ]
    pipeline._parser = MagicMock()
    pipeline._parser.parse_async = AsyncMock(return_value=fake_pages)

    # Mock vision channel (HTTP) — 返回 None 跳过 vision channel
    pipeline._vision_ch.embed_async = AsyncMock(return_value=None)

    # Mock description channel — should_process 返回 False 跳过
    pipeline._desc_ch = MagicMock()
    pipeline._desc_ch.should_process = MagicMock(return_value=False)

    # Mock document classifier 让它通过
    with patch("kb.ingestion.pipeline.classify_document",
               return_value=MagicMock(is_processable=True, page_count=1)):
        ingest_result = await pipeline.ingest(fake_pdf)

    assert ingest_result["pages_processed"] == 1
    assert ingest_result["chunks_stored"] >= 1
    doc_id = ingest_result["doc_id"]

    # ── Search ──────────────────────────────────────────────────────────────
    qdrant = pipeline._qdrant
    meta_db = pipeline._meta_db
    engine = SearchEngine(settings=integration_settings, qdrant=qdrant, meta_db=meta_db)

    with patch.object(engine, "_get_reranker") as mock_rr:
        mock_rr.return_value.rerank = lambda q, r, k: r[:k]
        resp, cache_hit = await engine.search("CEO approval", top_k=3)

    assert isinstance(resp, SearchResponse)
    assert cache_hit is False
    # 至少应该召回包含 doc_id 的结果
    if resp.total > 0:
        assert any(r.doc_id == doc_id for r in resp.results)


@pytest.mark.asyncio
async def test_search_empty_kb_cold_start(integration_settings):
    _cache.clear()
    qdrant = QdrantStore(settings=integration_settings)
    await qdrant.ensure_collections()
    meta_db = MetadataDB(
        db_url=f"sqlite+aiosqlite:///{integration_settings.qdrant_path}/kb_metadata.db"
    )
    await meta_db.init()

    engine = SearchEngine(settings=integration_settings, qdrant=qdrant, meta_db=meta_db)
    with patch.object(engine, "_get_reranker") as mock_rr:
        mock_rr.return_value.rerank = lambda q, r, k: r[:k]
        resp, _ = await engine.search("anything", top_k=3)

    assert resp.total == 0
    # 空 KB 应触发 cold-start UX
    assert resp.message != ""
    assert len(resp.suggested_upload_topics) > 0
