# tests/test_parser.py
"""MinerULocalParser tests.

不需要真实 MinerU 服务，所有 HTTP 调用均 mock。`_make_parser()` constructs
the parser with ingest_use_gpu_mineru=False so no real subprocess is spawned.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from kb.config import Settings
from kb.parser.mineru_local import MinerULocalParser
from kb.models import ParsedPage


def _make_parser(tmp_path=None, server_url: str = "http://test:8001") -> MinerULocalParser:
    s = Settings(
        qdrant_path=str(tmp_path / "q") if tmp_path else "./_qdrant_unused",
        image_storage_path=str(tmp_path / "i") if tmp_path else "./_img_unused",
        mineru_server_url=server_url,
        ingest_use_gpu_mineru=False,  # do not spawn a real GPU worker
    )
    return MinerULocalParser(settings=s)


# 模拟一个最小的 MinerU middle_json 输出
SAMPLE_MIDDLE_JSON = {
    "pdf_info": [
        {
            "page_idx": 0,
            "page_size": [612, 792],
            "preproc_blocks": [
                {
                    "type": "title",
                    "level": 1,
                    "lines": [{"spans": [{"type": "text", "content": "Chapter 1"}]}],
                },
                {
                    "type": "text",
                    "lines": [{"spans": [{"type": "text", "content": "Approval needed within 3 days."}]}],
                },
            ],
            "discarded_blocks": [],
        },
        {
            "page_idx": 1,
            "page_size": [612, 792],
            "preproc_blocks": [
                {
                    "type": "table",
                    "blocks": [
                        {
                            "type": "table_body",
                            "lines": [{"spans": [{"type": "table", "html": "<table><tr><td>A</td></tr></table>"}]}],
                        },
                    ],
                },
            ],
            "discarded_blocks": [],
        },
    ],
}


@pytest.fixture
def fake_pdf(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 fake content for hash purposes")
    return p


@pytest.mark.asyncio
async def test_parser_returns_parsed_pages(fake_pdf):
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=SAMPLE_MIDDLE_JSON)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert len(pages) == 2
    assert all(isinstance(p, ParsedPage) for p in pages)


@pytest.mark.asyncio
async def test_parser_extracts_text(fake_pdf):
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=SAMPLE_MIDDLE_JSON)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert "Chapter 1" in pages[0].structured_text
    assert "Approval needed" in pages[0].structured_text


@pytest.mark.asyncio
async def test_parser_extracts_table_html(fake_pdf):
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=SAMPLE_MIDDLE_JSON)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert len(pages[1].tables) >= 1
    assert "<table>" in pages[1].tables[0]["html"]


@pytest.mark.asyncio
async def test_parser_uses_md5_doc_id(fake_pdf):
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=SAMPLE_MIDDLE_JSON)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    import hashlib
    expected_doc_id = hashlib.md5(fake_pdf.read_bytes()).hexdigest()
    assert all(p.doc_id == expected_doc_id for p in pages)


@pytest.mark.asyncio
async def test_parser_extracts_heading_path(fake_pdf):
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=SAMPLE_MIDDLE_JSON)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    # Page 0 has "Chapter 1" title (level 1) — heading_path 应该包含它
    assert pages[0].heading_path == ["Chapter 1"]


@pytest.mark.asyncio
async def test_parser_heading_path_accumulates_across_pages(fake_pdf):
    """页 0 设置 heading 后，页 1 的 heading_path 应该继承（heading 栈跨页累积）。"""
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=SAMPLE_MIDDLE_JSON)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    # 页 1 没有标题块，但应继承页 0 的 "Chapter 1"
    assert "Chapter 1" in pages[1].heading_path


@pytest.mark.asyncio
async def test_parser_decodes_page_image_b64(fake_pdf):
    """如果 middle_json 包含 page_image_b64，parser 应解码到 page_image bytes。"""
    import base64
    fake_image_bytes = b"\x89PNG\r\n\x1a\nfake_png_data"
    middle_json_with_images = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [612, 792],
                "preproc_blocks": [],
                "page_image_b64": base64.b64encode(fake_image_bytes).decode("ascii"),
            },
        ],
    }
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=middle_json_with_images)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert pages[0].page_image == fake_image_bytes


@pytest.mark.asyncio
async def test_parser_handles_missing_page_image(fake_pdf):
    """如果 middle_json 没有 page_image_b64，page_image 应为空 bytes。"""
    parser = _make_parser()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=SAMPLE_MIDDLE_JSON)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert pages[0].page_image == b""


# ── BlockType schema 对齐测试 (probe_mineru.py 验证 2026-05-09) ───────────────


@pytest.mark.asyncio
async def test_parser_prefers_para_blocks_over_preproc(fake_pdf):
    """page_info 同时有 para_blocks 和 preproc_blocks 时,优先 para_blocks (post-processed)。"""
    parser = _make_parser()
    middle_json = {
        "pdf_info": [{
            "page_idx": 0,
            "para_blocks": [
                {"type": "text", "lines": [{"spans": [{"type": "text", "content": "from para_blocks"}]}]},
            ],
            "preproc_blocks": [
                {"type": "text", "lines": [{"spans": [{"type": "text", "content": "from preproc_blocks"}]}]},
            ],
        }],
    }
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=middle_json)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert "from para_blocks" in pages[0].structured_text
    assert "from preproc_blocks" not in pages[0].structured_text
    assert pages[0].metadata["block_field"] == "para_blocks"


@pytest.mark.asyncio
async def test_parser_falls_back_to_preproc_when_para_missing(fake_pdf):
    """没有 para_blocks 时回退 preproc_blocks (旧 fixture / 异常情况)。"""
    parser = _make_parser()
    middle_json = {
        "pdf_info": [{
            "page_idx": 0,
            "preproc_blocks": [
                {"type": "text", "lines": [{"spans": [{"type": "text", "content": "fallback"}]}]},
            ],
        }],
    }
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=middle_json)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert pages[0].structured_text == "fallback"
    assert pages[0].metadata["block_field"] == "preproc_blocks"


@pytest.mark.asyncio
async def test_parser_recognizes_paragraph_title_as_subheading(fake_pdf):
    """paragraph_title (子标题, MinerU 3.x BlockType) 必须识别为 heading level 3。"""
    parser = _make_parser()
    middle_json = {
        "pdf_info": [{
            "page_idx": 0,
            "para_blocks": [
                {"type": "title", "lines": [{"spans": [{"type": "text", "content": "Chapter A"}]}]},
                {"type": "paragraph_title", "lines": [{"spans": [{"type": "text", "content": "Section 1"}]}]},
                {"type": "text", "lines": [{"spans": [{"type": "text", "content": "body"}]}]},
            ],
        }],
    }
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=middle_json)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert pages[0].heading_path == ["Chapter A", "Section 1"]


@pytest.mark.asyncio
async def test_parser_extracts_equation_content(fake_pdf):
    """interline_equation 的 latex 文本要进 structured_text(原代码漏过此 type)。"""
    parser = _make_parser()
    middle_json = {
        "pdf_info": [{
            "page_idx": 0,
            "para_blocks": [
                {"type": "interline_equation", "lines": [
                    {"spans": [{"type": "interline_equation", "content": "E = mc^2"}]},
                ]},
                {"type": "text", "lines": [{"spans": [{"type": "text", "content": "Einstein"}]}]},
            ],
        }],
    }
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=middle_json)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert "E = mc^2" in pages[0].structured_text
    assert "Einstein" in pages[0].structured_text


@pytest.mark.asyncio
async def test_parser_skips_unrecognized_block_types(fake_pdf):
    """seal / page_number / header 等不在任何 set 的 type 静默跳过,不漏入正文。"""
    parser = _make_parser()
    middle_json = {
        "pdf_info": [{
            "page_idx": 0,
            "para_blocks": [
                {"type": "seal", "lines": []},
                {"type": "page_number", "lines": [{"spans": [{"type": "text", "content": "1"}]}]},
                {"type": "header", "lines": [{"spans": [{"type": "text", "content": "Doc Title"}]}]},
                {"type": "text", "lines": [{"spans": [{"type": "text", "content": "real content"}]}]},
            ],
        }],
    }
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=middle_json)
    with patch.object(parser._client, "post",
                      new_callable=AsyncMock, return_value=fake_resp):
        pages = await parser.parse_async(fake_pdf)
    assert pages[0].structured_text == "real content"
    assert "Doc Title" not in pages[0].structured_text  # header 不进正文
