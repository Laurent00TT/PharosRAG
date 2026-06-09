# tests/test_models.py
from kb.models import ParsedPage, TextChunk, VisionPage, DescPage, SearchResult

def test_parsed_page_fields():
    p = ParsedPage(
        doc_id="abc123",
        doc_name="test.pdf",
        page_num=0,
        structured_text="审批流程",
        heading_path=["手册", "第一章"],
        page_image=b"\x89PNG",
        image_blocks=[],
        tables=[],
        metadata={"size": 1024},
    )
    assert p.doc_id == "abc123"
    assert p.heading_path == ["手册", "第一章"]

def test_text_chunk_parent_flag():
    c = TextChunk(
        doc_id="abc123", doc_name="test.pdf", page_num=0,
        chunk_index=0, text="...", heading_path=[],
        has_visual_on_page=False, page_type="text",
        parent_chunk_id="abc123_p0", is_parent=False
    )
    assert c.is_parent is False

def test_search_result_defaults():
    r = SearchResult(
        doc_id="x", doc_name="x.pdf", page_num=1,
        heading_path=[], text_chunk="", vision_description=None,
        figure_index=None, figure_caption=None,
        image_url="", page_type="text",
        doc_version=None, effective_date=None,
        score=0.9, match_channels=["dense"]
    )
    assert r.match_channels == ["dense"]

from kb.models import DocMeta, DocumentSummary
from datetime import date

def test_doc_meta_fields():
    m = DocMeta(
        doc_id="abc", doc_name="test.pdf", version="v1",
        status="active", effective_date=date(2026, 1, 1), expiry_date=None,
    )
    assert m.doc_id == "abc"
    assert m.status == "active"
    assert m.expiry_date is None

def test_document_summary_defaults():
    s = DocumentSummary(
        doc_id="abc", doc_name="test.pdf", version=None,
        status="active", effective_date=None, expiry_date=None,
    )
    assert s.page_count == 0
    assert s.has_vision is False
