from datetime import date

from kb.models import SearchResponse, SearchResult
from kb.tool_server.mcp_formatters import evidence_response_to_mcp_payload


def test_evidence_response_payload_has_resource_uri():
    hit = SearchResult(
        doc_id="doc-1",
        doc_name="manual.pdf",
        page_num=3,
        heading_path=[],
        text_chunk="source text",
        vision_description=None,
        figure_index=None,
        figure_caption=None,
        image_url="local://page.png",
        page_type="text",
        doc_version="v1",
        effective_date=date(2026, 1, 1),
        score=0.9,
        match_channels=["dense"],
    )
    payload = evidence_response_to_mcp_payload(SearchResponse(results=[hit], total=1))
    result = payload["structuredContent"]["results"][0]
    assert result["resource_uri"] == "kb://documents/doc-1/pages/3"
    assert result["cite_as"] == "[doc-1:3]"


def test_evidence_hit_labels_text_as_evidence_not_hint():
    """Security review P0 #2: a hit with text_chunk gets the 'evidence'
    result_type + 'Text preview (evidence)' label in the MCP content
    channel — not conflated with generated description."""
    hit = SearchResult(
        doc_id="d", doc_name="d.pdf", page_num=0,
        heading_path=[], text_chunk="parsed page content",
        vision_description="LLM said something",
        figure_index=None, figure_caption=None,
        image_url="local://img.png", page_type="text",
        doc_version=None, effective_date=None,
        score=0.5, match_channels=["dense"],
    )
    payload = evidence_response_to_mcp_payload(SearchResponse(results=[hit], total=1))
    structured = payload["structuredContent"]["results"][0]
    assert structured["result_type"] == "evidence"
    assert structured["has_text_evidence"] is True
    assert structured["has_generated_description"] is True
    text_channel = payload["content"]
    assert "Text preview (evidence)" in text_channel
    assert "parsed page content" in text_channel
    # Description must NOT be rendered when real text is present.
    assert "LLM said something" not in text_channel


def test_evidence_hit_labels_description_as_hint_not_evidence():
    """Vision-only hit (no text_chunk, has vision_description) is a
    recall hint — must be flagged 'hint', not labeled 'Text preview'."""
    hit = SearchResult(
        doc_id="d", doc_name="d.pdf", page_num=1,
        heading_path=[], text_chunk="",   # no parsed text
        vision_description="A bar chart of Q3 sales",
        figure_index="Fig 2", figure_caption=None,
        image_url="local://img.png", page_type="flowchart",
        doc_version=None, effective_date=None,
        score=0.7, match_channels=["vision"],
    )
    payload = evidence_response_to_mcp_payload(SearchResponse(results=[hit], total=1))
    structured = payload["structuredContent"]["results"][0]
    assert structured["result_type"] == "hint"
    assert structured["has_text_evidence"] is False
    assert structured["has_generated_description"] is True
    text_channel = payload["content"]
    # Crucial: NOT labeled "Text preview"; explicitly tagged as hint.
    assert "Text preview" not in text_channel
    assert "RECALL HINT only" in text_channel
    assert "not evidence" in text_channel
    assert "A bar chart of Q3 sales" in text_channel
