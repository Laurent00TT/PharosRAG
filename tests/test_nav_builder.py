from __future__ import annotations

from kb.models import ParsedPage
from kb.nav.builder import NavIndexBuilder


def make_page(page_num: int, heading: list[str], metadata: dict | None = None) -> ParsedPage:
    return ParsedPage(
        doc_id="doc",
        doc_name="manual.pdf",
        page_num=page_num,
        structured_text="text",
        heading_path=heading,
        page_image=b"",
        image_blocks=[],
        tables=[],
        metadata=metadata or {},
    )


def test_builder_creates_document_and_page_entries():
    result = NavIndexBuilder().build_from_pages([make_page(1, ["Intro"]), make_page(2, ["Approval"])])
    assert [entry.entry_type for entry in result.entries] == ["document", "page", "page"]
    assert any(edge.edge_type == "next" for edge in result.edges)
    assert all(entry.contains_generated_content is False for entry in result.entries)


def test_builder_empty_pages_returns_empty_result():
    result = NavIndexBuilder().build_from_pages([])
    assert result.entries == []
    assert result.edges == []


def test_builder_creates_parent_child_edges_to_doc_root():
    pages = [
        make_page(1, ["A"]),
        make_page(2, ["B"]),
    ]
    result = NavIndexBuilder().build_from_pages(pages)
    doc_entry = next(e for e in result.entries if e.entry_type == "document")
    parent_child = [edge for edge in result.edges if edge.edge_type == "parent_child"]
    assert len(parent_child) == 2  # one for each page
    assert all(edge.from_entry_id == doc_entry.entry_id for edge in parent_child)


def test_builder_flowchart_metadata_creates_flowchart_entry():
    pages = [make_page(1, ["Flow"], metadata={"page_type": "flowchart"})]
    result = NavIndexBuilder().build_from_pages(pages)
    flowchart_entries = [e for e in result.entries if e.entry_type == "flowchart"]
    assert len(flowchart_entries) == 1


def test_builder_doc_root_matches_actual_page_range_zero_based():
    # MinerU parser emits 0-based page_num; doc root must span the actual
    # min..max of the pages, NOT 1..len(pages). Otherwise the root resource
    # URI points at a non-existent page and misses page 0.
    pages = [make_page(0, ["A"]), make_page(1, ["B"]), make_page(2, ["C"])]
    result = NavIndexBuilder().build_from_pages(pages)
    doc_entry = next(e for e in result.entries if e.entry_type == "document")
    assert doc_entry.page_start == 0
    assert doc_entry.page_end == 2
    assert doc_entry.resource_uris == ["kb://documents/doc/ranges/0-2"]


def test_builder_doc_root_handles_sparse_or_non_zero_page_range():
    # Defensive: if parser ever returns non-contiguous pages, doc root still
    # reflects min/max rather than len-based math.
    pages = [make_page(3, ["A"]), make_page(7, ["B"])]
    result = NavIndexBuilder().build_from_pages(pages)
    doc_entry = next(e for e in result.entries if e.entry_type == "document")
    assert doc_entry.page_start == 3
    assert doc_entry.page_end == 7
