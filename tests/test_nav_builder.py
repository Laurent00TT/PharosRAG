from __future__ import annotations

import pytest

from kb.models import ParsedPage
from kb.nav.builder import NavIndexBuilder
from kb.nav.lint import lint_nav_entries


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


def test_builder_creates_document_section_and_page_entries():
    # Each heading now materializes a section the page nests under (multi-level
    # tree). Before the section layer this was ["document", "page", "page"].
    result = NavIndexBuilder().build_from_pages([make_page(1, ["Intro"]), make_page(2, ["Approval"])])
    assert [entry.entry_type for entry in result.entries] == [
        "document", "section", "page", "section", "page",
    ]
    assert any(edge.edge_type == "next" for edge in result.edges)
    assert all(entry.contains_generated_content is False for entry in result.entries)


def test_builder_empty_pages_returns_empty_result():
    result = NavIndexBuilder().build_from_pages([])
    assert result.entries == []
    assert result.edges == []


def test_builder_top_sections_parent_to_doc_root_pages_parent_to_section():
    # doc → section(A) → page ; doc → section(B) → page.
    # Top-level sections hang off the doc root; pages hang off their section,
    # NOT the doc root (that was the old two-level behavior).
    pages = [make_page(1, ["A"]), make_page(2, ["B"])]
    result = NavIndexBuilder().build_from_pages(pages)
    doc_entry = next(e for e in result.entries if e.entry_type == "document")
    sections = [e for e in result.entries if e.entry_type == "section"]
    page_entries = [e for e in result.entries if e.entry_type == "page"]
    assert len(sections) == 2
    assert all(s.parent_entry_id == doc_entry.entry_id for s in sections)
    section_ids = {s.entry_id for s in sections}
    assert all(p.parent_entry_id in section_ids for p in page_entries)
    assert all(p.parent_entry_id != doc_entry.entry_id for p in page_entries)


def test_builder_nests_multi_level_heading_path():
    # 5 Security  ─┬─ 5.3 Escalation (pages 1,2)
    #              └─ 5.4 Review     (page 3)
    pages = [
        make_page(0, ["5 Security"]),
        make_page(1, ["5 Security", "5.3 Escalation"]),
        make_page(2, ["5 Security", "5.3 Escalation"]),
        make_page(3, ["5 Security", "5.4 Review"]),
    ]
    result = NavIndexBuilder().build_from_pages(pages)
    doc = next(e for e in result.entries if e.entry_type == "document")
    s5 = next(e for e in result.entries if e.label == "5 Security" and e.entry_type == "section")
    s53 = next(e for e in result.entries if e.label == "5.3 Escalation" and e.entry_type == "section")
    s54 = next(e for e in result.entries if e.label == "5.4 Review" and e.entry_type == "section")
    assert s5.parent_entry_id == doc.entry_id
    assert s53.parent_entry_id == s5.entry_id
    assert s54.parent_entry_id == s5.entry_id
    # coverage-based page_end: section 5 spans through its last child page
    assert (s5.page_start, s5.page_end) == (0, 3)
    assert (s53.page_start, s53.page_end) == (1, 2)
    # two pages nest under 5.3
    under_53 = [e for e in result.entries if e.entry_type == "page" and e.parent_entry_id == s53.entry_id]
    assert len(under_53) == 2
    # expand contract: 5.4 is a sibling of 5.3 (same parent), reachable via
    # kb_expand_nav_entry's parent_entry_id-based sibling lookup
    assert s53.parent_entry_id == s54.parent_entry_id


def test_builder_same_named_independent_sections_do_not_collapse():
    # Two un-numbered "Overview" H1s separated by a "Methods" H1 must remain
    # two distinct sections, NOT one section spanning page 0..2. Guards the
    # stack-based (not global-dict) prefix resolution.
    pages = [make_page(0, ["Overview"]), make_page(1, ["Methods"]), make_page(2, ["Overview"])]
    result = NavIndexBuilder().build_from_pages(pages)
    overviews = [e for e in result.entries if e.label == "Overview" and e.entry_type == "section"]
    assert len(overviews) == 2
    assert len({o.entry_id for o in overviews}) == 2
    assert sorted((o.page_start, o.page_end) for o in overviews) == [(0, 0), (2, 2)]


def test_builder_empty_heading_path_degrades_to_page_under_doc():
    # Parser produced no headings → pages hang directly off the doc root
    # (degraded page-as-entry fallback), exactly the pre-section behavior.
    pages = [make_page(0, []), make_page(1, [])]
    result = NavIndexBuilder().build_from_pages(pages)
    assert [e.entry_type for e in result.entries] == ["document", "page", "page"]
    doc = next(e for e in result.entries if e.entry_type == "document")
    assert all(
        p.parent_entry_id == doc.entry_id for p in result.entries if p.entry_type == "page"
    )


def test_builder_page_label_is_anchor_not_heading_path():
    # Pages carry a plain "Page N" anchor label, NOT the heading path — the
    # heading levels live on the section ancestors (reachable via
    # parent_entry_id). This stops a page from duplicating its section in nav
    # title search: a page under "5.3 Escalation" must NOT also match a search
    # for "escalation". Guards the second-step page.label de-duplication.
    pages = [make_page(0, ["5 Security"]), make_page(1, ["5 Security", "5.3 Escalation"])]
    result = NavIndexBuilder().build_from_pages(pages)
    page_entries = [e for e in result.entries if e.entry_type == "page"]
    assert [p.label for p in page_entries] == ["Page 0", "Page 1"]
    assert all(p.source == "page_label" for p in page_entries)
    assert all("Escalation" not in p.label for p in page_entries)


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


def test_builder_adjacent_same_name_sections_merge_known_limitation():
    # KNOWN LIMITATION (locked on purpose): two CONSECUTIVE same-named top-level
    # headings with no different heading between them cannot be told apart by the
    # prefix-stack and merge into ONE section spanning both pages. Contrast with
    # test_builder_same_named_independent_sections_do_not_collapse, where a
    # "Methods" page between the two "Overview"s DOES separate them. This test
    # makes any future change to that behavior a deliberate, reviewed decision.
    pages = [make_page(0, ["Overview"]), make_page(1, ["Overview"])]
    result = NavIndexBuilder().build_from_pages(pages)
    overviews = [e for e in result.entries if e.label == "Overview" and e.entry_type == "section"]
    assert len(overviews) == 1
    assert (overviews[0].page_start, overviews[0].page_end) == (0, 1)


def test_builder_normalized_prefix_survives_whitespace_jitter():
    # OCR/parser whitespace or case jitter ("5 Security" vs "5 Security ") between
    # consecutive pages must NOT fragment one section into two — the prefix stack
    # compares NORMALIZED labels. Regression guard for the raw-string-compare bug.
    pages = [make_page(0, ["5 Security"]), make_page(1, ["5 Security "])]
    result = NavIndexBuilder().build_from_pages(pages)
    sections = [e for e in result.entries if e.entry_type == "section"]
    assert len(sections) == 1
    assert (sections[0].page_start, sections[0].page_end) == (0, 1)


def test_builder_sections_carry_backfilled_range_resource_uris():
    # Every section must end with exactly one page-range resource URI matching its
    # final coverage-based page_end. Guards the post-loop backfill: if it regresses,
    # sections ship with resource_uris=[] and kb_search_nav crashes on a section hit.
    pages = [
        make_page(0, ["5 Security"]),
        make_page(1, ["5 Security", "5.3 Escalation"]),
        make_page(2, ["5 Security", "5.3 Escalation"]),
    ]
    result = NavIndexBuilder().build_from_pages(pages)
    sections = [e for e in result.entries if e.entry_type == "section"]
    assert sections, "expected at least one section"
    for s in sections:
        assert s.resource_uris == [
            f"kb://documents/{s.doc_id}/ranges/{s.page_start}-{s.page_end}"
        ]
    # also cover a genuinely single-page section (page_start == page_end)
    sep = NavIndexBuilder().build_from_pages(
        [make_page(0, ["Overview"]), make_page(1, ["Methods"]), make_page(2, ["Overview"])]
    )
    for s in (e for e in sep.entries if e.entry_type == "section"):
        assert len(s.resource_uris) == 1 and s.resource_uris[0].startswith("kb://")


def test_builder_section_label_is_leaf_heading_not_full_path():
    # section.label / normalized_label must be the LEAF heading only, never the
    # joined ancestor path — otherwise a child section would match nav searches for
    # any ancestor's terms (the full path would leak into title search).
    pages = [make_page(0, ["5 Security", "5.3 Escalation"])]
    result = NavIndexBuilder().build_from_pages(pages)
    s53 = next(
        e for e in result.entries
        if e.entry_type == "section" and e.label == "5.3 Escalation"
    )
    assert ">" not in s53.label
    assert "security" not in s53.normalized_label


def test_builder_output_passes_lint():
    # The full builder output (doc + sections + pages, URIs backfilled) must be
    # clean under lint_nav_entries: valid page ranges, non-empty resource_uris, no
    # generated content. Catches a backfill regression that field-level asserts miss.
    pages = [
        make_page(0, ["5 Security"]),
        make_page(1, ["5 Security", "5.3 Escalation"]),
        make_page(2, ["6 Review"]),
    ]
    result = NavIndexBuilder().build_from_pages(pages)
    assert lint_nav_entries(result.entries) == []


def test_builder_rejects_duplicate_page_num():
    # Duplicate page_num collides page entry_ids and the store's INSERT OR REPLACE
    # silently drops one + orphans its next/prev edges. The builder fails loud.
    pages = [make_page(3, ["A"]), make_page(3, ["B"])]
    with pytest.raises(ValueError, match="duplicate page_num"):
        NavIndexBuilder().build_from_pages(pages)
