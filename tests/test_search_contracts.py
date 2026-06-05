from __future__ import annotations

from kb.models import SearchResult
from kb.search.contracts import filter_results_by_contract


def _make(doc_id: str, doc_name: str = "d", page_type: str = "text") -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        doc_name=doc_name,
        page_num=1,
        heading_path=[],
        text_chunk="t",
        vision_description=None,
        figure_index=None,
        figure_caption=None,
        image_url="",
        page_type=page_type,
        doc_version=None,
        effective_date=None,
        score=1.0,
        match_channels=["dense"],
    )


def test_filter_by_doc_id_drops_other_docs():
    results = [_make("a"), _make("b"), _make("a")]
    filtered = filter_results_by_contract(results, {"doc_id": "a"})
    assert all(r.doc_id == "a" for r in filtered)
    assert len(filtered) == 2


def test_filter_by_doc_id_combined_with_page_type():
    results = [
        _make("a", page_type="text"),
        _make("a", page_type="image"),
        _make("b", page_type="text"),
    ]
    filtered = filter_results_by_contract(
        results, {"doc_id": "a", "page_type": "text"}
    )
    assert len(filtered) == 1
    assert filtered[0].doc_id == "a" and filtered[0].page_type == "text"


def test_filter_by_doc_id_empty_returns_all():
    """Empty filter dict acts as identity (existing behavior preserved)."""
    results = [_make("a"), _make("b")]
    assert filter_results_by_contract(results, {}) == results
