from __future__ import annotations

from kb.eval.golden import ExpectedSource, GoldenCase
from kb.eval.hybrid_eval import source_hit


def _case(*sources):
    return GoldenCase(
        id="case",
        query="query",
        intent="fact",
        expected_mode="evidence",
        expected_sources=[ExpectedSource(doc_id=d, page_num=p) for d, p in sources],
    )


def test_source_hit_detects_overlap():
    case = _case(("doc", 1))
    assert source_hit(case, [("doc", 1)]) is True
    assert source_hit(case, [("other", 1)]) is False


def test_source_hit_returns_false_on_empty_returned():
    case = _case(("doc", 1))
    assert source_hit(case, []) is False


def test_source_hit_returns_false_on_empty_expected():
    case = _case()
    assert source_hit(case, [("doc", 1)]) is False


def test_source_hit_finds_overlap_among_multiple():
    case = _case(("a", 1), ("b", 2), ("c", 3))
    assert source_hit(case, [("z", 9), ("b", 2)]) is True
    assert source_hit(case, [("a", 99), ("b", 99)]) is False  # page_num mismatch
