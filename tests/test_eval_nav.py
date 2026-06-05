from __future__ import annotations

from kb.eval.nav_eval import page_coverage


def test_page_coverage_full_match():
    assert page_coverage([3, 4, 5], [(3, 5)]) == 1.0


def test_page_coverage_partial_match():
    assert page_coverage([3, 4, 5], [(3, 3)]) == 1 / 3


def test_page_coverage_no_match():
    assert page_coverage([3, 4, 5], [(10, 12)]) == 0.0


def test_page_coverage_empty_expected_is_perfect():
    assert page_coverage([], [(1, 5)]) == 1.0
