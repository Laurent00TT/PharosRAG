from __future__ import annotations

from kb.nav.normalize import nav_entry_id, normalize_label


def test_normalize_label_collapses_space():
    assert normalize_label(" Approval   Flow ") == "approval flow"


def test_nav_entry_id_is_stable():
    assert nav_entry_id("doc", "section", "Approval", 1, 3) == nav_entry_id("doc", "section", "Approval", 1, 3)


def test_normalize_label_is_case_insensitive():
    assert normalize_label("Foo Bar") == normalize_label("foo bar") == "foo bar"


def test_nav_entry_id_changes_with_inputs():
    a = nav_entry_id("doc", "section", "Approval", 1, 3)
    b = nav_entry_id("doc", "section", "Approval", 1, 4)  # page_end differs
    c = nav_entry_id("doc", "page", "Approval", 1, 3)  # entry_type differs
    d = nav_entry_id("other", "section", "Approval", 1, 3)  # doc_id differs
    # Set-cardinality check: chained `a != b != c != d` only asserts
    # adjacent pairs (a≠b, b≠c, c≠d) — a hash collision between e.g.
    # a and c would slip past. Comparing the set size catches every pair.
    assert len({a, b, c, d}) == 4
