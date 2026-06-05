from __future__ import annotations

from kb.nav.lint import lint_nav_entries, lint_nav_entry
from kb.nav.models import NavEntry


def _make_entry(**overrides) -> NavEntry:
    defaults = dict(
        entry_id="e1",
        label="Approval",
        normalized_label="approval",
        entry_type="section",
        source="parser_heading",
        doc_id="doc",
        doc_name="m.pdf",
        page_start=1,
        page_end=3,
        order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/ranges/1-3"],
        source_ref_ids=[],
    )
    defaults.update(overrides)
    return NavEntry(**defaults)


def test_lint_clean_entry_returns_no_issues():
    assert lint_nav_entry(_make_entry()) == []


def test_lint_flags_invalid_page_range():
    issues = lint_nav_entry(_make_entry(page_start=5, page_end=3))
    assert any(issue.code == "invalid_page_range" for issue in issues)


def test_lint_flags_missing_resource_uri():
    issues = lint_nav_entry(_make_entry(resource_uris=[]))
    assert any(issue.code == "missing_resource_uri" for issue in issues)


def test_lint_nav_entries_batches():
    bad = _make_entry(entry_id="bad", page_start=5, page_end=3)
    good = _make_entry(entry_id="good")
    issues = lint_nav_entries([bad, good])
    assert len(issues) == 1
    assert issues[0].entry_id == "bad"
