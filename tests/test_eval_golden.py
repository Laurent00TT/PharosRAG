from pathlib import Path

import pytest

from kb.eval.golden import load_golden_cases


_DATA_DIR = Path(__file__).parent / "data"


def test_load_rag_wiki_mcp_golden_cases():
    cases = load_golden_cases(_DATA_DIR / "rag_wiki_mcp_golden.jsonl")
    assert len(cases) >= 5
    assert {case.expected_mode for case in cases} >= {"evidence", "hybrid"}
    assert all(case.expected_sources for case in cases)


def test_nav_golden_loads_and_has_3_modes():
    cases = load_golden_cases(_DATA_DIR / "nav_golden.jsonl")
    assert len(cases) >= 10
    modes = {c.expected_mode for c in cases}
    # Should cover all 3 routing modes
    assert "navigation" in modes
    assert "hybrid" in modes
    assert "evidence" in modes
    assert all(case.expected_sources for case in cases)


def test_load_golden_cases_reports_line_on_malformed_record(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(
        '{"id":"ok","query":"q","intent":"i","expected_mode":"evidence","expected_sources":[{"doc_id":"d","page_num":1}]}\n'
        'not json at all\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bad.jsonl:2:.*malformed"):
        load_golden_cases(bad_file)


def test_load_golden_cases_rejects_wiki_expected_mode(tmp_path):
    """The navigation-only refactor removed the 'wiki' mode. Anti-regression
    guard: a hand-edited golden file that still carries expected_mode='wiki'
    must fail loud, not be silently accepted."""
    bad_file = tmp_path / "wiki.jsonl"
    bad_file.write_text(
        '{"id":"x","query":"q","intent":"i","expected_mode":"wiki","expected_sources":[{"doc_id":"d","page_num":1}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wiki.jsonl:1:.*wiki.*removed"):
        load_golden_cases(bad_file)
