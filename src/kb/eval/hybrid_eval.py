from __future__ import annotations

from kb.eval.golden import GoldenCase


def source_hit(case: GoldenCase, returned_sources: list[tuple[str, int]]) -> bool:
    expected = {(source.doc_id, source.page_num) for source in case.expected_sources}
    returned = set(returned_sources)
    return bool(expected & returned)
