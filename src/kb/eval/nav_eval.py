from __future__ import annotations


def page_coverage(expected_pages: list[int], returned_ranges: list[tuple[int, int]]) -> float:
    """Fraction of expected pages covered by the returned page ranges.

    Used in nav eval: did NavIndexBuilder + NavSearchEngine + HybridNavigator
    together surface enough of the right pages? Returns 1.0 if all expected
    pages are within returned ranges; 0.0 if none; partial if some.
    """
    expected = set(expected_pages)
    returned: set[int] = set()
    for start, end in returned_ranges:
        returned.update(range(start, end + 1))
    if not expected:
        return 1.0
    return len(expected & returned) / len(expected)
