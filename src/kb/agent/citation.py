# src/kb/agent/citation.py
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    doc_id: str
    page_num: int


# Format: [doc_id:page_num] e.g., [abc123:3]. doc_id is alphanumeric (matches MD5 hash).
_CITATION_RE = re.compile(r"\[([a-zA-Z0-9_-]+):(\d+)\]")


def parse_citations(text: str) -> list[Citation]:
    """Extract all citation markers from LLM output text.
    Returns citations in order of appearance (no deduplication)."""
    return [
        Citation(doc_id=m.group(1), page_num=int(m.group(2)))
        for m in _CITATION_RE.finditer(text)
    ]


def validate_citations(
    citations: list[Citation],
    retrieved_pages: list[tuple[str, int]],
) -> tuple[list[Citation], list[Citation]]:
    """Split citations into (valid, hallucinated).

    A citation is valid only if its (doc_id, page_num) appears in
    retrieved_pages. Anything else is the LLM fabricating a reference
    — useful as a quality signal even when we don't reject it outright.
    """
    retrieved_set = set(retrieved_pages)
    valid: list[Citation] = []
    hallucinated: list[Citation] = []
    for c in citations:
        if (c.doc_id, c.page_num) in retrieved_set:
            valid.append(c)
        else:
            hallucinated.append(c)
    return valid, hallucinated
