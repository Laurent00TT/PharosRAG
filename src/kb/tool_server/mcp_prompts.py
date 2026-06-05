"""Wiki/nav compile prompt templates."""
from __future__ import annotations


KB_ANSWER_FROM_ORIGINALS = (
    "Call kb_hybrid_search first. Treat nav_hits as navigation pointers only, not evidence. "
    "Read suggested_fetches resources before answering. Use only original document/page/image resources as evidence. "
    "If a fetched page's navigation.looks_continued is true (the prose runs onto the next page), "
    "also fetch navigation.next_page_uri / prev_page_uri or the enclosing section's range before answering. "
    "But if text_truncated is true, the remainder is on the SAME page beyond the inline cap — "
    "fetch the page image or kb_get_document_source for the full page, not the next page. "
    "Cite answers as [doc_id:page_num]."
)


def register_prompts(mcp, state) -> None:
    @mcp.prompt()
    def kb_answer_from_originals() -> str:
        return KB_ANSWER_FROM_ORIGINALS
