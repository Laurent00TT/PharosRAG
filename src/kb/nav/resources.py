from __future__ import annotations


def document_uri(doc_id: str) -> str:
    return f"kb://documents/{doc_id}"


def page_uri(doc_id: str, page_num: int) -> str:
    return f"kb://documents/{doc_id}/pages/{page_num}"


def page_image_uri(doc_id: str, page_num: int) -> str:
    return f"kb://documents/{doc_id}/pages/{page_num}/image"


def page_range_uri(doc_id: str, page_start: int, page_end: int) -> str:
    return f"kb://documents/{doc_id}/ranges/{page_start}-{page_end}"
