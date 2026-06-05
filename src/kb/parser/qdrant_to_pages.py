"""Re-derive minimal ParsedPage stubs from qdrant text chunks.

Shared by the one-off `scripts/backfill_nav_index.py` recovery tool and
the `kb_rebuild_nav_index` MCP tool. Both want "rebuild nav without
re-parsing" — synthesize one ParsedPage per (doc_id, page_num) group,
populated with the first chunk's text+heading_path so NavIndexBuilder
has something to label pages with. Cheaper than re-parsing through
MinerU; sufficient for nav (which is title-based, not body-based).
"""
from __future__ import annotations

import asyncio
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from kb.models import ParsedPage


def _scroll_chunks(qdrant: Any, doc_id: str) -> list[dict]:
    """Pull every text_chunks point for one doc as a list of payloads.

    Synchronous; callers wrap with asyncio.to_thread when needed. The
    scroll loop is bounded by qdrant's own next_page_offset so a doc with
    no chunks returns [] without an extra round-trip.
    """
    payloads: list[dict] = []
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    offset = None
    while True:
        points, next_offset = qdrant.client.scroll(
            collection_name=qdrant.text_collection,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payloads.append(p.payload or {})
        if next_offset is None:
            break
        offset = next_offset
    return payloads


async def synthesize_pages_from_qdrant(
    qdrant: Any, *, doc_id: str, doc_name: str,
) -> list[ParsedPage]:
    """Return ParsedPage list for ``doc_id`` reconstructed from qdrant.

    One ParsedPage per distinct page_num seen in text_chunks. The first
    chunk encountered per page contributes heading_path + structured_text;
    additional chunks on the same page are ignored (nav doesn't need
    them — it labels by heading, not body). page_image bytes are left
    empty since nav builder only consumes metadata.
    """
    payloads = await asyncio.to_thread(_scroll_chunks, qdrant, doc_id)
    by_page: dict[int, dict] = {}
    for payload in payloads:
        page_num = payload.get("page_num")
        if page_num is None:
            continue
        if page_num not in by_page:
            heading = payload.get("heading_path") or []
            if not isinstance(heading, list):
                heading = []
            by_page[page_num] = {
                "heading_path": heading,
                "text": payload.get("text", "") or "",
                "page_type": payload.get("page_type", "text"),
            }
    pages: list[ParsedPage] = []
    for page_num in sorted(by_page.keys()):
        info = by_page[page_num]
        pages.append(ParsedPage(
            doc_id=doc_id,
            doc_name=doc_name,
            page_num=page_num,
            structured_text=info["text"][:500],
            heading_path=info["heading_path"],
            page_image=b"",
            image_blocks=[],
            tables=[],
            metadata={"page_type": info["page_type"]},
        ))
    return pages
