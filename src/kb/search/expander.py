# src/kb/search/expander.py
import asyncio
import logging

from kb.models import SearchResult
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


async def enrich_results(
    fused: list[tuple[str, int, float, list[str]]],
    qdrant: QdrantStore,
    meta_db: MetadataDB,
) -> list[SearchResult]:
    """Expand small chunks to parent text and join desc + document metadata."""
    tasks = [
        _enrich_one(doc_id, page_num, score, channels, qdrant, meta_db)
        for doc_id, page_num, score, channels in fused
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def _enrich_one(
    doc_id: str,
    page_num: int,
    score: float,
    channels: list[str],
    qdrant: QdrantStore,
    meta_db: MetadataDB,
) -> SearchResult | None:
    text_rec, desc_rec, vision_rec, doc_rec = await asyncio.gather(
        asyncio.to_thread(qdrant.get_parent_chunk, doc_id, page_num),
        asyncio.to_thread(qdrant.get_desc_record, doc_id, page_num),
        asyncio.to_thread(qdrant.get_vision_record, doc_id, page_num),
        meta_db.get_document(doc_id),
    )

    # Skip if no text content AND no description — nothing useful to return
    if text_rec is None and desc_rec is None:
        logger.warning("No text or desc found for doc_id=%s page_num=%d; skipping", doc_id, page_num)
        return None

    # image_url priority: desc_rec (image-rich page with full description) >
    # vision_rec (image page where desc was skipped) > "" (pure text page).
    # Pure-text pages: DescriptionChannel.should_process returns False, so
    # desc_rec is None — but ImageStore still saved the page image, so
    # vision_rec.image_url is a reliable secondary fallback source.
    image_url = (
        (desc_rec or {}).get("image_url")
        or (vision_rec or {}).get("image_url")
        or ""
    )

    return SearchResult(
        doc_id=doc_id,
        doc_name=(text_rec or doc_rec or {}).get("doc_name", ""),
        page_num=page_num,
        heading_path=(text_rec or {}).get("heading_path", []),
        text_chunk=(text_rec or {}).get("text", ""),
        vision_description=(desc_rec or {}).get("description"),
        figure_index=(desc_rec or {}).get("figure_index"),
        figure_caption=(desc_rec or {}).get("figure_caption"),
        image_url=image_url,
        page_type=(desc_rec or {}).get("page_type", "text"),
        doc_version=(doc_rec or {}).get("version"),
        effective_date=(doc_rec or {}).get("effective_date"),
        score=score,
        match_channels=channels,
    )
