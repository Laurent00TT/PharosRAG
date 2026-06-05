"""Backfill Channel 4 descriptions for already-ingested docs WITHOUT re-parsing.

The 304 docs were ingested while Channel 4 was disabled, so desc_pages holds
only empty stubs. Re-running full ingest would redo mineru parsing (hours,
wasteful). Instead this reuses the ingest pipeline's desc channel + text embed
+ qdrant upsert, but sources each page image from ImageStore (saved at the
original ingest) — no mineru.

Run INSIDE a container so DESCRIPTION_URL / embed URL / qdrant / image path are
already wired:
    docker exec kb-main python scripts/backfill_descriptions.py

Idempotent: skips (doc_id, page_num) pairs that already have a non-empty
description. Safe to re-run after an interruption.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "/app/src")

from kb.config import Settings
from kb.ingestion.pipeline import IngestionPipeline
from kb.models import TextChunk


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _scroll_all(client, collection, fields):
    """Scroll an entire collection, yielding payloads (with the given fields)."""
    out = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            with_payload=fields,
            with_vectors=False,
            offset=offset,
        )
        out.extend(p.payload for p in points)
        if offset is None:
            break
    return out


async def main() -> int:
    settings = Settings()
    pipeline = IngestionPipeline(settings=settings)
    await pipeline._qdrant.ensure_collections()
    desc_ch = pipeline._desc_ch
    if desc_ch is None:
        log("❌ description_enabled is False — nothing to do. Set DESCRIPTION_ENABLED=true.")
        return 1
    text_ch = pipeline._text_ch
    qdrant = pipeline._qdrant
    client = qdrant.client
    img_root = Path(settings.image_storage_path)

    # 1. all page records (one per page) from vision_pages
    pages = _scroll_all(
        client, qdrant.vision_collection,
        ["doc_id", "doc_name", "page_num", "image_url", "heading_path", "owner_id"],
    )
    log(f"vision_pages: {len(pages)} page records")

    # 2. pages that already have a NON-EMPTY desc → skip set
    done = set()
    for pl in _scroll_all(client, qdrant.desc_collection, ["doc_id", "page_num", "description"]):
        if (pl.get("description") or "").strip():
            done.add((pl.get("doc_id"), pl.get("page_num")))
    log(f"already-described pages: {len(done)} (will skip)")

    todo = [p for p in pages if (p.get("doc_id"), p.get("page_num")) not in done]
    log(f"to backfill: {len(todo)} pages")

    # Concurrency: serial describe is generation-bound (~16s/page at enforce-eager
    # for a ~700-char structured JSON → 30h for the corpus). vllm does continuous
    # batching, so running CONCURRENCY describe calls in flight lets it batch them
    # and multiplies throughput. KV-cache headroom (Qwen3.6 gpu-util 0.40 →
    # ~11GB KV) comfortably holds this many short-output sequences.
    CONCURRENCY = 12
    sem = asyncio.Semaphore(CONCURRENCY)
    counter = {"ok": 0, "fail": 0, "noimg": 0, "n": 0}
    t0 = time.time()

    async def process(pl):
        doc_id = pl.get("doc_id")
        page_num = pl.get("page_num")
        image_url = pl.get("image_url") or ""
        img_path = Path(image_url) if image_url else (img_root / doc_id / f"page_{page_num}.png")
        if not img_path.exists():
            img_path = img_root / doc_id / f"page_{page_num}.png"
        async with sem:
            if not img_path.exists():
                counter["noimg"] += 1
            else:
                try:
                    # Full-res page image (no downscale): the bottleneck is
                    # token GENERATION, not image encoding, so downscaling gave
                    # zero speedup (concurrency is the real lever) while risking
                    # fidelity on dense financial pages (small numbers, analyst
                    # names). vllm/Qwen-VL resizes internally anyway. Keep full-res.
                    image_bytes = img_path.read_bytes()
                    desc = await desc_ch.describe(
                        doc_id=doc_id,
                        doc_name=pl.get("doc_name", ""),
                        page_num=page_num,
                        image_bytes=image_bytes,
                        heading_path=pl.get("heading_path") or [],
                        image_url=image_url,
                    )
                    if not (desc.description or "").strip():
                        counter["fail"] += 1
                    else:
                        chunk = TextChunk(
                            doc_id=desc.doc_id, doc_name=desc.doc_name,
                            page_num=desc.page_num, chunk_index=0,
                            text=desc.description, heading_path=desc.heading_path,
                            has_visual_on_page=True, page_type=desc.page_type,
                            parent_chunk_id="", is_parent=False,
                        )
                        dense, _, _ = await asyncio.to_thread(text_ch.embed, chunk)
                        await qdrant.upsert_desc_page(desc, dense, owner_id=pl.get("owner_id"))
                        counter["ok"] += 1
                except Exception as e:
                    counter["fail"] += 1
                    log(f"  page {doc_id[:8]}/p{page_num} failed: {type(e).__name__}: {str(e)[:120]}")
        counter["n"] += 1
        i = counter["n"]
        if i % 50 == 0 or i == len(todo):
            rate = i / max(time.time() - t0, 1)
            eta = (len(todo) - i) / max(rate, 0.01)
            log(f"  [{i}/{len(todo)}] ok={counter['ok']} fail={counter['fail']} "
                f"noimg={counter['noimg']} rate={rate:.2f}/s eta={eta/60:.0f}min")

    await asyncio.gather(*[process(pl) for pl in todo])

    log(f"DONE: ok={counter['ok']} fail={counter['fail']} noimg={counter['noimg']} "
        f"in {(time.time()-t0)/60:.1f}min")
    await desc_ch.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
