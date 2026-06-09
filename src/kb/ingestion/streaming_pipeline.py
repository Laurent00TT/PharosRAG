# src/kb/ingestion/streaming_pipeline.py
"""Multi-document streaming ingest — semaphore-bounded concurrency, failure-tolerant.

Inspired by MinerU's doc_analyze_streaming but lighter (no GPU batch coordination —
just concurrency control). Suitable when the user imports hundreds of documents at once.
"""
import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


async def streaming_ingest(
    paths: list[Path],
    ingest_fn: Callable[[Path], Awaitable[dict]],
    on_doc_done: Callable[[Path, dict], Awaitable[None]],
    max_concurrent: int = 4,
) -> None:
    """Run ingest_fn on each path, fire on_doc_done when finished.

    - max_concurrent: max simultaneously-running docs (limited by GPU VRAM)
    - A single-doc failure does not abort others — the error goes into result["error"].
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _process(path: Path):
        async with sem:
            try:
                result = await ingest_fn(path)
            except Exception as e:
                logger.exception("Document %s ingest failed", path.name)
                result = {"error": str(e), "doc_id": "", "pages_processed": 0}
            await on_doc_done(path, result)

    await asyncio.gather(*[_process(p) for p in paths])
