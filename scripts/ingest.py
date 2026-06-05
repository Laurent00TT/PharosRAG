# scripts/ingest.py
"""
Usage:
  python scripts/ingest.py --path ./docs
  python scripts/ingest.py --path ./docs/manual.pdf
  python scripts/ingest.py --path ./docs --dry-run
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.config import Settings
from kb.ingestion.pipeline import IngestionPipeline
from kb.observability import new_trace, setup_tracing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".doc"}


async def main(path: Path, dry_run: bool) -> None:
    files = (
        [path] if path.is_file()
        else [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED]
    )

    if not files:
        logger.warning("No supported files found in %s", path)
        return

    logger.info("Found %d file(s) to process", len(files))

    if dry_run:
        for f in files:
            print(f"[dry-run] Would ingest: {f}")
        return

    settings = Settings()
    setup_tracing(
        log_dir=settings.log_dir,
        max_mb=settings.log_max_mb,
        backup_count=settings.log_backup_count,
    )
    pipeline = IngestionPipeline(settings=settings)
    total_chunks = 0

    for i, f in enumerate(files, 1):
        new_trace()  # one trace_id per file — makes per-file aggregation easy
        logger.info("[%d/%d] Processing: %s", i, len(files), f.name)
        try:
            result = await pipeline.ingest(f)
            if result["skipped"]:
                logger.info("  → Skipped (unchanged)")
            else:
                logger.info(
                    "  → %d pages, %d chunks stored",
                    result["pages_processed"], result["chunks_stored"]
                )
                total_chunks += result["chunks_stored"]
        except Exception as e:
            logger.error("  → Failed: %s", e, exc_info=True)

    logger.info("Done. Total chunks stored: %d", total_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into knowledge base")
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.path, args.dry_run))
