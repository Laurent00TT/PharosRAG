"""Smoke: drive MinerUAPIParser on one real PDF, report timing & first-page sample.

Requires MINERU_API_TOKEN in .env (or env). Hits mineru.net SaaS.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb.config import Settings
from kb.parser.mineru_api import MinerUAPIParser


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(REPO_ROOT / "test_doc"))
    ap.add_argument("--pattern", default="*266.pdf")
    args = ap.parse_args()

    # Don't depend on PARSER_BACKEND in .env — explicitly point at API mode
    settings = Settings().model_copy(update={"parser_backend": "mineru_api"})
    if not settings.mineru_api_token:
        raise SystemExit("MINERU_API_TOKEN missing in .env")

    parser = MinerUAPIParser(settings=settings)

    pdf = next(Path(args.input_dir).glob(args.pattern))
    print(f"PDF: {pdf.name}, {pdf.stat().st_size/1024:.0f} KB")
    print(f"API base: {settings.mineru_api_url}")

    t0 = time.monotonic()
    pages = await parser.parse_async(pdf)
    dt = time.monotonic() - t0
    print(f"\nDone in {dt:.1f}s, {len(pages)} pages")

    p0 = pages[0]
    print(f"\nPage 0 doc_id: {p0.doc_id}")
    print(f"Page 0 doc_name: {p0.doc_name}")
    print(f"Page 0 heading_path: {p0.heading_path}")
    print(f"Page 0 has page_image bytes: {len(p0.page_image)} bytes")
    print(f"Page 0 tables: {len(p0.tables)}, image_blocks: {len(p0.image_blocks)}")
    print(f"\nPage 0 structured_text (first 400 chars):")
    print(p0.structured_text[:400])
    print()
    if len(pages) > 1:
        print(f"Page 1 heading_path: {pages[1].heading_path}")
        print(f"Page 1 structured_text (first 200 chars):")
        print(pages[1].structured_text[:200])


if __name__ == "__main__":
    asyncio.run(main())
