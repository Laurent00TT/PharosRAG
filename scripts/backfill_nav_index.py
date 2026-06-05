"""Backfill nav index for documents ingested before the nav layer existed.

Scrolls the Qdrant text_chunks collection (one pass per active doc), groups
chunks by page_num to synthesize a minimal per-page sequence, builds
ParsedPage stubs, and runs NavIndexBuilder + NavIndexStore upsert.

This script is a one-off recovery tool. New ingestions auto-build via the
pipeline (see kb.nav.auto_build.auto_build_nav_for_doc invoked from
IngestionPipeline.ingest); existing pre-nav docs need this backfill once.

Usage:
    PYTHONPATH=src python scripts/backfill_nav_index.py
    PYTHONPATH=src python scripts/backfill_nav_index.py --doc-id <one_doc>
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.config import Settings
from kb.nav.auto_build import auto_build_nav_for_doc
from kb.nav.builder import NavIndexBuilder
from kb.nav.store import NavIndexStore
from kb.parser.qdrant_to_pages import synthesize_pages_from_qdrant
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", help="Only build for this doc (default: all active)")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8765",
        help="KB server base URL for maintenance mode probe (default: http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--admin-key", default=None,
        help="Admin API key for maintenance check (or set KB_ADMIN_KEY env var)",
    )
    parser.add_argument(
        "--skip-maintenance-check", action="store_true",
        help="Skip the HTTP maintenance mode probe (useful when server is down)",
    )
    args = parser.parse_args()

    # Maintenance probe — refuse to backfill during backup / maintenance windows
    if not args.skip_maintenance_check:
        admin_key = args.admin_key or os.environ.get("KB_ADMIN_KEY")
        if not admin_key:
            print("ERROR: provide --admin-key or set KB_ADMIN_KEY (or pass "
                  "--skip-maintenance-check)", file=sys.stderr)
            return 2
        from scripts.gc_deleted_docs import _check_maintenance_via_http
        try:
            on = await _check_maintenance_via_http(args.server_url, admin_key)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if on:
            print("ERROR: server in maintenance mode; refusing to backfill nav.",
                  file=sys.stderr)
            return 3

    settings = Settings()
    qdrant = QdrantStore(settings=settings)
    await qdrant.ensure_collections()
    meta_db = MetadataDB(db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db")
    await meta_db.init()
    nav_store = NavIndexStore(settings.nav_db_path)
    await nav_store.init()
    nav_builder = NavIndexBuilder()

    # Wrap the main loop so cleanup runs even on Ctrl-C / unhandled
    # exception mid-iteration. Otherwise a crashed backfill leaves a
    # qdrant HTTP client and two SQLite engines open, which on local
    # qdrant mode can prevent the next backfill attempt from acquiring
    # the file lock.
    try:
        if args.doc_id:
            doc_ids = [args.doc_id]
        else:
            doc_ids = await meta_db.list_active_doc_ids()

        print(f"Backfilling nav index for {len(doc_ids)} doc(s)")
        total_entries = 0
        total_docs_built = 0
        for doc_id in doc_ids:
            doc = await meta_db.get_document(doc_id)
            if not doc:
                print(f"  SKIP doc_id={doc_id} — not in metadata")
                continue
            doc_name = doc.get("doc_name", doc_id)
            pages = await synthesize_pages_from_qdrant(
                qdrant, doc_id=doc_id, doc_name=doc_name,
            )
            if not pages:
                print(f"  SKIP doc_id={doc_id[:12]} ({doc_name[:40]}) — no pages found in qdrant")
                continue
            count = await auto_build_nav_for_doc(
                nav_store=nav_store,
                nav_builder=nav_builder,
                pages=pages,
                enabled=True,
            )
            total_entries += count
            total_docs_built += 1
            print(f"  BUILT doc_id={doc_id[:12]} ({doc_name[:40]}) — {len(pages)} pages -> {count} entries")

        print(f"\nDone. Docs built: {total_docs_built}/{len(doc_ids)}  Total entries: {total_entries}")
    finally:
        # Best-effort close all three. Wrap each individually so one
        # stuck handle doesn't block the others.
        try:
            await nav_store.close()
        except Exception as exc:
            print(f"  WARN nav_store.close failed: {exc}")
        try:
            await meta_db.aclose()
        except Exception as exc:
            print(f"  WARN meta_db.aclose failed: {exc}")
        client = getattr(qdrant, "client", None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception as exc:
                print(f"  WARN qdrant client.close failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
