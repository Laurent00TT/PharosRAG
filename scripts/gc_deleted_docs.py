"""Purge soft-deleted documents after retention period expires (T5-7).

Scans the metadata DB for documents whose ``deleted_at`` timestamp is older
than ``retention_days`` (default 30) and physically removes them from all
five storage layers in order:

  1. Qdrant vectors (text / vision / desc collections)
  2. Image files on disk
  3. Source PDF blob (retained original, if STORE_SOURCE_DOCUMENTS was on)
  4. Nav index entries
  5. Metadata DB row  ← last, so a crash mid-purge is retried next run

Pre-T5 rows with ``deleted_at IS NULL`` are skipped unless ``--force-purge``
is explicitly passed — admin acknowledges there is no timestamp truth source
and accepts the data loss.

Usage:
    PYTHONPATH=src python scripts/gc_deleted_docs.py
    PYTHONPATH=src python scripts/gc_deleted_docs.py --dry-run
    PYTHONPATH=src python scripts/gc_deleted_docs.py --force-purge
    PYTHONPATH=src python scripts/gc_deleted_docs.py --skip-maintenance-check
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
from sqlalchemy import select

from kb.storage.metadata_db import (
    DocumentRecord,
    DOC_STATUS_DELETED,
    MetadataDB,
)

logger = logging.getLogger(__name__)


async def _check_maintenance_via_http(base_url: str, admin_key: str) -> bool:
    """Return True if server reports maintenance ON. Raise RuntimeError
    if server unreachable — operator must resolve before GC."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{base_url}/admin/maintenance_mode",
                headers={"X-API-Key": admin_key},
            )
            resp.raise_for_status()
            return bool(resp.json().get("on"))
    except httpx.HTTPError as exc:
        raise RuntimeError(f"failed to reach server at {base_url}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Purge soft-deleted KB documents past retention.")
    parser.add_argument("--retention-days", type=int, default=30,
                        help="Days a doc stays in trash before purge (default 30).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List candidates, don't delete.")
    parser.add_argument("--force-purge", action="store_true",
                        help="Also purge pre-T5 rows with NULL deleted_at.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8765",
                        help="Server URL for maintenance probe.")
    parser.add_argument("--admin-key", default=None,
                        help="Admin API key (or read from env KB_ADMIN_KEY).")
    parser.add_argument("--skip-maintenance-check", action="store_true",
                        help="DANGEROUS: skip the maintenance HTTP probe.")
    return parser


async def gc_main(
    *,
    retention_days: int = 30,
    dry_run: bool = False,
    force: bool = False,
    kb_metadata_url: str,
    qdrant_store: Any = None,
    image_store: Any = None,
    source_store: Any = None,
    nav_store: Any = None,
) -> list[dict]:
    """Core GC logic — returns list of dicts for docs that were (or would be)
    purged.  Injectable store args allow unit tests to pass mocks.

    Args:
        retention_days: docs deleted more than this many days ago are eligible.
        dry_run: if True, return candidates without modifying anything.
        force: if True, also include pre-T5 rows with NULL deleted_at.
        kb_metadata_url: SQLAlchemy async URL for the metadata database.
        qdrant_store: optional QdrantStore instance (skipped if None).
        image_store: optional ImageStore instance (skipped if None).
        source_store: optional SourceDocStore instance (skipped if None).
        nav_store: optional NavIndexStore instance (skipped if None).

    Returns:
        List of {"doc_id": str, "deleted_at": datetime | None} dicts for each
        doc that was (or in dry-run mode: would be) processed.
    """
    meta = MetadataDB(db_url=kb_metadata_url)
    await meta.init()

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=retention_days)

    try:
        async with meta._session_factory() as session:
            # Fetch all rows with status='deleted'
            result = await session.execute(
                select(DocumentRecord)
                .where(DocumentRecord.status == DOC_STATUS_DELETED)
            )
            rows = result.scalars().all()

        candidates: list[dict] = []
        for row in rows:
            deleted_at = row.deleted_at
            if deleted_at is None:
                # Pre-T5 row: no timestamp truth source
                if not force:
                    logger.debug("Skipping pre-T5 row %s (NULL deleted_at, use --force-purge)", row.doc_id)
                    continue
                # force=True: include it, mark deleted_at as None in result
                candidates.append({"doc_id": row.doc_id, "deleted_at": None})
            else:
                if deleted_at <= cutoff:
                    candidates.append({"doc_id": row.doc_id, "deleted_at": deleted_at})
                else:
                    logger.debug("Skipping %s: deleted_at=%s within retention window", row.doc_id, deleted_at)

        if dry_run:
            for c in candidates:
                print(f"[DRY-RUN] would purge doc_id={c['doc_id']} deleted_at={c['deleted_at']}")
            return candidates

        purged: list[dict] = []
        for c in candidates:
            doc_id = c["doc_id"]
            print(f"Purging doc_id={doc_id} deleted_at={c['deleted_at']} ...")

            # Step 1: Qdrant vectors — abort this doc on failure (vectors
            # would linger, but retrying next GC run is safe)
            if qdrant_store is not None:
                try:
                    await qdrant_store.delete_by_doc_id(doc_id)
                except Exception as exc:
                    logger.error("Failed to delete Qdrant vectors for %s: %s — skipping doc", doc_id, exc)
                    continue

            # Step 2: image files — abort this doc on failure
            if image_store is not None:
                try:
                    await image_store.delete_doc_images(doc_id)
                except Exception as exc:
                    logger.error("Failed to delete images for %s: %s — skipping doc", doc_id, exc)
                    continue

            # Step 2b: source PDF blob — non-fatal disk cleanup (like images).
            # delete() is sync + best-effort; absence is fine (flag-off docs).
            if source_store is not None:
                try:
                    source_store.delete(doc_id)
                except Exception as exc:
                    logger.warning("Failed to delete source PDF for %s: %s — continuing", doc_id, exc)

            # Step 3: nav index — non-fatal; nav can be rebuilt, and the
            # metadata row is deleted regardless so there is no orphan risk
            if nav_store is not None:
                try:
                    await nav_store.delete_entries_for_doc(doc_id)
                except Exception as exc:
                    logger.warning("Failed to delete nav entries for %s: %s — continuing to purge metadata", doc_id, exc)

            # Step 4: metadata row — last, so crash-safety: a partially
            # cleaned doc retains its row and will be retried next GC run
            async with meta._engine.begin() as conn:
                await conn.execute(
                    DocumentRecord.__table__.delete().where(
                        DocumentRecord.doc_id == doc_id
                    )
                )

            purged.append(c)
            print(f"  PURGED doc_id={doc_id}")

        return purged

    finally:
        await meta.aclose()


async def amain(argv: list[str] | None = None) -> int:
    """CLI entry point — constructs real stores and runs GC."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Maintenance probe — refuse to run during backup / maintenance windows
    if not args.skip_maintenance_check:
        admin_key = args.admin_key or os.environ.get("KB_ADMIN_KEY")
        if not admin_key:
            print(
                "ERROR: provide --admin-key or set KB_ADMIN_KEY "
                "(or pass --skip-maintenance-check)",
                file=sys.stderr,
            )
            return 2
        try:
            on = await _check_maintenance_via_http(args.server_url, admin_key)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if on:
            print("ERROR: server is in maintenance mode; refusing to run GC.",
                  file=sys.stderr)
            return 3

    from kb.config import Settings
    from kb.storage.image_store import ImageStore
    from kb.storage.qdrant_store import QdrantStore
    from kb.storage.source_store import SourceDocStore
    from kb.nav.store import NavIndexStore

    settings = Settings()

    qdrant_store = QdrantStore(settings=settings)
    image_store = ImageStore(settings=settings)
    source_store = SourceDocStore(settings=settings)
    nav_store = NavIndexStore(settings.nav_db_path)
    await nav_store.init()

    kb_metadata_url = f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"

    try:
        purged = await gc_main(
            retention_days=args.retention_days,
            dry_run=args.dry_run,
            force=args.force_purge,
            kb_metadata_url=kb_metadata_url,
            qdrant_store=qdrant_store,
            image_store=image_store,
            source_store=source_store,
            nav_store=nav_store,
        )
        action = "would purge" if args.dry_run else "purged"
        print(f"\nDone. GC {action} {len(purged)} doc(s).")
    finally:
        try:
            await nav_store.close()
        except Exception as exc:
            print(f"  WARN nav_store.close failed: {exc}", file=sys.stderr)
        client_handle = getattr(qdrant_store, "client", None)
        if client_handle is not None and hasattr(client_handle, "close"):
            try:
                client_handle.close()
            except Exception as exc:
                print(f"  WARN qdrant client.close failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
