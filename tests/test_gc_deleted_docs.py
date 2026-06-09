"""GC script — purge expired soft-deleted docs (T5-7)."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio

import pytest


@pytest.fixture
async def seeded(tmp_path):
    """Two soft-deleted docs: one within retention (alive in trash),
    one expired (should be purged). Plus a pre-T5 doc with NULL deleted_at."""
    from kb.storage.metadata_db import MetadataDB
    from sqlalchemy import update
    from kb.storage.metadata_db import DocumentRecord

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db")
    await meta.init()

    for did in ("alive", "expired", "pret5"):
        await meta.upsert_document(
            doc_id=did, doc_name=f"{did}.pdf", version=None,
            effective_date=None, expiry_date=None, supersedes=None,
            owner_id="u-x",
        )
        await meta.mark_deleted(did)

    long_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=40)
    async with meta._engine.begin() as conn:
        await conn.execute(
            update(DocumentRecord)
            .where(DocumentRecord.doc_id == "expired")
            .values(deleted_at=long_ago)
        )
        await conn.execute(
            update(DocumentRecord)
            .where(DocumentRecord.doc_id == "pret5")
            .values(deleted_at=None)
        )

    yield meta, tmp_path
    await meta.aclose()


@pytest.mark.asyncio
async def test_gc_dry_run_lists_expired_does_not_delete(seeded):
    """--dry-run: prints what would be purged, doesn't touch DB."""
    from scripts.gc_deleted_docs import gc_main
    meta, tmp_path = seeded
    candidates = await gc_main(retention_days=30, dry_run=True, force=False, kb_metadata_url=meta._engine.url.render_as_string(False))
    assert {c["doc_id"] for c in candidates} == {"expired"}
    for did in ("alive", "expired", "pret5"):
        doc = await meta.get_document(did, include_deleted_at=True)
        assert doc is not None
        assert doc["status"] == "deleted"


@pytest.mark.asyncio
async def test_gc_purges_expired_only(seeded):
    """Default behavior: only `expired` is purged. `alive` (within
    retention) and `pret5` (NULL deleted_at) are left alone."""
    from scripts.gc_deleted_docs import gc_main
    meta, _ = seeded
    purged = await gc_main(retention_days=30, dry_run=False, force=False, kb_metadata_url=meta._engine.url.render_as_string(False))
    assert {p["doc_id"] for p in purged} == {"expired"}
    assert (await meta.get_document("expired", include_deleted_at=True)) is None
    assert (await meta.get_document("alive", include_deleted_at=True))["status"] == "deleted"
    assert (await meta.get_document("pret5", include_deleted_at=True))["status"] == "deleted"


@pytest.mark.asyncio
async def test_gc_force_purge_includes_null_deleted_at(seeded):
    """--force-purge: also purges pre-T5 rows with NULL deleted_at
    (admin acknowledges no truth source, accepts the loss)."""
    from scripts.gc_deleted_docs import gc_main
    meta, _ = seeded
    purged = await gc_main(retention_days=30, dry_run=False, force=True, kb_metadata_url=meta._engine.url.render_as_string(False))
    purged_ids = {p["doc_id"] for p in purged}
    assert "expired" in purged_ids
    assert "pret5" in purged_ids
    assert "alive" not in purged_ids


@pytest.mark.asyncio
async def test_gc_calls_qdrant_image_nav_cleanup_for_each_purge(seeded):
    """v2 review P0 #4: GC must drive the actual physical cleanup
    across all three stores (Qdrant, images, nav). The metadata-only
    test above doesn't exercise this — without these calls, the doc
    row is deleted but vectors/images/nav linger forever."""
    from unittest.mock import AsyncMock, MagicMock
    from scripts.gc_deleted_docs import gc_main
    meta, _ = seeded

    qdrant_mock = MagicMock()
    qdrant_mock.delete_by_doc_id = AsyncMock()
    image_mock = MagicMock()
    image_mock.delete_doc_images = AsyncMock()
    nav_mock = MagicMock()
    nav_mock.delete_entries_for_doc = AsyncMock()

    purged = await gc_main(
        retention_days=30, dry_run=False, force=False,
        kb_metadata_url=meta._engine.url.render_as_string(False),
        qdrant_store=qdrant_mock, image_store=image_mock, nav_store=nav_mock,
    )
    assert {p["doc_id"] for p in purged} == {"expired"}
    qdrant_mock.delete_by_doc_id.assert_awaited_once_with("expired")
    image_mock.delete_doc_images.assert_awaited_once_with("expired")
    nav_mock.delete_entries_for_doc.assert_awaited_once_with("expired")
