import pytest


def test_doctor_result_counts_errors():
    from kb.ops.doctor import DoctorResult

    result = DoctorResult()
    result.error("qdrant", "missing collection")

    assert result.summary()["errors"] == 1
    assert result.summary()["warnings"] == 0


def test_doctor_result_serializes_findings():
    from kb.ops.doctor import DoctorResult

    result = DoctorResult()
    result.ok("config", "loaded")
    result.warn("security", "api key not set")

    data = result.to_dict()

    assert data["summary"]["warnings"] == 1
    assert data["findings"][0]["level"] == "OK"


def test_check_settings_reports_config_findings(test_settings):
    from kb.ops.doctor import DoctorResult, check_settings

    result = DoctorResult()
    check_settings(test_settings, result)

    assert any(f.area == "config" for f in result.findings)


def test_check_storage_paths_reports_missing_qdrant_parent(test_settings, tmp_path):
    from kb.ops.doctor import DoctorResult, check_storage_paths

    result = DoctorResult()
    test_settings.qdrant_path = str(tmp_path / "missing" / "qdrant")

    check_storage_paths(test_settings, result)

    assert any(f.area == "storage" and f.level == "WARN" for f in result.findings)


@pytest.mark.asyncio
async def test_check_http_service_reports_ok(monkeypatch):
    from kb.ops.doctor import DoctorResult, check_http_service

    class FakeResponse:
        status_code = 200

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr("kb.ops.doctor.httpx.AsyncClient", lambda timeout: FakeClient())
    result = DoctorResult()

    await check_http_service(result, "mineru", "http://localhost:8001")

    assert any(f.area == "mineru" and f.level == "OK" for f in result.findings)


async def test_doctor_includes_nav_path_finding(tmp_path):
    from kb.config import Settings
    from kb.ops.doctor import DoctorResult, check_nav_paths

    settings = Settings(nav_db_path=str(tmp_path / "nav.db"))
    result = DoctorResult()
    check_nav_paths(settings, result)
    areas = {f.area for f in result.findings}
    assert "nav_index" in areas


async def test_doctor_warns_when_mcp_write_tools_enabled(tmp_path):
    from kb.config import Settings
    from kb.ops.doctor import DoctorResult, check_nav_paths

    settings = Settings(nav_db_path=str(tmp_path / "nav.db"), mcp_enable_write_tools=True)
    result = DoctorResult()
    check_nav_paths(settings, result)
    sec_findings = [f for f in result.findings if "mcp_enable_write_tools" in f.message]
    assert len(sec_findings) == 1
    assert sec_findings[0].level == "WARN"


async def test_doctor_nav_orphan_check_clean_when_all_match(tmp_path):
    """No orphans = OK finding. Nav has doc-1, metadata says doc-1 is active."""
    from kb.config import Settings
    from kb.nav.models import NavEntry
    from kb.nav.store import NavIndexStore
    from kb.ops.doctor import DoctorResult, check_nav_orphans
    from kb.storage.metadata_db import MetadataDB

    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    nav_db = tmp_path / "nav.db"

    # Seed nav index with doc-1
    nav_store = NavIndexStore(str(nav_db))
    await nav_store.init()
    await nav_store.upsert_entries([NavEntry(
        entry_id="e-1", label="X", normalized_label="x", entry_type="section",
        source="parser_heading", doc_id="doc-1", doc_name="d.pdf",
        page_start=1, page_end=1, order_index=1, parent_entry_id=None,
        resource_uris=[], source_ref_ids=[],
    )])
    await nav_store.close()

    # Seed metadata with doc-1 as active
    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{qdrant_path}/kb_metadata.db")
    await meta.init()
    await meta.upsert_document(
        doc_id="doc-1", doc_name="d.pdf",
        version=None, effective_date=None, expiry_date=None, supersedes=None,
    )
    await meta.aclose()

    settings = Settings(qdrant_path=str(qdrant_path), nav_db_path=str(nav_db))
    result = DoctorResult()
    await check_nav_orphans(settings, result)

    orphan_findings = [f for f in result.findings if f.area == "nav_orphan"]
    assert len(orphan_findings) == 1
    assert orphan_findings[0].level == "OK"
    assert "match active metadata" in orphan_findings[0].message


async def test_doctor_nav_orphan_check_warns_on_orphan(tmp_path):
    """Doc in nav but not in active metadata = WARN. Sets up doc-active +
    doc-orphan in nav, only doc-active in metadata as active."""
    from kb.config import Settings
    from kb.nav.models import NavEntry
    from kb.nav.store import NavIndexStore
    from kb.ops.doctor import DoctorResult, check_nav_orphans
    from kb.storage.metadata_db import MetadataDB

    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    nav_db = tmp_path / "nav.db"

    nav_store = NavIndexStore(str(nav_db))
    await nav_store.init()
    await nav_store.upsert_entries([
        NavEntry(
            entry_id="a", label="A", normalized_label="a", entry_type="section",
            source="parser_heading", doc_id="doc-active", doc_name="a.pdf",
            page_start=1, page_end=1, order_index=1, parent_entry_id=None,
            resource_uris=[], source_ref_ids=[],
        ),
        NavEntry(
            entry_id="o", label="O", normalized_label="o", entry_type="section",
            source="parser_heading", doc_id="doc-orphan", doc_name="o.pdf",
            page_start=1, page_end=1, order_index=1, parent_entry_id=None,
            resource_uris=[], source_ref_ids=[],
        ),
    ])
    await nav_store.close()

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{qdrant_path}/kb_metadata.db")
    await meta.init()
    await meta.upsert_document(
        doc_id="doc-active", doc_name="a.pdf",
        version=None, effective_date=None, expiry_date=None, supersedes=None,
    )
    await meta.aclose()

    settings = Settings(qdrant_path=str(qdrant_path), nav_db_path=str(nav_db))
    result = DoctorResult()
    await check_nav_orphans(settings, result)

    orphan_findings = [f for f in result.findings if f.area == "nav_orphan"]
    assert len(orphan_findings) == 1
    assert orphan_findings[0].level == "WARN"
    assert "1 doc_ids" in orphan_findings[0].message
    assert "doc-orphan" in orphan_findings[0].message


# ── Phase 5.2 deep reconcile checks ─────────────────────────────────────


def _make_fake_qdrant(text_col="text_chunks_v1",
                      vision_col="vision_pages_v1",
                      desc_col="desc_pages_v1",
                      *,
                      collections: list[str] | None = None,
                      doc_id_counts: dict[str, int] | None = None,
                      scroll_pages: list[list[dict]] | None = None):
    """Mock QdrantStore + .client for reconcile tests.

    collections: list of names get_collections returns. Defaults to the
      three text/vision/desc names = 'all present, all good'.
    doc_id_counts: per-doc vector count returned by client.count.
    scroll_pages: optional list-of-pages-of-payloads used by scroll.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    collections = collections if collections is not None else [text_col, vision_col, desc_col]
    coll_list = SimpleNamespace(collections=[SimpleNamespace(name=n) for n in collections])

    client = MagicMock()
    client.get_collections.return_value = coll_list

    def _count(collection_name, count_filter, exact=True):
        # The filter object's must[0].match.value carries the doc_id.
        try:
            doc_id = count_filter.must[0].match.value
        except Exception:
            doc_id = None
        n = (doc_id_counts or {}).get(doc_id, 0)
        return SimpleNamespace(count=n)
    client.count.side_effect = _count

    # scroll returns (points, next_offset). Convert each dict page into
    # a list of point-like objects with .payload.
    pages = list(scroll_pages or [[]])
    def _scroll(collection_name, limit, offset=None, with_payload=True, with_vectors=False):
        if not pages:
            return ([], None)
        page = pages.pop(0)
        points = [SimpleNamespace(payload=p) for p in page]
        next_offset = "more" if pages else None
        return (points, next_offset)
    client.scroll.side_effect = _scroll

    store = SimpleNamespace(
        client=client,
        text_collection=text_col,
        vision_collection=vision_col,
        desc_collection=desc_col,
    )
    return store


@pytest.mark.asyncio
async def test_check_qdrant_collections_ok_when_all_present():
    from kb.ops.doctor import DoctorResult, check_qdrant_collections
    qdrant = _make_fake_qdrant()
    result = DoctorResult()
    await check_qdrant_collections(None, qdrant, result)
    # 3 OK findings, 0 errors.
    ok_qdrant = [f for f in result.findings if f.area == "qdrant" and f.level == "OK"]
    assert len(ok_qdrant) == 3
    assert result.summary()["errors"] == 0


@pytest.mark.asyncio
async def test_check_qdrant_collections_errors_on_missing():
    from kb.ops.doctor import DoctorResult, check_qdrant_collections
    # vision collection missing
    qdrant = _make_fake_qdrant(collections=["text_chunks_v1", "desc_pages_v1"])
    result = DoctorResult()
    await check_qdrant_collections(None, qdrant, result)
    errs = [f for f in result.findings if f.level == "ERROR"]
    assert len(errs) == 1
    assert "vision_pages_v1" in errs[0].message


@pytest.mark.asyncio
async def test_check_active_docs_have_vectors_errors_on_doc_with_zero_vectors(tmp_path):
    from kb.ops.doctor import DoctorResult, check_active_docs_have_vectors
    from kb.storage.metadata_db import MetadataDB

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await meta.init()
    try:
        await meta.upsert_document(
            doc_id="d-with-vec", doc_name="a.pdf",
            version=None, effective_date=None, expiry_date=None,
            supersedes=None, owner_id="u-x",
        )
        await meta.upsert_document(
            doc_id="d-no-vec", doc_name="b.pdf",
            version=None, effective_date=None, expiry_date=None,
            supersedes=None, owner_id="u-x",
        )
        qdrant = _make_fake_qdrant(doc_id_counts={"d-with-vec": 42})
        result = DoctorResult()
        await check_active_docs_have_vectors(meta, qdrant, result)
        errs = [f for f in result.findings if f.level == "ERROR"]
        assert len(errs) == 1
        assert "d-no-vec" in errs[0].message
        assert "NO text vectors" in errs[0].message
    finally:
        await meta.aclose()


@pytest.mark.asyncio
async def test_check_vectors_have_metadata_flags_orphans(tmp_path):
    from kb.ops.doctor import DoctorResult, check_vectors_have_metadata
    from kb.storage.metadata_db import MetadataDB

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await meta.init()
    try:
        await meta.upsert_document(
            doc_id="known", doc_name="k.pdf",
            version=None, effective_date=None, expiry_date=None,
            supersedes=None, owner_id="u-x",
        )
        # Qdrant has 2 doc_ids: 'known' (matches meta) + 'orphan' (no meta).
        qdrant = _make_fake_qdrant(
            scroll_pages=[[{"doc_id": "known"}, {"doc_id": "orphan"}]],
        )
        result = DoctorResult()
        await check_vectors_have_metadata(meta, qdrant, result)
        errs = [f for f in result.findings if f.level == "ERROR"]
        assert len(errs) == 1
        assert "orphan" in errs[0].message
        assert "1 doc_id" in errs[0].message
    finally:
        await meta.aclose()


def test_check_active_docs_have_images_warns_on_missing(tmp_path, test_settings):
    from kb.ops.doctor import DoctorResult, check_active_docs_have_images
    test_settings.image_storage_path = str(tmp_path / "images")
    # Seed: one doc has an image dir, one doesn't.
    img_root = tmp_path / "images"
    (img_root / "d-with-img").mkdir(parents=True)
    (img_root / "d-with-img" / "page_0.png").write_bytes(b"x")
    docs = [
        {"doc_id": "d-with-img", "doc_name": "a.pdf", "status": "active"},
        {"doc_id": "d-no-img",  "doc_name": "b.pdf", "status": "active"},
    ]
    result = DoctorResult()
    check_active_docs_have_images(test_settings, docs, result)
    warns = [f for f in result.findings if f.level == "WARN"]
    assert len(warns) == 1
    assert "d-no-img" in warns[0].message


@pytest.mark.asyncio
async def test_check_deleted_docs_have_no_residue_warns_with_gc_hint(tmp_path):
    from kb.ops.doctor import DoctorResult, check_deleted_docs_have_no_residue
    from kb.storage.metadata_db import MetadataDB

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await meta.init()
    try:
        await meta.upsert_document(
            doc_id="d-deleted-with-vec", doc_name="g.pdf",
            version=None, effective_date=None, expiry_date=None,
            supersedes=None, owner_id="u-x",
        )
        await meta.mark_deleted("d-deleted-with-vec")
        qdrant = _make_fake_qdrant(doc_id_counts={"d-deleted-with-vec": 17})
        result = DoctorResult()
        await check_deleted_docs_have_no_residue(meta, qdrant, result)
        warns = [f for f in result.findings if f.level == "WARN"]
        assert len(warns) == 1
        # Useful operator hint included
        assert "gc_deleted_docs.py" in warns[0].message
        assert "d-deleted-with-vec" in warns[0].message
    finally:
        await meta.aclose()


@pytest.mark.asyncio
async def test_check_orphan_image_dirs_warns_on_extras(tmp_path, test_settings):
    from kb.ops.doctor import DoctorResult, check_orphan_image_dirs
    from kb.storage.metadata_db import MetadataDB

    img_root = tmp_path / "images"
    img_root.mkdir()
    (img_root / "known-doc").mkdir()
    (img_root / "orphan-doc").mkdir()
    test_settings.image_storage_path = str(img_root)

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await meta.init()
    try:
        await meta.upsert_document(
            doc_id="known-doc", doc_name="k.pdf",
            version=None, effective_date=None, expiry_date=None,
            supersedes=None, owner_id="u-x",
        )
        result = DoctorResult()
        await check_orphan_image_dirs(test_settings, meta, result)
        warns = [f for f in result.findings if f.level == "WARN"]
        assert len(warns) == 1
        assert "orphan-doc" in warns[0].message
    finally:
        await meta.aclose()


@pytest.mark.asyncio
async def test_run_doctor_no_reconcile_skips_deep_checks(tmp_path, test_settings):
    """deep_reconcile=False short-circuits the 5 deep checks. Useful
    for fast operator probes or when Qdrant is intentionally down."""
    from kb.ops.doctor import run_doctor
    test_settings.qdrant_path = str(tmp_path)
    test_settings.image_storage_path = str(tmp_path)
    test_settings.nav_db_path = str(tmp_path / "nav.db")

    result = await run_doctor(test_settings, deep_reconcile=False)
    # Reconcile area must NOT appear when skipped.
    reconcile_findings = [f for f in result.findings if f.area == "reconcile"]
    assert reconcile_findings == []
    # Pre-reconcile checks still run.
    assert any(f.area == "config" for f in result.findings)
