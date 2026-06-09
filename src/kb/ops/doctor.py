from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from kb.config import Settings


@dataclass
class DoctorFinding:
    level: str
    area: str
    message: str


@dataclass
class DoctorResult:
    findings: list[DoctorFinding] = field(default_factory=list)

    def ok(self, area: str, message: str) -> None:
        self.findings.append(DoctorFinding("OK", area, message))

    def warn(self, area: str, message: str) -> None:
        self.findings.append(DoctorFinding("WARN", area, message))

    def error(self, area: str, message: str) -> None:
        self.findings.append(DoctorFinding("ERROR", area, message))

    def summary(self) -> dict:
        return {
            "errors": sum(1 for f in self.findings if f.level == "ERROR"),
            "warnings": sum(1 for f in self.findings if f.level == "WARN"),
            "total": len(self.findings),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "findings": [asdict(f) for f in self.findings],
        }

    def render_text(self) -> str:
        lines = ["Knowledge Base Doctor"]
        summary = self.summary()
        lines.append(
            f"Summary: {summary['errors']} error(s), "
            f"{summary['warnings']} warning(s), {summary['total']} finding(s)"
        )
        for finding in self.findings:
            lines.append(f"[{finding.level}] {finding.area}: {finding.message}")
        return "\n".join(lines)


def check_settings(settings: Settings, result: DoctorResult) -> None:
    result.ok("config", f"kb_profile={settings.kb_profile}")
    result.ok("config", f"parser_backend={settings.parser_backend}")


async def _check_auth(settings: Settings) -> DoctorFinding:
    from kb.auth.users import UsersStore

    store = UsersStore(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db",
    )
    await store.init()
    try:
        users = await store.list_users(include_disabled=True)
    finally:
        await store.aclose()
    if not users:
        return DoctorFinding(
            level="ERROR",
            area="auth",
            message="users table empty — server would refuse to start",
        )
    actives = [u for u in users if not u.is_disabled]
    admins = [u for u in actives if u.role == "admin"]
    if not admins:
        return DoctorFinding(
            level="ERROR",
            area="auth",
            message=f"{len(actives)} active user(s) but no active admin",
        )
    return DoctorFinding(
        level="OK",
        area="auth",
        message=f"{len(actives)} active user(s), {len(admins)} admin(s)",
    )


def check_storage_paths(settings: Settings, result: DoctorResult) -> None:
    qdrant_path = Path(settings.qdrant_path)
    image_path = Path(settings.image_storage_path)
    if qdrant_path.exists():
        result.ok("storage", f"Qdrant path exists: {qdrant_path}")
    elif qdrant_path.parent.exists():
        result.warn("storage", f"Qdrant path does not exist yet: {qdrant_path}")
    else:
        result.warn("storage", f"Qdrant parent path is missing: {qdrant_path.parent}")

    if image_path.exists():
        result.ok("storage", f"Image storage path exists: {image_path}")
    elif image_path.parent.exists():
        result.warn("storage", f"Image storage path does not exist yet: {image_path}")
    else:
        result.warn("storage", f"Image storage parent path is missing: {image_path.parent}")


def check_nav_paths(settings: Settings, result: DoctorResult) -> None:
    """Verify navigation index file paths are writable.

    Mirrors the storage-path checks. Does NOT touch the actual sqlite file
    or query the nav index — that's a runtime concern, doctor only checks
    config + filesystem precondition.
    """
    import os
    nav_path = Path(settings.nav_db_path)
    parent = nav_path.parent
    if not parent.exists():
        result.warn("nav_index", f"parent directory missing: {parent}")
    elif not os.access(parent, os.W_OK):
        result.warn("nav_index", f"parent directory not writable: {parent}")
    else:
        result.ok("nav_index", f"parent directory writable: {parent}")

    if settings.mcp_enable_write_tools:
        result.warn("security", "mcp_enable_write_tools=True — agents can write; ensure access control")
    else:
        result.ok("security", "mcp_enable_write_tools=False (default; agents read-only)")


async def check_nav_orphans(settings: Settings, result: DoctorResult) -> None:
    """Detect nav entries pointing to docs that no longer exist in MetadataDB.

    Orphans accumulate when ``nav_store.delete_entries_for_doc`` is not
    called on doc delete (e.g., if the delete went through a path that
    does not know about nav, or if a future code-path forgets to wire
    nav cleanup). This check warns but does NOT auto-fix — the operator
    decides whether to rebuild nav or run an explicit cleanup script.

    "Active" here means MetadataDB.list_active_doc_ids() — deprecated /
    deleted / failed docs are considered absent from the nav index too.
    """
    if not settings.nav_enabled:
        return
    from kb.nav.store import NavIndexStore
    from kb.storage.metadata_db import MetadataDB

    nav_store_path = settings.nav_db_path
    if not Path(nav_store_path).exists():
        result.ok("nav_orphan", f"nav index file not yet created: {nav_store_path}")
        return

    nav_store = NavIndexStore(nav_store_path)
    try:
        await nav_store.init()
        nav_doc_ids = await nav_store.list_doc_ids()
    finally:
        await nav_store.close()

    if not nav_doc_ids:
        result.ok("nav_orphan", "nav index is empty (no orphans possible)")
        return

    meta = MetadataDB(db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db")
    try:
        await meta.init()
        active_ids = set(await meta.list_active_doc_ids())
    finally:
        await meta.aclose()

    orphans = nav_doc_ids - active_ids
    if orphans:
        sample = sorted(orphans)[:5]
        suffix = f" (+{len(orphans) - 5} more)" if len(orphans) > 5 else ""
        result.warn(
            "nav_orphan",
            f"{len(orphans)} doc_ids in nav index have no active metadata; "
            f"consider rebuilding nav or running orphan cleanup. "
            f"Sample: {sample}{suffix}",
        )
    else:
        result.ok("nav_orphan", f"all {len(nav_doc_ids)} nav doc_ids match active metadata")


async def check_http_service(
    result: DoctorResult,
    area: str,
    base_url: str,
    *,
    path: str = "/health",
    timeout_s: float = 2.0,
) -> None:
    url = base_url.rstrip("/") + path
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
        if 200 <= response.status_code < 300:
            result.ok(area, f"reachable: {url}")
        else:
            result.warn(area, f"{url} returned HTTP {response.status_code}")
    except Exception as exc:
        result.warn(area, f"{url} not reachable: {exc}")


# ── Phase 5.2 deep reconcile checks ────────────────────────────────────
# Spec: doctor must surface 5 consistency invariants between metadata,
# Qdrant, and image storage. Each check is independently best-effort —
# a failed Qdrant connection produces one ERROR, NOT a panel of WARNs.


async def _get_qdrant_doc_ids(qdrant_store) -> set[str]:
    """Scroll the text collection and collect distinct payload['doc_id'].
    Uses scroll (not search) — no embedding needed; just pages of points.
    Returns empty set if collection is empty or unreachable."""
    seen: set[str] = set()
    offset = None
    # Hard cap to bound runtime in huge KBs; doctor is interactive.
    PAGE = 500
    HARD_CAP = 50000
    while len(seen) < HARD_CAP:
        try:
            points, offset = qdrant_store.client.scroll(
                collection_name=qdrant_store.text_collection,
                limit=PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            break
        for p in points:
            did = (p.payload or {}).get("doc_id")
            if did:
                seen.add(did)
        if offset is None:
            break
    return seen


def _count_vectors_for_doc(qdrant_store, doc_id: str) -> int:
    """Cheap count: scroll with filter, returning only count. Falls back
    to 0 if Qdrant rejects the filter (e.g. collection missing)."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    try:
        result = qdrant_store.client.count(
            collection_name=qdrant_store.text_collection,
            count_filter=flt,
            exact=True,
        )
        return int(getattr(result, "count", 0))
    except Exception:
        return 0


async def check_qdrant_collections(
    settings: Settings,
    qdrant_store,
    result: DoctorResult,
) -> None:
    """Verify the 3 versioned collections exist (text / vision / desc).
    Missing collection = ERROR — ingest/search both broken."""
    try:
        existing = {c.name for c in qdrant_store.client.get_collections().collections}
    except Exception as exc:
        result.error("qdrant", f"cannot list collections: {exc}")
        return
    for name in (qdrant_store.text_collection,
                 qdrant_store.vision_collection,
                 qdrant_store.desc_collection):
        if name in existing:
            result.ok("qdrant", f"collection present: {name}")
        else:
            result.error("qdrant", f"collection MISSING: {name}")


async def check_active_docs_have_vectors(
    meta_db,
    qdrant_store,
    result: DoctorResult,
    *,
    sample_limit: int | None = None,
) -> None:
    """For each active document, the text_chunks collection should have
    at least one vector with that doc_id. Zero = ingest broken / partial.

    sample_limit caps how many active docs we count; doctor is meant to
    be quick — for KBs with 10k+ active docs, set this to e.g. 100."""
    try:
        docs = await meta_db.list_documents(status="active")
    except Exception as exc:
        result.error("reconcile", f"failed to list active docs: {exc}")
        return
    if sample_limit:
        docs = docs[:sample_limit]
    missing: list[str] = []
    for d in docs:
        if _count_vectors_for_doc(qdrant_store, d["doc_id"]) == 0:
            missing.append(d["doc_id"])
    if missing:
        result.error(
            "reconcile",
            f"{len(missing)} active doc(s) have NO text vectors "
            f"(ingest broken?): {missing[:5]}"
            + (" ..." if len(missing) > 5 else ""),
        )
    else:
        result.ok(
            "reconcile",
            f"all {len(docs)} active docs have text vectors",
        )


async def check_vectors_have_metadata(
    meta_db,
    qdrant_store,
    result: DoctorResult,
) -> None:
    """Each distinct doc_id appearing in the text collection should
    have a corresponding documents row. Orphans = GC didn't fire or
    metadata DB was rolled back without the vectors."""
    qdrant_ids = await _get_qdrant_doc_ids(qdrant_store)
    if not qdrant_ids:
        result.ok("reconcile", "qdrant text collection empty — nothing to check for orphans")
        return
    # Pull ALL doc_ids regardless of status so a soft-deleted doc with
    # vectors doesn't get falsely flagged as orphan here (we'll catch
    # those separately in check_deleted_docs_have_no_residue).
    try:
        all_docs = await meta_db.list_documents()
    except Exception as exc:
        result.error("reconcile", f"failed to list all docs: {exc}")
        return
    known = {d["doc_id"] for d in all_docs}
    orphans = sorted(qdrant_ids - known)
    if orphans:
        result.error(
            "reconcile",
            f"{len(orphans)} doc_id(s) have vectors but NO metadata row: "
            f"{orphans[:5]}" + (" ..." if len(orphans) > 5 else ""),
        )
    else:
        result.ok("reconcile", f"all {len(qdrant_ids)} qdrant doc_ids match metadata")


def check_active_docs_have_images(
    settings: Settings,
    meta_db_active_docs: list[dict],
    result: DoctorResult,
) -> None:
    """Active doc should have an image directory at
    ``image_storage_path/<doc_id>/`` containing at least one file.

    Caller passes the list rather than re-fetching — saves one query
    when other checks already loaded it. Empty dir is a WARN not ERROR
    because text-only docs legitimately have no rendered pages (e.g.
    pre-MinerU bootstrap data, plain-text imports)."""
    root = Path(settings.image_storage_path)
    missing: list[str] = []
    for d in meta_db_active_docs:
        doc_dir = root / d["doc_id"]
        if not doc_dir.exists() or not any(doc_dir.iterdir()):
            missing.append(d["doc_id"])
    if missing:
        result.warn(
            "reconcile",
            f"{len(missing)} active doc(s) have no image files "
            f"(may be text-only; admin should verify): {missing[:5]}"
            + (" ..." if len(missing) > 5 else ""),
        )
    else:
        result.ok(
            "reconcile",
            f"all {len(meta_db_active_docs)} active docs have image dirs",
        )


async def check_deleted_docs_have_no_residue(
    meta_db,
    qdrant_store,
    result: DoctorResult,
) -> None:
    """Spec: deprecated/deleted docs should NOT have vectors. If they
    do, GC hasn't run or failed mid-loop. WARN (not ERROR) because
    soft-delete is a transient state — vectors disappear when retention
    expires + gc_deleted_docs.py runs. Suggests the GC command."""
    try:
        deleted_docs = await meta_db.list_documents(status="deleted")
    except Exception as exc:
        result.error("reconcile", f"failed to list deleted docs: {exc}")
        return
    with_residue: list[str] = []
    for d in deleted_docs:
        if _count_vectors_for_doc(qdrant_store, d["doc_id"]) > 0:
            with_residue.append(d["doc_id"])
    if with_residue:
        result.warn(
            "reconcile",
            f"{len(with_residue)} soft-deleted doc(s) still have vectors "
            f"(retention pending GC): {with_residue[:5]}"
            + (" ..." if len(with_residue) > 5 else "")
            + " — run: python scripts/gc_deleted_docs.py",
        )
    else:
        result.ok("reconcile", f"no vector residue from {len(deleted_docs)} soft-deleted doc(s)")


async def check_orphan_image_dirs(
    settings: Settings,
    meta_db,
    result: DoctorResult,
) -> None:
    """Image directories under ``image_storage_path`` with no matching
    documents row. Indicates a partial cleanup (DELETE failed before
    image purge, OR GC bug). WARN — disk leak is recoverable."""
    root = Path(settings.image_storage_path)
    if not root.exists():
        result.ok("reconcile", "image_storage_path empty — no orphan dirs to check")
        return
    try:
        all_docs = await meta_db.list_documents()
    except Exception as exc:
        result.error("reconcile", f"failed to list all docs: {exc}")
        return
    known = {d["doc_id"] for d in all_docs}
    orphans: list[str] = []
    for entry in root.iterdir():
        if entry.is_dir() and entry.name not in known:
            orphans.append(entry.name)
    if orphans:
        result.warn(
            "reconcile",
            f"{len(orphans)} image dir(s) without metadata row: "
            f"{orphans[:5]}" + (" ..." if len(orphans) > 5 else ""),
        )
    else:
        result.ok("reconcile", "no orphan image directories")


async def run_doctor(
    settings: Settings | None = None,
    *,
    deep_reconcile: bool = True,
    reconcile_sample_limit: int | None = None,
) -> DoctorResult:
    settings = settings or Settings()
    result = DoctorResult()
    check_settings(settings, result)
    check_storage_paths(settings, result)
    check_nav_paths(settings, result)
    await check_nav_orphans(settings, result)
    result.findings.append(await _check_auth(settings))
    await check_http_service(result, "mineru", settings.mineru_server_url)
    # PoC v1.1 model stack (replaces the legacy colqwen :8002 probe — that
    # service was retired when vision embedding moved onto the shared
    # multimodal embedding server). These cloud-hosted vllm servers reach a
    # local doctor run via SSH tunnel, so an unreachable result is a WARN
    # (check_http_service downgrades failures), not a hard ERROR.
    await check_http_service(result, "embedding", settings.multimodal_embedding_server_url)
    await check_http_service(result, "reranker", settings.reranker_server_url)
    await check_http_service(result, "sparse", settings.sparse_server_url)
    # Most OpenAI-compatible model servers do not expose /health consistently,
    # so use a config-level finding rather than a network probe for the LLM.
    result.ok("llm", f"agent endpoint configured: {settings.agent_url}")

    # Deep reconcile — opt-out (--no-reconcile) for fast operator checks.
    if deep_reconcile:
        await _run_reconcile_block(settings, result, reconcile_sample_limit)
    return result


async def _run_reconcile_block(
    settings: Settings,
    result: DoctorResult,
    sample_limit: int | None,
) -> None:
    """Five-check sequence. Each check is wrapped so one failing import
    or transient error doesn't abort the whole reconcile pass."""
    try:
        from kb.storage.metadata_db import MetadataDB
        from kb.storage.qdrant_store import QdrantStore
    except Exception as exc:
        result.error("reconcile", f"cannot import storage modules: {exc}")
        return

    meta = MetadataDB(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db",
    )
    try:
        await meta.init()
    except Exception as exc:
        result.error("reconcile", f"meta_db init failed: {exc}")
        return

    try:
        qdrant = QdrantStore(settings=settings)
    except Exception as exc:
        result.error("reconcile", f"qdrant client init failed: {exc}")
        await meta.aclose()
        return

    try:
        await check_qdrant_collections(settings, qdrant, result)
        try:
            active = await meta.list_documents(status="active")
        except Exception as exc:
            result.error("reconcile", f"list_documents(active) failed: {exc}")
            active = []
        if sample_limit:
            active_sampled = active[:sample_limit]
        else:
            active_sampled = active
        # Reuse list to avoid two trips for the image check.
        await check_active_docs_have_vectors(
            meta, qdrant, result, sample_limit=sample_limit,
        )
        check_active_docs_have_images(settings, active_sampled, result)
        await check_vectors_have_metadata(meta, qdrant, result)
        await check_deleted_docs_have_no_residue(meta, qdrant, result)
        await check_orphan_image_dirs(settings, meta, result)
    finally:
        await meta.aclose()
