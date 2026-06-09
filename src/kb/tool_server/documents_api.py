from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from kb.audit.log import AuditWriteFailure
from kb.auth.context import get_current_user
from kb.auth.permissions import require_owner_or_admin
from kb.admin.maintenance import require_no_maintenance
from kb.observability import emit, emit_error
from kb.tool_server.security import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, request: Request):
    """Return doc metadata. T3 I-6: 404 for any non-active status —
    deleted (soft-delete), deprecated (replaced by newer version), or
    failed/staging. Read paths uniformly hide non-active rows; the row
    itself stays in the DB for audit + restore."""
    meta_db = request.app.state.meta_db
    doc = await meta_db.get_document(doc_id)
    if doc is None or doc.get("status") != "active":
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{doc_id}", dependencies=[Depends(require_no_maintenance)])
async def delete_document(doc_id: str, request: Request):
    """Soft-delete with two-phase audit (T5-4 refactor — see C-8).

    T5 change: this handler used to physically delete Qdrant vectors,
    image files, and nav entries during the main op. That made restore
    impossible — flipping status back to active left no content for
    the Agent to retrieve. v2 design moves all physical cleanup to
    scripts/gc_deleted_docs.py, run after retention expires.

    Phase 1: attempted (best-effort)
    Main op: mark_deleted (writes status='deleted' AND deleted_at=now,
             per Task 3 storage layer)
    Cache:   invalidate_all so cached search results drop the now-
             soft-deleted doc; the I-6 active gate also masks it from
             new reads
    Phase 2: completed (mandatory — DB → JSONL → 503)
    """
    user = get_current_user()
    meta_db = request.app.state.meta_db
    audit_log = request.app.state.audit_log

    doc = await meta_db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await require_owner_or_admin(
        doc, user,
        audit_log=audit_log,
        target_kind="document", target_id=doc_id,
    )

    # Phase 1: attempted (best-effort — failure must not block main op)
    await audit_log.write_attempted(
        "doc.delete",
        user_id=user.user_id,
        target_kind="document",
        target_id=doc_id,
        payload={
            "doc_name": doc.get("doc_name"),
            "doc_status_before": doc.get("status"),
        },
    )

    # ── Main op: pure soft delete ───────────────────────────────────
    # mark_deleted writes deleted_at (Task 3 change); GC consumes it.
    # Qdrant vectors, image files, and nav entries are LEFT INTACT so
    # restore (T5-6) is meaningful. GC (T5-7) does the physical purge
    # after retention expires.
    await meta_db.mark_deleted(doc_id)

    # Search cache must drop — keys are (query, filters) hashes, can't
    # evict per-doc. In-process scope.
    from kb.search.cache import invalidate_all
    invalidate_all()

    emit(
        "document.soft_deleted",   # NB: was "document.deleted" pre-T5
        doc_id=doc_id,             # so log scanners can distinguish
        doc_name=doc.get("doc_name", ""),   # soft vs GC's eventual hard delete
    )
    # ── End main op ─────────────────────────────────────────────────

    # Phase 2: completed (mandatory — DB → JSONL → 503)
    try:
        await audit_log.write_completed(
            "doc.delete",
            user_id=user.user_id,
            target_kind="document",
            target_id=doc_id,
        )
    except AuditWriteFailure as exc:
        emit_error(
            "audit.completed_lost", exc,
            action="doc.delete", user_id=user.user_id, target_id=doc_id,
        )
        raise HTTPException(
            status_code=503,
            detail="operation completed but audit failed; "
                   "admin must reconstruct from trace",
        )

    # mark_deleted just wrote deleted_at; re-fetch to surface it in the
    # response. The if-None branch defends against a race where T5-7 GC
    # (separate process) could physically purge the row between mark_deleted
    # and this read — extremely unlikely under SQLite, but cheap insurance.
    _row = await meta_db.get_document(doc_id, include_deleted_at=True)
    return {
        "doc_id": doc_id,
        "status": "deleted",
        "deleted_at": _row["deleted_at"] if _row else None,
    }


@router.post(
    "/documents/{doc_id}/restore",
    dependencies=[Depends(require_no_maintenance)],
)
async def restore_document(doc_id: str, request: Request):
    """Restore a soft-deleted document.

    Permission: owner OR admin. NULL-owner (pre-T3) → admin only.
    Audit: best-effort doc.restore action (NOT two-phase — it's the
    inverse of delete; restore failure means the row stays deleted
    which is a safe state, no audit-truth gap to worry about).

    Body / args: none. URL path only.
    Response: {doc_id, status} on 200; 404 if doc missing; 400 if doc
    is not in 'deleted' state; 403 if caller is not owner/admin.

    T5-4 invariant: this endpoint is meaningful ONLY because the DELETE
    handler stopped physically purging Qdrant/images/nav. If you change
    DELETE's behavior, this comment is your reminder to revisit C-8.
    """
    user = get_current_user()
    meta_db = request.app.state.meta_db
    audit_log = request.app.state.audit_log

    # include_deleted_at=True so we can see soft-deleted rows and the
    # restore-target's audit row knows the original deleted_at timestamp.
    doc = await meta_db.get_document(doc_id, include_deleted_at=True)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc["status"] != "deleted":
        raise HTTPException(
            status_code=400,
            detail="Document is not in 'deleted' state",
        )

    await require_owner_or_admin(
        doc, user,
        audit_log=audit_log,
        target_kind="document", target_id=doc_id,
    )

    await meta_db.restore_document(doc_id)

    # T5: invalidate search cache so the now-active doc shows up in
    # subsequent queries (cache keys are query+filter hashes; we can't
    # evict per-doc, so drop the whole cache like DELETE does).
    from kb.search.cache import invalidate_all
    invalidate_all()

    # Best-effort audit (A-3). doc.restore is NOT in TWO_PHASE_ACTIONS
    # by design — restore failure leaves the row deleted, which is a
    # safe state; no two-phase audit-truth gap to worry about.
    await audit_log.write(
        "doc.restore",
        user_id=user.user_id,
        target_kind="document",
        target_id=doc_id,
        payload={"doc_name": doc.get("doc_name")},
    )

    return {"doc_id": doc_id, "status": "active"}


@router.get("/documents")
async def list_docs(request: Request, topic: Optional[str] = Query(None)):
    meta_db = request.app.state.meta_db
    active_ids = await meta_db.list_active_doc_ids()
    docs = []
    for doc_id in active_ids:
        doc = await meta_db.get_document(doc_id)
        if doc:
            docs.append(doc)
    if topic:
        topic_lower = topic.lower()
        docs = [d for d in docs if topic_lower in d["doc_name"].lower()]
    return docs
