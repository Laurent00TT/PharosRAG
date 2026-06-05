from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kb.admin.maintenance import require_no_maintenance
from kb.auth.context import get_current_user
from kb.tool_server.ask_api import AskRequest, ask as ask_endpoint
from kb.tool_server.documents_api import (
    delete_document as _delete_document,
    restore_document as _restore_document,
)
from kb.tool_server.feedback_api import FeedbackRequest
from kb.tool_server.mcp_resources import _fetch_page_payload
from kb.tool_server.security import verify_api_key

router = APIRouter(
    prefix="/ui/api",
    tags=["ui"],
    dependencies=[Depends(verify_api_key)],
)


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class PageRangePayload(BaseModel):
    page_start: int = Field(ge=0)
    page_end: int = Field(ge=0)


def _safe_upload_dest(filename: str) -> Path:
    """Relative destination ``<uuid>/<original-name>.pdf`` under the UI upload dir.

    The ORIGINAL filename is preserved (Unicode included) so the ingestion
    pipeline's version-supersession — which deprecates prior docs whose
    ``doc_name == path.name`` for the SAME owner — treats re-uploading a
    same-named file as a NEW VERSION that replaces the old one (matching
    CLI/worker ingest). The UUID lives in the PARENT directory rather than
    the filename, so two same-named uploads never collide on disk yet still
    share the doc_name that drives replacement.
    """
    # Path(...).name strips directory components → no path traversal. Keep
    # Unicode letters (e.g. 中文名); drop only control chars / stray separators.
    base = Path(filename or "upload.pdf").name
    base = re.sub(r"[\x00-\x1f/\\]+", "", base).strip()
    stem = Path(base).stem.strip() or "upload"
    return Path(uuid.uuid4().hex) / f"{stem}.pdf"


def _text_preview(raw: str, limit: int = 180) -> str:
    text = raw.strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("text", "caption", "title", "heading"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
            else:
                html = parsed.get("html")
                if isinstance(html, str) and html.strip():
                    text = re.sub(r"<[^>]*>", " ", html)
                else:
                    text = ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


async def _require_active_doc(request: Request, doc_id: str) -> dict:
    meta_db = request.app.state.meta_db
    doc = await meta_db.get_document(doc_id)
    if doc is None or doc.get("status") != "active":
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


async def _nav_entries(request: Request, doc_id: str | None = None):
    nav_store = getattr(request.app.state, "nav_store", None)
    if nav_store is None:
        return []
    return await nav_store.list_entries(doc_id=doc_id)


@router.get("/me")
async def me() -> dict:
    user = get_current_user()
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "key_prefix": user.key_prefix,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.get("/status")
async def status(request: Request) -> dict:
    meta_db_ready = getattr(request.app.state, "meta_db", None) is not None
    evidence_ready = getattr(request.app.state, "engine", None) is not None
    nav_ready = (
        getattr(request.app.state, "nav_store", None) is not None
        and getattr(request.app.state, "nav_search", None) is not None
    )
    hybrid_ready = getattr(request.app.state, "hybrid_navigator", None) is not None
    active_ids: list[str] = []
    document_count = 0
    nav_entry_count = 0
    nav_doc_count = 0
    if meta_db_ready:
        active_ids = await request.app.state.meta_db.list_active_doc_ids()
        document_count = len(active_ids)
    if nav_ready:
        if meta_db_ready:
            counts_by_doc = await request.app.state.nav_store.count_entries_by_doc(active_ids)
            nav_entry_count = sum(counts_by_doc.values())
            nav_doc_count = len(counts_by_doc)
        else:
            nav_entry_count = await request.app.state.nav_store.count_entries()
            nav_doc_count = len(await request.app.state.nav_store.list_doc_ids())
    maintenance = getattr(request.app.state, "maintenance", None)
    maintenance_state = await maintenance.get_state() if maintenance is not None else None
    return {
        "meta_db_ready": meta_db_ready,
        "evidence_ready": evidence_ready,
        "nav_ready": nav_ready,
        "hybrid_ready": hybrid_ready,
        "document_count": document_count,
        "nav_entry_count": nav_entry_count,
        "nav_doc_count": nav_doc_count,
        "mcp_write_tools_enabled": bool(
            getattr(request.app.state.settings, "mcp_enable_write_tools", False)
        ),
        "maintenance": maintenance_state,
    }


@router.get("/documents")
async def documents(request: Request) -> dict:
    meta_db = request.app.state.meta_db
    active_ids = await meta_db.list_active_doc_ids()
    nav_counts = await request.app.state.nav_store.count_entries_by_doc(active_ids)
    docs: list[dict] = []
    for doc_id in active_ids:
        doc = await meta_db.get_document(doc_id, include_ingested_at=True)
        if doc is None:
            continue
        nav_entry_count = nav_counts.get(doc_id, 0)
        ingested_at = doc.get("ingested_at")
        docs.append({
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "version": doc.get("version"),
            "status": doc.get("status"),
            "owner_id": doc.get("owner_id"),
            "nav_indexed": nav_entry_count > 0,
            "nav_entry_count": nav_entry_count,
            "resource_uri": f"kb://documents/{doc_id}",
            "ingested_at": ingested_at.isoformat() if ingested_at else None,
            "has_source": bool(doc.get("has_source")),
            "source_bytes": doc.get("source_bytes"),
        })
    return {"documents": docs, "total": len(docs)}


@router.get("/documents/{doc_id}/toc")
async def document_toc(doc_id: str, request: Request) -> dict:
    await _require_active_doc(request, doc_id)
    entries = await _nav_entries(request, doc_id=doc_id)
    return {
        "doc_id": doc_id,
        "entries": [
            {
                "entry_id": entry.entry_id,
                "label": entry.label,
                "entry_type": entry.entry_type,
                "page_start": entry.page_start,
                "page_end": entry.page_end,
                "resource_uris": entry.resource_uris,
                "parent_entry_id": entry.parent_entry_id,
                "order_index": entry.order_index,
            }
            for entry in entries
        ],
        "total": len(entries),
        "contains_generated_content": False,
    }


@router.post("/hybrid_search")
async def hybrid_search(req: HybridSearchRequest, request: Request) -> dict:
    navigator = getattr(request.app.state, "hybrid_navigator", None)
    if navigator is None:
        raise HTTPException(status_code=503, detail="Hybrid navigator is not initialized")
    response = await navigator.search(query=req.query, top_k=req.top_k)
    payload = asdict(response)
    for hit in payload.get("evidence_hits", []):
        page_payload = await _fetch_page_payload(
            request.app.state, hit["doc_id"], hit["page_num"]
        )
        if page_payload.get("status") == "ok":
            hit["text_preview"] = _text_preview(page_payload.get("text", ""))
            hit["section"] = " / ".join(page_payload.get("heading_path", []))
    payload["contains_generated_content"] = False
    return payload


def _page_preview_from_payload(payload: dict) -> dict:
    return {
        "status": payload.get("status"),
        "safety": payload.get("safety"),
        "doc_id": payload.get("doc_id"),
        "doc_name": payload.get("doc_name", ""),
        "page_num": payload.get("page_num"),
        "page_type": payload.get("page_type", "text"),
        "heading_path": payload.get("heading_path", []),
        "resource_uri": (
            f"kb://documents/{payload.get('doc_id')}/pages/{payload.get('page_num')}"
        ),
        "image_resource_uri": payload.get("image_resource_uri", ""),
        "evidence": {
            "text": payload.get("text", ""),
            "text_truncated": payload.get("text_truncated", False),
            "figure_caption": payload.get("figure_caption", ""),
            "figure_index": payload.get("figure_index"),
            "image_url": payload.get("image_url", ""),
        },
        "hints": {
            "generated_description": payload.get("generated_description", ""),
        },
        "evidence_fields": payload.get("evidence_fields", []),
        "hint_fields": payload.get("hint_fields", []),
    }


@router.get("/documents/{doc_id}/pages/{page_num}")
async def page_preview(doc_id: str, page_num: int, request: Request) -> dict:
    await _require_active_doc(request, doc_id)
    payload = await _fetch_page_payload(request.app.state, doc_id, page_num)
    if payload.get("status") == "page_not_found":
        raise HTTPException(status_code=404, detail="Page not found")
    if payload.get("status") != "ok":
        raise HTTPException(status_code=503, detail=payload)
    return _page_preview_from_payload(payload)


@router.get("/documents/{doc_id}/pages/{page_num}/image")
async def page_image(doc_id: str, page_num: int, request: Request):
    """Stream the rendered PNG for a page so the browser can <img> it.

    The MCP resource at the same URI returns a JSON envelope with a base64
    data URL; that's the right shape for agents but heavy for browsers.
    This endpoint returns the raw bytes via FileResponse so the browser
    can cache it normally.
    """
    await _require_active_doc(request, doc_id)
    image_store = getattr(request.app.state, "image_store", None)
    if image_store is None:
        raise HTTPException(status_code=503, detail="Image store not initialized")
    path_str = await image_store.get_url(doc_id, page_num)
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.get("/documents/{doc_id}/source")
async def document_source(doc_id: str, request: Request):
    """Stream the retained original PDF so the workbench can view/download it.

    Navigation-to-source: the KB stores the source PDF at ingest (gated by
    STORE_SOURCE_DOCUMENTS); this serves it inline so the browser's native
    viewer renders it. 404 when no source was kept (older docs / flag off).
    """
    doc = await _require_active_doc(request, doc_id)
    source_store = getattr(request.app.state, "source_store", None)
    if source_store is None:
        raise HTTPException(status_code=503, detail="Source store not initialized")
    path = source_store.get_path(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No source document stored for this document")
    # Serve under the human-readable original name. FileResponse encodes it
    # per RFC 5987 (filename* for Unicode), so non-ASCII doc names stay safe
    # in the header — no manual escaping needed.
    download_name = doc.get("doc_name") or f"{doc_id}.pdf"
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=download_name,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.post("/documents/{doc_id}/range_preview")
async def range_preview(doc_id: str, body: PageRangePayload, request: Request) -> dict:
    await _require_active_doc(request, doc_id)
    if body.page_start > body.page_end:
        raise HTTPException(status_code=400, detail="page_start must be <= page_end")
    if body.page_end - body.page_start + 1 > 20:
        raise HTTPException(status_code=400, detail="range preview supports up to 20 pages")
    pages = []
    for page_num in range(body.page_start, body.page_end + 1):
        payload = await _fetch_page_payload(request.app.state, doc_id, page_num)
        if payload.get("status") == "ok":
            pages.append(_page_preview_from_payload(payload))
    return {
        "doc_id": doc_id,
        "page_start": body.page_start,
        "page_end": body.page_end,
        "pages": pages,
    }


@router.delete("/documents/{doc_id}", dependencies=[Depends(require_no_maintenance)])
async def ui_delete_document(doc_id: str, request: Request) -> dict:
    """Soft-delete a document (UI wrapper over documents_api.delete_document).

    Owner-or-admin is enforced inside the delegate; Qdrant vectors, page
    images and nav entries are retained so a later restore is meaningful.
    """
    return await _delete_document(doc_id, request)


@router.post("/documents/{doc_id}/restore", dependencies=[Depends(require_no_maintenance)])
async def ui_restore_document(doc_id: str, request: Request) -> dict:
    """Restore a soft-deleted document (UI wrapper over documents_api.restore_document)."""
    return await _restore_document(doc_id, request)


@router.post("/ask")
async def ui_ask(req: AskRequest, request: Request) -> dict:
    return await ask_endpoint(req, request)


@router.post("/feedback")
async def ui_feedback(req: FeedbackRequest, request: Request) -> dict:
    feedback_db = getattr(request.app.state, "feedback_db", None)
    if feedback_db is None:
        raise HTTPException(status_code=503, detail="Feedback DB is not initialized")
    updated = await feedback_db.update_feedback(
        query_id=req.query_id,
        was_helpful=req.was_helpful,
        comment=req.comment,
        expected_doc=req.expected_doc,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="query_id not found")
    return {"updated": True}


@router.post("/upload", dependencies=[Depends(require_no_maintenance)])
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")
    user = get_current_user()
    upload_dir = Path(request.app.state.settings.qdrant_path) / "ui_uploads"
    stored_path = upload_dir / _safe_upload_dest(file.filename)
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")
    stored_path.write_bytes(contents)
    job = await request.app.state.job_store.create_job(
        paths=[str(stored_path)],
        config={"max_attempts": 3, "source": "ui_upload"},
        owner_id=user.user_id,
    )
    return {
        "stored_path": str(stored_path),
        "job": asdict(job),
    }


# Cap a single multipart batch so one request can't queue an unbounded number
# of files (mirrors the ingestion API's CreateIngestionJobRequest max_length).
_MAX_BATCH_FILES = 100


@router.post("/upload_batch", dependencies=[Depends(require_no_maintenance)])
async def upload_pdf_batch(
    request: Request, files: list[UploadFile] = File(...)
) -> dict:
    """Store many PDFs at once and queue them as ONE job with one item per
    file. Same per-file validation + same-name version replacement as /upload,
    but reuses the job store's job->items model so the whole batch is tracked
    under a single job_id (the UI polls one job for N/M progress).

    Invalid files are SKIPPED (reported in ``rejected``) rather than failing
    the whole batch — one typo'd non-PDF shouldn't sink 49 good uploads. The
    request only 400s when NOTHING valid remains.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)} (max {_MAX_BATCH_FILES} per batch)",
        )
    user = get_current_user()
    upload_dir = Path(request.app.state.settings.qdrant_path) / "ui_uploads"
    stored: list[dict] = []
    paths: list[str] = []
    rejected: list[dict] = []
    for f in files:
        name = f.filename or "upload.pdf"
        if Path(name).suffix.lower() != ".pdf":
            rejected.append({"filename": name, "error": "not a .pdf"})
            continue
        contents = await f.read()
        if not contents:
            rejected.append({"filename": name, "error": "empty file"})
            continue
        if not contents.startswith(b"%PDF-"):
            rejected.append({"filename": name, "error": "not a valid PDF"})
            continue
        dest = upload_dir / _safe_upload_dest(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(contents)
        paths.append(str(dest))
        stored.append({"filename": name, "stored_path": str(dest)})
    if not paths:
        raise HTTPException(
            status_code=400,
            detail=f"No valid PDFs in upload ({len(rejected)} rejected)",
        )
    job = await request.app.state.job_store.create_job(
        paths=paths,
        config={"max_attempts": 3, "source": "ui_upload_batch"},
        owner_id=user.user_id,
    )
    return {
        "stored": stored,
        "rejected": rejected,
        "job": asdict(job),
    }
