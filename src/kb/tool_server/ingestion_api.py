from dataclasses import asdict, is_dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from kb.admin.maintenance import require_no_maintenance

SUPPORTED_INGESTION_SUFFIXES = {".pdf"}

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


class CreateIngestionJobRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=100)
    max_attempts: int = Field(default=3, ge=1, le=10)


def _parse_allowed_roots(raw: str) -> list[Path]:
    """Parse the comma-separated allowed_roots config into resolved Paths."""
    if not raw or not raw.strip():
        return []
    return [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]


def _validate_path_under_roots(raw_path: str, allowed_roots: list[Path]) -> Path:
    """Verify ``raw_path`` lies inside one of ``allowed_roots``.

    Returns the resolved Path on success. Raises HTTPException(400) on:
      - non-existent path
      - unsupported suffix
      - path outside every allowed root (when guard is configured)

    When allowed_roots is empty the guard is OFF — any existing PDF passes.

    TODO(user): decide the normalisation strategy for symlinks and case
    sensitivity. Two valid choices:

      (a) Use ``path.resolve()`` which follows symlinks. This protects
          against a symlink INSIDE an allowed root pointing OUTSIDE
          (e.g. ./test_doc/escape -> /etc/secrets.pdf). Tighter security
          but may break legitimate symlink usage (e.g. a symlink to a
          shared corpus on a different drive).

      (b) Use ``path.absolute()`` (no symlink follow). Looser — allows
          symlink workflows but a hostile symlink inside an allowed root
          becomes an escape vector. On Windows this is rare; on Linux
          more relevant.

    Implement the check in the body below (5-8 lines).
    """
    path = Path(raw_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {raw_path}")
    if path.suffix.lower() not in SUPPORTED_INGESTION_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {path.suffix}",
        )
    if not allowed_roots:
        # Guard disabled — accept any existing supported file.
        return path

    # Use resolve() so symlinks INSIDE an allowed root that point OUTSIDE
    # cannot be used as escape vectors. allowed_roots are already resolved
    # in _parse_allowed_roots, so the comparison is symlink-to-symlink safe.
    # On Windows this is rarely a problem (symlinks are uncommon); on Linux
    # this trades the symlink-as-shortcut workflow for tighter protection
    # against prompt-injection-driven path attacks via the agent.
    resolved = path.resolve()
    for root in allowed_roots:
        if resolved.is_relative_to(root):
            return path
    raise HTTPException(
        status_code=403,
        detail=f"Path outside ingestion_allowed_roots: {raw_path}",
    )


@router.post("/jobs", dependencies=[Depends(require_no_maintenance)])
async def create_ingestion_job(req: CreateIngestionJobRequest, request: Request):
    from kb.auth.context import get_current_user

    # B-1: owner_id MUST come from the auth-bound current_user, NEVER from
    # the request body. CreateIngestionJobRequest deliberately omits an
    # owner_id field; Pydantic's default extra="ignore" silently discards
    # any owner_id a client sends so a member cannot impersonate.
    user = get_current_user()
    settings = request.app.state.settings
    allowed_roots = _parse_allowed_roots(settings.ingestion_allowed_roots)
    for raw_path in req.paths:
        _validate_path_under_roots(raw_path, allowed_roots)
    job = await request.app.state.job_store.create_job(
        paths=req.paths,
        config={"max_attempts": req.max_attempts},
        owner_id=user.user_id,
    )
    return _to_dict(job)


@router.get("/jobs/{job_id}")
async def get_ingestion_job(job_id: str, request: Request):
    store = request.app.state.job_store
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    items = await store.list_job_items(job_id)
    return {
        "job": _to_dict(job),
        "items": [_to_dict(item) for item in items],
    }


@router.get("/jobs/{job_id}/events")
async def list_ingestion_job_events(
    job_id: str,
    request: Request,
    after_id: int = Query(default=0, ge=0),
):
    events = await request.app.state.job_store.list_events(
        job_id,
        after_id=after_id,
    )
    return {"events": [_to_dict(event) for event in events]}


@router.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_no_maintenance)])
async def cancel_ingestion_job(job_id: str, request: Request):
    from kb.auth.context import get_current_user
    from kb.auth.permissions import require_owner_or_admin

    user = get_current_user()
    store = request.app.state.job_store
    audit_log = request.app.state.audit_log

    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    # T3: ownership check BEFORE state mutation. Denial does not call
    # request_cancel — previously T2 would request_cancel on a missing
    # job_id and only then 404; the new order cleans that up and
    # prevents members from inadvertently mutating job state by hitting
    # the endpoint.
    await require_owner_or_admin(
        job, user,
        audit_log=audit_log,
        target_kind="job", target_id=job_id,
    )

    await store.request_cancel(job_id)
    job = await store.get_job(job_id)  # re-fetch post-cancel for status

    # T2 best-effort audit (A-3). Failure here must not break the 200.
    await audit_log.write(
        "job.cancel",
        user_id=user.user_id,
        target_kind="job",
        target_id=job_id,
        payload={"job_status_after": job.status},
    )
    return {"job": _to_dict(job)}


@router.post("/jobs/{job_id}/retry", dependencies=[Depends(require_no_maintenance)])
async def retry_ingestion_job(job_id: str, request: Request):
    """Requeue all failed items in this job.

    T3 follow-up (P1a review): mirrors the cancel handler's permission
    bracket — 404 first, then require_owner_or_admin, then mutate.
    Without this gate any member could resurrect another member's
    failed job, re-occupy the GPU queue, and clear the original owner's
    ``cancel_requested`` flag (sqlite_store.retry_failed_items resets it).
    """
    from kb.auth.context import get_current_user
    from kb.auth.permissions import require_owner_or_admin

    user = get_current_user()
    store = request.app.state.job_store
    audit_log = request.app.state.audit_log

    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    await require_owner_or_admin(
        job, user,
        audit_log=audit_log,
        target_kind="job", target_id=job_id,
    )

    retried = await store.retry_failed_items(job_id)
    return {"retried_items": retried}


def _to_dict(value) -> dict:
    if is_dataclass(value):
        return asdict(value)
    return dict(value)
