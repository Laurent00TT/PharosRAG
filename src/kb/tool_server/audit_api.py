"""GET /audit — paginated audit log read endpoint.

Scoping (spec §6.2 permission matrix):
- member: forced to current_user.user_id regardless of ?user_id (silently
  ignored, no 403 — admin can't cause data leak by typoing a query param)
- admin: free to use ?user_id=... or omit for all rows

Scoping happens in the handler; AuditLog.query stays role-agnostic so it
can be reused by scripts, doctor CLI (T4), and other backends.

Time parsing (P1-2): ISO-8601 with optional Z suffix -> aware datetime ->
astimezone(timezone.utc) -> drop tzinfo to match DB-naive UTC storage.
The wrong way (astimezone(tz=None)) converts to LOCAL time and produces
an N-hour skew on non-UTC machines.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from kb.auth.context import get_current_user

router = APIRouter()


def _parse_since(since: str) -> datetime:
    try:
        dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad since: {exc}")
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@router.get("/audit")
async def list_audits(
    request: Request,
    action: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    target_id: Optional[str] = Query(default=None),
    since: Optional[str] = Query(default=None, description="ISO-8601 datetime"),
    limit: Optional[int] = Query(default=None, ge=1),
):
    user = get_current_user()
    audit_log = request.app.state.audit_log
    settings = request.app.state.settings

    if user.role == "admin":
        effective_user_id: Optional[str] = user_id   # None = all
    else:
        effective_user_id = user.user_id              # forced, ?user_id ignored

    since_dt: Optional[datetime] = _parse_since(since) if since else None

    requested = limit if limit is not None else settings.audit_default_limit
    effective_limit = min(requested, settings.audit_max_limit)

    rows = await audit_log.query(
        user_id=effective_user_id,
        action=action,
        target_id=target_id,
        since=since_dt,
        limit=effective_limit,
    )
    return {
        "rows": [
            {
                "event_id": r["event_id"],
                "user_id":  r["user_id"],
                "action":   r["action"],
                "target_kind": r["target_kind"],
                "target_id": r["target_id"],
                "ts":       r["ts"],
                "payload":  json.loads(r["payload_json"] or "{}"),
            }
            for r in rows
        ],
        "limit": effective_limit,
    }
