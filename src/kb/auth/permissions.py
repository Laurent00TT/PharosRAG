"""Auth permission helpers — require_role / require_owner_or_admin.

Spec §6.2 + I-11 + I-13. Two layers:

- ``require_role(role)`` is a FastAPI dependency factory. Used on
  admin-only endpoints (POST /admin/maintenance_mode, etc.). Admin
  passes any role check (role hierarchy).

- ``require_owner_or_admin(target, user, *, audit_log, target_kind,
  target_id)`` is an async helper called from handler bodies after
  the relevant target (document / job) has been fetched. On the 403
  path it emits a best-effort ``authz.denied`` audit row (I-13:
  authentication vs authorization distinction). Owner-check applies
  to:
   - documents.owner_id  (DELETE /documents/{id})
   - ingestion_jobs.owner_id  (POST /jobs/{id}/cancel)
   - nav entries' parent doc owner_id  (MCP write tools)

NULL or '_system' owner_id (pre-T3 backfill rows) means "admin only"
per I-11 — silent member 403 path emits ``authz.denied`` with
``reason: null_owner_admin_only`` so doctor can distinguish.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from kb.auth.context import get_current_user
from kb.auth.users import User
from kb.observability.trace import emit_error

logger = logging.getLogger(__name__)


_NULL_OWNER_SENTINELS = frozenset({None, "_system"})


def require_role(role: str):
    """FastAPI dependency factory: ensure current_user has the given
    role. Admin passes any check (admin > member hierarchy)."""
    async def _check() -> None:
        user = get_current_user()
        if user.role == "admin":
            return
        if user.role != role:
            raise HTTPException(
                status_code=403,
                detail=f"requires role={role}",
            )
    return _check


async def require_owner_or_admin(
    target: dict[str, Any] | Any,
    user: User,
    *,
    audit_log,
    target_kind: str,
    target_id: str,
) -> None:
    """Allow if user is admin OR the target's owner_id matches user.user_id.
    NULL / '_system' owner_id is admin-only (I-11).

    On 403:
      - emit best-effort ``authz.denied`` audit (I-13 — note the action
        name is ``authz.denied``, NOT ``auth.failed``: authn succeeded)
      - raise HTTPException(403)

    ``target`` is treated as a mapping-like: it can be a dict (from
    ``meta_db.get_document`` return) or a dataclass (IngestionJob).
    We probe ``.get('owner_id')`` first, then attribute access.
    """
    if user.role == "admin":
        return

    owner_id = _extract_owner_id(target)
    if owner_id in _NULL_OWNER_SENTINELS:
        await _emit_authz_denied(
            audit_log, user, target_kind, target_id,
            reason="null_owner_admin_only",
        )
        raise HTTPException(
            status_code=403,
            detail="pre-T3 document/job (no owner) — admin only",
        )

    if owner_id == user.user_id:
        return

    await _emit_authz_denied(
        audit_log, user, target_kind, target_id,
        reason="not_owner",
    )
    raise HTTPException(
        status_code=403,
        detail=f"not the owner of this {target_kind}",
    )


def _extract_owner_id(target: dict | Any) -> str | None:
    if isinstance(target, dict):
        return target.get("owner_id")
    return getattr(target, "owner_id", None)


async def _emit_authz_denied(
    audit_log, user: User, target_kind: str, target_id: str, *, reason: str,
) -> None:
    """Best-effort (A-3): swallow audit failure so 403 is the visible
    outcome even when audit DB blips. emit_error trace already records
    the audit failure for forensics."""
    if audit_log is None:
        return
    try:
        await audit_log.write(
            "authz.denied",
            user_id=user.user_id,
            target_kind=target_kind,
            target_id=target_id,
            payload={"reason": reason, "attempted_role": user.role},
        )
    except Exception as exc:
        emit_error(
            "audit.best_effort_failed", exc,
            action="authz.denied", target_id=target_id, user_id=user.user_id,
        )
