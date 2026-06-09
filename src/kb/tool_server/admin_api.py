"""Admin endpoints — maintenance_mode flag + drain loop (T5).

POST /admin/maintenance_mode  body {on: bool}  admin-only
GET  /admin/maintenance_mode                   admin-only, ALWAYS available

The POST flips the flag in MaintenanceState (SQLite-backed). On ON, it
also waits up to 5min for active worker claims to drain so backups see
a quiescent system. drained:false in the response means a worker is
stuck — admin / backup script must decide.

Neither endpoint uses Depends(require_no_maintenance) — otherwise once
maintenance is ON you couldn't turn it off (C-3).
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from kb.auth.context import get_current_user
from kb.auth.permissions import require_role
from kb.tool_server.security import verify_api_key

if TYPE_CHECKING:
    from kb.jobs.sqlite_store import SQLiteJobStore


# Tunables exposed at module level so tests can monkeypatch.
_DRAIN_TIMEOUT_SECONDS: float = 300.0
_DRAIN_POLL_SECONDS: float = 1.0


router = APIRouter(dependencies=[Depends(verify_api_key)])


class MaintenancePayload(BaseModel):
    on: bool


async def _count_active_claims_via_app(request: Request) -> int:
    """Single-arity wrapper so tests can replace the count source
    without monkeypatching SQLiteJobStore directly. Production reads
    request.app.state.job_store.count_active_claims() — only items with
    a NON-EXPIRED lease, matching C-4 drain semantic."""
    store: SQLiteJobStore = request.app.state.job_store
    return await store.count_active_claims()


@router.get(
    "/admin/maintenance_mode",
    dependencies=[Depends(require_role("admin"))],
)
async def get_maintenance_mode(request: Request) -> dict:
    """Read the flag. ALWAYS available, even when maintenance is on —
    backup scripts probe this to verify the flag they set is live."""
    state = request.app.state.maintenance
    return await state.get_state()


@router.post(
    "/admin/maintenance_mode",
    dependencies=[Depends(require_role("admin"))],
)
async def set_maintenance_mode(
    body: MaintenancePayload,
    request: Request,
) -> dict:
    """Flip the flag. On ON, also wait up to 5min for active workers
    to drain. Response includes ``drained: bool`` so callers can
    decide whether to proceed with the backup."""
    user = get_current_user()
    state = request.app.state.maintenance

    if body.on:
        # Sticky design: set the flag BEFORE draining. If the endpoint
        # crashes (SIGKILL / OOM) during drain, maintenance stays ON.
        # Operators recover by re-issuing the same POST with {on: false}.
        # Reversing the order (drain then set_on) would let writes slip
        # through during the drain window — exactly what we're avoiding.
        await state.set_on(user_id=user.user_id)
        # Drain loop — bounded by _DRAIN_TIMEOUT_SECONDS.
        # Use time.monotonic() rather than asyncio.get_event_loop().time()
        # to avoid the Python 3.12 deprecation warning for the latter when
        # called outside of a running event loop context.
        deadline = time.monotonic() + _DRAIN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                active = await _count_active_claims_via_app(request)
            except Exception:  # noqa: BLE001 — drain MUST not crash the endpoint
                # Counting failed transiently; treat as "still active" and
                # let the loop try again. We do NOT short-circuit out — a
                # transient DB blip shouldn't claim drain succeeded.
                active = None
            if active == 0:
                snap = await state.get_state()
                return {"on": True, "drained": True, "set_at": snap["set_at"]}
            await asyncio.sleep(_DRAIN_POLL_SECONDS)
        # Timed out
        count_failed = False
        try:
            active = await _count_active_claims_via_app(request)
        except Exception:  # noqa: BLE001
            active = None
            count_failed = True
        return {
            "on": True,
            "drained": False,
            "still_active": active,
            "count_failed": count_failed,
        }
    else:
        await state.set_off(user_id=user.user_id)
        snap = await state.get_state()
        return {"on": False, "set_at": snap["set_at"]}
