"""Doctor team view — admin's at-a-glance status panel.

Pure aggregation over existing stores. NEVER writes; never holds locks
across multiple statements. If a single section's data source is
unavailable (e.g. audit_log inaccessible), the section renders as
"[— data unavailable]" but the rest still works (C-5 invariant).

The module exposes two faces:
  - ``gather_team_view(...)`` collects everything into a dict; tests
    can assert structure directly without touching the renderer.
  - ``render_team_text(view)`` and ``team_view_to_dict(view)`` shape
    the dict for human/machine output.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from kb.storage.metadata_db import OWNER_ID_SYSTEM

logger = logging.getLogger(__name__)


# Spec sample uses these labels — keep them stable so doctor output stays
# greppable in incident reports.
_DESTRUCTIVE_ACTIONS = (
    "doc.delete.attempted", "doc.delete.completed",
    "doc.reassign.attempted", "doc.reassign.completed",
    "user.create", "user.disable", "user.role_change", "user.key_reset",
)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def gather_team_view(
    *,
    meta,
    audit,
    jobs,
    retention_days: int = 30,
    maintenance=None,
) -> dict[str, Any]:
    """Aggregate the team view dict.

    Each section is computed independently and wrapped in a try/except
    so one broken source doesn't kill the whole panel (C-5).

    ``maintenance`` (optional MaintenanceState) is read once for the
    "Maintenance" section. None = section unavailable (caller didn't
    wire it; not a bug — pre-T5 doctors didn't have it).
    """
    now = _now()
    one_day_ago = now - timedelta(days=1)
    seven_days_ago = now - timedelta(days=7)

    view: dict[str, Any] = {"as_of": now, "retention_days": retention_days}

    # ── maintenance flag (T5 carry-forward) ────────────────────────────
    # Single point of truth for "is server in maintenance right now" +
    # who/when toggled it. Backup/GC scripts also probe the HTTP endpoint
    # but operators looking at doctor want it inline alongside queue/
    # destructive views. Best-effort: a misconfigured state shouldn't
    # blank the whole panel.
    if maintenance is None:
        view["maintenance"] = None
    else:
        try:
            snap = await maintenance.get_state()
            view["maintenance"] = {
                "on": bool(snap.get("on")),
                "set_at": snap.get("set_at"),
                "set_by_user_id": snap.get("set_by_user_id"),
            }
        except Exception as exc:  # noqa: BLE001 — C-5 best-effort
            logger.warning("doctor team maintenance failed: %s", exc)
            view["maintenance"] = None

    # ── per-user 24h activity (P1-6 review fix: server-side GROUP BY) ──
    try:
        action_to_label = {
            "doc.ingest": "ingest",
            "job.cancel": "cancel",
            "job.failed": "failed",
        }
        counts = await audit.count_by_action_user(
            actions=list(action_to_label.keys()),
            since=one_day_ago,
        )
        per_user: dict[str | None, dict[str, int]] = {}
        for user_id, by_action in counts.items():
            bucket = {"ingest": 0, "cancel": 0, "failed": 0}
            for action, n in by_action.items():
                bucket[action_to_label[action]] = n
            per_user[user_id] = bucket
        view["activity_24h"] = [
            {"user_id": uid, **bucket}
            for uid, bucket in sorted(per_user.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))
        ]
    except Exception as exc:  # noqa: BLE001 — C-5 best-effort
        logger.warning("doctor team activity_24h failed: %s", exc)
        view["activity_24h"] = None

    # ── queue status (active workers + stale claims + queued by owner) ─
    # Review fix P1#2: jobs may be None when ingestion_jobs.db is absent
    # (doctor's caller pre-checked). Treat that as "section unavailable"
    # without spamming a warning, since None is intentional not erroneous.
    # Review fix P1#1: split CLAIMED items into active vs stale
    # (lease-expired) so a dead worker doesn't show as still running.
    if jobs is None:
        view["active_workers"] = None
        view["stale_claims"] = None
        view["queue_by_owner"] = None
    else:
        try:
            claimed = await jobs.list_claimed_items()
            view["active_workers"] = [
                {
                    "item_id": row["item_id"],
                    "job_id": row["job_id"],
                    "owner_id": row["owner_id"],
                    "locked_by": row["locked_by"],
                    "claimed_at": row["claimed_at"],
                }
                for row in claimed if not row["is_stale"]
            ]
            view["stale_claims"] = [
                {
                    "item_id": row["item_id"],
                    "job_id": row["job_id"],
                    "owner_id": row["owner_id"],
                    "locked_by": row["locked_by"],
                    "claimed_at": row["claimed_at"],
                    "locked_until": row["locked_until"],
                }
                for row in claimed if row["is_stale"]
            ]
            # include_reclaimable=True so "Queued" matches what the next
            # claim_next_item round will actually pick up (queued + stale).
            view["queue_by_owner"] = await jobs.count_queued_by_owner(
                include_reclaimable=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("doctor team queue failed: %s", exc)
            view["active_workers"] = None
            view["stale_claims"] = None
            view["queue_by_owner"] = None

    # ── destructive actions 7d ─────────────────────────────────────────
    # Single audit.query(actions=[...]) — collapses the pre-T4 8-serial-
    # query pattern via `WHERE action IN (...)`. The DB sort is already
    # newest-first; we cap at 50 in the SELECT (limit) and skip the
    # Python-side sort. 8x fewer round-trips on every doctor invocation.
    try:
        view["destructive_7d"] = await audit.query(
            actions=list(_DESTRUCTIVE_ACTIONS),
            since=seven_days_ago,
            limit=50,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("doctor team destructive_7d failed: %s", exc)
        view["destructive_7d"] = None

    # ── soft-deleted inventory (T5: deleted_at is source of truth) ─────
    # T5 closed the T4 P1-5 carry-forward: documents.deleted_at is now
    # written by mark_deleted() and is the primary timestamp source.
    # Audit fallback still exists for pre-T5 rows whose deleted_at is
    # NULL (they were deleted before the column existed). The limit=2000
    # is kept to bound the fallback query; KBs with >2000 lifetime deletes
    # will silently show "unknown delete time" only for the oldest rows.
    try:
        deleted_docs = await meta.list_documents(status="deleted")
        # T5: prefer documents.deleted_at as truth (closes T4 P1-5
        # carry-forward). Fall back to audit_log lookup only for
        # pre-T5 rows whose deleted_at is still NULL.
        # Audit fallback only fires if there's at least one pre-T5 row.
        # Fresh post-T5 installs skip the query entirely.
        has_pre_t5_rows = any(d.get("deleted_at") is None for d in deleted_docs)
        delete_ts_by_doc: dict[str, str] = {}
        if has_pre_t5_rows:
            delete_audits = await audit.query(action="doc.delete.completed", limit=2000)
            for r in delete_audits:
                tid = r.get("target_id")
                if tid:
                    delete_ts_by_doc.setdefault(tid, r["ts"])
        trash = []
        for d in deleted_docs:
            ts = d.get("deleted_at")
            if ts is None:
                ts_str = delete_ts_by_doc.get(d["doc_id"])
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts.tzinfo:
                            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                    except ValueError:
                        ts = None
            if ts is not None:
                days_elapsed = (now - ts).days
                remaining = max(retention_days - days_elapsed, 0)
            else:
                remaining = retention_days
            trash.append({
                "doc_id": d["doc_id"], "doc_name": d["doc_name"],
                "owner_id": d["owner_id"],
                "deleted_at": ts,
                "retention_days_remaining": remaining,
            })
        view["trash_inventory"] = sorted(
            trash, key=lambda d: d["retention_days_remaining"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("doctor team trash failed: %s", exc)
        view["trash_inventory"] = None

    # ── orphan owners (active docs with NULL/_system owner) ────────────
    try:
        orphans_all = await meta.list_documents(owner_filter="null_or_system")
        view["orphan_owners"] = [d for d in orphans_all if d["status"] == "active"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("doctor team orphans failed: %s", exc)
        view["orphan_owners"] = None

    return view


def render_team_text(view: dict[str, Any]) -> str:
    """Spec-style human-readable output.

    Stable section markers ("▎Today's Activity", etc.) — incident
    runbooks grep for these to extract numbers.
    """
    lines: list[str] = []
    lines.append("Knowledge Base Doctor (Team View)            "
                 f"{view['as_of']:%Y-%m-%d %H:%M}")
    lines.append("═" * 63)
    lines.append("")

    # Today's activity
    lines.append("▎Today's Activity (last 24h)")
    if view["activity_24h"] is None:
        lines.append("  [— data unavailable]")
    elif not view["activity_24h"]:
        lines.append("  (no audit rows in window)")
    else:
        lines.append(f"  {'user':<20} {'ingest':>7} {'cancel':>7} {'failed':>7}")
        for row in view["activity_24h"]:
            uid = row["user_id"] or "(none)"
            lines.append(
                f"  {uid:<20} {row['ingest']:>7} {row['cancel']:>7} {row['failed']:>7}"
            )
    lines.append("")

    # Maintenance flag (T5 carry-forward) — placed BEFORE queue status
    # so an operator notices "server is paused" before being confused
    # by "Active workers: 0".
    lines.append("▎Maintenance")
    m = view.get("maintenance")
    if m is None:
        lines.append("  [— data unavailable]")
    else:
        if m["on"]:
            ts = str(m.get("set_at") or "")[:19]
            who = m.get("set_by_user_id") or "(unknown)"
            lines.append(f"  ON  (set at {ts or '(unknown)'} by {who})")
            lines.append("  • writes blocked: claim_next_item, DELETE/restore, ingest, cancel, retry, MCP write tools")
        else:
            ts = str(m.get("set_at") or "")[:19]
            who = m.get("set_by_user_id") or "(none)"
            if ts:
                lines.append(f"  OFF  (last toggle {ts} by {who})")
            else:
                lines.append("  OFF  (never toggled)")
    lines.append("")

    # Queue status
    lines.append("▎Queue Status")
    if view["queue_by_owner"] is None:
        lines.append("  [— data unavailable]")
    else:
        active = view["active_workers"] or []
        stale = view.get("stale_claims") or []
        lines.append(f"  Active workers: {len(active)}")
        for a in active:
            lines.append(
                f"    - {a['locked_by']} on item {a['item_id'][:12]} "  # 12 hex chars — enough for unambiguous display
                f"(claimed at {a['claimed_at']})"
            )
        # Review fix P1#1: stale claims rendered separately so admin
        # doesn't mistake a dead worker for an active one.
        if stale:
            lines.append(f"  Stale (reclaimable): {len(stale)}")
            for s in stale[:20]:  # cap display same as destructive_7d
                lines.append(
                    f"    - {s['locked_by']} on item {s['item_id'][:12]} "
                    f"(lease expired {s['locked_until']})"
                )
        total_queued = sum(view["queue_by_owner"].values())
        # "Queued + reclaimable" wording reflects include_reclaimable=True
        lines.append(f"  Queued + reclaimable: {total_queued} items")
        for owner, count in sorted(view["queue_by_owner"].items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
            tag = "(orphan)" if owner in (None, OWNER_ID_SYSTEM) else ""
            lines.append(f"    - {owner or '(none)':<20} {count:>5} {tag}")
    lines.append("")

    # Destructive actions
    lines.append("▎Recent Destructive Actions (last 7 days)")
    if view["destructive_7d"] is None:
        lines.append("  [— data unavailable]")
    elif not view["destructive_7d"]:
        lines.append("  (no destructive actions in window)")
    else:
        for r in view["destructive_7d"][:20]:  # text mode caps at 20; JSON carries the full 50
            ts = (r.get("ts") or "")[:19]
            uid = r.get("user_id") or "system"
            lines.append(
                f"  {ts}  {uid:<12}  {r['action']:<24}  "
                f"{r.get('target_id') or '-'}"
            )
    lines.append("")

    # Trash inventory
    lines.append(f"▎Soft-Deleted Inventory (retention={view['retention_days']} days)")
    if view["trash_inventory"] is None:
        lines.append("  [— data unavailable]")
    elif not view["trash_inventory"]:
        lines.append("  (trash empty)")
    else:
        lines.append(f"  {len(view['trash_inventory'])} documents in trash")
        for d in view["trash_inventory"]:
            owner = d.get("owner_id") or "(none)"
            del_ts = (str(d.get("deleted_at")) or "")[:19] if d.get("deleted_at") else "(unknown)"
            lines.append(
                f"  - {d['doc_name']:<30} (deleted {del_ts} by {owner})  "
                f"[{d['retention_days_remaining']} days left]"
            )
    lines.append("")

    # Orphan owners
    lines.append("▎Orphan Doc Owners")
    if view["orphan_owners"] is None:
        lines.append("  [— data unavailable]")
    elif not view["orphan_owners"]:
        lines.append("  (no orphan-owner docs)")
    else:
        lines.append(
            f"  {len(view['orphan_owners'])} active docs have owner_id "
            "NULL or '_system' (pre-T3 data — admin-only writes)"
        )
        for d in view["orphan_owners"][:10]:
            lines.append(f"    - {d['doc_id']:<14}  {d['doc_name']}")

    return "\n".join(lines) + "\n"


def team_view_to_dict(view: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable shape — same keys as gather_team_view but with
    ``as_of`` ISO-string and datetimes throughout converted via str().

    Note: ``queue_by_owner`` contains ``None`` as a dict key (CLI ingest's
    NULL owner). This is preserved by the normalizer because doctor's
    Python consumer cares about the key. Callers serializing with
    ``json.dumps`` MUST pass ``default=str`` — the standard JSON encoder
    rejects ``None`` keys with TypeError. ``scripts/doctor.py`` already
    does this; if you add another consumer, mirror the pattern.
    """
    def _norm(v):
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, dict):
            return {k: _norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        return v
    return _norm(view)
