"""Archive audit_log rows older than retention to cold-storage JSONL.

Spec §8 carry-forward: audit_log grows monotonically. Eventually rows
older than ~1 year are unlikely to be queried interactively but must
NOT be lost (incidents from years past, compliance reviews, etc.).
This script moves them off the hot SQLite table into year-bucketed
JSONL files under ``logs/audit_archive/`` so:

  - DB stays small → GET /audit and per-user query latency stays bounded
  - Archived rows stay greppable (matches existing audit_fallback.jsonl)
  - No new dependencies (no pyarrow, no separate DB)

Output layout (year-bucketed so operators can quickly grep one year):

  logs/audit_archive/audit_archive_2024.jsonl
  logs/audit_archive/audit_archive_2025.jsonl
  ...

Each line is the same shape AuditLog.query() returns:

  {"event_id": 12345, "user_id": "u-alice", "action": "doc.ingest",
   "target_kind": "document", "target_id": "d-42",
   "ts": "2024-03-15T10:22:08", "payload_json": "{}"}

Safety:
  - Maintenance probe via /admin/maintenance_mode (same pattern as gc).
    Refuses to run while maintenance is ON (avoids racing backup script).
  - Default retention 365 days; runs as DRY-RUN unless --confirm passed.
  - Two-phase per row: append JSONL line + fsync THEN delete DB row.
    Crash between phases = duplicate row in archive (safe) not lost row.
  - Year-bucket file is opened append-only; existing content preserved.

Exit codes:
  0 — success (or dry-run completion)
  2 — admin key missing / server unreachable / file IO error
  3 — server in maintenance mode (refused to run)
  4 — dry-run had work to do; pass --confirm to actually archive
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx

from kb.config import Settings
from kb.storage.metadata_db import AuditLogRecord
from sqlalchemy import delete as sql_delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


async def _check_maintenance_via_http(base_url: str, admin_key: str) -> bool:
    """Reuse the gc pattern. Returns True if server is in maintenance."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{base_url}/admin/maintenance_mode",
                headers={"X-API-Key": admin_key},
            )
            resp.raise_for_status()
            return bool(resp.json().get("on"))
    except httpx.HTTPError as exc:
        raise RuntimeError(f"failed to reach server at {base_url}: {exc}")


def _row_to_dict(r: AuditLogRecord) -> dict:
    """Match AuditLog.query() output shape so archived rows can be diffed
    against hot-table queries during forensic work."""
    return {
        "event_id": r.event_id,
        "user_id": r.user_id,
        "action": r.action,
        "target_kind": r.target_kind,
        "target_id": r.target_id,
        "ts": r.ts.isoformat() if r.ts else None,
        "payload_json": r.payload_json,
    }


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic-enough append: open in 'a', write each row, fsync, close.
    Year-bucket file is shared across runs so we append rather than
    rewrite. Lost rows on power loss are bounded by the OS fsync window
    (~ one fdatasync per archive call, not per row — acceptable trade
    for an archive workload that runs maybe once a month)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


async def archive_main(
    *,
    retention_days: int = 365,
    confirm: bool = False,
    kb_metadata_url: Optional[str] = None,
    archive_dir: Optional[Path] = None,
) -> dict:
    """Core archive routine — test-callable.

    Returns: {"candidates": int, "archived": int, "deleted": int,
              "by_year": {year: count}, "files": [paths]}.
    confirm=False = dry run (no JSONL writes, no DB deletes).
    """
    settings = Settings()
    db_url = kb_metadata_url or f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    archive_root = Path(archive_dir) if archive_dir else Path(settings.log_dir) / "audit_archive"

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=retention_days)
    engine = create_async_engine(db_url, poolclass=NullPool)
    sf = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    by_year: dict[int, list[dict]] = defaultdict(list)
    event_ids_to_delete: list[int] = []
    async with sf() as session:
        result = await session.execute(
            select(AuditLogRecord)
            .where(AuditLogRecord.ts < cutoff)
            .order_by(AuditLogRecord.ts.asc())
        )
        for r in result.scalars().all():
            year = r.ts.year if r.ts else 0
            by_year[year].append(_row_to_dict(r))
            event_ids_to_delete.append(r.event_id)

    total = len(event_ids_to_delete)
    if total == 0:
        await engine.dispose()
        return {"candidates": 0, "archived": 0, "deleted": 0,
                "by_year": {}, "files": []}

    if not confirm:
        # Dry run — report what WOULD happen, touch nothing.
        await engine.dispose()
        return {
            "candidates": total, "archived": 0, "deleted": 0,
            "by_year": {y: len(rows) for y, rows in sorted(by_year.items())},
            "files": [str(archive_root / f"audit_archive_{y}.jsonl") for y in sorted(by_year)],
        }

    # Confirmed — append each year-bucket to its file. Append-and-fsync
    # for each year BEFORE the DB delete, so a crash mid-loop leaves
    # rows duplicated in archive (safe) rather than lost.
    files: list[str] = []
    archived = 0
    for year, rows in sorted(by_year.items()):
        path = archive_root / f"audit_archive_{year}.jsonl"
        _append_jsonl(path, rows)
        archived += len(rows)
        files.append(str(path))

    # All archive writes durably on disk; now DELETE the rows in batches
    # (single statement on SQLite — atomic via the IN clause).
    deleted = 0
    async with sf() as session:
        # SQLite IN-list practical limit is 999; batch to be safe.
        BATCH = 500
        for i in range(0, len(event_ids_to_delete), BATCH):
            chunk = event_ids_to_delete[i:i + BATCH]
            await session.execute(
                sql_delete(AuditLogRecord).where(AuditLogRecord.event_id.in_(chunk))
            )
            deleted += len(chunk)
        await session.commit()

    await engine.dispose()
    return {
        "candidates": total, "archived": archived, "deleted": deleted,
        "by_year": {y: len(rows) for y, rows in sorted(by_year.items())},
        "files": files,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Archive audit_log rows older than retention to JSONL.",
    )
    p.add_argument("--retention-days", type=int, default=365,
                   help="Rows older than this are archived (default 365).")
    p.add_argument("--confirm", action="store_true",
                   help="REQUIRED for actual archive. Default is dry-run.")
    p.add_argument("--archive-dir", default=None,
                   help="Output dir (default <log_dir>/audit_archive).")
    p.add_argument("--server-url", default="http://127.0.0.1:8765",
                   help="Server URL for maintenance probe.")
    p.add_argument("--admin-key", default=None,
                   help="Admin API key (or env KB_ADMIN_KEY).")
    p.add_argument("--skip-maintenance-check", action="store_true",
                   help="DANGEROUS: skip the maintenance HTTP probe.")
    return p


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.skip_maintenance_check:
        admin_key = args.admin_key or os.environ.get("KB_ADMIN_KEY")
        if not admin_key:
            print("ERROR: provide --admin-key or set KB_ADMIN_KEY (or pass "
                  "--skip-maintenance-check if running offline)", file=sys.stderr)
            return 2
        try:
            on = await _check_maintenance_via_http(args.server_url, admin_key)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if on:
            print("ERROR: server in maintenance mode; refusing to archive.",
                  file=sys.stderr)
            return 3

    result = await archive_main(
        retention_days=args.retention_days,
        confirm=args.confirm,
        archive_dir=Path(args.archive_dir) if args.archive_dir else None,
    )

    label = "DRY RUN — would archive" if not args.confirm else "Archived"
    print(f"{label} {result['candidates']} audit rows older than "
          f"{args.retention_days} days")
    if result["by_year"]:
        for year, count in result["by_year"].items():
            print(f"  - {year}: {count} rows")
        if args.confirm:
            print(f"Deleted from hot table: {result['deleted']} rows")
            print("Archive files:")
            for f in result["files"]:
                print(f"  - {f}")
    # Dry-run with work to do exits 4 so cron jobs can decide to re-run
    # with --confirm — also flags "you forgot --confirm" in CI scripts.
    if not args.confirm and result["candidates"] > 0:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
