import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check local knowledge-base health.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    # Phase 5.2 deep reconcile: scrolls Qdrant + checks per-doc consistency.
    # On by default; --no-reconcile skips the 5 deep checks for fast
    # config/HTTP-only probes (also useful when Qdrant is intentionally down).
    parser.add_argument(
        "--no-reconcile", action="store_true",
        help="Skip deep metadata/Qdrant/image consistency checks.",
    )
    parser.add_argument(
        "--reconcile-sample-limit", type=int, default=None,
        help="Cap active-doc reconcile to first N (default: all).",
    )
    sub = parser.add_subparsers(dest="cmd")
    team = sub.add_parser("team", help="Team-wide admin view (T4).")
    # Review fix P2#1: also accept `--json` AFTER the team keyword.
    # Different dest so the subparser default (False) doesn't overwrite
    # a True set by the global `--json` before the keyword. _run_team
    # ORs the two; either ordering yields the same result.
    team.add_argument(
        "--json", action="store_true", dest="team_json",
        help="Same as global --json; accepted after the `team` keyword.",
    )
    team.add_argument(
        "--retention-days", type=int, default=None,
        help="Override settings.deleted_doc_retention_days (informational only).",
    )
    return parser


async def _run_default(args: argparse.Namespace) -> int:
    """Default invocation (no subcommand) — single-host health check.

    Back-compat path for any cron/CI job that calls
    ``python scripts/doctor.py`` without arguments. Output shape is
    unchanged from pre-T4.
    """
    from kb.ops.doctor import run_doctor

    result = await run_doctor(
        Settings(),
        deep_reconcile=not args.no_reconcile,
        reconcile_sample_limit=args.reconcile_sample_limit,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.render_text())
    return 1 if result.summary()["errors"] else 0


async def _run_team(args: argparse.Namespace) -> int:
    from kb.admin.maintenance import MaintenanceState
    from kb.audit.log import AuditLog
    from kb.jobs.sqlite_store import SQLiteJobStore
    from kb.ops.doctor_team import (
        gather_team_view, render_team_text, team_view_to_dict,
    )
    from kb.storage.metadata_db import MetadataDB

    settings = Settings()
    meta_url = f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    jobs_url = f"sqlite+aiosqlite:///{settings.qdrant_path}/ingestion_jobs.db"
    meta = MetadataDB(db_url=meta_url)
    await meta.init()
    audit = AuditLog(db_url=meta_url, fallback_path=settings.audit_fallback_path)
    await audit.init()
    # T5 carry-forward: surface the maintenance flag on the team panel.
    # Read-only — doctor never sets/clears the flag, only displays it.
    maintenance = MaintenanceState(db_url=meta_url)
    await maintenance.init()
    # Review fix P1#2: `readonly=True` only suppresses BEGIN IMMEDIATE;
    # it doesn't open SQLite in true read-only mode, so a missing
    # `ingestion_jobs.db` would silently create a 0-byte file and then
    # fail at "no such table". Pre-check existence here so the queue
    # section is cleanly reported as unavailable without touching disk.
    # T5 follow-up: open the store via `?mode=ro` URI for hard guarantee.
    jobs_db_path = Path(settings.qdrant_path) / "ingestion_jobs.db"
    if jobs_db_path.exists():
        # T4 P0-1: readonly=True skips BEGIN IMMEDIATE so doctor's
        # SELECTs don't contend with worker write locks.
        jobs = SQLiteJobStore(db_url=jobs_url, readonly=True)
    else:
        jobs = None  # gather_team_view treats None as section unavailable

    use_json = args.json or getattr(args, "team_json", False)
    try:
        retention = (
            args.retention_days
            if args.retention_days is not None
            else settings.deleted_doc_retention_days
        )
        view = await gather_team_view(
            meta=meta, audit=audit, jobs=jobs,
            retention_days=retention,
            maintenance=maintenance,
        )
        if use_json:
            print(json.dumps(team_view_to_dict(view), ensure_ascii=False, indent=2))
        else:
            # P2-8 review note: spec sample uses Unicode box markers
            # (▎ ═). PowerShell 7+ / Windows Terminal handle UTF-8;
            # legacy cmd.exe may need `chcp 65001` first. For
            # `--json` output the markers don't apply.
            print(render_team_text(view))
        return 0
    finally:
        if jobs is not None:
            await jobs.close()
        await maintenance.aclose()
        await audit.aclose()
        await meta.aclose()


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "team":
        return await _run_team(args)
    return await _run_default(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
