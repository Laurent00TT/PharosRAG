import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.admin.maintenance import MaintenanceState
from kb.audit.log import AuditLog
from kb.auth.users import UsersStore
from kb.config import Settings
from kb.ingestion.job_executor import IngestionJobExecutor
from kb.ingestion.pipeline import IngestionPipeline
from kb.jobs.resource_manager import ResourceManager
from kb.jobs.runner import IngestionJobRunner
from kb.jobs.sqlite_store import SQLiteJobStore
from kb.nav.builder import NavIndexBuilder
from kb.nav.store import NavIndexStore
from kb.observability import setup_tracing


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    defaults = settings or Settings()
    parser = argparse.ArgumentParser(description="Run local knowledge-base workers.")
    parser.add_argument(
        "--queue",
        choices=["ingestion"],
        default="ingestion",
        help="Queue to consume. Only ingestion is supported.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=defaults.job_worker_poll_interval_s,
        help="Seconds to sleep when no job item is available.",
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=defaults.job_worker_lease_seconds,
        help="Lease duration for claimed job items.",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=None,
        help="Seconds between lease heartbeats. Defaults to one third of lease.",
    )
    return parser


async def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    args = build_parser(settings).parse_args(argv)
    setup_tracing(
        log_dir=settings.log_dir,
        max_mb=settings.log_max_mb,
        backup_count=settings.log_backup_count,
    )
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    # AuditLog targets the shared kb_metadata.db (same file UsersStore
    # and MetadataDB use, per spec P1-2). Worker runs alongside the
    # tool_server; both write into the same audit_log table.
    metadata_db_url = (
        f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    )
    audit_log = AuditLog(
        db_url=metadata_db_url,
        fallback_path=settings.audit_fallback_path,
    )
    await audit_log.init()
    # T5: MaintenanceState must be constructed BEFORE SQLiteJobStore so
    # we can pass maintenance.is_on as the maintenance_check callback.
    # Both read kb_metadata.db (same DB as audit_log).
    maintenance = MaintenanceState(db_url=metadata_db_url)
    await maintenance.init()
    store = SQLiteJobStore(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/ingestion_jobs.db",
        maintenance_check=maintenance.is_on,
    )
    await store.init()
    # T3: UsersStore lets the executor materialize the owner User from
    # job.owner_id before invoking the pipeline. Same DB as tool_server
    # (kb_metadata.db) so trace/audit attribution stays consistent
    # across server-created jobs and worker-executed items.
    users_store = UsersStore(db_url=metadata_db_url)
    await users_store.init()
    # Nav layer is wired into the pipeline ONLY when both feature flags
    # are on. Mirrors the server lifespan gate (`if settings.nav_enabled`)
    # so config flips affect worker + server consistently — otherwise
    # users could turn nav off and still find the worker writing nav db.
    nav_enabled = settings.nav_enabled and settings.nav_auto_build_on_ingest
    if nav_enabled:
        nav_store = NavIndexStore(settings.nav_db_path)
        await nav_store.init()
        nav_builder = NavIndexBuilder()
    else:
        nav_store = None
        nav_builder = None
    resource_manager = ResourceManager()
    executor = IngestionJobExecutor(
        store=store,
        # ``cancel_check`` keyword is detected by the executor and bound
        # to the item's job_id at dispatch time — the lambda just needs
        # to accept and forward it. When nav is disabled, nav_store and
        # nav_builder are None and auto_build_nav_for_doc silently no-ops.
        pipeline_factory=lambda *, cancel_check=None: IngestionPipeline(
            settings, nav_store=nav_store, nav_builder=nav_builder,
            cancel_check=cancel_check, audit_log=audit_log,
        ),
        resource_manager=resource_manager,
        resources=list(resource_manager.default_resources),
        audit_log=audit_log,
        users_store=users_store,
    )
    runner = IngestionJobRunner(
        store=store,
        executor=executor,
        poll_interval_s=args.poll_interval,
        lease_seconds=args.lease_seconds,
        heartbeat_interval_s=(
            args.heartbeat_interval
            or settings.job_worker_heartbeat_interval_s
            or None
        ),
    )
    try:
        await runner.run_forever()
    finally:
        await store.close()
        if nav_store is not None:
            await nav_store.close()
        await users_store.aclose()
        # T5: maintenance closed before audit_log (same close-order
        # convention as tool_server/app.py — audit_log must outlive
        # everything else so any shutdown-time emit still has a live
        # engine to write through).
        await maintenance.aclose()
        # audit_log closed AFTER other stores so a shutdown-time audit
        # emit from any of the above closes still has a live engine to
        # write through. Mirrors the lifespan shutdown order in
        # tool_server/app.py (close all stores before audit_log).
        await audit_log.aclose()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
