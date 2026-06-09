from dataclasses import dataclass, field
from datetime import datetime

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_PARTIALLY_SUCCEEDED = "partially_succeeded"
JOB_FAILED = "failed"
JOB_CANCELLING = "cancelling"
JOB_CANCELLED = "cancelled"

ITEM_QUEUED = "queued"
ITEM_CLAIMED = "claimed"
ITEM_SUCCEEDED = "succeeded"
ITEM_SKIPPED = "skipped"
ITEM_FAILED = "failed"
ITEM_CANCELLED = "cancelled"


@dataclass
class IngestionJob:
    job_id: str
    status: str
    total_items: int
    succeeded_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    cancel_requested: bool = False
    config: dict = field(default_factory=dict)
    # T3: surface the ORM-level owner_id on the dataclass so worker
    # (execute_item) can read it without a second SQL query when binding
    # the ContextVar. NULL = pre-T1a backfill (`_system`) → admin-only
    # writes per I-11; the worker treats this as "no current_user" and
    # the auth path treats it as admin-only at decision time.
    owner_id: str | None = None


@dataclass
class IngestionJobItem:
    item_id: str
    job_id: str
    path: str
    status: str
    attempt: int = 0
    max_attempts: int = 3
    claimed_at: datetime | None = None    # T4: surfaces the ORM column
    # Lease fields — surfaced for diagnostics. Their ABSENCE here previously
    # caused a false "lease orphan" diagnosis (2026-05-28): a claimed item read
    # via the API showed locked_by/locked_until as missing → looked like NULL in
    # the DB → looked like a stuck orphan, when really the item was being
    # processed by a live, heartbeating worker. Expose them so lease state is
    # visible and "claimed by dead worker (expired lease)" vs "claimed + in
    # progress (fresh lease)" are distinguishable without a raw SQL probe.
    locked_by: str | None = None
    locked_until: datetime | None = None


@dataclass
class IngestionJobEvent:
    event_id: int
    job_id: str
    item_id: str | None
    ts: datetime
    level: str
    event: str
    message: str
    payload: dict = field(default_factory=dict)
