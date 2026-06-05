from typing import Protocol

from kb.jobs.errors import JobError
from kb.jobs.models import IngestionJob, IngestionJobItem


class JobStore(Protocol):
    async def init(self) -> None: ...
    async def create_job(
        self,
        paths: list[str],
        config: dict,
        *,
        owner_id: str | None = None,
    ) -> IngestionJob: ...
    async def get_job(self, job_id: str) -> IngestionJob | None: ...
    async def list_job_items(self, job_id: str) -> list[IngestionJobItem]: ...
    async def claim_next_item(
        self,
        worker_id: str,
        lease_seconds: int,
    ) -> IngestionJobItem | None: ...
    async def heartbeat(
        self,
        item_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> bool: ...
    async def complete_item(self, item_id: str, result: dict) -> None: ...
    async def fail_item(
        self,
        item_id: str,
        error: JobError,
        retry_at,
    ) -> None: ...
    async def request_cancel(self, job_id: str) -> None: ...
    async def retry_failed_items(self, job_id: str) -> int: ...
