# src/kb/parser/mineru_local.py
"""Local MinerU parser — HTTP client for the in-tree mineru_server.

Two sub-modes are controlled by Settings.ingest_use_gpu_mineru:

- True (default): per-PDF GPU worker. mineru_gpu_worker context manager
  spawns a fresh mineru_server.py subprocess in cuda mode, drives one
  /parse, then /exits the worker so the OS reclaims GPU memory before the
  next PDF (MinerU has no unload API). ~2.4x speedup vs CPU path on the
  5070 + 11 PDF benchmark (see
  internal design notes (not published)).

- False: assume one or more long-lived mineru_server instances are
  reachable. When Settings.mineru_server_urls is non-empty (comma-
  separated), round-robin /parse calls across all listed URLs to scale
  ingest throughput; otherwise fall back to single-instance behavior on
  Settings.mineru_server_url. See internal design notes (not published)
  for the multi-instance design.

Conversion of mineru's middle_json into ParsedPage lives in
kb.parser._middle_to_pages so MinerUAPIParser can share it.
"""
import asyncio
import hashlib
import itertools
import logging
import os
import socket
from pathlib import Path

import httpx

from kb.config import Settings
from kb.models import ParsedPage
from kb.parser._middle_to_pages import middle_to_pages

logger = logging.getLogger(__name__)


class _AffinityRouter:
    """Hostname-affinity round-robin router with concurrency cap.

    Why hostname affinity (vs pure random):
    - KB worker runs as N container replicas; each container has a stable
      hostname (e.g. "docker-kb_worker-1"). Hashing it gives a stable
      base offset into the URL list.
    - With B base offsets uniformly distributed and round-robin from each
      base, short bursts of M requests across N replicas distribute
      ~deterministically across all backends, instead of clustering by
      random luck (random.choice with N=4, 8 requests can give 0/4/2/0).
    - Falls back to PID hash if hostname is generic.

    Each replica still round-robins WITHIN its own counter so multi-request
    bursts from one worker also use multiple backends.

    Concurrency cap (asyncio.Semaphore sized to len(urls)) pushes
    backpressure to the caller when in-flight count exceeds available
    backend slots, rather than queueing inside any single backend.
    """

    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise ValueError("urls must be non-empty")
        self._urls = [u.rstrip("/") for u in urls]
        self._sema = asyncio.Semaphore(len(self._urls))

        # Stable per-replica base offset
        seed = os.environ.get("HOSTNAME") or socket.gethostname() or str(os.getpid())
        h = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16)
        self._base_idx = h % len(self._urls)
        self._counter = itertools.count()
        logger.info(
            "MinerULocalParser router: %d urls, hostname=%r → base_idx=%d",
            len(self._urls), seed, self._base_idx,
        )

    def __len__(self) -> int:
        return len(self._urls)

    def acquire_url(self) -> "_AcquiredURL":
        """Return an async context manager that yields the picked URL and
        holds the in-flight slot for the duration of the request."""
        return _AcquiredURL(self)


class _AcquiredURL:
    def __init__(self, router: _AffinityRouter) -> None:
        self._router = router

    async def __aenter__(self) -> str:
        await self._router._sema.acquire()
        n = len(self._router._urls)
        offset = next(self._router._counter)
        return self._router._urls[(self._router._base_idx + offset) % n]

    async def __aexit__(self, *_exc) -> None:
        self._router._sema.release()


def _resolve_mineru_urls(settings: Settings) -> list[str]:
    """Settings.mineru_server_urls (plural, comma-separated) overrides
    Settings.mineru_server_url (single). Empty plural falls back to single.
    """
    plural = (getattr(settings, "mineru_server_urls", "") or "").strip()
    if plural:
        urls = [u.strip() for u in plural.split(",") if u.strip()]
        if urls:
            return urls
    return [settings.mineru_server_url]


class MinerULocalParser:
    """Local-mineru implementation of the Parser Protocol."""

    def __init__(
        self,
        settings: Settings,
        mode: str | None = None,
        lang: str = "ch",
        timeout: float = 600.0,
    ) -> None:
        self._settings = settings
        urls = _resolve_mineru_urls(settings)
        self._router = _AffinityRouter(urls)
        if len(self._router) > 1:
            logger.info(
                "MinerULocalParser routing across %d mineru instances: %s",
                len(self._router), urls,
            )
        # Mode resolution: explicit arg > settings.mineru_parse_mode.
        # See Settings.mineru_parse_mode for the choice rationale (notably,
        # "pipeline" sidesteps the hybrid backend's multiprocessing.spawn
        # deadlock seen in some containerised cloud envs).
        self._mode = mode or settings.mineru_parse_mode
        self._lang = lang
        # Persistent client; tests patch `parser._client.post`. Lifetime is
        # tied to the IngestionPipeline (process-long).
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def parse_async(self, path: Path) -> list[ParsedPage]:
        """Parse a single document and return a list of ParsedPage.

        When ingest_use_gpu_mineru is True we spawn a per-PDF GPU worker
        that owns the GPU for the duration of /parse; the worker /exits
        afterwards so the OS reclaims VRAM before the next PDF.
        """
        if self._settings.ingest_use_gpu_mineru:
            # Imported lazily so non-GPU deployments don't pull mineru_worker
            # (and its subprocess plumbing) at import time.
            from kb.ingestion.mineru_worker import mineru_gpu_worker
            async with mineru_gpu_worker(self._settings):
                return await self._http_parse(path)
        return await self._http_parse(path)

    async def _http_parse(self, path: Path) -> list[ParsedPage]:
        pdf_bytes = path.read_bytes()
        doc_id = hashlib.md5(pdf_bytes).hexdigest()
        doc_name = path.name

        data = {
            "mode": self._mode,
            "lang": self._lang,
            "formula_enable": "true",
            "table_enable": "true",
        }
        # Busy-aware dispatch: a mineru instance returns 503 when it's already
        # parsing (see scripts/mineru_server.py). On 503, rotate to the next URL
        # via the router instead of queueing on a busy instance. Try each
        # instance at most once (len(router) attempts); if ALL are busy, surface
        # the last 503 so the job runner retries later rather than hanging.
        n = len(self._router)
        last_response = None
        for _ in range(n):
            # Fresh files dict per attempt — httpx consumes the upload tuple.
            files = {"file": (doc_name, pdf_bytes, "application/pdf")}
            async with self._router.acquire_url() as url:
                response = await self._client.post(
                    f"{url}/parse",
                    files=files,
                    data=data,
                )
            if response.status_code == 503:
                last_response = response
                continue  # instance busy — try the next URL
            response.raise_for_status()
            return middle_to_pages(response.json(), doc_id, doc_name)

        # Every instance was busy this pass.
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("mineru: no instances available")
