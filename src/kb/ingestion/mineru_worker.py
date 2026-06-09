# src/kb/ingestion/mineru_worker.py
"""Per-PDF MinerU subprocess worker.

MinerU has no public unload API — its layout/OCR/table singletons release
VRAM only when the process dies. So to bound GPU memory on a single local
card during ingest we run MinerU as a per-PDF subprocess: spawn a fresh
mineru_server in cuda mode, drive it through one /parse, then call /exit so
the OS reclaims the GPU before the next PDF.

(PoC v1.0 also unloaded a co-resident ColQwen2.5 here to fit both on an 8GB
5070; that vision model was retired in v1.1 — embeddings now run on the
shared multimodal embedding server — so only the MinerU lifecycle remains.)
"""
import asyncio
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from kb.config import Settings
from kb.observability import emit

logger = logging.getLogger(__name__)


async def _wait_for_health(url: str, timeout_s: float, poll_interval_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            await asyncio.sleep(poll_interval_s)
    raise TimeoutError(
        f"mineru worker did not become healthy at {url} within {timeout_s}s "
        f"(last error: {last_err})"
    )


async def _post_safe(url: str, timeout_s: float = 10.0) -> bool:
    """Best-effort POST; return True on 2xx, False otherwise. Never raises."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url)
            return 200 <= r.status_code < 300
    except Exception as e:
        logger.warning("POST %s failed: %s", url, e)
        return False


async def _is_port_free(url: str, timeout_s: float = 1.0) -> bool:
    """True if /health does NOT respond — i.e. nothing is listening on this port."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"{url}/health")
            return r.status_code != 200
    except Exception:
        return True


@asynccontextmanager
async def mineru_gpu_worker(settings: Settings):
    """Spawn a fresh mineru_server in cuda mode for the duration of one PDF.

    On enter: spawn mineru worker, wait for /health.
    On exit:  call mineru /exit, wait for process termination (with kill fallback)
              so the OS reclaims the GPU before the next PDF.

    Yields the mineru server URL the caller should use for /parse.
    """
    mineru_url = settings.mineru_server_url.rstrip("/")
    health_timeout = settings.ingest_mineru_worker_health_timeout_s

    # 1. Spawn mineru worker subprocess (cuda mode)
    repo_root = Path(__file__).resolve().parents[3]
    server_script = repo_root / "scripts" / "mineru_server.py"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    parsed_port = mineru_url.rsplit(":", 1)[-1]
    env = {**os.environ, "MINERU_DEVICE_MODE": "cuda", "PYTHONIOENCODING": "utf-8"}

    # Refuse to spawn if the port is already taken — a stale worker would
    # be picked up by /health and cause silent state mixing.
    if not await _is_port_free(mineru_url):
        raise RuntimeError(
            f"port {parsed_port} is already in use — another mineru server "
            f"is responding. Kill it before spawning a new GPU worker."
        )

    t_spawn = time.monotonic()
    # No CREATE_NEW_PROCESS_GROUP: we want Windows job-object inheritance so
    # an abnormal exit of the parent kills the worker (avoids orphaned 8001
    # listeners holding the GPU).
    proc = subprocess.Popen(
        [python_exe, str(server_script), "--port", parsed_port],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        await _wait_for_health(mineru_url, timeout_s=health_timeout)
        emit(
            "ingest.mode_switch",
            action="mineru_worker_ready",
            pid=proc.pid,
            duration_ms=round((time.monotonic() - t_spawn) * 1000, 1),
        )

        yield mineru_url

    finally:
        # 2. Tell mineru to exit, then wait for process death
        t_exit = time.monotonic()
        await _post_safe(f"{mineru_url}/exit", timeout_s=5.0)
        # Give it ~5s to flush and exit cleanly; otherwise kill.
        try:
            proc.wait(timeout=5.0)
            exit_kind = "graceful"
        except subprocess.TimeoutExpired:
            logger.warning("mineru worker pid %s did not exit in 5s; terminating", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
                exit_kind = "terminated"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
                exit_kind = "killed"
        emit(
            "ingest.mode_switch",
            action="mineru_worker_exit",
            exit_kind=exit_kind,
            duration_ms=round((time.monotonic() - t_exit) * 1000, 1),
        )
