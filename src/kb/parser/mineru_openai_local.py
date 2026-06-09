# src/kb/parser/mineru_openai_local.py
"""Local MinerU parser — client for the `mineru-openai-server` (OpenAI-compatible).

Paired with: a remote GPU host running
  mineru-openai-server --backend vlm-vllm-async-engine --port 8101

⚠️ STATUS — INCOMPLETE: HTTP protocol details unverified
============================================================
`mineru-openai-server` ships with MinerU 3.1.x but its HTTP request /
response schema is **not documented in MinerU upstream docs**. The
community is asking the same question — see RAGflow issue #11252
("How to use Mineru API with backend mode vlm-vllm-async-engine"),
which has been open since 2025 with no upstream answer.

Until we can run the server against a real GPU and inspect its actual
endpoints (Day 0 PoC verification, see
internal design notes (not published) §11.1 #4),
the `parse_async` implementation here raises NotImplementedError so
nobody silently runs against a broken client.

Once verified, the contract this class needs to satisfy:
    async def parse_async(path: Path) -> list[ParsedPage]

Inputs:
    - A PDF file on local disk
    - settings.mineru_server_url pointing at mineru-openai-server :8101

Probable protocol shapes to investigate (in priority order):
  1. OpenAI Files API style:
        POST /v1/files      (upload PDF, returns file_id)
        POST /v1/chat/completions   (multimodal request referencing file_id)
  2. SaaS-compatible:
        POST /v4/extract/task  (matches existing mineru_api.py SaaS client)
  3. Native /parse multipart (matches in-tree mineru_server.py /parse)

Verification command (run on the GPU host after starting the server):
    curl -X OPTIONS http://localhost:8101/
    curl http://localhost:8101/openapi.json | jq '.paths | keys'
    # Find the endpoint, then trace its schema in the response.

Decision logged: internal design notes (not published) [2026-05-22] [P-10A]
"""
import logging
from pathlib import Path

import httpx

from kb.config import Settings
from kb.models import ParsedPage

logger = logging.getLogger(__name__)


class MinerUOpenAILocalParser:
    """Client for mineru-openai-server. Currently a placeholder.

    See module docstring for the verification work blocking full
    implementation.
    """

    def __init__(
        self,
        settings: Settings,
        mode: str = "hybrid",
        lang: str = "ch",
        timeout: float = 600.0,
    ) -> None:
        self._settings = settings
        # Reuses mineru_server_url for now; if mineru-openai-server runs
        # on a different port from in-tree mineru_server.py, add a new
        # Settings field. Default is 8101 per spec deployment design.
        self._url = settings.mineru_server_url.rstrip("/")
        self._mode = mode
        self._lang = lang
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health_check(self) -> bool:
        """Best-effort health check. The /health (or /v1/models) endpoint
        is one of the safest assumptions for any OpenAI-compatible server.
        """
        for path in ("/health", "/v1/models"):
            try:
                resp = await self._client.get(f"{self._url}{path}", timeout=10.0)
                if resp.status_code == 200:
                    logger.info("mineru-openai-server reachable via %s", path)
                    return True
            except Exception as e:
                logger.debug("health check %s failed: %s", path, e)
                continue
        return False

    async def parse_async(self, path: Path) -> list[ParsedPage]:
        """Parse a PDF via mineru-openai-server.

        ⚠️ NOT IMPLEMENTED — see module docstring. The HTTP protocol of
        mineru-openai-server is not documented upstream and cannot be
        guessed without running the server against a real GPU.
        """
        raise NotImplementedError(
            "P-10A pending verification — mineru-openai-server HTTP "
            "protocol must be inspected against a live deployment "
            "(see module docstring and internal design notes (not published) "
            "entry [2026-05-22] [P-10A])."
        )
