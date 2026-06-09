# src/kb/parser/mineru_api.py
"""OpenDataLab MinerU SaaS API client (mineru.net).

Hard API limits (verified at https://mineru.net/apiManage/docs):
- File size: 200 MB
- Page count: 200 pages
- Daily quota per account: 1000 priority pages, 5000 file submissions
The size/page limits are enforced server-side; we pre-check and raise
ParseError before submitting to avoid wasting daily quota on a request that
will be rejected. Files exceeding limits should be processed via the local
backend (parser_backend=mineru_local) — there is no automatic fallback.

Implementation calibrated against the real API by token probe on
2026-05-10 — see internal design notes (not published)
section "API Reality Check" for the exact request/response shapes.

Per-PDF flow:
  1. POST /api/v4/file-urls/batch  → {batch_id, file_urls[0]} (presigned OSS PUT)
  2. PUT pdf bytes to file_urls[0]  ⚠️ no Content-Type header (OSS rejects)
  3. GET /api/v4/extract-results/batch/{batch_id} loop until our data_id is "done"
  4. GET full_zip_url (CDN, no auth) and read layout.json
  5. Render page images locally (pypdfium2) — API zip has no page-rendered PNGs
  6. Reuse middle_to_pages() to build ParsedPage list

Failure handling: every error path raises ParseError or TimeoutError. There
is no automatic fallback to the local parser by design (caller chose
parser_backend at startup; pipeline records ingest.error and moves on).
"""
import asyncio
import hashlib
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import httpx

from kb.config import Settings
from kb.models import ParsedPage
from kb.observability import emit
from kb.parser._middle_to_pages import middle_to_pages
from kb.parser._page_render import render_page_images
from kb.parser.base import ParseError

logger = logging.getLogger(__name__)


# MinerU SaaS hard limits (mineru.net/apiManage/docs)
_API_MAX_FILE_BYTES = 200 * 1024 * 1024   # 200 MB
_API_MAX_PAGES = 200


class MinerUAPIParser:
    """MinerU SaaS API implementation of the Parser Protocol."""

    def __init__(self, settings: Settings) -> None:
        self._base = settings.mineru_api_url.rstrip("/")
        # Resolve token list: prefer mineru_api_tokens (multi), fall back to
        # mineru_api_token (single). Both empty → factory has already raised
        # before construction, but we re-check here defensively.
        if settings.mineru_api_tokens:
            tokens = [t.strip() for t in settings.mineru_api_tokens.split(",") if t.strip()]
        elif settings.mineru_api_token:
            tokens = [settings.mineru_api_token]
        else:
            tokens = []
        if not tokens:
            raise ValueError("MinerUAPIParser requires at least one token")
        self._tokens = tokens
        # Accumulated page-count per token. Used by _pick_least_loaded to
        # balance API quota usage by pages (the actual MinerU rate limit),
        # not by raw request count. PDFs vary 1-100+ pages so round-robin
        # by request would let a "long" token absorb 10x the load of a "short" one.
        self._token_load = [0] * len(tokens)
        self._lang = settings.mineru_api_lang
        self._poll_initial_s = settings.mineru_api_poll_interval_s
        self._max_wait_s = settings.mineru_api_max_wait_s
        self._enable_ocr = settings.mineru_api_enable_ocr
        self._enable_formula = settings.mineru_api_enable_formula
        self._enable_table = settings.mineru_api_enable_table

    def _pick_least_loaded(self, expected_pages: int, exclude: set[int] | None = None) -> int:
        """Return the index of the token with the lowest accumulated page load.

        Eligible = all token indices not in `exclude`. Adds expected_pages to
        that token's running total so subsequent picks see the new state. If
        every token is excluded (e.g. all attempted and failed), raises.
        """
        excl = exclude or set()
        eligible = [i for i in range(len(self._tokens)) if i not in excl]
        if not eligible:
            raise RuntimeError("no tokens left to try")
        # Tie-break by lowest index for determinism in tests.
        idx = min(eligible, key=lambda i: (self._token_load[i], i))
        self._token_load[idx] += max(1, expected_pages)
        return idx

    @staticmethod
    def _count_pages(pdf_bytes: bytes) -> int:
        """Quick page-count via pypdfium2. Used only for load-balancing
        decisions; sub-100ms even on 100-page PDFs."""
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            return len(pdf)
        finally:
            pdf.close()

    @staticmethod
    def _auth_headers_for(token: str) -> dict:
        # Used only for mineru.net endpoints; OSS PUT must NOT carry these.
        return {"Authorization": f"Bearer {token}"}

    async def parse_async(self, path: Path) -> list[ParsedPage]:
        """Parse one PDF via the SaaS API.

        Token strategy:
          - PAGE-WEIGHTED LEAST-LOADED: pick the token whose running page
            total is currently smallest (ties broken by index). After
            picking, add this PDF's page count to that token's load.
            This gives true even distribution across MinerU's per-account
            page quota even when PDFs have wildly different page counts.
          - If a token fails (HTTP error, network error, ParseError from the
            API response), exclude it and pick the next-least-loaded one.
            Up to N-1 retries.
          - Same token is used for the whole upload + poll lifecycle of one
            PDF (mid-flow switching is unsafe — file is uploaded against
            the picked account).
          - If every token fails, raise the LAST error.
        """
        pdf_bytes = path.read_bytes()
        doc_id = hashlib.md5(pdf_bytes).hexdigest()
        data_id = doc_id

        # Pre-check API hard limits. Raising here avoids wasting daily quota
        # on a submission the server will reject with 4xx anyway.
        if len(pdf_bytes) > _API_MAX_FILE_BYTES:
            size_mb = len(pdf_bytes) / (1024 * 1024)
            raise ParseError(
                f"PDF '{path.name}' is {size_mb:.1f}MB, exceeds MinerU API limit "
                f"({_API_MAX_FILE_BYTES // (1024*1024)}MB). "
                f"Set PARSER_BACKEND=mineru_local in .env for this file (no size limit)."
            )
        expected_pages = self._count_pages(pdf_bytes)
        if expected_pages > _API_MAX_PAGES:
            raise ParseError(
                f"PDF '{path.name}' has {expected_pages} pages, exceeds MinerU API limit "
                f"({_API_MAX_PAGES}). "
                f"Set PARSER_BACKEND=mineru_local in .env for this file."
            )

        last_err: Exception | None = None
        excluded: set[int] = set()
        for attempt in range(len(self._tokens)):
            token_idx_used = self._pick_least_loaded(expected_pages, excluded)
            token = self._tokens[token_idx_used]
            try:
                result = await self._parse_with_token(
                    path, pdf_bytes, doc_id, data_id, token, token_idx_used
                )
                emit(
                    "parse.api.token_load_snapshot",
                    token_loads=list(self._token_load),
                    token_used=token_idx_used,
                    expected_pages=expected_pages,
                )
                return result
            except (httpx.HTTPStatusError, httpx.RequestError, ParseError) as e:
                last_err = e
                excluded.add(token_idx_used)
                emit(
                    "parse.api.token_failed",
                    token_idx=token_idx_used,
                    attempt=attempt + 1,
                    total_tokens=len(self._tokens),
                    error_kind=type(e).__name__,
                    error=str(e)[:200],
                )
                if attempt < len(self._tokens) - 1:
                    logger.warning(
                        "parse failed with token #%d (%s); falling over to next least-loaded token: %s",
                        token_idx_used, type(e).__name__, e,
                    )
                    continue
                # All tokens exhausted → re-raise.
                raise
        # Defensive: shouldn't reach here, but mypy/lint friendly.
        if last_err is not None:
            raise last_err
        raise ParseError("no tokens available")

    async def _parse_with_token(
        self,
        path: Path,
        pdf_bytes: bytes,
        doc_id: str,
        data_id: str,
        token: str,
        token_idx_used: int,
    ) -> list[ParsedPage]:
        """Single-token parse attempt. Raises on any HTTP/parse failure;
        the outer parse_async handles fail-over."""
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=180.0) as client:
            batch_id, upload_url = await self._request_upload(
                client, path.name, data_id, token
            )
            emit(
                "parse.api.upload_url_ready",
                batch_id=batch_id,
                token_idx=token_idx_used,
                duration_ms=round((time.monotonic() - t0) * 1000, 1),
            )

            await self._upload_pdf(client, upload_url, pdf_bytes)
            emit(
                "parse.api.uploaded",
                batch_id=batch_id,
                duration_ms=round((time.monotonic() - t0) * 1000, 1),
            )

            zip_url = await self._poll_until_done(client, batch_id, data_id, token)
            emit(
                "parse.api.task_done",
                batch_id=batch_id,
                duration_ms=round((time.monotonic() - t0) * 1000, 1),
            )

            zip_resp = await client.get(zip_url, follow_redirects=True)
            zip_resp.raise_for_status()

        layout_json = self._extract_layout_json(zip_resp.content)
        page_images_b64 = render_page_images(pdf_bytes)
        for idx, b64 in enumerate(page_images_b64):
            if idx < len(layout_json.get("pdf_info", [])):
                layout_json["pdf_info"][idx]["page_image_b64"] = b64

        return middle_to_pages(layout_json, doc_id, path.name)

    async def _request_upload(
        self, client: httpx.AsyncClient, filename: str, data_id: str, token: str
    ) -> tuple[str, str]:
        """POST /api/v4/file-urls/batch → (batch_id, presigned upload URL).

        Request body shape (verified by 2026-05-10 probe):
            {"enable_formula": bool, "enable_table": bool, "language": str,
             "files": [{"name": str, "is_ocr": bool, "data_id": str}]}
        Response shape:
            {"code": 0, "msg": "ok", "data": {"batch_id": str, "file_urls": [str]}}
        """
        r = await client.post(
            f"{self._base}/api/v4/file-urls/batch",
            headers={**self._auth_headers_for(token), "Content-Type": "application/json"},
            json={
                "enable_formula": self._enable_formula,
                "enable_table": self._enable_table,
                "language": self._lang,
                "files": [{
                    "name": filename,
                    "is_ocr": self._enable_ocr,
                    "data_id": data_id,
                }],
            },
        )
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0:
            raise ParseError(f"upload-urls API failed: {body!r}")
        data = body["data"]
        return data["batch_id"], data["file_urls"][0]

    async def _upload_pdf(
        self, client: httpx.AsyncClient, upload_url: str, pdf_bytes: bytes
    ) -> None:
        """PUT to OSS presigned URL.

        ⚠️ No Content-Type / Authorization header is allowed: the OSS
        signature was computed without them, and adding any breaks the
        signature ("SignatureDoesNotMatch" 403).
        """
        put = await client.put(upload_url, content=pdf_bytes, headers={})
        if put.status_code != 200:
            raise ParseError(
                f"OSS upload failed (HTTP {put.status_code}): {put.text[:300]}"
            )

    async def _poll_until_done(
        self, client: httpx.AsyncClient, batch_id: str, data_id: str, token: str
    ) -> str:
        """Poll /extract-results/batch/{batch_id} until our data_id reaches done.

        Returns full_zip_url. Raises ParseError on failed state, TimeoutError
        if max_wait_s elapses without resolution.
        """
        deadline = time.monotonic() + self._max_wait_s
        wait_s = self._poll_initial_s  # exponential backoff capped at 30s
        url = f"{self._base}/api/v4/extract-results/batch/{batch_id}"

        while time.monotonic() < deadline:
            r = await client.get(url, headers=self._auth_headers_for(token))
            r.raise_for_status()
            body = r.json()
            if body.get("code") != 0:
                raise ParseError(f"batch-results API failed: {body!r}")
            results = body["data"]["extract_result"]
            mine = next(
                (it for it in results if it.get("data_id") == data_id), None
            )
            if mine is None:
                raise ParseError(
                    f"data_id {data_id} not found in batch {batch_id} results"
                )
            state = mine.get("state")
            if state == "done":
                return mine["full_zip_url"]
            if state == "failed":
                raise ParseError(
                    f"task failed for {data_id}: "
                    f"{mine.get('err_msg', '<no message>')}"
                )
            # state in {"waiting-file", "running"} → keep polling
            await asyncio.sleep(wait_s)
            wait_s = min(wait_s * 1.5, 30.0)

        raise TimeoutError(
            f"batch {batch_id} did not finish for data_id {data_id} "
            f"within {self._max_wait_s}s"
        )

    @staticmethod
    def _extract_layout_json(zip_bytes: bytes) -> dict:
        """Read layout.json from the result zip.

        File is named exactly 'layout.json' at the zip root, NOT '*_middle.json'
        as the official MinerU local docs would suggest. Verified 2026-05-10.
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            try:
                with zf.open("layout.json") as f:
                    return json.load(f)
            except KeyError:
                raise ParseError(
                    f"layout.json missing from zip; zip contents: {zf.namelist()}"
                )
