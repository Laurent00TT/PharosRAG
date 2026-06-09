# tests/test_mineru_api_unit.py
"""MinerUAPIParser unit tests.

No real API calls. We mock at the method-call boundary
(_request_upload / _upload_pdf / _poll_until_done / outbound httpx GET)
because the API parser constructs httpx clients per-call, which makes
patching `parser._client` infeasible.
"""
import asyncio
import io
import json
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kb.config import Settings
from kb.parser.base import ParseError
from kb.parser.mineru_api import MinerUAPIParser


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "qdrant_path": str(tmp_path / "q"),
        "image_storage_path": str(tmp_path / "i"),
        "parser_backend": "mineru_api",
        "mineru_api_url": "https://mineru.test",
        "mineru_api_token": "t0k3n",
        "mineru_api_poll_interval_s": 0.001,
        "mineru_api_max_wait_s": 1.0,
    }
    base.update(overrides)
    return Settings(**base)


def _layout_json(num_pages: int) -> dict:
    return {
        "pdf_info": [
            {
                "page_idx": i,
                "page_size": [612, 792],
                "preproc_blocks": [],
                "discarded_blocks": [],
                "para_blocks": [{
                    "type": "text",
                    "bbox": [0, 0, 100, 50],
                    "lines": [{"bbox": [0, 0, 100, 50],
                               "spans": [{"type": "text",
                                          "bbox": [0, 0, 100, 50],
                                          "content": f"page {i} text"}]}],
                    "index": 0,
                }],
            }
            for i in range(num_pages)
        ],
        "_backend": "test",
        "_version_name": "test",
    }


def _zip_with_layout(num_pages: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("layout.json", json.dumps(_layout_json(num_pages)))
        zf.writestr("full.md", "# stub")
    return buf.getvalue()


def _zip_missing_layout() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", "no layout here")
    return buf.getvalue()


# Minimal valid 1-page PDF so pypdfium2 can render in render_page_images
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
    b"4 0 obj << /Length 0 >> stream\nendstream endobj\n"
    b"xref\n0 5\n0000000000 65535 f\n"
    b"0000000010 00000 n\n0000000056 00000 n\n"
    b"0000000104 00000 n\n0000000178 00000 n\n"
    b"trailer << /Size 5 /Root 1 0 R >>\n"
    b"startxref 220\n%%EOF\n"
)


# ------------------------ _extract_layout_json ------------------------


def test_extract_layout_json_finds_file():
    parsed = MinerUAPIParser._extract_layout_json(_zip_with_layout(2))
    assert "pdf_info" in parsed
    assert len(parsed["pdf_info"]) == 2


def test_extract_layout_json_raises_when_missing():
    with pytest.raises(ParseError, match="layout.json missing"):
        MinerUAPIParser._extract_layout_json(_zip_missing_layout())


# ------------------------ _poll_until_done ------------------------


@pytest.mark.asyncio
async def test_poll_returns_zip_url_on_done(tmp_path):
    parser = MinerUAPIParser(_settings(tmp_path))
    client = MagicMock()
    client.get = AsyncMock(return_value=_resp_json({
        "code": 0, "msg": "ok",
        "data": {"batch_id": "B", "extract_result": [{
            "data_id": "myid", "file_name": "x.pdf",
            "state": "done", "full_zip_url": "https://cdn/z.zip",
            "err_msg": "",
        }]},
    }))
    url = await parser._poll_until_done(client, "B", "myid", "tok")
    assert url == "https://cdn/z.zip"


@pytest.mark.asyncio
async def test_poll_transitions_running_to_done(tmp_path):
    parser = MinerUAPIParser(_settings(tmp_path))
    responses = [
        _resp_json({"code": 0, "data": {"batch_id": "B",
            "extract_result": [{"data_id": "id", "state": "running"}]}}),
        _resp_json({"code": 0, "data": {"batch_id": "B",
            "extract_result": [{"data_id": "id", "state": "running"}]}}),
        _resp_json({"code": 0, "data": {"batch_id": "B",
            "extract_result": [{"data_id": "id", "state": "done",
                                "full_zip_url": "https://cdn/z.zip"}]}}),
    ]
    client = MagicMock()
    client.get = AsyncMock(side_effect=responses)
    url = await parser._poll_until_done(client, "B", "id", "tok")
    assert url == "https://cdn/z.zip"
    assert client.get.await_count == 3


@pytest.mark.asyncio
async def test_poll_raises_on_failed_state(tmp_path):
    parser = MinerUAPIParser(_settings(tmp_path))
    client = MagicMock()
    client.get = AsyncMock(return_value=_resp_json({
        "code": 0, "data": {"batch_id": "B",
            "extract_result": [{"data_id": "id", "state": "failed",
                                "err_msg": "OCR exploded"}]},
    }))
    with pytest.raises(ParseError, match="OCR exploded"):
        await parser._poll_until_done(client, "B", "id", "tok")


@pytest.mark.asyncio
async def test_poll_times_out(tmp_path):
    s = _settings(tmp_path, mineru_api_max_wait_s=0.05,
                  mineru_api_poll_interval_s=0.01)
    parser = MinerUAPIParser(s)
    client = MagicMock()
    client.get = AsyncMock(return_value=_resp_json({
        "code": 0, "data": {"batch_id": "B",
            "extract_result": [{"data_id": "id", "state": "running"}]},
    }))
    with pytest.raises(TimeoutError, match="did not finish"):
        await parser._poll_until_done(client, "B", "id", "tok")


@pytest.mark.asyncio
async def test_poll_raises_when_data_id_missing(tmp_path):
    parser = MinerUAPIParser(_settings(tmp_path))
    client = MagicMock()
    client.get = AsyncMock(return_value=_resp_json({
        "code": 0, "data": {"batch_id": "B",
            "extract_result": [{"data_id": "OTHER", "state": "done"}]},
    }))
    with pytest.raises(ParseError, match="not found"):
        await parser._poll_until_done(client, "B", "id", "tok")


# ------------------------ _request_upload ------------------------


@pytest.mark.asyncio
async def test_request_upload_returns_batch_and_url(tmp_path):
    parser = MinerUAPIParser(_settings(tmp_path))
    client = MagicMock()
    client.post = AsyncMock(return_value=_resp_json({
        "code": 0, "msg": "ok",
        "data": {"batch_id": "BATCH123",
                 "file_urls": ["https://oss.test/upload/abc?signature=xyz"]},
    }))
    batch_id, upload_url = await parser._request_upload(
        client, "x.pdf", "data123", "tok"
    )
    assert batch_id == "BATCH123"
    assert upload_url == "https://oss.test/upload/abc?signature=xyz"


@pytest.mark.asyncio
async def test_request_upload_raises_on_non_zero_code(tmp_path):
    parser = MinerUAPIParser(_settings(tmp_path))
    client = MagicMock()
    client.post = AsyncMock(return_value=_resp_json({
        "code": 401, "msg": "auth failed", "data": None,
    }))
    with pytest.raises(ParseError, match="upload-urls API failed"):
        await parser._request_upload(client, "x.pdf", "data123", "tok")


# ------------------------ end-to-end parse_async (heavily mocked) ------------------------


@pytest.mark.asyncio
async def test_parse_async_full_flow(tmp_path):
    """parse_async drives upload → put → poll → zip download → middle_to_pages."""
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(_MINIMAL_PDF)

    parser = MinerUAPIParser(_settings(tmp_path))

    fake_zip = _zip_with_layout(num_pages=1)

    # Patch the methods we don't want to actually hit the network on.
    async def fake_request_upload(self, client, filename, data_id, token):
        return ("BATCH-X", "https://oss.test/upload/x")

    async def fake_upload_pdf(self, client, upload_url, pdf_bytes):
        return None

    async def fake_poll(self, client, batch_id, data_id, token):
        return "https://cdn/result.zip"

    fake_zip_resp = MagicMock()
    fake_zip_resp.raise_for_status = MagicMock()
    fake_zip_resp.content = fake_zip

    # Patch the AsyncClient instance's `.get` so the only network call
    # (the zip download) returns our prepared bytes.
    real_async_client = httpx.AsyncClient

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def get(self, url, **kw):
            return fake_zip_resp

    with patch.object(MinerUAPIParser, "_request_upload", fake_request_upload), \
         patch.object(MinerUAPIParser, "_upload_pdf", fake_upload_pdf), \
         patch.object(MinerUAPIParser, "_poll_until_done", fake_poll), \
         patch("kb.parser.mineru_api.httpx.AsyncClient", _FakeClient):
        pages = await parser.parse_async(pdf)

    assert len(pages) == 1
    assert pages[0].page_num == 0
    assert "page 0 text" in pages[0].structured_text
    assert pages[0].doc_name == "fake.pdf"
    # page_image bytes are populated by render_page_images locally
    assert isinstance(pages[0].page_image, bytes)
    assert len(pages[0].page_image) > 0


# ------------------------ API limits pre-check ------------------------


@pytest.mark.asyncio
async def test_parse_async_rejects_file_over_size_limit(tmp_path):
    s = _settings(tmp_path)
    parser = MinerUAPIParser(settings=s)

    pdf = tmp_path / "big.pdf"
    # Write 201 MB (just over the 200 MB API limit)
    pdf.write_bytes(b"%PDF-1.4 " + b"\x00" * (201 * 1024 * 1024))

    with pytest.raises(ParseError, match="exceeds MinerU API limit"):
        await parser.parse_async(pdf)


@pytest.mark.asyncio
async def test_parse_async_rejects_file_over_page_limit(tmp_path):
    s = _settings(tmp_path)
    parser = MinerUAPIParser(settings=s)
    pdf = tmp_path / "many_pages.pdf"
    pdf.write_bytes(_MINIMAL_PDF)

    # Patch _count_pages so we don't have to construct a 200-page PDF
    with patch.object(MinerUAPIParser, "_count_pages", staticmethod(lambda b: 250)):
        with pytest.raises(ParseError, match="250 pages"):
            await parser.parse_async(pdf)


# ------------------------ token fail-over ------------------------


@pytest.mark.asyncio
async def test_parse_async_fails_over_to_next_token(tmp_path):
    """When the first token's parse raises, the next token is tried.

    Uses a counter that returns ParseError on the first call and a successful
    layout_json zip on the second call. We patch _parse_with_token because
    the inner state of httpx mocking is fiddly; this is the unit boundary
    parse_async actually calls.
    """
    s = _settings(tmp_path, mineru_api_token="",
                  mineru_api_tokens="t_first,t_second")
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(_MINIMAL_PDF)

    parser = MinerUAPIParser(settings=s)
    assert parser._tokens == ["t_first", "t_second"]

    call_log = []

    async def fake_parse_with_token(self, path, pdf_bytes, doc_id, data_id, token, token_idx):
        call_log.append((token, token_idx))
        if token == "t_first":
            raise ParseError("simulated quota for t_first")
        # Second token succeeds: return a single ParsedPage
        from kb.models import ParsedPage
        return [ParsedPage(
            doc_id=doc_id, doc_name=path.name, page_num=0,
            structured_text="ok", heading_path=[],
            page_image=b"", image_blocks=[], tables=[], metadata={},
        )]

    with patch.object(MinerUAPIParser, "_parse_with_token", fake_parse_with_token):
        pages = await parser.parse_async(pdf)

    assert len(pages) == 1
    assert [c[0] for c in call_log] == ["t_first", "t_second"]


@pytest.mark.asyncio
async def test_parse_async_raises_when_all_tokens_fail(tmp_path):
    s = _settings(tmp_path, mineru_api_token="",
                  mineru_api_tokens="t_a,t_b")
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(_MINIMAL_PDF)

    parser = MinerUAPIParser(settings=s)

    async def always_fail(self, *a, **kw):
        raise ParseError("everything is broken")

    with patch.object(MinerUAPIParser, "_parse_with_token", always_fail):
        with pytest.raises(ParseError, match="broken"):
            await parser.parse_async(pdf)


# ------------------------ helpers ------------------------


def _resp_json(payload: dict, status_code: int = 200) -> MagicMock:
    """Build a minimal httpx-Response-shaped MagicMock for AsyncMock returns."""
    r = MagicMock()
    r.status_code = status_code
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value=payload)
    return r
