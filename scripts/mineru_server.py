# scripts/mineru_server.py
"""MinerU parsing service — FastAPI wrapper. Built on the real MinerU 3.x API
(verified by probe_mineru.py on 2026-05-09).

Backend entry-point mapping (the function name in all three backends is doc_analyze;
NOT the older hybrid_analyze / vlm_analyze names):
  hybrid    -> mineru.backend.hybrid.hybrid_analyze.doc_analyze
              return: (middle_json, model_list, _vlm_ocr_enable)
  vlm       -> mineru.backend.vlm.vlm_analyze.doc_analyze
              return: (middle_json, results)
  pipeline  -> mineru.backend.pipeline.pipeline_analyze.doc_analyze_streaming
              callback: on_doc_ready(doc_index, model_list, middle_json, ocr_enable)

Usage:
  pip install ".[mineru,server]"
  python scripts/mineru_server.py --port 8001

Verify:
  curl http://localhost:8001/health
  curl -X POST http://localhost:8001/parse -F "file=@sample.pdf" -F "mode=hybrid"
"""
import base64
import io
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, UploadFile, HTTPException, Form
    from fastapi.responses import JSONResponse, Response
except ImportError as e:
    raise SystemExit(f"Install server extras: pip install '.[server]'\n{e}")


def _configure_model_source() -> None:
    """Read mineru_model_source from Settings and inject it as the
    MINERU_MODEL_SOURCE env var.

    Must be called BEFORE importing mineru.* — by the time MinerU starts up,
    the default has already been resolved.
    """
    # If already set in the environment, respect it (shell export wins)
    if os.environ.get("MINERU_MODEL_SOURCE"):
        logger.info("Using MINERU_MODEL_SOURCE from env: %s",
                    os.environ["MINERU_MODEL_SOURCE"])
        return
    try:
        from kb.config import Settings
        os.environ["MINERU_MODEL_SOURCE"] = Settings().mineru_model_source
        logger.info("Set MINERU_MODEL_SOURCE=%s from Settings",
                    os.environ["MINERU_MODEL_SOURCE"])
    except Exception as e:
        # If Settings cannot be read, do not block startup — let MinerU use
        # its built-in default (huggingface).
        logger.warning("Could not read Settings.mineru_model_source: %s", e)


def _configure_device() -> None:
    """Default to CUDA. Per-PDF subprocess worker model: this server is
    expected to be spawned by the ingestion pipeline for one PDF, then
    killed via /exit so the OS reclaims the ~6.6GB GPU footprint before
    the next PDF (MinerU has no unload API).

    Override with MINERU_DEVICE_MODE=cpu when running long-lived for
    debugging or when GPU is unavailable.
    """
    if os.environ.get("MINERU_DEVICE_MODE"):
        logger.info("Using MINERU_DEVICE_MODE from env: %s",
                    os.environ["MINERU_DEVICE_MODE"])
        return
    os.environ["MINERU_DEVICE_MODE"] = "cuda"
    logger.info("Set MINERU_DEVICE_MODE=cuda (default — per-PDF worker model)")


_configure_model_source()
_configure_device()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MinerU server starting (models load on first /parse request)")
    yield


app = FastAPI(title="MinerU Parsing Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "mineru_server"}


@app.post("/exit")
async def exit_server():
    """Trigger a hard process exit so the OS reclaims GPU memory.

    MinerU's doc_analyze() loads layout/OCR/table singletons that have no
    public unload API; the only reliable way to free ~6.6GB GPU is to let
    the process die. The pipeline's per-PDF worker spawns a fresh server,
    runs /parse, then calls /exit so the OS reclaims the GPU before the
    next PDF.
    """
    import asyncio
    async def _bye():
        await asyncio.sleep(0.1)  # let the response flush first
        os._exit(0)
    asyncio.create_task(_bye())
    return {"status": "exiting"}


# Busy flag for 503 pushback. A mineru instance processes one PDF at a time
# (the model is a singleton; concurrent /parse would queue in the threadpool
# and contend for GPU). Returning 503 when busy lets the KB-side router retry
# a DIFFERENT instance instead of piling onto a busy one — turns dumb
# round-robin into busy-aware dispatch with no shared state (the 503 signal IS
# the coordination). Safe without a lock: the `if`+set below run synchronously
# in the single-threaded event loop with no await between them.
_parse_busy = False


@app.post("/parse")
async def parse(
    file: UploadFile,
    mode: str = Form("hybrid"),         # "hybrid" | "vlm" | "pipeline"
    lang: str = Form("ch"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    include_page_images: bool = Form(True),
):
    """Parse the uploaded document and return the MinerU middle_json structure.
    Optionally injects each rendered page image (PNG base64) into
    middle_json["pdf_info"][i]["page_image_b64"].

    Returns 503 if this instance is already parsing — caller should retry a
    different instance (see src/kb/parser/mineru_local.py).
    """
    if mode not in ("hybrid", "vlm", "pipeline"):
        raise HTTPException(400, f"Unknown mode: {mode}")

    global _parse_busy
    if _parse_busy:
        raise HTTPException(503, "mineru instance busy: another /parse in progress")
    _parse_busy = True
    try:
        return await _parse_impl(
            file, mode, lang, formula_enable, table_enable, include_page_images
        )
    finally:
        _parse_busy = False


async def _parse_impl(
    file, mode, lang, formula_enable, table_enable, include_page_images,
):
    pdf_bytes = await file.read()

    # Offload the sync parse to a worker thread so a long-running parse
    # does not block this server's event loop. Without this, a single
    # /parse call would freeze /health and any other concurrent request
    # for the duration of the parse (tens of seconds to minutes).
    import asyncio
    try:
        middle_json = await asyncio.to_thread(
            _run_mineru,
            pdf_bytes,
            mode,
            lang,
            formula_enable,
            table_enable,
        )
    except Exception as e:
        logger.exception("MinerU parsing failed")
        raise HTTPException(500, f"Parsing failed: {e}")

    if include_page_images:
        try:
            page_images = _render_page_images(pdf_bytes)
            for idx, png_b64 in enumerate(page_images):
                if idx < len(middle_json.get("pdf_info", [])):
                    middle_json["pdf_info"][idx]["page_image_b64"] = png_b64
        except Exception as e:
            logger.warning("Page image rendering failed (%s); middle_json has no page_image_b64", e)

    # Serialize manually with lone-surrogate scrubbing. MinerU's text
    # extraction can emit unpaired UTF-16 surrogates (e.g. '\ud835' from a
    # mathematical-bold char like 𝐀 U+1D400 split mid-pair) — common in
    # math-heavy arxiv PDFs. JSONResponse → str.encode('utf-8') then raises
    # "surrogates not allowed" and 500s the whole multi-hundred-page doc.
    # json.dumps tolerates surrogates in the str; the single
    # .encode('utf-8', 'replace') pass scrubs any lone surrogate to '?'.
    # (Caught 2026-05-28 on 2311.16502v3.pdf, 117pp.)
    body = json.dumps(middle_json, ensure_ascii=False).encode("utf-8", "replace")
    return Response(content=body, media_type="application/json")


def _run_mineru(
    pdf_bytes: bytes,
    mode: str,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
) -> dict:
    """Synchronously run MinerU and return middle_json.

    All backends share DummyDataWriter (we only want middle_json, not
    side-effect files). The per-page rendered image is generated separately
    by the caller using pypdfium2, so it does not depend on image_writer.
    """
    try:
        from mineru.data.data_reader_writer import DummyDataWriter
    except ImportError as e:
        raise RuntimeError(
            "MinerU is not installed correctly: pip install mineru\n"
            f"Original error: {e}"
        )

    writer = DummyDataWriter()

    if mode == "hybrid":
        from mineru.backend.hybrid.hybrid_analyze import doc_analyze
        middle_json, _model_list, _vlm_ocr = doc_analyze(
            pdf_bytes=pdf_bytes,
            image_writer=writer,
            backend="transformers",        # local GPU; switch to "http-client" + server_url for an external vLLM
            parse_method="auto",
            language=lang,
            inline_formula_enable=formula_enable,
        )
        return middle_json

    if mode == "vlm":
        from mineru.backend.vlm.vlm_analyze import doc_analyze
        # transformers backend = in-process model.generate(). vllm-engine
        # and http-client backends were explored 2026-05-27 (see
        # internal design notes (not published)
        # §11 "vllm path abandoned") but produced garbage `!!!!!` output
        # on MinerU2.5-Pro-2605-1.2B via vllm 0.21. We get throughput by
        # running multiple full-mineru instances instead — see _start_
        # mineru.sh PORT param + MINERU_SERVER_URLS round-robin on the
        # client side.
        middle_json, _results = doc_analyze(
            pdf_bytes=pdf_bytes,
            image_writer=writer,
            backend="transformers",
        )
        return middle_json

    # pipeline mode — streaming API, collect synchronously
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
    results: list[dict] = []

    def collect(doc_index, model_list, middle_json, ocr_enable):
        results.append(middle_json)

    doc_analyze_streaming(
        pdf_bytes_list=[pdf_bytes],
        image_writer_list=[writer],
        lang_list=[lang],
        on_doc_ready=collect,
        parse_method="auto",
        formula_enable=formula_enable,
        table_enable=table_enable,
    )
    return results[0] if results else {}


# Page rendering. Logically shared with kb.parser._page_render, but kept
# inline here so this script can run in an isolated conda env without the
# kb.* package installed (e.g. when transformers version pinning forces
# mineru into its own env separate from the main KB stack).
# If you edit this, mirror the change to src/kb/parser/_page_render.py.
def _render_page_images(pdf_bytes: bytes, scale: float | None = None) -> list[str]:
    """Render each PDF page as base64-PNG string. Used to inject
    middle_json["pdf_info"][i]["page_image_b64"].

    scale = env PAGE_RENDER_SCALE (default 3.0 ≈ 216 DPI). Raised from 2.0
    (144 DPI) on 2026-05-30 so the VL embedder (Ch3) + description VLM (Ch4)
    get a crisper page image. ⚠️ Higher DPI = more vision tokens per page —
    keep under the embed server's --max-model-len (currently 4096) AND the
    model's internal max-pixels cap, else vision embed 400s / truncates.
    Tune the exact value together with a re-ingest (it only affects newly
    rendered pages). See internal design notes (not published).
    If you edit this, mirror to src/kb/parser/_page_render.py."""
    import base64
    import io
    import os
    import pypdfium2 as pdfium

    if scale is None:
        scale = float(os.environ.get("PAGE_RENDER_SCALE", "3.0"))

    pdf = pdfium.PdfDocument(pdf_bytes)
    out: list[str] = []
    try:
        for page in pdf:
            bitmap = page.render(scale=scale)
            pil_img = bitmap.to_pil()
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            page.close()
    finally:
        pdf.close()
    return out


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
