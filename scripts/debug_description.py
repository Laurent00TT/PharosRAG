"""Debug Description Channel: reproduce ingest's exact call sequence.

Independent test returned proper JSON (337 chars), but during ingest the same
LLM call returned empty content. This script narrows down the cause:
  1. Calls real mineru_server to get the actual rendered page image
  2. Replicates the ingest sequence: vision embed → description
  3. Logs the raw LLM response content

Run with mineru_server (8001), multimodal embedding server (8003), llama-server (1235) up.
"""
import asyncio
import base64
import io
import logging
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.adapters.base import (
    AdapterMessage, ImagePart, build_multimodal_content,
    DISABLE_THINKING_EXTRA_BODY,
)
from kb.adapters.openai_compat import OpenAICompatAdapter
from kb.config import Settings
from kb.ingestion.channels.description import DescriptionChannel, _PROMPT
from kb.ingestion.channels.vision import VisionChannel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def fetch_real_page_image(pdf_path: Path) -> bytes:
    """Call mineru_server /parse and extract first page image bytes."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            "http://localhost:8001/parse",
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
            data={"mode": "hybrid", "include_page_images": "true"},
        )
        resp.raise_for_status()
        middle_json = resp.json()
    pages = middle_json.get("pdf_info", [])
    if not pages or "page_image_b64" not in pages[0]:
        raise RuntimeError("No page_image_b64 in mineru response")
    return base64.b64decode(pages[0]["page_image_b64"])


async def variant_a_minimal(image_bytes: bytes) -> str:
    """Variant A: only OpenAICompatAdapter + description prompt (no other steps).
    This is what the standalone test did — it succeeded."""
    settings = Settings()
    adapter = OpenAICompatAdapter(
        base_url=settings.description_url,
        api_key=settings.description_api_key,
        model=settings.description_model,
        supports_vision=settings.description_supports_vision,
        extra_body=DISABLE_THINKING_EXTRA_BODY,
    )
    msgs = [
        AdapterMessage(role="system", content=_PROMPT),
        AdapterMessage(role="user", content=build_multimodal_content(
            text="Analyze this page (heading: none):",
            images=[ImagePart(data=image_bytes, mime_type="image/png")],
        )),
    ]
    r = await adapter.chat(msgs)
    return r.content, r.thinking_chars, r.content_chars


async def variant_b_with_vision_first(image_bytes: bytes) -> str:
    """Variant B: vision embed first (HTTP to the multimodal embedding
    server), then description. Same order as ingest pipeline."""
    settings = Settings()

    # Step 1: vision embed (mimics IngestionPipeline.ingest line 88)
    vision_ch = VisionChannel(settings=settings)
    logger.info("Step 1: Qwen3-VL single-vector vision embed...")
    vision_emb = await vision_ch.embed_async(image_bytes)
    logger.info(f"  vision_emb: {'OK' if vision_emb else 'None'}, "
                f"dim={len(vision_emb) if vision_emb else None}")

    # Step 2: description (mimics IngestionPipeline.ingest line 105)
    logger.info("Step 2: Description channel...")
    adapter = OpenAICompatAdapter(
        base_url=settings.description_url,
        api_key=settings.description_api_key,
        model=settings.description_model,
        supports_vision=settings.description_supports_vision,
        extra_body=DISABLE_THINKING_EXTRA_BODY,
    )
    msgs = [
        AdapterMessage(role="system", content=_PROMPT),
        AdapterMessage(role="user", content=build_multimodal_content(
            text="Analyze this page (heading: none):",
            images=[ImagePart(data=image_bytes, mime_type="image/png")],
        )),
    ]
    r = await adapter.chat(msgs)
    return r.content, r.thinking_chars, r.content_chars


async def main():
    pdf = Path("smoke_test.pdf")
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    logger.info("Fetching real page image from mineru_server...")
    img_bytes = await fetch_real_page_image(pdf)
    logger.info(f"Got page image: {len(img_bytes)} bytes")

    print()
    print("=" * 70)
    print("Variant A: standalone (no vision step before)")
    print("=" * 70)
    try:
        content, thinking, content_chars = await variant_a_minimal(img_bytes)
        print(f"thinking_chars: {thinking}")
        print(f"content_chars : {content_chars}")
        print(f"content       : {content[:300]!r}")
    except Exception as e:
        print(f"FAIL: {e}")

    print()
    print("=" * 70)
    print("Variant B: ingest-like (vision embed first, then description)")
    print("=" * 70)
    try:
        content, thinking, content_chars = await variant_b_with_vision_first(img_bytes)
        print(f"thinking_chars: {thinking}")
        print(f"content_chars : {content_chars}")
        print(f"content       : {content[:300]!r}")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    asyncio.run(main())
