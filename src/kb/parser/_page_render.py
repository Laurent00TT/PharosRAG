# src/kb/parser/_page_render.py
"""Per-page PDF rendering via pypdfium2.

Used by both parser backends:
- MinerULocalParser: mineru_server already injects rendered pages into
  middle_json, so it doesn't call this directly. Kept as a shared helper
  for symmetry and to allow callers that bypass the server (tests).
- MinerUAPIParser: API result zip does NOT contain page-rendered PNGs
  (only embedded figure crops), so the API parser must render locally
  and inject into layout_json the same way mineru_server does.

Centralised so behaviour does not drift between paths.
"""
import base64
import io


def render_page_images(pdf_bytes: bytes, scale: float | None = None) -> list[str]:
    """Render each PDF page as PNG and return a list of base64 strings.

    The base64 encoding matches mineru_server's existing format so
    middle_json["pdf_info"][i]["page_image_b64"] can be set without
    further conversion.

    scale = env PAGE_RENDER_SCALE (default 3.0 ≈ 216 DPI). Raised from 2.0
    (144 DPI) on 2026-05-30 for crisper VL embed (Ch3) + descriptions (Ch4).
    ⚠️ Higher DPI = more vision tokens; keep under the embed --max-model-len
    (currently 4096) + the model's max-pixels cap. Mirror of
    scripts/mineru_server.py:_render_page_images.
    See internal design notes (not published).
    """
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
