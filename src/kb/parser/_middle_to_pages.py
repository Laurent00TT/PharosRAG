# src/kb/parser/_middle_to_pages.py
"""Convert MinerU middle_json (or API zip's layout.json — same schema) to ParsedPage list.

Centralised so both MinerULocalParser and MinerUAPIParser produce identical
ParsedPage shape from identical middle_json / layout.json input. Field
mapping is based on the MinerU 3.x BlockType enum (verified by
probe_mineru.py on 2026-05-09 and re-verified against API zip on 2026-05-10):

  - Prefer the post-finalize `para_blocks` (cleaned + cross-page merged).
  - Fall back to raw `preproc_blocks` if the former is missing.
  - `discarded_blocks` (header / footer / page_number) are already
    separated by MinerU; we do not pull from them.
"""
import base64
import logging

from kb.models import ParsedPage

logger = logging.getLogger(__name__)


# Derived from MinerU 3.x mineru.utils.enum_class.BlockType:
# heading hierarchy — doc_title > title > paragraph_title
_HEADING_TYPES = {"doc_title", "title", "paragraph_title"}

# Body text types (formulas / code / list / abstract belong here too)
_TEXT_TYPES = {
    "text", "abstract", "list", "ref_text", "aside_text",
    "code", "code_body",
    "equation", "interline_equation",
    "algorithm",
}

# Tables (body + caption — caption goes through _extract_caption for context)
_TABLE_TYPES = {"table", "table_body"}

# Image-like blocks (image / chart only, but we also pick up image_body)
_IMAGE_TYPES = {"image", "image_body", "chart", "chart_body"}

# Caption sub-block types — extracted as the caption field of their parent
_CAPTION_TYPES = {
    "caption", "image_caption", "table_caption", "chart_caption",
    "algorithm_caption", "code_caption",
}

# Heading levels (smaller = more important)
_HEADING_LEVEL = {
    "doc_title": 1,
    "title": 2,
    "paragraph_title": 3,
}


def middle_to_pages(
    middle_json: dict,
    doc_id: str,
    doc_name: str,
) -> list[ParsedPage]:
    """Top-level conversion.

    `middle_json` shape (both local middle_json and API layout.json):
        {"pdf_info": [{"page_idx": int, "page_size": [w,h],
                       "para_blocks": [...], "preproc_blocks": [...],
                       "discarded_blocks": [...],
                       "page_image_b64": <optional, injected by caller>}, ...]}
    """
    pages_info = middle_json.get("pdf_info", [])
    # Document-level heading stack (used to build heading_path per page)
    heading_stack: list[str] = []
    result: list[ParsedPage] = []

    for page_info in pages_info:
        page_num = page_info.get("page_idx", 0)
        # Prefer para_blocks (post-processed); fall back to preproc_blocks (raw)
        blocks = page_info.get("para_blocks") or page_info.get("preproc_blocks", [])

        structured_text_parts: list[str] = []
        tables: list[dict] = []
        image_blocks: list[dict] = []

        for block in blocks:
            btype = block.get("type", "")

            if btype in _HEADING_TYPES:
                text = _extract_text(block)
                if not text:
                    continue
                level = _HEADING_LEVEL.get(btype, 1)
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(text)
                structured_text_parts.append(f"{'#' * level} {text}")
            elif btype in _TABLE_TYPES:
                html = _extract_table_html(block)
                caption = _extract_caption(block)
                tables.append({
                    "html": html,
                    "page_num": page_num,
                    "caption": caption,
                })
                structured_text_parts.append(f"[TABLE]\n{html}")
            elif btype in _IMAGE_TYPES:
                caption = _extract_caption(block)
                image_blocks.append({
                    "page_num": page_num,
                    "caption": caption,
                })
                if caption:
                    structured_text_parts.append(f"[IMAGE: {caption}]")
            elif btype in _TEXT_TYPES:
                text = _extract_text(block)
                if text:
                    structured_text_parts.append(text)
            # Other types (discarded / seal / page_number / etc.) are intentionally skipped

        structured_text = "\n\n".join(p for p in structured_text_parts if p)

        # Decode page_image_b64 (injected by mineru_server via pypdfium2,
        # or by MinerUAPIParser which renders locally with the same helper).
        page_image_bytes = b""
        if "page_image_b64" in page_info:
            try:
                page_image_bytes = base64.b64decode(page_info["page_image_b64"])
            except Exception as e:
                logger.warning("page_image_b64 decode failed (%s); ignoring", e)

        result.append(ParsedPage(
            doc_id=doc_id,
            doc_name=doc_name,
            page_num=page_num,
            structured_text=structured_text,
            heading_path=list(heading_stack),
            page_image=page_image_bytes,
            image_blocks=image_blocks,
            tables=tables,
            metadata={
                "page_size": page_info.get("page_size"),
                "block_count": len(blocks),
                "block_field": "para_blocks" if page_info.get("para_blocks") else "preproc_blocks",
            },
        ))

    return result


def _extract_text(block: dict) -> str:
    """Extract text from a block's lines/spans.
    Recurses into nested blocks (e.g. image_body containing a caption).
    MinerU 3.x span types: 'text' / 'inline_equation' / 'interline_equation' / 'image' / 'table'
    """
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            stype = span.get("type", "")
            # text + equations both expose .content (LaTeX for the equation cases)
            if stype in ("text", "inline_equation", "interline_equation"):
                content = span.get("content", "")
                if content:
                    parts.append(content)
    # Nested blocks (image_body with caption / table_body with caption / etc.)
    for sub in block.get("blocks", []):
        sub_text = _extract_text(sub)
        if sub_text:
            parts.append(sub_text)
    return " ".join(parts).strip()


def _extract_table_html(block: dict) -> str:
    """Extract HTML content from a table block."""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == "table":
                html = span.get("html", "")
                if html:
                    return html
    for sub in block.get("blocks", []):
        html = _extract_table_html(sub)
        if html:
            return html
    return ""


def _extract_caption(block: dict) -> str:
    """Extract caption from an image/table block (sub-block types listed in _CAPTION_TYPES)."""
    for sub in block.get("blocks", []):
        sub_type = sub.get("type", "")
        if sub_type in _CAPTION_TYPES:
            return _extract_text(sub)
    return ""
