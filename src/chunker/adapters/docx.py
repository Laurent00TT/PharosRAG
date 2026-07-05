"""Adapter: Word .docx -> list[Element].  ⚠️ FALLBACK ONLY (deprecated as primary).

The project's COMMITTED docx/pptx path is MinerU's native office backend (scripts/parse_office.py
-> content_list -> from_mineru). A fair, same-metric head-to-head (the earlier "this adapter beats
MinerU 94% vs 87%" was a MEASUREMENT BUG — the comparison harness skipped MinerU's `list_items`
field) showed this adapter ties MinerU and Tika on docx text coverage (~95%) with NO fidelity
edge, AND has known gaps MinerU handles: nested tables (`_table_html` cell.text doesn't recurse)
and inline content controls (`w:sdt`). It is kept only as a zero-dependency fallback for
environments without the MinerU office backend.

OOXML is already structured, so this needs no VLM: heading *styles* give text_level (with a
bold/format fallback for the ~64% of real docx that use no heading styles — one thing MinerU does
NOT do). Running headers/footers (separate parts) are excluded by iterating the document body."""
from __future__ import annotations

import re

from ..types import Element

_HEADING_NAME_RE = re.compile(r"^Heading\s+(\d+)$", re.I)
_HEADING_ID_RE = re.compile(r"^Heading(\d+)$")
_TERMINAL = ("。", "！", "？", "；", ".", "!", "?", ";", "，", ",")
# leading enumerator (legal/numbered headings may end in a period and still be headings)
_ENUM_RE = re.compile(r"^\s*(?:section|article|clause|sec\.|art\.|chapter|part|rule|item|"
                      r"artículo|artikel|§)\b|^\s*\(?\d{1,3}[.)）]|^\s*[A-Z]\.\s", re.I)


def _para_level(para):
    """Heading level from style — prefer language-independent style_id (Heading1..9)."""
    st = para.style
    if st is None:
        return None
    sid = (getattr(st, "style_id", "") or "").strip()
    m = _HEADING_ID_RE.match(sid)
    if m:
        return int(m.group(1))
    name = (getattr(st, "name", "") or "").strip()
    m = _HEADING_NAME_RE.match(name)
    if m:
        return int(m.group(1))
    if sid.lower() == "title" or name.lower() == "title":
        return 1
    return None


def _is_format_heading(para):
    """Fallback when a doc uses NO heading styles (~56% of real-world docx): a paragraph that is
    *wholly bold*, short, and not sentence-like is a visually-formatted heading. Conservative —
    whole-para bold + short + not sentence-like. (Size-rank for sub-levels deferred.)"""
    txt = (para.text or "").strip()
    if not txt:
        return False
    toks = txt.split()
    cjk = len(toks) == 1 and any("　" <= c <= "鿿" for c in txt)   # CJK has no spaces
    if (len(txt) if cjk else len(toks)) > (25 if cjk else 14):
        return False
    if txt.endswith(_TERMINAL) and not _ENUM_RE.match(txt):    # numbered/legal headings may end in '.'
        return False
    runs = [r for r in para.runs if (r.text or "").strip()]
    return bool(runs) and all(r.bold for r in runs)


def _textbox_texts(para):
    """Text inside Word text boxes (w:txbxContent) anchored to this paragraph — nested deep in
    w:drawing / w:pict / mc:AlternateContent, so never a body child (review F1). Caller dedups
    the mc:Choice/mc:Fallback duplicate copies by text."""
    try:
        return [t for p in para._p.xpath(".//w:txbxContent//w:p")
                if (t := (p.xpath("string(.)") or "").strip())]
    except Exception:
        return []


def _table_html(table):
    rows = []
    for r in table.rows:
        cells = "".join(f"<td>{(c.text or '').strip()}</td>" for c in r.cells)
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _has_image(para):
    try:
        return bool(para._p.xpath(".//w:drawing") or para._p.xpath(".//w:pict"))
    except Exception:
        return False


def _iter_blocks(doc):
    """Yield Paragraph / Table in true document (reading) order, descending into block-level
    content controls (w:sdt) — these are direct body children whose tag is neither w:p nor
    w:tbl, so they (and their form-field content) are otherwise skipped (review)."""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    def walk(parent):
        for child in parent.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, doc)
            elif child.tag == qn("w:tbl"):
                yield Table(child, doc)
            elif child.tag == qn("w:sdt"):
                content = child.find(qn("w:sdtContent"))
                if content is not None:
                    yield from walk(content)

    yield from walk(doc.element.body)


def from_docx(path) -> list[Element]:
    """Read a .docx into Element[] (paragraphs+headings, tables interleaved in order, images).
    `page` is 0 throughout (docx is reflowable / has no fixed pages) — page-grouped and the
    per-page banner guard are intentionally inert for this format."""
    from docx import Document
    from docx.table import Table

    doc = Document(str(path))
    # if the doc uses NO heading styles, fall back to bold/format-based heading inference
    use_format = not any(_para_level(p) is not None for p in doc.paragraphs)

    out, idx, seen_tb = [], 0, set()
    for block in _iter_blocks(doc):
        if isinstance(block, Table):
            out.append(Element(idx=idx, kind="table", text=None, table_body=_table_html(block), page=0))
            idx += 1
            continue
        # Paragraph
        txt = (block.text or "").strip()
        lvl = _para_level(block)
        if lvl is None and use_format and _is_format_heading(block):
            lvl = 1
        img = _has_image(block)
        if txt:
            out.append(Element(idx=idx, kind="text", text=txt, text_level=lvl, page=0))
            idx += 1
        for tb in _textbox_texts(block):          # text-box content (review F1); dedup Choice/Fallback
            if tb not in seen_tb:
                seen_tb.add(tb)
                out.append(Element(idx=idx, kind="text", text=tb, text_level=None, page=0))
                idx += 1
        if img:                                   # a paragraph may carry an inline image too
            out.append(Element(idx=idx, kind="image", text=None, page=0))
            idx += 1
    return out
