"""Adapter: MinerU output -> list[Element]. This is the seam where the parse stage plugs in;
swapping MinerU for another parser means writing another adapter, not touching the core."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..types import Element

_NOISE = {"header", "footer", "page_number"}


def _key(s):
    return re.sub(r"\s+", "", (s or "")).lower()[:24]


def _merge_lookup(layout):
    """page -> [(text_key, merge_prev)] from layout para_blocks (cross-page continuation)."""
    table = {}
    for pg in (layout or {}).get("pdf_info", []):
        rows = []
        for b in pg.get("para_blocks", []) or []:
            txt = " ".join(s.get("content", "") for ln in b.get("lines", []) or []
                           for s in ln.get("spans", []) or [])
            rows.append((_key(txt), bool(b.get("merge_prev"))))
        table[pg.get("page_idx", 0)] = rows
    return table


def _merge_prev(text, page, lookup):
    k = _key(text)
    if len(k) < 6:
        return False
    for tk, mp in lookup.get(page, []):
        if tk and (tk.startswith(k[:12]) or k.startswith(tk[:12])):
            return mp
    return False


def _join(v):
    return " ".join(v) if isinstance(v, list) else (str(v) if v else None)


def from_mineru(content_list, layout=None) -> list[Element]:
    """content_list: parsed *_content_list.json (list). layout: parsed layout.json (dict, optional;
    needed only for cross-page merge_prev)."""
    lookup = _merge_lookup(layout) if layout else {}
    out = []
    for i, el in enumerate(content_list):
        kind = el.get("type", "text")
        page = el.get("page_idx", 0)
        text = el.get("text")
        cap = _join(el.get("table_caption") or el.get("image_caption") or el.get("chart_caption"))
        foot = _join(el.get("table_footnote") or el.get("image_footnote") or el.get("chart_footnote"))
        merge = (_merge_prev(text, page, lookup)
                 if kind in ("text", "list") and text and kind not in _NOISE else False)
        out.append(Element(
            idx=i, kind=kind, text=text, text_level=el.get("text_level"), page=page,
            bbox=el.get("bbox"), list_items=el.get("list_items"), caption=cap, footnote=foot,
            table_body=el.get("table_body"),
            asset_content=el.get("content") if kind in ("image", "chart") else None,
            sub_type=el.get("sub_type"), merge_prev=merge,
            image_path=el.get("img_path")))       # verbatim; embed stage resolves vs MinerU output root
    return out


def from_mineru_dir(doc_dir) -> list[Element]:
    """Convenience: read a MinerU output dir (the *_content_list.json + layout.json)."""
    doc_dir = Path(doc_dir)
    cl = [p for p in doc_dir.rglob("*content_list.json") if "_v2" not in p.name][0]
    lay = next(iter(doc_dir.rglob("layout.json")), None)
    content = json.loads(cl.read_text(encoding="utf-8"))
    layout = json.loads(lay.read_text(encoding="utf-8")) if lay else None
    return from_mineru(content, layout)
