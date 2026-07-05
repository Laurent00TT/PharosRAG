"""Ingest-side helper: extract DOCUMENT-level metadata + carry an access policy. This is NOT part
of the format-agnostic chunker core — extraction is per-format (office core.xml, PDF /Info), so it
lives at the ingest seam. The chunker only STAMPS the result onto every chunk (see core.chunk's
doc_meta/acl params), which then ride into the vector-store payload for FILTER + citation (doc_meta)
and HARD access pre-filter (acl)."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

_JUNK_META = {"", "admin", "user", "windows user", "unknown", "untitled"}
_OFFICE = {".docx", ".pptx", ".xlsx"}
# doc_meta is CONVENIENCE (filter + citation), never a security channel. Access policy travels in the
# SEPARATE `acl` param of Chunker.chunk, NOT here. So we refuse to copy any access-semantic key out of
# **overrides** — a poisoned manifest splatted in (see INTEGRATION §1) must not be able to smuggle
# `acl`/`tenant`/`allow` into doc_meta, where a buggy downstream might misread it as the policy.
_ACL_KEYS = {"acl", "tenant", "allow", "deny", "visibility", "classification", "principals", "unset"}


def _clean(v):
    return "" if (v or "").strip().lower() in _JUNK_META else (v or "").strip()


def _office_core(path):
    with zipfile.ZipFile(path) as z:
        if "docProps/core.xml" not in z.namelist():
            return {}
        x = z.read("docProps/core.xml").decode("utf-8", "ignore")
    g = lambda t: _clean((re.search(rf"<{t}[^>]*>([^<]*)</{t}>", x) or [None, ""])[1])
    return {"title": g("dc:title"), "author": g("dc:creator"),
            "created": g("dcterms:created")[:10], "modified": g("dcterms:modified")[:10]}


def _pdf_info(path):
    try:
        from pypdf import PdfReader
    except Exception:
        return {}
    r = PdfReader(str(path))
    info = r.metadata or {}
    cre = str(info.get("/CreationDate") or "")
    m = re.search(r"(\d{4})(\d\d)(\d\d)", cre)
    return {"title": _clean(str(info.get("/Title") or "")),
            "author": _clean(str(info.get("/Author") or "")),
            "created": f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "",
            "page_count": len(r.pages)}


def extract_doc_meta(path, **overrides) -> dict:
    """Document metadata from the source file, merged with caller overrides (e.g. source / doc_type /
    domain / lang from a manifest — these WIN over file-embedded values). Empty/junk values dropped."""
    path = Path(path)
    meta = {"doc_id": path.stem, "format": path.suffix.lstrip(".").lower()}
    try:
        if path.suffix.lower() in _OFFICE:
            meta.update(_office_core(path))
        elif path.suffix.lower() == ".pdf":
            meta.update(_pdf_info(path))
    except Exception:
        pass
    meta.update({k: v for k, v in overrides.items()
                 if v not in (None, "") and k not in _ACL_KEYS})   # never carry access-semantic keys
    return {k: v for k, v in meta.items() if v not in (None, "", [])}
