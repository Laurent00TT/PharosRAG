from __future__ import annotations

import hashlib
import re


def normalize_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.strip().lower())
    return normalized


def nav_entry_id(doc_id: str, entry_type: str, label: str, page_start: int, page_end: int) -> str:
    raw = f"{doc_id}:{entry_type}:{normalize_label(label)}:{page_start}:{page_end}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
