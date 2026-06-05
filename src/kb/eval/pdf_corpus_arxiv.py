"""arXiv Atom API collector helpers for Corpus v1."""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .pdf_corpus import make_manifest_record


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_entries(
    *,
    categories: list[str],
    max_results: int,
    start: int = 0,
    timeout_s: int = 45,
) -> list[dict[str, Any]]:
    query = " OR ".join(f"cat:{category}" for category in categories)
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "NaviKB corpus builder"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        text = response.read().decode("utf-8")
    # Keep a little distance from arXiv if callers loop in the future.
    time.sleep(1.0)
    return parse_arxiv_feed(text)


def parse_arxiv_feed(text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(text)
    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = _text(entry, "atom:id")
        arxiv_id = _strip_version(raw_id.rsplit("/", 1)[-1])
        categories = [
            cat.attrib.get("term", "")
            for cat in entry.findall("atom:category", ATOM_NS)
            if cat.attrib.get("term")
        ]
        entries.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(_text(entry, "atom:title").split()),
            "summary": " ".join(_text(entry, "atom:summary").split()),
            "published": _text(entry, "atom:published"),
            "categories": categories,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        })
    return entries


def build_arxiv_records(entries: list[dict[str, Any]], *, root: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for entry in entries:
        arxiv_id = entry["arxiv_id"]
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        key_id = arxiv_id.replace(".", "-").replace("/", "-")
        title = entry.get("title") or arxiv_id
        cats = ",".join(entry.get("categories") or [])
        records.append(make_manifest_record(
            doc_key=f"arxiv-{key_id}",
            source_family="academic",
            source_url=entry.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}",
            download_url=entry.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            local_path=f"datasets/pdf_corpus_v1/raw/academic/arxiv-{key_id}.pdf",
            language="en",
            domain="research_paper",
            doc_type="academic_paper",
            layout_tags=["figures", "tables", "multi_column", "text"],
            quality_tier="candidate",
            eval_role="background",
            gold_priority=0,
            license_note="arXiv public paper; follow paper license before redistribution",
            collection_policy="download_ok",
            notes=f"{title}; categories={cats}; published={entry.get('published', '')}",
            root=root,
            ingest_status="downloaded" if (root / f"datasets/pdf_corpus_v1/raw/academic/arxiv-{key_id}.pdf").exists() else "pending",
        ))
    return records


def _text(entry: ET.Element, path: str) -> str:
    node = entry.find(path, ATOM_NS)
    return "" if node is None or node.text is None else node.text.strip()


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)
