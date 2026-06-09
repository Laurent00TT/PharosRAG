"""PDF Corpus v1 manifest helpers.

The corpus workflow is intentionally manifest-first: scripts can compose these
small helpers without importing ingestion/model code, while tests can validate
the metadata contract without touching external services.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "pdf_corpus_manifest_v1"

REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "doc_key",
    "source_family",
    "source_url",
    "download_url",
    "local_path",
    "sha256",
    "file_size_mb",
    "page_count",
    "language",
    "domain",
    "doc_type",
    "layout_tags",
    "quality_tier",
    "eval_role",
    "gold_priority",
    "license_note",
    "collection_policy",
    "ingest_status",
    "parse_status",
    "notes",
}

CONTROLLED_VALUES = {
    "quality_tier": {"smoke", "gold", "medium", "candidate", "rejected"},
    "eval_role": {"smoke", "gold", "background", "stress_only", "parse_only"},
    "source_family": {
        "mmdocir",
        "govinfo",
        "sec",
        "manual",
        "academic",
        "docvqa_like",
        "omnidocbench",
        "synthetic",
        "eastmoney",
        "goldman_sachs",
        "jpmorgan",
        "msci",
        "sia",
        "spglobal",
        "ubs",
        "other_financial",
        "other",
    },
    "ingest_status": {"pending", "downloaded", "ingested", "failed", "skipped"},
    "parse_status": {"unknown", "text_pdf", "scanned", "mixed", "failed"},
    "collection_policy": {
        "download_ok",
        "manual_download",
        "metadata_only",
        "do_not_redistribute",
    },
}


@dataclass(frozen=True)
class ManifestIssue:
    line_no: int
    field: str
    message: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def load_manifest_or_empty(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_record(record: dict[str, Any], line_no: int) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    for field in sorted(REQUIRED_MANIFEST_FIELDS):
        if field not in record:
            issues.append(ManifestIssue(
                line_no=line_no,
                field=field,
                message=f"missing required field: {field}",
            ))

    if record.get("schema_version") not in (None, SCHEMA_VERSION):
        issues.append(ManifestIssue(
            line_no=line_no,
            field="schema_version",
            message=f"unsupported schema_version: {record.get('schema_version')!r}",
        ))

    for field, allowed in CONTROLLED_VALUES.items():
        if field not in record:
            continue
        value = record[field]
        if value not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            issues.append(ManifestIssue(
                line_no=line_no,
                field=field,
                message=f"invalid {field}: {value!r}; expected one of: {allowed_str}",
            ))

    if "layout_tags" in record and not isinstance(record["layout_tags"], list):
        issues.append(ManifestIssue(
            line_no=line_no,
            field="layout_tags",
            message="layout_tags must be a list",
        ))

    if "gold_priority" in record and not isinstance(record["gold_priority"], int):
        issues.append(ManifestIssue(
            line_no=line_no,
            field="gold_priority",
            message="gold_priority must be an integer",
        ))

    return issues


def validate_manifest(path: Path) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    if not path.exists():
        return [ManifestIssue(0, "path", f"manifest not found: {path}")]

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(ManifestIssue(line_no, "json", f"malformed json: {exc.msg}"))
            continue
        issues.extend(validate_record(record, line_no=line_no))
    return issues


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_count_pdf_pages(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:
        return None


def enrich_local_pdf_metadata(record: dict[str, Any], root: Path) -> dict[str, Any]:
    enriched = dict(record)
    local_path = enriched.get("local_path")
    if not local_path:
        return enriched

    pdf_path = _resolve_under_root(root, local_path)
    if not pdf_path.exists():
        return enriched

    enriched["sha256"] = compute_sha256(pdf_path)
    enriched["file_size_mb"] = round(pdf_path.stat().st_size / (1024 * 1024), 6)
    enriched["page_count"] = try_count_pdf_pages(pdf_path)
    return enriched


def make_manifest_record(
    *,
    doc_key: str,
    source_family: str,
    source_url: str,
    download_url: str,
    local_path: str,
    language: str,
    domain: str,
    doc_type: str,
    layout_tags: list[str],
    quality_tier: str,
    eval_role: str,
    gold_priority: int,
    license_note: str,
    collection_policy: str,
    notes: str,
    root: Path,
    ingest_status: str = "pending",
    parse_status: str = "unknown",
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "doc_key": doc_key,
        "source_family": source_family,
        "source_url": source_url,
        "download_url": download_url,
        "local_path": local_path,
        "sha256": "",
        "file_size_mb": 0.0,
        "page_count": None,
        "language": language,
        "domain": domain,
        "doc_type": doc_type,
        "layout_tags": sorted(set(layout_tags)),
        "quality_tier": quality_tier,
        "eval_role": eval_role,
        "gold_priority": gold_priority,
        "license_note": license_note,
        "collection_policy": collection_policy,
        "ingest_status": ingest_status,
        "parse_status": parse_status,
        "notes": notes,
    }
    return enrich_local_pdf_metadata(record, root=root)


def merge_records_by_doc_key(
    existing: Iterable[dict[str, Any]],
    new_records: Iterable[dict[str, Any]],
    *,
    replace_source: str | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in existing:
        if replace_source is not None and record.get("source_family") == replace_source:
            continue
        merged[str(record["doc_key"])] = record
    for record in new_records:
        merged[str(record["doc_key"])] = record
    return sort_for_subset(merged.values())


def sort_for_subset(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            -int(r.get("gold_priority", 0) or 0),
            str(r.get("source_family", "")),
            str(r.get("doc_key", "")),
        ),
    )


def select_subset(
    records: Iterable[dict[str, Any]],
    *,
    tier: str | None = None,
    eval_role: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if tier is not None and record.get("quality_tier") != tier:
            continue
        if eval_role is not None and record.get("eval_role") != eval_role:
            continue
        selected.append(record)

    selected = sort_for_subset(selected)
    if limit is not None:
        return selected[:limit]
    return selected


def filter_existing_local_pdfs(
    records: Iterable[dict[str, Any]],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    existing = []
    for record in records:
        local_path = record.get("local_path")
        if not local_path:
            continue
        if _resolve_under_root(root, local_path).exists():
            existing.append(record)
    return existing


def write_pdf_list(path: Path, records: Iterable[dict[str, Any]], *, root: Path) -> None:
    lines: list[str] = []
    for record in records:
        local_path = record.get("local_path")
        if not local_path:
            raise ValueError(f"record {record.get('doc_key', '<unknown>')} has no local_path")
        lines.append(str(_resolve_under_root(root, local_path).resolve()))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def normalize_mmdocir_doc(
    doc: dict[str, Any],
    *,
    pdf_path: Path,
    root: Path,
    quality_tier: str,
    eval_role: str,
) -> dict[str, Any]:
    doc_name = str(doc["doc_name"])
    domain = _slugify(str(doc.get("domain") or "unknown")).replace("-", "_")
    return make_manifest_record(
        doc_key=f"mmdocir-{_slugify(Path(doc_name).stem)}",
        source_family="mmdocir",
        source_url="https://mmdocrag.github.io/MMDocIR/",
        download_url="",
        local_path=_relative_posix(pdf_path, root),
        language="en",
        domain=domain,
        doc_type="benchmark_pdf",
        layout_tags=_layout_tags_from_mmdocir_questions(doc.get("questions", [])),
        quality_tier=quality_tier,
        eval_role=eval_role,
        gold_priority=int(doc.get("gold_priority", 0) or 0),
        license_note="MMDocIR benchmark asset; check upstream terms before redistribution",
        collection_policy="metadata_only",
        notes=f"MMDocIR domain={doc.get('domain', '')}; questions={len(doc.get('questions', []))}",
        root=root,
    )


def _resolve_under_root(root: Path, local_path: str) -> Path:
    path = Path(local_path)
    if path.is_absolute():
        return path
    return root / path


def _relative_posix(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "document"


def _layout_tags_from_mmdocir_questions(questions: Iterable[dict[str, Any]]) -> list[str]:
    tags: set[str] = set()
    for question in questions:
        raw = str(question.get("type", ""))
        lowered = raw.lower()
        if "table" in lowered:
            tags.add("tables")
        if "chart" in lowered:
            tags.add("charts")
        if "figure" in lowered:
            tags.add("figures")
        if "multimodal" in lowered:
            tags.add("visual")
        if "text" in lowered or not tags:
            tags.add("text")
    return sorted(tags or {"text"})
