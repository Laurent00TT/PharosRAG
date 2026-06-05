import hashlib
import json
from pathlib import Path

from kb.eval.pdf_corpus import (
    compute_md5,
    enrich_local_pdf_metadata,
    filter_existing_local_pdfs,
    load_manifest_or_empty,
    make_manifest_record,
    merge_records_by_doc_key,
    normalize_mmdocir_doc,
    read_jsonl,
    select_subset,
    validate_manifest,
    validate_record,
    write_jsonl,
    write_pdf_list,
)
from kb.eval.pdf_corpus_synthetic import generate_synthetic_corpus
from kb.eval.pdf_corpus_public_sources import build_public_seed_records
from kb.eval.pdf_corpus_arxiv import build_arxiv_records, parse_arxiv_feed


def _record(**overrides):
    record = {
        "schema_version": "pdf_corpus_manifest_v1",
        "doc_key": "doc-a",
        "source_family": "mmdocir",
        "source_url": "https://mmdocrag.github.io/MMDocIR/",
        "download_url": "",
        "local_path": "datasets/pdf_corpus_v1/raw/mmdocir/doc-a.pdf",
        "sha256": "",
        "file_size_mb": 0.0,
        "page_count": None,
        "language": "en",
        "domain": "finance",
        "doc_type": "report",
        "layout_tags": ["tables"],
        "quality_tier": "candidate",
        "eval_role": "background",
        "gold_priority": 0,
        "license_note": "benchmark metadata only",
        "collection_policy": "metadata_only",
        "ingest_status": "pending",
        "parse_status": "unknown",
        "notes": "",
    }
    record.update(overrides)
    return record


def test_validate_record_accepts_valid_manifest_record():
    assert validate_record(_record(), line_no=1) == []


def test_validate_record_reports_missing_required_field_with_line_number():
    record = _record()
    record.pop("doc_key")

    issues = validate_record(record, line_no=7)

    assert len(issues) == 1
    assert issues[0].line_no == 7
    assert "doc_key" in issues[0].message


def test_validate_record_reports_invalid_controlled_value():
    issues = validate_record(_record(quality_tier="important"), line_no=3)

    assert len(issues) == 1
    assert issues[0].field == "quality_tier"
    assert "important" in issues[0].message


def test_validate_manifest_reads_jsonl_and_reports_line_numbers(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    valid = _record(doc_key="ok")
    invalid = _record(doc_key="bad", eval_role="qa")
    manifest.write_text(
        json.dumps(valid, ensure_ascii=False) + "\n"
        + json.dumps(invalid, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    issues = validate_manifest(manifest)

    assert len(issues) == 1
    assert issues[0].line_no == 2
    assert issues[0].field == "eval_role"


def test_read_write_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [_record(doc_key="a"), _record(doc_key="b")]

    write_jsonl(path, records)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert [r["doc_key"] for r in read_jsonl(path)] == ["a", "b"]


def test_load_manifest_or_empty_returns_empty_for_missing_file(tmp_path):
    assert load_manifest_or_empty(tmp_path / "missing.jsonl") == []


def test_enrich_local_pdf_metadata_computes_hash_and_size(tmp_path):
    pdf = tmp_path / "doc.pdf"
    data = b"%PDF-1.4\nnot a complete pdf\n"
    pdf.write_bytes(data)
    record = _record(local_path="doc.pdf")

    enriched = enrich_local_pdf_metadata(record, root=tmp_path)

    assert enriched["sha256"] == hashlib.sha256(data).hexdigest()
    assert enriched["file_size_mb"] == round(len(data) / (1024 * 1024), 6)
    assert "page_count" in enriched


def test_compute_md5_matches_ingestion_doc_id(tmp_path):
    pdf = tmp_path / "doc.pdf"
    data = b"%PDF-1.4\nstable bytes\n"
    pdf.write_bytes(data)

    assert compute_md5(pdf) == hashlib.md5(data).hexdigest()


def test_make_manifest_record_fills_defaults_and_enriches_local_pdf(tmp_path):
    pdf = tmp_path / "raw" / "synthetic" / "policy.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n")

    record = make_manifest_record(
        doc_key="synthetic-policy",
        source_family="synthetic",
        source_url="local://synthetic/policy",
        download_url="",
        local_path="raw/synthetic/policy.pdf",
        language="en",
        domain="internal_policy",
        doc_type="policy",
        layout_tags=["tables", "text"],
        quality_tier="gold",
        eval_role="gold",
        gold_priority=100,
        license_note="generated fixture",
        collection_policy="download_ok",
        notes="unit test",
        root=tmp_path,
    )

    assert record["schema_version"] == "pdf_corpus_manifest_v1"
    assert record["sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert record["ingest_status"] == "pending"
    assert record["parse_status"] == "unknown"


def test_merge_records_by_doc_key_replaces_selected_source_only():
    existing = [
        _record(doc_key="m1", source_family="mmdocir", notes="old"),
        _record(doc_key="s1", source_family="synthetic", notes="keep"),
    ]
    new = [
        _record(doc_key="m2", source_family="mmdocir", notes="new"),
    ]

    merged = merge_records_by_doc_key(existing, new, replace_source="mmdocir")

    assert [r["doc_key"] for r in merged] == ["m2", "s1"]
    assert {r["notes"] for r in merged} == {"new", "keep"}


def test_select_subset_is_deterministic_and_respects_tier_role_priority():
    records = [
        _record(doc_key="z", quality_tier="gold", eval_role="gold", gold_priority=1),
        _record(doc_key="a", quality_tier="gold", eval_role="gold", gold_priority=5),
        _record(doc_key="b", quality_tier="candidate", eval_role="background", gold_priority=100),
        _record(doc_key="c", quality_tier="gold", eval_role="gold", gold_priority=5),
    ]

    selected = select_subset(records, tier="gold", eval_role="gold", limit=2)

    assert [r["doc_key"] for r in selected] == ["a", "c"]


def test_filter_existing_local_pdfs_and_write_pdf_list(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    existing_pdf = pdf_dir / "a.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4\n")
    records = [
        _record(doc_key="a", local_path="pdfs/a.pdf"),
        _record(doc_key="missing", local_path="pdfs/missing.pdf"),
    ]

    existing = filter_existing_local_pdfs(records, root=tmp_path)
    out = tmp_path / "subset_pdfs.txt"
    write_pdf_list(out, existing, root=tmp_path)

    assert [r["doc_key"] for r in existing] == ["a"]
    assert out.read_text(encoding="utf-8").strip() == str(existing_pdf.resolve())


def test_write_pdf_list_rejects_record_without_local_path(tmp_path):
    out = tmp_path / "subset_pdfs.txt"
    record = _record(local_path="")

    try:
        write_pdf_list(out, [record], root=tmp_path)
    except ValueError as exc:
        assert "local_path" in str(exc)
    else:
        raise AssertionError("write_pdf_list should reject records without local_path")


def test_normalize_mmdocir_doc_builds_corpus_manifest_record(tmp_path):
    pdf_dir = tmp_path / "datasets" / "mmdocir" / "extracted" / "doc_pdfs"
    pdf_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "Annual Report 2024.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    doc = {
        "doc_name": "Annual Report 2024.pdf",
        "domain": "Finance",
        "questions": [
            {"type": "text-only"},
            {"type": "['Table']"},
        ],
    }

    record = normalize_mmdocir_doc(
        doc,
        pdf_path=pdf_path,
        root=tmp_path,
        quality_tier="candidate",
        eval_role="background",
    )

    assert record["schema_version"] == "pdf_corpus_manifest_v1"
    assert record["doc_key"] == "mmdocir-annual-report-2024"
    assert record["source_family"] == "mmdocir"
    assert record["local_path"] == "datasets/mmdocir/extracted/doc_pdfs/Annual Report 2024.pdf"
    assert record["domain"] == "finance"
    assert record["doc_type"] == "benchmark_pdf"
    assert record["layout_tags"] == ["tables", "text"]
    assert record["quality_tier"] == "candidate"
    assert record["eval_role"] == "background"


def test_generate_synthetic_corpus_creates_gold_records_and_queries(tmp_path):
    records, queries = generate_synthetic_corpus(root=tmp_path)

    assert len(records) == 20
    assert len(queries) == 60
    assert all(record["source_family"] == "synthetic" for record in records)
    assert all(record["quality_tier"] == "gold" for record in records)
    assert all(record["eval_role"] == "gold" for record in records)
    assert validate_record(records[0], line_no=1) == []

    first_pdf = tmp_path / records[0]["local_path"]
    assert first_pdf.exists()
    first_doc_id = compute_md5(first_pdf)
    first_queries = [
        query for query in queries
        if query["metadata"]["doc_key"] == records[0]["doc_key"]
    ]
    assert {q["relevant"][0][0] for q in first_queries} == {first_doc_id}
    assert {q["metadata"]["query_type"] for q in first_queries} == {
        "fact_lookup",
        "table_lookup",
        "exception_rule",
    }


def test_build_public_seed_records_creates_valid_candidate_records(tmp_path):
    records = build_public_seed_records(root=tmp_path, download=False, limit=5)

    assert len(records) == 5
    assert all(record["quality_tier"] == "candidate" for record in records)
    assert all(record["eval_role"] == "background" for record in records)
    assert all(validate_record(record, line_no=1) == [] for record in records)
    assert {record["source_family"] for record in records} <= {
        "govinfo",
        "academic",
        "sec",
    }


def test_parse_arxiv_feed_and_build_records(tmp_path):
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.01234v2</id>
        <title> A Test Paper About Retrieval </title>
        <summary>Example abstract.</summary>
        <published>2024-01-01T00:00:00Z</published>
        <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
      </entry>
    </feed>
    """

    entries = parse_arxiv_feed(feed)
    records = build_arxiv_records(entries, root=tmp_path)

    assert entries[0]["arxiv_id"] == "2401.01234"
    assert records[0]["doc_key"] == "arxiv-2401-01234"
    assert records[0]["download_url"] == "https://arxiv.org/pdf/2401.01234.pdf"
    assert records[0]["domain"] == "research_paper"
    assert validate_record(records[0], line_no=1) == []
