# src/kb/eval/testset.py
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestQuery:
    query: str
    relevant: list[tuple[str, int]]   # ground-truth (doc_id, page_num) pairs
    # Optional analysis metadata. Common keys: query_type, expected_channel,
    # difficulty, domain, notes.
    metadata: dict | None = None


def load_testset(path: Path) -> list[TestQuery]:
    """Load a JSONL test set. One query per line:
    {"query": "...", "relevant": [["doc_id", page_num], ...], "metadata": {...}}

    NOTE: `doc_id` must match the actual MD5 hash assigned by the ingestion pipeline
    (IngestionPipeline.ingest() computes md5 of file bytes). The bundled
    tests/data/eval_testset.jsonl uses placeholder ids ("sample") that work for
    loader unit tests but won't match anything in a real ingested KB. To build a
    production test set:
      1. Ingest your seed PDFs and note the assigned doc_ids (`scripts/ingest.py`
         logs them, or query `metadata_db.list_active_doc_ids()`).
      2. Author queries with the real doc_id + page_num as ground truth.
    """
    queries: list[TestQuery] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            relevant = [(r[0], int(r[1])) for r in data["relevant"]]
            queries.append(TestQuery(
                query=data["query"],
                relevant=relevant,
                metadata=data.get("metadata"),
            ))
    return queries
