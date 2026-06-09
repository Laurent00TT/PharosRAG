"""Generate deterministic synthetic internal PDFs and gold queries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb.eval.pdf_corpus import (  # noqa: E402
    load_manifest_or_empty,
    merge_records_by_doc_key,
    write_jsonl,
)
from kb.eval.pdf_corpus_synthetic import generate_synthetic_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--gold-manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1" / "gold_manifest.jsonl",
    )
    parser.add_argument(
        "--gold-queries",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1" / "gold_queries.jsonl",
    )
    parser.add_argument("--replace-source", action="store_true")
    args = parser.parse_args()

    synthetic_records, synthetic_queries = generate_synthetic_corpus(root=args.root)

    manifest = load_manifest_or_empty(args.manifest)
    manifest = merge_records_by_doc_key(
        manifest,
        synthetic_records,
        replace_source="synthetic" if args.replace_source else None,
    )
    write_jsonl(args.manifest, manifest)

    gold_manifest = load_manifest_or_empty(args.gold_manifest)
    gold_manifest = merge_records_by_doc_key(
        gold_manifest,
        synthetic_records,
        replace_source="synthetic" if args.replace_source else None,
    )
    write_jsonl(args.gold_manifest, gold_manifest)

    write_jsonl(args.gold_queries, synthetic_queries)

    print(f"Wrote synthetic PDFs/sources under: {args.root / 'datasets' / 'pdf_corpus_v1'}")
    print(f"  synthetic records: {len(synthetic_records)}")
    print(f"  synthetic queries: {len(synthetic_queries)}")
    print(f"  manifest total: {len(manifest)}")
    print(f"  gold manifest total: {len(gold_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
