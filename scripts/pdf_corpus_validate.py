"""Validate and optionally enrich a PDF Corpus v1 manifest.

Examples:
  python scripts/pdf_corpus_validate.py
  python scripts/pdf_corpus_validate.py --manifest datasets/pdf_corpus_v1/manifest.jsonl
  python scripts/pdf_corpus_validate.py --enrich-output datasets/pdf_corpus_v1/derived/pdf_metadata.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb.eval.pdf_corpus import (  # noqa: E402
    enrich_local_pdf_metadata,
    load_manifest_or_empty,
    validate_manifest,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1" / "manifest.jsonl",
        help="Corpus v1 manifest JSONL path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve relative local_path values.",
    )
    parser.add_argument(
        "--enrich-output",
        type=Path,
        default=None,
        help="Optional output path for records enriched with local sha256/size/page_count.",
    )
    args = parser.parse_args()

    issues = validate_manifest(args.manifest)
    if issues:
        print(f"Manifest validation failed: {args.manifest}")
        for issue in issues:
            print(f"  line {issue.line_no}: {issue.field}: {issue.message}")
        return 1

    records = load_manifest_or_empty(args.manifest)
    print(f"Manifest valid: {args.manifest} ({len(records)} records)")

    if args.enrich_output:
        enriched = [
            enrich_local_pdf_metadata(record, root=args.root)
            for record in records
        ]
        write_jsonl(args.enrich_output, enriched)
        print(f"Wrote enriched metadata: {args.enrich_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
