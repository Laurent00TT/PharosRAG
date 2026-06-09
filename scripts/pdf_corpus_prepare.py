"""Prepare Corpus v1 smoke/gold/medium manifests and local PDF path lists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb.eval.pdf_corpus import (  # noqa: E402
    filter_existing_local_pdfs,
    load_manifest_or_empty,
    sort_for_subset,
    validate_manifest,
    write_jsonl,
    write_pdf_list,
)


def _eligible(records: list[dict], *, tiers: set[str], roles: set[str] | None = None) -> list[dict]:
    out = []
    for record in records:
        if record.get("quality_tier") == "rejected":
            continue
        if record.get("quality_tier") not in tiers:
            continue
        if roles is not None and record.get("eval_role") not in roles:
            continue
        out.append(record)
    return sort_for_subset(out)


def _prefer_local(records: list[dict], *, root: Path) -> list[dict]:
    existing = []
    missing = []
    for record in records:
        local_path = record.get("local_path")
        if local_path and (root / local_path).exists():
            existing.append(record)
        else:
            missing.append(record)
    return sort_for_subset(existing) + sort_for_subset(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1" / "manifest.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--smoke-limit", type=int, default=10)
    parser.add_argument("--gold-limit", type=int, default=80)
    parser.add_argument("--medium-limit", type=int, default=250)
    args = parser.parse_args()

    issues = validate_manifest(args.manifest)
    if issues:
        print(f"Manifest validation failed: {args.manifest}", file=sys.stderr)
        for issue in issues:
            print(f"  line {issue.line_no}: {issue.field}: {issue.message}", file=sys.stderr)
        return 1

    records = load_manifest_or_empty(args.manifest)
    smoke = _eligible(records, tiers={"smoke"}, roles={"smoke"})[:args.smoke_limit]
    if not smoke:
        smoke = _prefer_local(_eligible(
            records,
            tiers={"gold", "medium", "candidate"},
            roles={"gold", "background", "stress_only"},
        ), root=args.root)[:args.smoke_limit]

    gold = _eligible(records, tiers={"gold"}, roles={"gold"})[:args.gold_limit]
    medium = _prefer_local(_eligible(
        records,
        tiers={"smoke", "gold", "medium", "candidate"},
        roles={"smoke", "gold", "background", "stress_only"},
    ), root=args.root)[:args.medium_limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    derived = args.out_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    outputs = {
        "smoke": smoke,
        "gold": gold,
        "medium": medium,
    }
    for name, selected in outputs.items():
        manifest_path = args.out_dir / f"{name}_manifest.jsonl"
        pdf_list_path = derived / f"{name}_pdfs.txt"
        existing = filter_existing_local_pdfs(selected, root=args.root)
        write_jsonl(manifest_path, selected)
        write_pdf_list(pdf_list_path, existing, root=args.root)
        print(f"{name}: {len(selected)} records, {len(existing)} local PDFs")
        print(f"  manifest: {manifest_path}")
        print(f"  pdf list: {pdf_list_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
