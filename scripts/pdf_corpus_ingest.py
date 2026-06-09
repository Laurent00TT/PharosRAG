"""Ingest a PDF Corpus v1 manifest subset into an isolated Qdrant store."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb.eval.pdf_corpus import (  # noqa: E402
    filter_existing_local_pdfs,
    load_manifest_or_empty,
    sort_for_subset,
    validate_manifest,
)


def _select(records: list[dict], quality_tier: str | None, eval_role: str | None, limit: int | None) -> list[dict]:
    selected = []
    for record in records:
        if record.get("quality_tier") == "rejected":
            continue
        if quality_tier and record.get("quality_tier") != quality_tier:
            continue
        if eval_role and record.get("eval_role") != eval_role:
            continue
        selected.append(record)
    selected = sort_for_subset(selected)
    if limit is not None:
        return selected[:limit]
    return selected


async def _run(args: argparse.Namespace) -> int:
    issues = validate_manifest(args.manifest)
    if issues:
        print(f"Manifest validation failed: {args.manifest}", file=sys.stderr)
        for issue in issues:
            print(f"  line {issue.line_no}: {issue.field}: {issue.message}", file=sys.stderr)
        return 1

    records = load_manifest_or_empty(args.manifest)
    selected = _select(records, args.quality_tier, args.eval_role, args.limit)
    existing = filter_existing_local_pdfs(selected, root=args.root)

    print(f"Selected records: {len(selected)}")
    print(f"Existing local PDFs: {len(existing)}")
    print(f"Qdrant path: {args.qdrant_path}")
    print(f"Image path: {args.image_path}")

    if args.dry_run:
        for record in existing:
            print(f"  {record['doc_key']} -> {record['local_path']}")
        return 0

    from kb.config import Settings
    from kb.ingestion.pipeline import IngestionPipeline
    from kb.observability import new_trace, setup_tracing

    base = Settings()
    settings = base.model_copy(update={
        "qdrant_path": str(args.qdrant_path),
        "image_storage_path": str(args.image_path),
    })
    setup_tracing(
        log_dir=settings.log_dir,
        max_mb=settings.log_max_mb,
        backup_count=settings.log_backup_count,
    )

    pipeline = IngestionPipeline(settings=settings)
    await pipeline._qdrant.ensure_collections()
    await pipeline._meta_db.init()

    active = await pipeline._meta_db.list_active_doc_ids()
    if args.skip_wipe:
        print(f"--skip-wipe set; keeping {len(active)} active docs")
    else:
        print(f"Active docs to wipe: {len(active)}")
        for doc_id in active:
            await pipeline._qdrant.delete_by_doc_id(doc_id)
            await pipeline._meta_db.deprecate_document(doc_id)

    args.run_report.parent.mkdir(parents=True, exist_ok=True)
    with args.run_report.open("w", encoding="utf-8") as report:
        for idx, record in enumerate(existing, start=1):
            pdf_path = (args.root / record["local_path"]).resolve()
            t0 = time.monotonic()
            new_trace()
            print(f"[{idx}/{len(existing)}] {record['doc_key']} {pdf_path.name}", flush=True)
            row = {
                "doc_key": record["doc_key"],
                "local_path": record["local_path"],
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                result = await pipeline.ingest(pdf_path)
                row.update({
                    "status": "ok",
                    "elapsed_s": round(time.monotonic() - t0, 3),
                    "result": result,
                })
                print(f"  ok in {row['elapsed_s']}s", flush=True)
            except Exception as exc:
                row.update({
                    "status": "failed",
                    "elapsed_s": round(time.monotonic() - t0, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(f"  failed in {row['elapsed_s']}s: {type(exc).__name__}: {exc}", flush=True)
            report.write(json.dumps(row, ensure_ascii=False) + "\n")
            report.flush()

    print(f"Run report: {args.run_report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1" / "manifest.jsonl",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--quality-tier", default=None)
    parser.add_argument("--eval-role", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wipe", action="store_true")
    parser.add_argument("--qdrant-path", type=Path, default=Path("./qdrant_data_pdf_corpus_v1"))
    parser.add_argument("--image-path", type=Path, default=Path("./storage/images_pdf_corpus_v1"))
    parser.add_argument(
        "--run-report",
        type=Path,
        default=REPO_ROOT / "datasets" / "pdf_corpus_v1" / "derived" / "ingest_runs" / "latest.jsonl",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
