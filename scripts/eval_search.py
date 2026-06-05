"""Run retrieval-quality eval over a JSONL test set.

Usage:
  python scripts/eval_search.py --testset tests/data/dk_grammar_eval.jsonl
  python scripts/eval_search.py --testset path.jsonl --top-k 5 --no-mq
  python scripts/eval_search.py --testset path.jsonl --baseline   # save snapshot

Output:
  - Aggregate Recall@K, NDCG@K, MRR
  - Per-query breakdown (which queries failed)
  - Optionally writes a JSON snapshot for diff comparison

This is the foundation for measuring all future quality changes:
  - chunker upgrades
  - synonym expansion
  - reranker swaps
  - description channel ablation
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.config import Settings
from kb.agent.prompts import AGENT_PROMPT_VERSION
from kb.eval.runner import EvalRunner
from kb.eval.testset import load_testset
from kb.ingestion.channels.description import DESCRIPTION_PROMPT_VERSION
from kb.observability import setup_tracing
from kb.search.engine import SearchEngine
from kb.search.hyde import HYDE_PROMPT_VERSION
from kb.search.multi_query import MULTI_QUERY_PROMPT_VERSION
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore


async def main(args) -> None:
    settings = Settings()

    # Allow CLI to flip MQ / HyDE for ablation (without editing .env)
    if args.no_mq:
        settings = settings.model_copy(update={"multi_query_enabled": False})
    if args.no_hyde:
        settings = settings.model_copy(update={"hyde_enabled": False})
    # Override qdrant_path for evaluating an isolated collection (e.g. MMDocIR
    # subset stored in ./qdrant_data_mmdocir).
    if args.qdrant_path:
        settings = settings.model_copy(update={"qdrant_path": args.qdrant_path})

    setup_tracing(
        log_dir=settings.log_dir,
        max_mb=settings.log_max_mb,
        backup_count=settings.log_backup_count,
    )

    qdrant = QdrantStore(settings=settings)
    await qdrant.ensure_collections()

    # Description-channel ablation: monkey-patch to disable desc retrieval.
    # Used to quantify the marginal value of the desc channel before deciding
    # whether to keep paying ~12s/page LLM description cost during ingest.
    if args.no_desc:
        original_search_desc = qdrant.search_desc
        def _no_desc_search(*a, **kw):
            return []
        qdrant.search_desc = _no_desc_search
        print("[ABLATION] DESC CHANNEL DISABLED for this run")
    meta_db = MetadataDB(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    )
    await meta_db.init()
    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)

    testset = load_testset(Path(args.testset))
    print(f"Loaded {len(testset)} queries from {args.testset}")
    print(f"Settings: hyde={settings.hyde_enabled}, "
          f"multi_query={settings.multi_query_enabled} (n={settings.multi_query_n})")
    print()

    runner = EvalRunner(engine)
    t0 = time.monotonic()
    result = await runner.run(testset, top_k=args.top_k)
    elapsed = time.monotonic() - t0

    # Aggregate
    print("=" * 70)
    print(f"Aggregate (n={result.n_queries}, k={result.k}, elapsed={elapsed:.1f}s)")
    print("=" * 70)
    print(f"  Recall@{result.k}: {result.avg_recall_at_k:.3f}")
    print(f"  NDCG@{result.k}:   {result.avg_ndcg_at_k:.3f}")
    print(f"  MRR:              {result.avg_mrr:.3f}")
    print()

    if result.group_metrics:
        print("Group metrics")
        for name, metrics in sorted(result.group_metrics.items()):
            print(
                f"  {name}: n={metrics['n_queries']} "
                f"R@{result.k}={metrics['avg_recall_at_k']:.3f} "
                f"NDCG={metrics['avg_ndcg_at_k']:.3f} "
                f"MRR={metrics['avg_mrr']:.3f}"
            )
        print()

    # Per-query
    if args.verbose:
        print("=" * 70)
        print("Per-query")
        print("=" * 70)
        for q in result.per_query:
            ok = "OK" if q["recall"] > 0 else "MISS"
            print(f"  [{ok}] {q['query'][:60]:<60s}  "
                  f"R={q['recall']:.2f}  NDCG={q['ndcg']:.3f}  MRR={q['mrr']:.2f}")

    # Failures
    misses = [q for q in result.per_query if q["recall"] == 0]
    if misses:
        print()
        print(f"--- {len(misses)} miss(es) ---")
        for q in misses:
            print(f"  Q: {q['query']}")
            print(f"     expected: {q['relevant']}")
            print(f"     got:      {q['retrieved'][:3]}")

    # Save snapshot
    if args.snapshot:
        snap = {
            "settings": {
                "hyde": settings.hyde_enabled,
                "multi_query": settings.multi_query_enabled,
                "multi_query_n": settings.multi_query_n,
                "embedding_version": settings.embedding_version,
                "reranker_model_id": settings.reranker_model_id,
                "reranker_disabled": settings.search_disable_reranker,
            },
            "k": result.k,
            "n_queries": result.n_queries,
            "elapsed_s": round(elapsed, 1),
            "avg_recall_at_k": result.avg_recall_at_k,
            "avg_ndcg_at_k": result.avg_ndcg_at_k,
            "avg_mrr": result.avg_mrr,
            "prompt_versions": {
                "agent": AGENT_PROMPT_VERSION,
                "description": DESCRIPTION_PROMPT_VERSION,
                "hyde": HYDE_PROMPT_VERSION,
                "multi_query": MULTI_QUERY_PROMPT_VERSION,
            },
            "group_metrics": result.group_metrics,
            "per_query": result.per_query,
        }
        Path(args.snapshot).write_text(
            json.dumps(snap, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSnapshot written: {args.snapshot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True, type=Path,
                        help="JSONL test set path")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-hyde", action="store_true",
                        help="Disable HyDE for this run (ablation)")
    parser.add_argument("--no-mq", action="store_true",
                        help="Disable Multi-Query for this run (ablation)")
    parser.add_argument("--no-desc", action="store_true",
                        help="Disable description channel (RRF will fuse 3 channels: dense+sparse+vision)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-query breakdown")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="Save result as JSON snapshot for later diff")
    parser.add_argument("--qdrant-path", default=None,
                        help="Override qdrant_path (evaluate an isolated "
                             "collection, e.g. ./qdrant_data_mmdocir)")
    args = parser.parse_args()
    asyncio.run(main(args))
