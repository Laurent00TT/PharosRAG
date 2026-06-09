# scripts/eval.py
"""Run knowledge base retrieval evaluation.

Usage:
  python scripts/eval.py --testset tests/data/eval_testset.jsonl
  python scripts/eval.py --testset path.jsonl --output results.json --top-k 10
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.config import Settings
from kb.eval.runner import EvalRunner
from kb.eval.testset import load_testset
from kb.search.engine import SearchEngine
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore


async def main() -> int:
    parser = argparse.ArgumentParser(description="Knowledge Base Evaluation")
    parser.add_argument("--testset", required=True, help="Path to JSONL test set")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", help="Path to write detailed results JSON")
    args = parser.parse_args()

    settings = Settings()
    qdrant = QdrantStore(settings=settings)
    await qdrant.ensure_collections()
    meta_db = MetadataDB(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    )
    await meta_db.init()
    engine = SearchEngine(settings=settings, qdrant=qdrant, meta_db=meta_db)

    testset = load_testset(Path(args.testset))
    print(f"Loaded {len(testset)} queries from {args.testset}")

    runner = EvalRunner(engine=engine)
    result = await runner.run(testset, top_k=args.top_k)

    print(f"\n=== Results ({result.n_queries} queries) ===")
    print(f"Avg Recall@{result.k}: {result.avg_recall_at_k:.3f}")
    print(f"Avg NDCG@{result.k}:   {result.avg_ndcg_at_k:.3f}")
    print(f"Avg MRR:        {result.avg_mrr:.3f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "n_queries": result.n_queries,
                "k": result.k,
                "avg_recall_at_k": result.avg_recall_at_k,
                "avg_ndcg_at_k": result.avg_ndcg_at_k,
                "avg_mrr": result.avg_mrr,
                "per_query": [
                    {**q, "retrieved": [list(t) for t in q["retrieved"]],
                     "relevant": [list(t) for t in q["relevant"]]}
                    for q in result.per_query
                ],
            }, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
