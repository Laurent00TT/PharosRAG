# src/kb/eval/runner.py
import logging
from dataclasses import dataclass, field

from kb.eval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from kb.eval.testset import TestQuery
from kb.search.engine import SearchEngine

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    n_queries: int
    k: int                       # the metric cut-off used
    avg_recall_at_k: float
    avg_ndcg_at_k: float
    avg_mrr: float
    per_query: list[dict] = field(default_factory=list)
    group_metrics: dict = field(default_factory=dict)


class EvalRunner:
    def __init__(self, engine: SearchEngine) -> None:
        self._engine = engine

    async def run(self, testset: list[TestQuery], top_k: int = 5) -> EvalResult:
        per_query: list[dict] = []
        for tq in testset:
            response, _ = await self._engine.search(tq.query, top_k=top_k)
            retrieved = [(r.doc_id, r.page_num) for r in response.results]
            entry = {
                "query": tq.query,
                "retrieved": retrieved,
                "relevant": tq.relevant,
                "recall": recall_at_k(retrieved, tq.relevant, top_k),
                "ndcg": ndcg_at_k(retrieved, tq.relevant, top_k),
                "mrr": mean_reciprocal_rank(retrieved, tq.relevant),
            }
            if entry["recall"] == 0:
                entry["failure_bucket"] = classify_failure(retrieved, tq.relevant)
            # Carry metadata through so downstream analysis can group by
            # query type / domain / etc.
            if tq.metadata:
                entry["metadata"] = tq.metadata
            per_query.append(entry)

        n = len(per_query)
        if n == 0:
            return EvalResult(n_queries=0, k=top_k, avg_recall_at_k=0.0,
                              avg_ndcg_at_k=0.0, avg_mrr=0.0, per_query=[])
        return EvalResult(
            n_queries=n,
            k=top_k,
            avg_recall_at_k=sum(q["recall"] for q in per_query) / n,
            avg_ndcg_at_k=sum(q["ndcg"] for q in per_query) / n,
            avg_mrr=sum(q["mrr"] for q in per_query) / n,
            per_query=per_query,
            group_metrics=_group_metrics(per_query),
        )


def _group_metrics(per_query: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for query_result in per_query:
        metadata = query_result.get("metadata") or {}
        key = metadata.get("query_type", "unclassified")
        groups.setdefault(key, []).append(query_result)
    return {
        name: {
            "n_queries": len(rows),
            "avg_recall_at_k": sum(r["recall"] for r in rows) / len(rows),
            "avg_ndcg_at_k": sum(r["ndcg"] for r in rows) / len(rows),
            "avg_mrr": sum(r["mrr"] for r in rows) / len(rows),
        }
        for name, rows in groups.items()
    }


def classify_failure(
    retrieved: list[tuple[str, int]],
    relevant: list[tuple[str, int]],
) -> str:
    """Categorize a recall==0 result.

    Only called when ``entry["recall"] == 0`` in ``EvalRunner.run`` —
    so a "hit" classification can never reach this function (a hit
    implies recall > 0). The bucket is derived purely from doc_id sets:
    same doc but wrong page is qualitatively different from missing the
    doc entirely, and the two bugs cluster around different root causes
    (chunking/page-routing vs index recall).
    """
    if not relevant:
        return "no_ground_truth"
    if not retrieved:
        return "no_recall"
    relevant_docs = {doc_id for doc_id, _ in relevant}
    retrieved_docs = {doc_id for doc_id, _ in retrieved}
    if relevant_docs & retrieved_docs:
        return "wrong_page"
    return "wrong_doc"
