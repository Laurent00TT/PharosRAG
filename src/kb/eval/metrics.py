# src/kb/eval/metrics.py
"""Pure-function retrieval metrics. Inputs are lists of identifiers (any hashable type).

Convention: retrieved is ordered by score descending; relevant is the unordered ground-truth set.
"""
from math import log2


def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Fraction of relevant items found in top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Normalized discounted cumulative gain at k. Binary relevance."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if r in relevant_set else 0.0) / log2(i + 2)
        for i, r in enumerate(top_k)
    )
    ideal_count = min(k, len(relevant))
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(retrieved: list, relevant: list) -> float:
    """Reciprocal rank of the first relevant item; 0 if none found."""
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved):
        if r in relevant_set:
            return 1.0 / (i + 1)
    return 0.0
