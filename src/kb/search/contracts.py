from kb.models import SearchResult


def filter_results_by_contract(
    results: list[SearchResult],
    doc_filter: dict | None,
) -> list[SearchResult]:
    if not doc_filter:
        return results
    filtered: list[SearchResult] = []
    for result in results:
        if "doc_name" in doc_filter and result.doc_name != doc_filter["doc_name"]:
            continue
        if "doc_id" in doc_filter and result.doc_id != doc_filter["doc_id"]:
            continue
        if "page_type" in doc_filter and result.page_type != doc_filter["page_type"]:
            continue
        filtered.append(result)
    return filtered


def diversify_results(
    results: list[SearchResult],
    max_per_doc: int = 3,
) -> list[SearchResult]:
    counts: dict[str, int] = {}
    diversified: list[SearchResult] = []
    for result in results:
        current = counts.get(result.doc_id, 0)
        if current >= max_per_doc:
            continue
        diversified.append(result)
        counts[result.doc_id] = current + 1
    return diversified


def adaptive_cutoff(
    results: list[SearchResult],
    top_k: int,
    *,
    enabled: bool,
    calibrated: bool,
    relative_ratio: float = 0.5,
    min_score: float = 0.3,
    min_results: int = 1,
) -> list[SearchResult]:
    """Trim a score-descending result list to an adaptive length.

    `top_k` is always the hard maximum. When `enabled`, the returned list may be
    SHORTER than top_k by cutting where relevance drops off — so an easy query
    with one clear answer returns few results instead of padding to top_k with
    noise, and a broad query can fill up to top_k.

      calibrated=True  (reranker sigmoid scores in [0,1]):
        keep results with score >= `min_score` — an absolute relevance gate,
        valid because reranker scores are calibrated P(relevant).
      calibrated=False (RRF scores, NOT comparable across queries):
        keep results with score >= `relative_ratio` * top_score — relative to
        THIS query's best, since absolute RRF magnitude is meaningless across
        queries.

    Always returns at least `min_results` (when available) and at most top_k.
    When `enabled` is False this is exactly `results[:top_k]` (legacy behaviour),
    so the flag is safe to leave off until validated.

    Assumes `results` is already sorted by score descending (engine guarantees
    this after RRF / rerank) — relies on it to break early.
    """
    capped = results[:top_k]
    if not enabled or not capped:
        return capped

    top_score = capped[0].score
    kept: list[SearchResult] = []
    for r in capped:
        if calibrated:
            passes = r.score >= min_score
        else:
            passes = top_score > 0 and r.score >= relative_ratio * top_score
        if not passes:
            break  # descending scores: once one fails, the rest fail too
        kept.append(r)

    if len(kept) < min_results:
        return capped[:min_results]
    return kept
