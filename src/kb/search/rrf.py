# src/kb/search/rrf.py


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, int, float]]],
    channel_names: list[str],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, int, float, list[str]]]:
    """
    ranked_lists: each list is [(doc_id, page_num, score), ...] sorted desc by score.
    weights: optional per-channel multiplier on the RRF contribution. Missing
        channels default to 1.0 (so weights=None == classic unweighted RRF).
        Lets us up-weight e.g. sparse for keyword-heavy CJK queries or dense
        for semantic EN queries without changing the fusion math.
    Returns: [(doc_id, page_num, fused_score, [channels]), ...] sorted desc.
    """
    weights = weights or {}
    fused: dict[tuple[str, int], float] = {}
    channels: dict[tuple[str, int], list[str]] = {}

    for name, ranked in zip(channel_names, ranked_lists):
        w = weights.get(name, 1.0)
        for rank, (doc_id, page_num, _) in enumerate(ranked):
            key = (doc_id, page_num)
            fused[key] = fused.get(key, 0.0) + w * (1.0 / (k + rank + 1))
            if key not in channels:
                channels[key] = []
            if name not in channels[key]:
                channels[key].append(name)

    return sorted(
        [
            (doc_id, page_num, score, channels[(doc_id, page_num)])
            for (doc_id, page_num), score in fused.items()
        ],
        key=lambda x: x[2],
        reverse=True,
    )
