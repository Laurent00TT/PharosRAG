# src/kb/search/engine.py
import asyncio
import logging
import time

from kb.config import Settings
from kb.models import SearchResponse
from kb.observability import (
    SearchEndEvent,
    SearchStartEvent,
    emit,
    preview,
)
from kb.search.cache import cache_key, get_cached, set_cached
from kb.search.constants import ALL_CHANNELS
from kb.search.contracts import adaptive_cutoff, diversify_results, filter_results_by_contract
from kb.search.reranker import RerankerUnavailable
from kb.search.expander import enrich_results
from kb.search.hyde import hyde_expand
from kb.search.planner import QueryPlanner
from kb.search.retriever import Retriever
from kb.search.rrf import reciprocal_rank_fusion
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantStore,
        meta_db: MetadataDB,
        cache_epoch=None,
    ) -> None:
        self._settings = settings
        self._retriever = Retriever(settings, qdrant, meta_db)
        self._reranker = None   # lazy: loaded on first search call
        self._llm_adapter = None     # lazy: shared by HyDE / Multi-Query
        self._qdrant = qdrant
        self._meta_db = meta_db
        # Review P1 #6: CacheEpochStore for cross-process cache
        # invalidation. None = single-process / test path (epoch=0
        # baked into the key; behavior unchanged).
        self._cache_epoch = cache_epoch

    async def aclose(self) -> None:
        await self._retriever.aclose()
        if self._llm_adapter is not None:
            await self._llm_adapter.aclose()
        if self._reranker is not None:
            await self._reranker.aclose()

    def _get_reranker(self):
        if self._reranker is None:
            from kb.search.reranker import Reranker
            # PoC v1.1: new Reranker is httpx client; accepts (settings, timeout)
            # not the legacy (model_name, device) signature.
            self._reranker = Reranker(settings=self._settings)
        return self._reranker

    def _get_llm_adapter(self):
        """Construct the short-task adapter shared by HyDE + Multi-Query.

        NOTE: This is NOT the Agent's adapter — that one is built in
        tool_server/app.py with thinking left on. HyDE and Multi-Query produce
        short structured output (one hypothetical doc / N query variants);
        thinking is harmful here:
          - wastes tokens, latency x5-10
          - short outputs may end up with content="" and only reasoning_content,
            so downstream parsing of an empty string fails
        Therefore we force thinking off here.
        """
        if self._llm_adapter is None:
            from kb.adapters.factory import make_chat_adapter
            s = self._settings
            self._llm_adapter = make_chat_adapter(
                adapter_type=s.query_rewrite_adapter,
                url=s.query_rewrite_url,
                api_key=s.query_rewrite_api_key,
                model=s.query_rewrite_model,
                supports_vision=s.query_rewrite_supports_vision,
                disable_thinking=True,    # HyDE/MQ are short structured tasks
                task_type="query_rewrite",  # Phase 6.4 — kb.llm.call attribution
            )
        return self._llm_adapter

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: dict | None = None,
        hyde: bool | None = None,                     # None = use settings default
        multi_query: bool | None = None,              # None = use settings; False = force off
        include_deprecated: bool = False,
    ) -> tuple[SearchResponse, bool]:
        """Returns (SearchResponse, cache_hit).
        Callers use cache_hit to populate observability logs."""
        t_total = time.monotonic()
        doc_filter = doc_filter or {}
        search_config = self._settings.effective_search_config()
        plan = (
            QueryPlanner(self._settings).plan(query, doc_filter)
            if self._settings.search_planner_enabled
            else None
        )
        # hyde param: None defers to settings; True/False overrides explicitly
        effective_hyde = (
            hyde
            if hyde is not None
            else (plan.use_hyde if plan else search_config["hyde_enabled"])
        )
        # Per-call override mirrors hyde param: None = settings/planner, False/True = explicit.
        if multi_query is False:
            multi_query_enabled = False
            multi_query_n = 1
        elif multi_query is True:
            multi_query_enabled = True
            multi_query_n = plan.multi_query_n if plan else search_config["multi_query_n"]
        else:
            multi_query_enabled = search_config["multi_query_enabled"]
            multi_query_n = plan.multi_query_n if plan else search_config["multi_query_n"]
        candidate_multiplier = plan.candidate_multiplier if plan else 3

        if plan is not None:
            emit(
                "search.plan",
                query_type=plan.query_type,
                use_hyde=plan.use_hyde,
                multi_query_n=plan.multi_query_n,
                enabled_channels=plan.enabled_channels,
                candidate_multiplier=plan.candidate_multiplier,
                needs_vision=plan.needs_vision,
                reason=plan.reason,
            )

        emit(SearchStartEvent(
            query=preview(query),
            top_k=top_k,
            hyde=effective_hyde,
            multi_query_enabled=multi_query_enabled,
            multi_query_n=multi_query_n,
            doc_filter=doc_filter,
            include_deprecated=include_deprecated,
        ))

        # Cache key includes every flag that affects retrieval — including the
        # multi_query toggle/N (silently changing Settings used to leak old cache).
        # Review P1 #6: epoch from CacheEpochStore lets a worker bump
        # the counter and invalidate this process's cache too.
        epoch = await self._cache_epoch.get_epoch() if self._cache_epoch else 0
        key = cache_key(
            query, doc_filter, include_deprecated, top_k, effective_hyde,
            multi_query_enabled=multi_query_enabled,
            multi_query_n=multi_query_n,
            epoch=epoch,
        )
        cached = get_cached(key)
        if cached is not None:
            logger.debug("Cache hit for query: %.50s", query)
            emit(SearchEndEvent(
                total_duration_ms=round((time.monotonic() - t_total) * 1000, 1),
                cache_hit=True,
                result_count=cached.total,
                top_score=cached.results[0].score if cached.results else 0.0,
                success=True,
            ))
            return cached, True

        if effective_hyde:
            search_query = await hyde_expand(query, adapter=self._get_llm_adapter())
        else:
            search_query = query

        seed_query = search_query

        # Multi-Query: generate query variants and retrieve in parallel
        if multi_query_enabled and multi_query_n > 1:
            from kb.search.multi_query import expand_queries
            queries = await expand_queries(
                seed_query,
                n=multi_query_n,
                adapter=self._get_llm_adapter(),
            )
        else:
            queries = [seed_query]

        # Retrieve all query variants in parallel — each retrieve hits 4 channels
        # internally; running 3 variants sequentially was the biggest source of search latency.
        retrieve_results = await asyncio.gather(*[
            self._retriever.retrieve(q, top_k * candidate_multiplier, doc_filter, include_deprecated)
            for q in queries
        ])
        all_dense  = [r[0] for r in retrieve_results]
        all_sparse = [r[1] for r in retrieve_results]
        all_vision = [r[2] for r in retrieve_results]
        all_desc   = [r[3] for r in retrieve_results]

        # RRF-merge each channel across the N query variants
        dense_h = _flatten_rrf(all_dense)
        sparse_h = _flatten_rrf(all_sparse)
        vision_h = _flatten_rrf(all_vision)
        desc_h = _flatten_rrf(all_desc)

        t_rrf = time.monotonic()
        rrf_weights = {
            "dense": self._settings.search_rrf_weight_dense,
            "sparse": self._settings.search_rrf_weight_sparse,
            "vision": self._settings.search_rrf_weight_vision,
            "desc": self._settings.search_rrf_weight_desc,
        }
        fused = reciprocal_rank_fusion(
            [dense_h, sparse_h, vision_h, desc_h],
            channel_names=ALL_CHANNELS,
            weights=rrf_weights,
        )
        top1_channels = fused[0][3] if fused else []
        emit(
            "rrf.fused",
            duration_ms=round((time.monotonic() - t_rrf) * 1000, 1),
            candidates=len(fused),
            top1_channels=top1_channels,
            n_variants=len(queries),
        )

        t_enrich = time.monotonic()
        candidates = await enrich_results(
            fused[: top_k * 2], self._qdrant, self._meta_db
        )
        emit(
            "enrich.end",
            duration_ms=round((time.monotonic() - t_enrich) * 1000, 1),
            candidates_in=min(len(fused), top_k * 2),
            results_count=len(candidates),
            with_text=sum(1 for c in candidates if c.text_chunk),
            with_desc=sum(1 for c in candidates if c.vision_description),
            with_image=sum(1 for c in candidates if c.image_url),
        )

        t_rerank = time.monotonic()
        # Adaptive cutoff trims to an adaptive length (≤ top_k) when enabled;
        # otherwise it is exactly [:top_k]. RRF-only scores are uncalibrated so
        # we gate relatively (vs this query's top score); reranker scores are
        # calibrated so we gate on an absolute relevance threshold.
        def _cutoff(rs):
            return adaptive_cutoff(
                rs, top_k,
                enabled=self._settings.search_adaptive_cutoff_enabled,
                calibrated=not self._settings.search_disable_reranker,
                relative_ratio=self._settings.search_adaptive_cutoff_ratio,
                min_score=self._settings.search_adaptive_cutoff_min_score,
                min_results=self._settings.search_adaptive_cutoff_min_results,
            )

        def _rrf_order():
            # RRF-ordered path: trim + diversify + adaptive cutoff, no reranker.
            kept = filter_results_by_contract(candidates, doc_filter)
            return _cutoff(diversify_results(kept, max_per_doc=3))

        if self._settings.search_disable_reranker:
            # Bypass reranker — keep raw RRF order.
            # See Settings.search_disable_reranker for context.
            reranked = _rrf_order()
            rerank_scores = [r.score for r in reranked]
            emit(
                "rerank.skip",
                duration_ms=round((time.monotonic() - t_rerank) * 1000, 1),
                candidates_in=len(candidates),
                kept=len(reranked),
            )
        else:
            # Rerank input pool: top_k * multiplier (capped by enriched count).
            # multiplier=1.0 → set-preserving (reranker can't evict RRF top-k).
            pool_n = max(top_k, round(top_k * self._settings.search_rerank_pool_multiplier))
            rerank_input = candidates[:pool_n]
            try:
                reranked = await asyncio.to_thread(
                    lambda: self._get_reranker().rerank(query, rerank_input, top_k)
                )
                reranked = filter_results_by_contract(reranked, doc_filter)
                reranked = _cutoff(diversify_results(reranked, max_per_doc=3))
                rerank_scores = [r.score for r in reranked]
                emit(
                    "rerank.call",
                    duration_ms=round((time.monotonic() - t_rerank) * 1000, 1),
                    candidates_in=len(candidates),
                    pool_n=pool_n,
                    kept=len(reranked),
                    score_max=max(rerank_scores) if rerank_scores else 0.0,
                    score_min=min(rerank_scores) if rerank_scores else 0.0,
                    score_p50=sorted(rerank_scores)[len(rerank_scores) // 2] if rerank_scores else 0.0,
                )
            except RerankerUnavailable as e:
                # Graceful degradation: rerank server down/erroring → serve RRF
                # order instead of failing the whole search. Without this, an
                # unhealthy reranker (e.g. the 2026-05-28 morning outage) 500s
                # every query. RRF order is a fully usable fallback.
                logger.warning("reranker unavailable, falling back to RRF order: %s", e)
                reranked = _rrf_order()
                rerank_scores = [r.score for r in reranked]
                emit(
                    "rerank.fallback_rrf",
                    duration_ms=round((time.monotonic() - t_rerank) * 1000, 1),
                    candidates_in=len(candidates),
                    kept=len(reranked),
                    error=str(e)[:200],
                )

        response = SearchResponse(results=reranked, total=len(reranked))
        if (
            response.results
            and response.results[0].score < self._settings.search_low_confidence_threshold
        ):
            response.low_confidence = True
            response.confidence_reason = "top_score_below_threshold"

        # Cold-start UX: if no results AND KB is empty, suggest upload topics
        if response.total == 0:
            active_docs = await self._meta_db.list_active_doc_ids()
            if not active_docs:
                response.message = (
                    "No matching content in the knowledge base. "
                    "Try uploading relevant documents and querying again."
                )
                response.suggested_upload_topics = self._suggest_upload_topics(query)

        set_cached(key, response)
        emit(SearchEndEvent(
            total_duration_ms=round((time.monotonic() - t_total) * 1000, 1),
            cache_hit=False,
            result_count=response.total,
            top_score=response.results[0].score if response.results else 0.0,
            cold_start=bool(response.message),
            success=True,
        ))
        return response, False

    def _suggest_upload_topics(self, query: str) -> list[str]:
        """Suggest document categories the user should upload.

        MVP: static common-categories list. The `query` parameter is kept for
        forward-compatibility — design doc §12 calls for query-aware suggestions
        (generated from query keywords), e.g. via Gemma 4 keyword extraction or
        embedding-similarity to a category vocabulary. Defer until we have
        usage data showing static suggestions are insufficient.
        """
        del query  # explicitly unused in MVP; suppress linter warnings
        return [
            "Product approval workflows",
            "Operations manuals",
            "Organization charts",
        ]


def _flatten_rrf(per_query_lists: list[list[tuple[str, int, float]]]) -> list[tuple[str, int, float]]:
    """RRF-merge same-channel results across N query variants.
    Returns [(doc_id, page_num, score), ...]."""
    if len(per_query_lists) == 1:
        return per_query_lists[0]
    # RRF needs a channel_names list — here each variant list acts as its own pseudo-channel
    channel_names = [f"q{i}" for i in range(len(per_query_lists))]
    fused = reciprocal_rank_fusion(per_query_lists, channel_names=channel_names)
    return [(doc_id, page_num, score) for (doc_id, page_num, score, _) in fused]
