# src/kb/search/reranker.py
"""Reranker — HTTP client for the reranker_server (Qwen3-VL-Reranker-8B).

PoC v1.1 (rev. after 2026-05-22 reviewer round 2):
  - Endpoint: ``POST /v1/rerank`` (Cohere-compatible, the standard vllm
    rerank path). Payload is a plain text query + list of string documents.
  - Response: ``{"results": [{"index": int, "relevance_score": float}, ...]}``
  - vllm /v1/rerank **does NOT accept multimodal document payloads**
    in 0.11/0.12 — for image candidates we fall back to
    ``vision_description`` (lossy but available). Multimodal rerank
    waits for vllm upstream support or a separate vision-rerank service.

Failure behaviour:
  Raises RerankerUnavailable on transport/server failure. The caller
  (SearchEngine) must decide whether to skip reranking and return RRF
  results directly. The old "silently return 0.0 for all candidates"
  was a reviewer-flagged anti-pattern (silent failure that makes search
  look healthy while quietly broken).
"""
import asyncio
import dataclasses
import logging

import httpx

from kb.config import Settings
from kb.models import SearchResult

logger = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    """Reranker server transport/protocol failure. Caller decides degrade path."""


class Reranker:
    """Cross-encoder reranker via Qwen3-VL-Reranker-8B (vllm /v1/rerank)."""

    def __init__(
        self,
        settings: Settings | None = None,
        timeout: float = 60.0,
    ) -> None:
        if settings is None:
            settings = Settings()
        self._url = settings.reranker_server_url.rstrip("/")
        self._model_id = settings.reranker_model_id
        self._timeout = timeout
        # No long-lived AsyncClient. The sync rerank() wrapper uses
        # asyncio.run() which creates a fresh event loop per call; a
        # module-scope client bound to the first loop fails on the second
        # call with "RuntimeError: Event loop is closed". Open inside the
        # async body instead — see ingestion/channels/*.py for the same
        # pattern.

    async def aclose(self) -> None:
        # No-op kept for caller API compatibility (used to be needed when
        # we held self._client).
        return None

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Cross-encoder rerank via Cohere-style /v1/rerank.

        Returns NEW SearchResult objects with cross-encoder scores; input
        list and its elements are not mutated.

        Raises RerankerUnavailable on transport/protocol failure — the
        caller (SearchEngine) decides whether to fall back to RRF order.
        """
        if not results or top_k <= 0:
            return []

        candidates = [
            r for r in results
            if r.text_chunk or r.vision_description or r.image_url
        ]
        if not candidates:
            return []

        scores = asyncio.run(self._rerank_async(query, candidates))
        rescored = [
            dataclasses.replace(r, score=float(s))
            for r, s in zip(candidates, scores)
        ]
        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_k]

    async def _rerank_async(
        self, query: str, candidates: list[SearchResult],
    ) -> list[float]:
        """Build documents (plain strings) and POST /v1/rerank.

        vllm /v1/rerank schema (from vllm/examples/online_serving/cohere_rerank_client.py):
            {"model": ..., "query": "<query>", "documents": ["<str>", ...]}
        Response:
            {"results": [{"index": int, "relevance_score": float}, ...]}
        """
        documents = [self._doc_text_for(r) for r in candidates]
        payload = {
            "model": self._model_id,
            "query": query,
            "documents": documents,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/v1/rerank", json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            raise RerankerUnavailable(
                f"rerank request to {self._url}/v1/rerank failed: {e}"
            ) from e

        # vllm/Cohere /v1/rerank response: {"results": [{"index", "relevance_score"}, ...]}
        # Order by index to align with input order.
        items = data.get("results", [])
        scores_by_idx: dict[int, float] = {}
        for item in items:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if idx is not None and score is not None:
                scores_by_idx[idx] = float(score)
        return [scores_by_idx.get(i, 0.0) for i in range(len(candidates))]

    @staticmethod
    def _doc_text_for(r: SearchResult) -> str:
        """Best textual representation of a candidate for /v1/rerank.

        vllm /v1/rerank only accepts text documents (no multimodal in
        0.11/0.12). For image candidates we fall back to
        ``vision_description`` (lossy, LLM-generated). This is a known
        regression vs the original P-4 design that wanted to pass
        image_url directly; will revisit once vllm supports multimodal
        documents in /v1/rerank (or we switch to a vision-aware rerank
        endpoint).
        """
        if r.text_chunk:
            return r.text_chunk
        if r.vision_description:
            return r.vision_description
        return ""
