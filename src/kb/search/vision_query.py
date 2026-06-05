# src/kb/search/vision_query.py
"""Query-time vision embedding via multimodal_embedding_server.

PoC v1.1 refactor (spec §3.1):
  - Legacy (removed in v1.1): ColQwen2.5 multi-vec text query (via the
    standalone colqwen_server /embed_query).
  - New (this file): Qwen3-VL-Embedding-8B single-vec text query via the
    same vllm-backed multimodal_embedding_server that text channel uses.

Note: In the new unified-vector-space architecture, text query and vision
query share the same model and same endpoint. Whether to apply the
``embedding_query_instruction`` prefix for vision-channel queries is a
trade-off — currently we DO apply it for consistency with text query
embedding (same prompt → same retrieval semantics). Verify in eval.

This module could be merged into TextChannel post-PoC (spec §5.2 suggests
"vision_query.py 可合并到 text channel 后删除"). Kept separate for now
to preserve retriever.py call sites unchanged.
"""
import asyncio
import logging

import httpx

from kb.config import Settings

logger = logging.getLogger(__name__)


class VisionQueryEmbedder:
    """Text query -> 4096-dim dense vector for vision-channel search."""

    def __init__(
        self,
        settings: Settings,
        timeout: float = 30.0,
    ) -> None:
        self._url = settings.multimodal_embedding_server_url.rstrip("/")
        self._model_id = settings.multimodal_embedding_model_id
        self._query_instruction = settings.embedding_query_instruction
        self._output_dim = settings.embedding_output_dim or None
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed_query_async(self, query: str) -> list[float] | None:
        """Text query -> 4096-dim vector. None on failure (caller degrades).

        Applies the same query instruction prefix as TextChannel.embed_query
        for consistency. Both text and vision channel search now use the same
        retrieval prompt; eval should verify whether vision-specific prompts
        improve recall on image-heavy queries.
        """
        try:
            text = (
                self._query_instruction + query
                if self._query_instruction
                else query
            )
            payload: dict = {
                "model": self._model_id,
                "input": [text],
            }
            if self._output_dim:
                payload["dimensions"] = self._output_dim
            resp = await self._client.post(
                f"{self._url}/v1/embeddings", json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.warning(
                "Vision query embedding failed (%s); skipping vision channel for query",
                e,
            )
            return None

    def embed_query(self, query: str) -> list[float] | None:
        """Sync entry-point kept for legacy retriever call sites."""
        return asyncio.run(self.embed_query_async(query))
