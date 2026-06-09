# src/kb/ingestion/channels/text.py
"""TextChannel — HTTP client for the multimodal_embedding_server.

PoC v1.1 refactor (spec §3.1):
  - Legacy (removed in v1.1): BGEM3FlagModel loaded in-process, returned
    (dense, sparse_idx, sparse_val).
  - New (this file): httpx client → multimodal_embedding_server
    (Qwen3-VL-Embedding-8B running via vllm with OpenAI-compatible /v1/embeddings).
    Returns dense only; sparse path moved to channels/sparse.py (SparseChannel, P-5).

Backward compatibility contract:
  ``embed`` / ``embed_batch`` / ``embed_query`` keep their 3-tuple signature
  ``(dense, [], [])`` with empty sparse placeholders. Callers (pipeline.py,
  retriever.py) continue to work but receive empty sparse vectors until a
  follow-up commit wires SparseChannel into the pipeline/retriever paths.
  See internal design notes (not published) [P-2] for the migration plan.

Instruction prefix:
  Qwen3-VL-Embedding requires a query-side instruction prefix for retrieval
  tasks (per the model card). Documents are embedded without prefix.
  Both behaviors honor ``settings.embedding_query_instruction``.
"""
import asyncio
import logging

import httpx

from kb.config import Settings
from kb.models import TextChunk

logger = logging.getLogger(__name__)


class TextChannel:
    """Dense text embedder backed by multimodal_embedding_server (vllm).

    The sparse counterpart lives in ``channels/sparse.py``; this class
    only handles dense embeddings.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        timeout: float = 60.0,
    ) -> None:
        if settings is None:
            settings = Settings()
        self._settings = settings
        self._url = settings.multimodal_embedding_server_url.rstrip("/")
        self._model_id = settings.multimodal_embedding_model_id
        self._query_instruction = settings.embedding_query_instruction
        # 0 means use native model dim; pass None to vllm to skip MRL truncation
        self._output_dim = settings.embedding_output_dim or None
        # NOTE: We deliberately do NOT keep a long-lived httpx.AsyncClient on
        # self. Each sync wrapper below calls asyncio.run() which creates a
        # fresh event loop; a client bound to a previous loop raises
        # "Event loop is closed". Instead we create a short-lived client per
        # invocation inside _embed_texts_async via `async with`.
        self._timeout = timeout

    async def aclose(self) -> None:  # no-op kept for caller compatibility
        return

    def embed(self, chunk: TextChunk) -> tuple[list[float], list[int], list[float]]:
        """Single-chunk embed. Returns ``(dense, [], [])`` — sparse moved to SparseChannel."""
        results = self.embed_batch([chunk])
        return results[0]

    def embed_batch(
        self,
        chunks: list[TextChunk],
        batch_size: int | None = None,
    ) -> list[tuple[list[float], list[int], list[float]]]:
        """Batch-embed chunks (sync entry).

        Returns:
            List of ``(dense, [], [])`` tuples. The two empty lists are
            historical sparse-vector placeholders kept for caller compatibility;
            real sparse vectors must be obtained from ``SparseChannel`` separately.

        Empty input returns empty list (no server call).
        ``batch_size`` is accepted for backward compatibility but currently
        ignored — the multimodal_embedding_server handles batching internally.
        """
        if not chunks:
            return []
        _ = batch_size  # explicitly unused; vllm batches internally
        dense_list = asyncio.run(
            self._embed_texts_async([c.text for c in chunks], is_query=False)
        )
        return [(d, [], []) for d in dense_list]

    def embed_query(
        self, query_text: str,
    ) -> tuple[list[float], list[int], list[float]]:
        """Query-time embed (sync). Returns ``(dense, [], [])`` — sparse via SparseChannel."""
        dense_list = asyncio.run(
            self._embed_texts_async([query_text], is_query=True)
        )
        return (dense_list[0], [], [])

    async def _embed_texts_async(
        self, texts: list[str], is_query: bool,
    ) -> list[list[float]]:
        """Call multimodal_embedding_server's /v1/embeddings (OpenAI-compatible).

        ``is_query`` controls whether the Qwen3-VL-Embedding query-side
        instruction prefix is prepended (per the model card).
        """
        # Qwen3-VL-Embedding query-side instruction prefix
        if is_query and self._query_instruction:
            texts = [self._query_instruction + t for t in texts]

        payload: dict = {
            "model": self._model_id,
            "input": texts,
            # Left-truncate inputs that tokenize past the embed server's
            # max-model-len instead of letting vllm 400 the whole document.
            # Chinese/mixed chunks occasionally exceed 4096 by a few tokens;
            # embedding tolerates losing a few tail tokens far better than the
            # document failing outright. See Settings.embedding_max_input_tokens.
            "truncate_prompt_tokens": self._settings.embedding_max_input_tokens,
        }
        if self._output_dim:
            # vllm's /v1/embeddings honors `dimensions` for MRL truncation
            payload["dimensions"] = self._output_dim

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._url}/v1/embeddings", json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        return [item["embedding"] for item in data["data"]]
