# src/kb/ingestion/channels/sparse.py
"""Sparse channel — fetch MILCO sparse embeddings via sparse_server HTTP.

Spec: internal design notes (not published) §3.2-3.3
Server: scripts/sparse_server.py (P-9)
Model: omai-research/milco-650m (Apache 2.0, audited 2026-05-22)

The two endpoints reflect MILCO doc/query pruning asymmetry which is
encoded into embedding_version (see spec §2). Any change to pruning
ratio or source_view setting MUST bump embedding_version and re-ingest.
"""
import asyncio
import dataclasses
import logging

import httpx

from kb.config import Settings

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SparseEmbedding:
    """Sparse vector in (indices, values) format.

    Directly compatible with Qdrant's
    ``SparseVector(indices=..., values=...)``.

    Token id semantics depend on the sparse model — MILCO uses its own
    dual-view vocabulary (English LSR vocab + multilingual vocab offset).
    Switching sparse models invalidates existing indices and requires a
    fresh embedding_version + re-ingest (see spec §8.2).
    """
    indices: list[int]
    values: list[float]


class SparseChannel:
    """Client for the sparse_server (MILCO HTTP wrapper).

    Two endpoints reflect doc/query pruning asymmetry:
      - ``POST /embed_doc``  — doc-side pruning (settings.sparse_doc_prune_ratio)
      - ``POST /embed_query`` — query-side pruning (settings.sparse_query_prune_ratio)

    The ``settings.sparse_use_source_view`` toggle is honored server-side
    (a global config; not per-request), to avoid query/doc inconsistency.

    Failure mode: returns ``None`` instead of raising, letting the caller
    decide whether to degrade gracefully (e.g. skip sparse channel for one
    query). Mirrors VisionChannel for consistency.
    """

    def __init__(self, settings: Settings, timeout: float = 60.0) -> None:
        self._url = settings.sparse_server_url.rstrip("/")
        # Same lazy-client pattern as TextChannel — see comment there.
        # Long-lived AsyncClient binds to creation-time event loop and breaks
        # when sync wrappers (asyncio.run) create fresh loops.
        self._timeout = timeout

    async def aclose(self) -> None:  # no-op kept for caller compatibility
        return

    async def embed_doc_batch_async(
        self, texts: list[str]
    ) -> list[SparseEmbedding] | None:
        """Batch embed documents at ingest time.

        Returns:
            list of SparseEmbedding, one per input text.
            None if the sparse_server is unavailable (caller decides degrade).
            Empty list if input is empty (no server call).
        """
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/embed_doc",
                    json={"texts": texts},
                )
                resp.raise_for_status()
                data = resp.json()
            return [
                SparseEmbedding(indices=e["indices"], values=e["values"])
                for e in data["embeddings"]
            ]
        except Exception as e:
            logger.warning(
                "Sparse embed_doc failed (%s); skipping sparse channel for batch", e
            )
            return None

    def embed_doc_batch(self, texts: list[str]) -> list[SparseEmbedding] | None:
        """Sync entry-point kept for legacy pipeline call sites."""
        return asyncio.run(self.embed_doc_batch_async(texts))

    async def embed_query_async(self, query: str) -> SparseEmbedding | None:
        """Query-time sparse embedding.

        Returns None if server unavailable (caller decides whether to skip
        the sparse channel in this query, mirroring VisionQueryEmbedder).
        """
        if not query:
            return None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/embed_query",
                    json={"query": query},
                )
                resp.raise_for_status()
                data = resp.json()
            return SparseEmbedding(
                indices=data["indices"], values=data["values"]
            )
        except Exception as e:
            logger.warning(
                "Sparse embed_query failed (%s); skipping sparse channel for query", e
            )
            return None

    def embed_query(self, query: str) -> SparseEmbedding | None:
        """Sync entry-point kept for legacy retriever call sites."""
        return asyncio.run(self.embed_query_async(query))
