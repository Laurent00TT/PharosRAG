# src/kb/search/retriever.py
import asyncio
import logging
import time

from qdrant_client.models import FieldCondition, MatchValue

from kb.config import Settings
from kb.ingestion.channels.text import TextChannel
from kb.observability import RetrieveVariantEvent, emit
from kb.search.constants import CHANNEL_TEXT_DENSE, CHANNEL_TEXT_SPARSE, CHANNEL_VISION, CHANNEL_DESC
from kb.search.vision_query import VisionQueryEmbedder
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantStore,
        meta_db: MetadataDB,
        text_channel: TextChannel | None = None,
    ) -> None:
        self._settings = settings
        self._qdrant = qdrant
        self._meta_db = meta_db
        # P1-2: explicit settings so the channel honors injected config.
        self._text = text_channel or TextChannel(settings=settings)
        # P0-4: SparseChannel wired into search; before this fix retriever
        # asked text.py for sparse and got an empty list, killing sparse hits.
        from kb.ingestion.channels.sparse import SparseChannel
        self._sparse = SparseChannel(settings=settings)
        self._vision_q = VisionQueryEmbedder(settings)

    async def aclose(self) -> None:
        await self._vision_q.aclose()
        await self._sparse.aclose()
        await self._text.aclose()

    async def retrieve(
        self,
        query: str,
        limit: int,
        doc_filter: dict,
        include_deprecated: bool = False,
    ) -> tuple[
        list[tuple[str, int, float]],
        list[tuple[str, int, float]],
        list[tuple[str, int, float]],
        list[tuple[str, int, float]],
    ]:
        """Returns (dense_hits, sparse_hits, vision_hits, desc_hits).
        Each hit is (doc_id, page_num, score)."""
        if include_deprecated:
            doc_ids = None
        else:
            valid = await self._meta_db.list_valid_doc_ids()
            if not valid:
                emit(RetrieveVariantEvent(
                    embed_ms=0.0,
                    vision_query_ms=0.0,
                    vision_query_ok=False,
                    search_ms=0.0,
                    dense_hits=0,
                    sparse_hits=0,
                    vision_hits=0,
                    desc_hits=0,
                    limit=limit,
                    include_deprecated=include_deprecated,
                ))
                return [], [], [], []
            doc_ids = valid

        extra_must = self._build_extra_must(doc_filter)

        # Run text dense, sparse, and vision query embeddings in parallel.
        # PoC v1.1: text channel returns (dense, [], []) — sparse comes
        # from the dedicated SparseChannel (MILCO HTTP service).
        t_embed = time.monotonic()
        text_embed_task = asyncio.to_thread(self._text.embed_query, query)
        sparse_embed_task = self._sparse.embed_query_async(query)
        vision_embed_task = self._vision_q.embed_query_async(query)
        (dense, _, _), sparse_emb, vision_emb = await asyncio.gather(
            text_embed_task, sparse_embed_task, vision_embed_task,
        )
        # Sparse may be None (server down) — fall back to empty so the
        # sparse channel returns zero hits without breaking the rest.
        if sparse_emb is None:
            s_idx: list[int] = []
            s_val: list[float] = []
        else:
            s_idx = sparse_emb.indices
            s_val = sparse_emb.values
        embed_ms = round((time.monotonic() - t_embed) * 1000, 1)
        # vision_query_ms is now folded into embed_ms (parallel); keep field for
        # back-compat in trace but mark it as combined.
        vision_query_ms = embed_ms
        vision_query_ok = vision_emb is not None

        dense_task  = asyncio.to_thread(self._qdrant.search_text_dense,
                                        dense, limit, doc_ids, extra_must)
        sparse_task = asyncio.to_thread(self._qdrant.search_text_sparse,
                                        s_idx, s_val, limit, doc_ids, extra_must)
        desc_task   = asyncio.to_thread(self._qdrant.search_desc,
                                        dense, limit, doc_ids, extra_must)

        t_search = time.monotonic()
        if vision_emb is not None:
            vision_task = asyncio.to_thread(self._qdrant.search_vision,
                                            vision_emb, limit, doc_ids,
                                            extra_must)
            dense_r, sparse_r, desc_r, vision_r = await asyncio.gather(
                dense_task, sparse_task, desc_task, vision_task
            )
        else:
            dense_r, sparse_r, desc_r = await asyncio.gather(
                dense_task, sparse_task, desc_task
            )
            vision_r = []
        search_ms = round((time.monotonic() - t_search) * 1000, 1)

        def to_hits(results: list[dict]) -> list[tuple[str, int, float]]:
            return [(r["doc_id"], r["page_num"], r["score"]) for r in results]

        d_hits = to_hits(dense_r)
        s_hits = to_hits(sparse_r)
        v_hits = to_hits(vision_r)
        dc_hits = to_hits(desc_r)
        emit(RetrieveVariantEvent(
            embed_ms=embed_ms,                      # combined text + vision query embed (parallel)
            vision_query_ms=vision_query_ms,        # back-compat alias for embed_ms
            vision_query_ok=vision_query_ok,        # False = vision channel skipped
            search_ms=search_ms,                    # parallel Qdrant query for all 4 channels
            dense_hits=len(d_hits),
            sparse_hits=len(s_hits),
            vision_hits=len(v_hits),
            desc_hits=len(dc_hits),
            limit=limit,
            include_deprecated=include_deprecated,
        ))
        return d_hits, s_hits, v_hits, dc_hits

    def _build_extra_must(self, doc_filter: dict) -> list:
        must = []
        if "doc_name" in doc_filter:
            must.append(FieldCondition(key="doc_name",
                                       match=MatchValue(value=doc_filter["doc_name"])))
        if "page_type" in doc_filter:
            must.append(FieldCondition(key="page_type",
                                       match=MatchValue(value=doc_filter["page_type"])))
        return must
