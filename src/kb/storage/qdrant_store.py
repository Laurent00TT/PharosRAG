# src/kb/storage/qdrant_store.py
#
# PoC v1.1 schema (spec §2):
#   - dense size = 4096(Qwen3-VL-Embedding-8B 原生)
#   - vision = single-vector(text/desc/vision 共用同一向量空间,
#     vector field 改名 colqwen → vision_dense 清晰区分)
#   - sparse 沿用 SparseVectorParams,token id/vocab 由 MILCO 决定
#
# 旧 collection(embedding_version=bge-m3_colqwen25_v02)在 Qdrant 中仍可读,
# 但当前代码不再访问其 vector field "colqwen" / size=128 / MultiVectorConfig。
# 走旧 collection 需要单独的 LegacyQdrantStore(本 PoC 不实现)。
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, Filter, FieldCondition, MatchValue, MatchAny,
)

from kb.config import Settings
from kb.models import TextChunk, VisionPage, DescPage

logger = logging.getLogger(__name__)

# Vector field name constants — change here propagates everywhere
DENSE_VEC_NAME = "dense"
SPARSE_VEC_NAME = "sparse"
VISION_VEC_NAME = "vision_dense"   # renamed from "colqwen" in PoC v1.1


class QdrantStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if not settings.qdrant_url:
            self.client = QdrantClient(path=settings.qdrant_path)
        else:
            self.client = QdrantClient(url=settings.qdrant_url)

        ver = settings.embedding_version
        self.text_collection   = f"text_chunks_{ver}"
        self.vision_collection = f"vision_pages_{ver}"
        self.desc_collection   = f"desc_pages_{ver}"

    async def ensure_collections(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}

        if self.text_collection not in existing:
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config={
                    DENSE_VEC_NAME: VectorParams(size=4096, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    SPARSE_VEC_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )

        if self.vision_collection not in existing:
            # PoC v1.1: single-vector dense (Qwen3-VL-Embedding-8B);
            # was multi-vector MAX_SIM (ColQwen2.5) — see commit history.
            self.client.create_collection(
                collection_name=self.vision_collection,
                vectors_config={
                    VISION_VEC_NAME: VectorParams(size=4096, distance=Distance.COSINE),
                },
            )

        if self.desc_collection not in existing:
            self.client.create_collection(
                collection_name=self.desc_collection,
                vectors_config={
                    DENSE_VEC_NAME: VectorParams(size=4096, distance=Distance.COSINE),
                },
            )

    async def upsert_text_chunk(
        self,
        chunk: TextChunk,
        dense: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        *,
        owner_id: str | None = None,
    ) -> None:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense,
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page_num": chunk.page_num,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "heading_path": chunk.heading_path,
                "has_visual_on_page": chunk.has_visual_on_page,
                "page_type": chunk.page_type,
                "parent_chunk_id": chunk.parent_chunk_id,
                "is_parent": chunk.is_parent,
                "owner_id": owner_id,   # T3: doctor reverse-lookup only (NOT for filter)
            },
        )
        self.client.upsert(collection_name=self.text_collection, points=[point])

    async def upsert_vision_page(
        self,
        page: VisionPage,
        embedding: list[float],
        *,
        owner_id: str | None = None,
    ) -> None:
        # PoC v1.1: embedding is single 4096-dim vector (was multi-vec in v1.0)
        vector = {VISION_VEC_NAME: embedding}

        # Deterministic id keyed on (doc_id, page_num): exactly one vision record
        # per page, so a re-ingest / backfill OVERWRITES rather than duplicating.
        # (A random uuid4 here let repeated backfills accrete duplicate records.)
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"vision:{page.doc_id}:{page.page_num}")),
            vector=vector,
            payload={
                "doc_id": page.doc_id, "doc_name": page.doc_name,
                "page_num": page.page_num, "image_url": page.image_url,
                "heading_path": page.heading_path, "vision_tier": page.vision_tier,
                "page_type": page.page_type,
                "owner_id": owner_id,   # T3: doctor reverse-lookup only (NOT for filter)
            },
        )
        self.client.upsert(collection_name=self.vision_collection, points=[point])

    async def upsert_desc_page(
        self,
        page: DescPage,
        dense: list[float],
        *,
        owner_id: str | None = None,
    ) -> None:
        # Deterministic id keyed on (doc_id, page_num): exactly one description
        # record per page, so a re-run backfill OVERWRITES rather than duplicating.
        # (A random uuid4 here let repeated backfills accrete duplicate records —
        # the desc collection had grown to ~2x its page count before this fix.)
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"desc:{page.doc_id}:{page.page_num}")),
            vector={"dense": dense},
            payload={
                "doc_id": page.doc_id, "doc_name": page.doc_name,
                "page_num": page.page_num, "page_type": page.page_type,
                "figure_index": page.figure_index, "figure_caption": page.figure_caption,
                "position_hint": page.position_hint, "description": page.description,
                "key_entities": page.key_entities, "image_url": page.image_url,
                "heading_path": page.heading_path,
                "owner_id": owner_id,   # T3: doctor reverse-lookup only (NOT for filter)
            },
        )
        self.client.upsert(collection_name=self.desc_collection, points=[point])

    async def delete_by_doc_id(self, doc_id: str) -> None:
        f = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        for col in (self.text_collection, self.vision_collection, self.desc_collection):
            try:
                self.client.delete(collection_name=col, points_selector=f)
            except Exception as e:
                logger.warning("Failed to delete from %s: %s", col, e)

    def search_text_dense(
        self,
        dense: list[float],
        limit: int,
        doc_ids: list[str] | None = None,
        extra_must: list | None = None,
    ) -> list[dict]:
        must = list(extra_must) if extra_must else []
        if doc_ids:
            must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
        must.append(FieldCondition(key="is_parent", match=MatchValue(value=False)))
        response = self.client.query_points(
            collection_name=self.text_collection,
            query=dense,
            using="dense",
            query_filter=Filter(must=must),
            limit=limit,
        )
        return [{"score": r.score, **r.payload} for r in response.points]

    def search_text_sparse(
        self,
        indices: list[int],
        values: list[float],
        limit: int,
        doc_ids: list[str] | None = None,
        extra_must: list | None = None,
    ) -> list[dict]:
        must = list(extra_must) if extra_must else []
        if doc_ids:
            must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
        must.append(FieldCondition(key="is_parent", match=MatchValue(value=False)))
        response = self.client.query_points(
            collection_name=self.text_collection,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            query_filter=Filter(must=must),
            limit=limit,
        )
        return [{"score": r.score, **r.payload} for r in response.points]

    def search_desc(
        self,
        dense: list[float],
        limit: int,
        doc_ids: list[str] | None = None,
        extra_must: list | None = None,
    ) -> list[dict]:
        must = list(extra_must) if extra_must else []
        if doc_ids:
            must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
        f = Filter(must=must) if must else None
        response = self.client.query_points(
            collection_name=self.desc_collection,
            query=dense,
            using="dense",
            query_filter=f,
            limit=limit,
        )
        return [{"score": r.score, **r.payload} for r in response.points]

    def search_vision(
        self,
        embedding: list[float],
        limit: int,
        doc_ids: list[str] | None = None,
        extra_must: list | None = None,
    ) -> list[dict]:
        # PoC v1.1: query is single 4096-dim vector (was multi-vec MAX_SIM in v1.0)
        must = list(extra_must) if extra_must else []
        if doc_ids:
            must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
        f = Filter(must=must) if must else None
        response = self.client.query_points(
            collection_name=self.vision_collection,
            query=embedding,
            using=VISION_VEC_NAME,
            query_filter=f,
            limit=limit,
        )
        return [{"score": r.score, **r.payload} for r in response.points]

    def get_parent_chunk(self, doc_id: str, page_num: int) -> dict | None:
        points, _ = self.client.scroll(
            collection_name=self.text_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="page_num", match=MatchValue(value=page_num)),
                FieldCondition(key="is_parent", match=MatchValue(value=True)),
            ]),
            limit=1,
            with_payload=True,
        )
        if points:
            return points[0].payload
        # Fallback: any chunk on this page (handles pages with only one chunk)
        points, _ = self.client.scroll(
            collection_name=self.text_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="page_num", match=MatchValue(value=page_num)),
            ]),
            limit=1,
            with_payload=True,
        )
        return points[0].payload if points else None

    def get_desc_record(self, doc_id: str, page_num: int) -> dict | None:
        points, _ = self.client.scroll(
            collection_name=self.desc_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="page_num", match=MatchValue(value=page_num)),
            ]),
            limit=1,
            with_payload=True,
        )
        return points[0].payload if points else None

    def get_vision_record(self, doc_id: str, page_num: int) -> dict | None:
        """Returns the vision_pages payload for a given page (image_url, vision_tier, ...).
        Used by expander as a fallback for image_url when desc channel did not run
        (e.g. pure-text pages skip DescriptionChannel but ImageStore still saved the page image)."""
        points, _ = self.client.scroll(
            collection_name=self.vision_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="page_num", match=MatchValue(value=page_num)),
            ]),
            limit=1,
            with_payload=True,
        )
        return points[0].payload if points else None
