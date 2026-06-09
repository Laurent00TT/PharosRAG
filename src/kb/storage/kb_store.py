import logging

from kb.storage.image_store import ImageStore
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class KnowledgeBaseStore:
    def __init__(
        self,
        qdrant: QdrantStore,
        meta_db: MetadataDB,
        image_store: ImageStore,
    ) -> None:
        self.qdrant = qdrant
        self.meta_db = meta_db
        self.image_store = image_store

    async def ensure_ready(self) -> None:
        await self.qdrant.ensure_collections()
        await self.meta_db.init()

    async def cleanup_doc_assets(self, doc_id: str) -> dict:
        vectors_deleted = False
        vector_error = ""
        try:
            await self.qdrant.delete_by_doc_id(doc_id)
            vectors_deleted = True
        except Exception as exc:
            vector_error = str(exc)
            logger.warning("Failed to delete vectors for %s: %s", doc_id, exc)

        try:
            images = await self.image_store.delete_doc_images(doc_id)
        except Exception as exc:
            images = {"deleted": False, "error": str(exc)}
            logger.warning("Failed to delete images for %s: %s", doc_id, exc)

        return {
            "doc_id": doc_id,
            "vectors_deleted": vectors_deleted,
            "vector_error": vector_error,
            "images": images,
        }

    async def cleanup_staging_doc(self, doc_id: str) -> dict:
        return await self.cleanup_doc_assets(doc_id)

    async def activate_new_version(self, new_doc_id: str, old_doc_ids: list[str]) -> None:
        await self.meta_db.activate_document(new_doc_id)
        for old_id in old_doc_ids:
            await self.meta_db.deprecate_document(old_id)
