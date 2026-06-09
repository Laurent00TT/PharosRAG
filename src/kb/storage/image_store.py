# src/kb/storage/image_store.py
import logging
import shutil
from pathlib import Path
from kb.config import Settings

logger = logging.getLogger(__name__)


class ImageStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Settings 已删除 blob/use_local_storage 字段，本地存储为唯一选项
        Path(settings.image_storage_path).mkdir(parents=True, exist_ok=True)

    async def save(self, doc_id: str, page_num: int, image_bytes: bytes) -> str:
        return await self._save_local(doc_id, page_num, image_bytes)

    async def _save_local(self, doc_id: str, page_num: int, image_bytes: bytes) -> str:
        doc_dir = Path(self._settings.image_storage_path) / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        path = doc_dir / f"page_{page_num}.png"
        path.write_bytes(image_bytes)
        return str(path)

    async def get_url(self, doc_id: str, page_num: int) -> str:
        path = Path(self._settings.image_storage_path) / doc_id / f"page_{page_num}.png"
        return str(path)

    async def delete_doc_images(self, doc_id: str) -> dict:
        doc_dir = Path(self._settings.image_storage_path) / doc_id
        if not doc_dir.exists():
            return {"deleted": False, "reason": "not_found"}
        shutil.rmtree(doc_dir)
        return {"deleted": True, "path": str(doc_dir)}
