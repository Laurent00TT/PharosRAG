# src/kb/storage/source_store.py
"""Original-document blob store.

Keeps the source PDF for each ingested document so the workbench (and MCP
agents) can navigate back to the original, and so pages can be re-rendered at
higher DPI on demand. Mirrors ImageStore: local filesystem, one file per
doc_id under settings.source_storage_path.
"""
import asyncio
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from kb.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredSource:
    path: str
    sha256: str
    bytes: int


class SourceDocStore:
    def __init__(self, settings: Settings) -> None:
        self._root = Path(settings.source_storage_path)
        self._root.mkdir(parents=True, exist_ok=True)

    def _doc_path(self, doc_id: str) -> Path:
        # Defense-in-depth: doc_id can originate from a URL path. Reject
        # anything that could escape the store root, even though current
        # callers already validate (the /source endpoint via _require_active_doc;
        # ingest passes a deterministic UUID).
        if not doc_id or ".." in doc_id or "/" in doc_id or "\\" in doc_id or "\x00" in doc_id:
            raise ValueError(f"unsafe doc_id: {doc_id!r}")
        path = self._root / f"{doc_id}.pdf"
        if not path.resolve().is_relative_to(self._root.resolve()):
            raise ValueError(f"doc_id escapes source root: {doc_id!r}")
        return path

    async def save(self, doc_id: str, src_path: Path) -> StoredSource:
        """Copy the source file into the store, returning its sha256 + size.

        The copy+hash is blocking IO, so it runs in a worker thread to keep
        the ingest event loop responsive.
        """
        return await asyncio.to_thread(self._save_sync, doc_id, Path(src_path))

    def _save_sync(self, doc_id: str, src_path: Path) -> StoredSource:
        dest = self._doc_path(doc_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        size = 0
        # Stream so large PDFs never sit fully in memory.
        with open(src_path, "rb") as fin, open(dest, "wb") as fout:
            while True:
                chunk = fin.read(1 << 20)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
                fout.write(chunk)
        return StoredSource(path=str(dest), sha256=hasher.hexdigest(), bytes=size)

    def get_path(self, doc_id: str) -> Path | None:
        try:
            path = self._doc_path(doc_id)
        except ValueError:
            logger.warning("rejected unsafe doc_id for source fetch: %r", doc_id)
            return None
        return path if path.exists() else None

    def delete(self, doc_id: str) -> bool:
        try:
            path = self._doc_path(doc_id)
        except ValueError:
            logger.warning("rejected unsafe doc_id for source delete: %r", doc_id)
            return False
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:  # noqa: BLE001 — best-effort cleanup
            logger.warning("source-doc delete failed for %s: %s", doc_id, exc)
            return False
