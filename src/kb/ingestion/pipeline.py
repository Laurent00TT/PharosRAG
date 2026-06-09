# src/kb/ingestion/pipeline.py
import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from kb.config import Settings
from kb.ingestion.channels.description import DescriptionChannel
from kb.ingestion.channels.text import TextChannel
from kb.ingestion.channels.vision import VisionChannel
from kb.ingestion.chunker import Chunker
from kb.jobs.cancel import JobCancelled
from kb.models import TextChunk, VisionPage
from kb.nav.auto_build import auto_build_nav_for_doc
from kb.nav.builder import NavIndexBuilder
from kb.nav.store import NavIndexStore
from kb.observability import emit, emit_error
from kb.parser import make_parser
from kb.parser.document_classifier import classify_document
from kb.storage.image_store import ImageStore
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore
from kb.storage.source_store import SourceDocStore

logger = logging.getLogger(__name__)

CancelCheck = Callable[[], Awaitable[bool]]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        nav_store: NavIndexStore | None = None,
        nav_builder: NavIndexBuilder | None = None,
        cancel_check: CancelCheck | None = None,
        audit_log=None,
    ) -> None:
        """Initialize the ingestion pipeline.

        ``nav_store`` and ``nav_builder`` are optional injection points
        for the navigation layer. When the server lifespan owns the
        nav layer it passes them through here; the pipeline calls
        ``auto_build_nav_for_doc`` at the end of each ``ingest()`` so
        every successful ingestion populates the nav index.

        When ``nav_store``/``nav_builder`` are None, auto-build silently
        no-ops — this preserves the standalone test path (tests that
        instantiate IngestionPipeline without nav wiring stay green).
        """
        self._settings = settings
        # Factory selects local (GPU/CPU) or cloud API based on
        # settings.parser_backend at startup. No automatic fallback.
        self._parser = make_parser(settings)
        self._chunker = Chunker(
            small_size=settings.chunker_small_size,
            parent_size=settings.chunker_parent_size,
            overlap=settings.chunker_overlap,
            min_headings_for_segmenting=settings.chunker_min_headings_for_segmenting,
        )
        # P1-2: pass settings explicitly so the channels honor injected
        # config (tests, runtime overrides) instead of re-reading .env.
        self._text_ch = TextChannel(settings=settings)
        # P0-4: SparseChannel wired into the ingest pipeline so MILCO sparse
        # vectors land in Qdrant. Before this fix the channel existed but
        # text.py returned (dense, [], []) and pipeline upserted empty sparse.
        from kb.ingestion.channels.sparse import SparseChannel
        self._sparse_ch = SparseChannel(settings=settings)
        self._vision_ch = VisionChannel(settings=settings)
        # description_enabled=False short-circuits the channel: no adapter
        # is built, no /describe calls are made, and pages are indexed
        # without the visual_description field. Saves the noisy "adapter
        # parse failed (Connection error)" log spam when running KB inside
        # docker without a host-side vision LM reachable.
        self._desc_ch = (
            self._build_description_channel(settings)
            if settings.description_enabled
            else None
        )
        self._qdrant = QdrantStore(settings=settings)
        self._meta_db = MetadataDB(
            db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
        )
        self._image_store = ImageStore(settings=settings)
        # Original-document store (kept so the workbench / MCP can navigate
        # back to the source PDF). Gated by settings.store_source_documents.
        self._source_store = SourceDocStore(settings=settings)
        self._store_source = settings.store_source_documents
        # Nav layer: caller-injected (lifespan-owned). Not lazily built
        # here — that would create a second connection to nav_index.db
        # and fight the lifespan-owned store for the same file. If a
        # caller wants auto-build outside the server context, they pass
        # in their own NavIndexStore explicitly.
        self._nav_store = nav_store
        self._nav_builder = nav_builder
        # Cooperative-cancel hook. The executor injects a closure that
        # reads `cancel_requested` on the owning job; pipeline polls it
        # at each page boundary (cheapest place that still yields a
        # responsive cancel — typically <60s wait worst-case). When
        # None, cancel polling no-ops and behavior is identical to the
        # pre-cancel implementation.
        self._cancel_check = cancel_check
        # Best-effort audit sink (spec §4.5 / A-3). Worker constructs the
        # AuditLog and injects via pipeline_factory; when None (legacy
        # call sites and tests), audit writes are silently skipped.
        # user_id is None for T2 — T3 will fill via ContextVar / job
        # owner once IngestionJob exposes owner_id.
        self._audit_log = audit_log

    async def _check_cancel(self, context: str) -> None:
        """Poll the cancel hook and raise ``JobCancelled`` if asserted.

        The caller is responsible for partial-write cleanup before
        letting the exception propagate (see the page loop in ingest()).
        Cancel-check hook errors are swallowed so a transient metadata
        DB hiccup never turns a normal ingest into a forced cancel —
        the lease-expiry fallback can still pick up real cancels later.
        """
        if self._cancel_check is None:
            return
        try:
            cancelled = await self._cancel_check()
        except Exception as exc:
            logger.warning("cancel_check raised %r — treating as not-cancelled", exc)
            return
        if cancelled:
            raise JobCancelled(context)

    async def _heal_nav_for_skipped_doc(
        self, *, doc_id: str, doc_name: str,
    ) -> int:
        """When skipping an unchanged-hash ingest, check whether nav has
        entries for this doc; if not, synthesize ParsedPage stubs from
        qdrant text chunks and run auto-build. Returns the number of nav
        entries built (0 = nav already healthy or self-heal disabled).

        Why this matters: the unchanged-hash skip path used to return
        immediately, so users who needed to rebuild nav (because the
        nav db was wiped, or the doc predated the nav layer) had to
        remember to run scripts/backfill_nav_index.py. Now main path
        self-heals.

        Wrapped in try/except so a backfill hiccup never blocks the
        normal skip behavior — the existing manual backfill script
        remains the deterministic recovery tool.
        """
        if self._nav_store is None or self._nav_builder is None:
            return 0
        if not self._settings.nav_auto_build_on_ingest:
            return 0
        try:
            existing_entries = await self._nav_store.list_entries(doc_id=doc_id)
        except Exception as exc:
            logger.warning("nav self-heal: list_entries failed for %s: %s", doc_id, exc)
            return 0
        if existing_entries:
            return 0  # nav already populated, nothing to repair
        try:
            from kb.parser.qdrant_to_pages import synthesize_pages_from_qdrant
            pages = await synthesize_pages_from_qdrant(
                self._qdrant, doc_id=doc_id, doc_name=doc_name,
            )
            if not pages:
                return 0  # no qdrant chunks either; nothing to build from
            count = await auto_build_nav_for_doc(
                nav_store=self._nav_store,
                nav_builder=self._nav_builder,
                pages=pages,
                enabled=True,
            )
            logger.info(
                "nav self-heal: built %d entries for skipped doc %s",
                count, doc_id,
            )
            return count
        except Exception as exc:
            logger.warning("nav self-heal failed for %s: %s", doc_id, exc)
            return 0

    async def _cleanup_partial_writes(self, doc_id: str) -> None:
        """Best-effort drop of qdrant points + image files for a doc that
        is being cancelled mid-ingest. Doc never reached active state in
        metadata (upsert_document runs after the page loop), so no
        metadata row needs deleting. Both calls are wrapped in try/except
        because a failed cleanup must not mask the cancel signal — the
        executor needs to see JobCancelled, not the cleanup error.
        """
        try:
            await self._qdrant.delete_by_doc_id(doc_id)
        except Exception as exc:
            logger.warning(
                "Partial qdrant cleanup failed for cancelled %s: %s", doc_id, exc,
            )
        try:
            await self._image_store.delete_doc_images(doc_id)
        except Exception as exc:
            logger.warning(
                "Partial image cleanup failed for cancelled %s: %s", doc_id, exc,
            )

    async def aclose(self) -> None:
        """Release every per-pipeline resource. Long-running workers
        instantiate one IngestionPipeline per job item, so missing any
        close here accumulates handles: HTTP keep-alives in the parser
        / vision / desc HTTP clients, the SQLAlchemy engine + SQLite
        FD inside MetadataDB, and the qdrant HTTP client.

        All closes are wrapped in try/except so one stuck resource
        can't block teardown of the others — the executor's finally
        block must always finish cleanly.
        """
        # Channels (HTTP clients to mineru / embedding server / vision_llm).
        # _desc_ch is None when settings.description_enabled=False — skip it.
        for resource in (self._parser, self._vision_ch, self._desc_ch):
            if resource is None:
                continue
            close = getattr(resource, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:
                logger.warning("aclose failed on %r: %s", resource, exc)
        # MetadataDB owns a SQLAlchemy async engine; dispose releases
        # the connection pool and underlying SQLite file handle.
        if self._meta_db is not None:
            try:
                await self._meta_db.aclose()
            except Exception as exc:
                logger.warning("meta_db aclose failed: %s", exc)
        # Qdrant client has an HTTP keepalive pool. close() is sync.
        qclient = getattr(self._qdrant, "client", None)
        if qclient is not None:
            close = getattr(qclient, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    logger.warning("qdrant client close failed: %s", exc)

    async def ingest(self, path: Path, *, owner_id: str | None = None) -> dict:
        """Run the full ingestion chain. owner_id is the user the resulting
        document(s) are attributed to — set documents.owner_id + qdrant
        payload owner_id on every chunk/page. None means a non-API ingest
        (CLI script, T1a backfill); the doc ends up with NULL owner_id and
        is admin-only-writable per I-11.
        """
        t_total = time.monotonic()
        file_size = path.stat().st_size if path.exists() else 0
        emit(
            "ingest.start",
            doc_path=str(path),
            doc_name=path.name,
            file_size_bytes=file_size,
            suffix=path.suffix.lower(),
        )

        # 0. Classification gate — avoid crash paths (encrypted/corrupt PDFs, etc.)
        classification = classify_document(path)
        emit(
            "classify.result",
            classification=classification.classification,
            is_processable=classification.is_processable,
            page_count=classification.page_count,
            signals=classification.signals,
            error=classification.error,
        )
        if not classification.is_processable:
            logger.warning(
                "Document %s classified as %s (not processable): %s",
                path.name, classification.classification, classification.error,
            )
            emit(
                "ingest.end",
                duration_ms=round((time.monotonic() - t_total) * 1000, 1),
                doc_id="",
                skipped=True,
                pages_processed=0,
                chunks_stored=0,
                reason=f"unprocessable_{classification.classification}",
                success=False,
            )
            return {
                "doc_id": "",
                "skipped": True,
                "pages_processed": 0,
                "chunks_stored": 0,
                "error": f"unprocessable: {classification.classification}",
            }

        await self._qdrant.ensure_collections()
        await self._meta_db.init()

        doc_id = hashlib.md5(path.read_bytes()).hexdigest()

        # Skip if a document with the same hash is already active.
        # When skipping, self-heal a missing NavIndex from existing
        # qdrant chunks — covers two real cases:
        #   1. Doc was ingested before the nav layer existed
        #   2. nav_index.db was deleted/recreated since the original ingest
        # Without this, the user has to remember to run
        # scripts/backfill_nav_index.py — main path silently leaves the
        # nav gap.
        existing = await self._meta_db.get_document(doc_id)
        if existing and existing["status"] == "active":
            logger.info("Skipping unchanged document: %s", path.name)
            nav_repaired = await self._heal_nav_for_skipped_doc(
                doc_id=doc_id, doc_name=path.name,
            )
            emit(
                "ingest.end",
                duration_ms=round((time.monotonic() - t_total) * 1000, 1),
                doc_id=doc_id,
                skipped=True,
                pages_processed=0,
                chunks_stored=0,
                reason="unchanged_hash",
                nav_repaired=nav_repaired,
                success=True,
            )
            return {
                "doc_id": doc_id, "skipped": True,
                "pages_processed": 0, "chunks_stored": 0,
                "nav_repaired": nav_repaired,
            }

        # Collect prior versions of the same filename. They are deprecated only
        # after the replacement has parsed and stored successfully.
        #
        # T3 critical fix (review P0): scope the deprecate set to the SAME
        # owner. Without this, Bob uploading a doc with the same filename
        # as Alice's would silently deprecate Alice's content — violating
        # the small-team invariant "大锅饭只覆盖 read，不覆盖 write".
        #
        # owner_id=None (CLI ingest / pre-T3 path) only matches other
        # None-owner docs. T1a "_system" backfill rows never match a real
        # user UUID and stay untouched (admin can clean them via
        # require_owner_or_admin's NULL-owner branch + explicit delete).
        old_doc_ids: list[str] = []
        all_active = await self._meta_db.list_active_doc_ids()
        for old_id in all_active:
            rec = await self._meta_db.get_document(old_id)
            if (
                rec
                and rec["doc_name"] == path.name
                and old_id != doc_id
                and rec.get("owner_id") == owner_id   # cross-owner protection
            ):
                old_doc_ids.append(old_id)

        # Cancel checkpoint #1: before paying the parser cost. If the
        # user cancelled while waiting in queue but lease started anyway,
        # bail before MinerU runs (which can be the most expensive step).
        # No partial cleanup needed — nothing has been written yet.
        try:
            await self._check_cancel(f"cancelled before parsing {path.name}")
        except JobCancelled:
            emit(
                "ingest.cancelled",
                duration_ms=round((time.monotonic() - t_total) * 1000, 1),
                doc_id=doc_id,
                pages_processed=0,
                chunks_stored=0,
                vision_pages_stored=0,
                desc_pages_stored=0,
            )
            raise

        t_parse = time.monotonic()
        # Backend selection (local GPU/CPU or — once Stage 5 lands — cloud API)
        # is encapsulated inside the parser. Pipeline just hands off the path.
        pages = await self._parser.parse_async(path)
        emit(
            "parse.mineru",
            duration_ms=round((time.monotonic() - t_parse) * 1000, 1),
            doc_id=doc_id,
            pages_returned=len(pages),
            has_image_b64=any(p.page_image for p in pages) if pages else False,
            gpu_mode=self._settings.ingest_use_gpu_mineru,
        )
        chunks_stored = 0
        vision_pages_stored = 0
        desc_pages_stored = 0

        for page in pages:
            # Cancel checkpoint #2: top of each page. Cooperative cancel
            # — worst-case wait is one page's processing time (~10–60s).
            # If asserted, raise JobCancelled; partial qdrant/image
            # writes for this doc are cleaned up by the JobCancelled
            # handler below before the exception propagates.
            try:
                await self._check_cancel(
                    f"cancelled at page {page.page_num} of {path.name}"
                )
            except JobCancelled:
                await self._cleanup_partial_writes(doc_id)
                emit(
                    "ingest.cancelled",
                    duration_ms=round((time.monotonic() - t_total) * 1000, 1),
                    doc_id=doc_id,
                    pages_processed=page.page_num,
                    chunks_stored=chunks_stored,
                    vision_pages_stored=vision_pages_stored,
                    desc_pages_stored=desc_pages_stored,
                )
                raise
            image_url = await self._image_store.save(
                doc_id=page.doc_id, page_num=page.page_num, image_bytes=page.page_image
            )

            # Channel 1: text chunking + Qwen3-VL-Embedding dense + MILCO sparse
            # (PoC v1.1: dense and sparse come from two separate services and
            # are dispatched in parallel; both feed Qdrant.upsert_text_chunk.)
            chunks = self._chunker.chunk_page(page)
            t_text = time.monotonic()
            if chunks:
                # Parallel: dense (Qwen3-VL via HTTP) + sparse (MILCO via HTTP).
                # text_ch.embed_batch returns (dense, [], []) — the trailing
                # empty sparse placeholders are intentionally ignored here
                # because real sparse comes from sparse_ch.embed_doc_batch_async.
                text_task = asyncio.to_thread(self._text_ch.embed_batch, chunks)
                sparse_task = self._sparse_ch.embed_doc_batch_async(
                    [c.text for c in chunks]
                )
                batch_results, sparse_results = await asyncio.gather(
                    text_task, sparse_task,
                )
                # SparseChannel returns None on server failure — degrade
                # gracefully (empty sparse) so dense + vision still index.
                if sparse_results is None:
                    sparse_results = [None] * len(chunks)
                for chunk, (dense, _, _), sparse in zip(
                    chunks, batch_results, sparse_results, strict=True,
                ):
                    if sparse is None:
                        s_idx: list[int] = []
                        s_val: list[float] = []
                    else:
                        s_idx = sparse.indices
                        s_val = sparse.values
                    await self._qdrant.upsert_text_chunk(
                        chunk, dense, s_idx, s_val, owner_id=owner_id,
                    )
                    chunks_stored += 1
            emit(
                "embed.text",
                duration_ms=round((time.monotonic() - t_text) * 1000, 1),
                doc_id=doc_id,
                page_num=page.page_num,
                chunks_count=len(chunks),
                has_table=bool(page.tables),
                has_image_block=bool(page.image_blocks),
            )

            # Channel 2: vision embedding (HTTP to the multimodal embedding server)
            t_vision = time.monotonic()
            vision_emb = await self._vision_ch.embed_async(page.page_image)
            vision_ok = vision_emb is not None
            if vision_ok:
                page_type = "table" if page.tables else (
                    "mixed" if page.image_blocks else "text"
                )
                vp = VisionPage(
                    doc_id=page.doc_id, doc_name=page.doc_name,
                    page_num=page.page_num, image_url=image_url,
                    heading_path=page.heading_path,
                    # Provenance only — no code branches on this value. Single-
                    # vector Qwen3-VL embedding replaced the retired ColQwen
                    # multi-vector tier (PoC v1.0 → v1.1).
                    vision_tier="qwen3vl",
                    page_type=page_type,
                )
                await self._qdrant.upsert_vision_page(vp, vision_emb, owner_id=owner_id)
                vision_pages_stored += 1
            emit(
                "embed.vision",
                duration_ms=round((time.monotonic() - t_vision) * 1000, 1),
                doc_id=doc_id,
                page_num=page.page_num,
                success=vision_ok,
                image_size_bytes=len(page.page_image),
            )

            # Channel 3: description (independent vision LLM pass).
            # Skipped entirely when self._desc_ch is None (settings.
            # description_enabled=False).
            total_len = len(page.structured_text) + sum(len(str(t)) for t in page.tables) + 100
            text_ratio = len(page.structured_text) / total_len
            should_describe = self._desc_ch is not None and self._desc_ch.should_process(
                has_image_blocks=bool(page.image_blocks), text_char_ratio=text_ratio
            )
            if should_describe:
                desc = await self._desc_ch.describe(
                    doc_id=page.doc_id, doc_name=page.doc_name,
                    page_num=page.page_num, image_bytes=page.page_image,
                    heading_path=page.heading_path, image_url=image_url,
                )
                desc_text_chunk = TextChunk(
                    doc_id=desc.doc_id, doc_name=desc.doc_name,
                    page_num=desc.page_num, chunk_index=0,
                    text=desc.description, heading_path=desc.heading_path,
                    has_visual_on_page=True, page_type=desc.page_type,
                    parent_chunk_id="", is_parent=False,
                )
                desc_dense, _, _ = await asyncio.to_thread(self._text_ch.embed, desc_text_chunk)
                await self._qdrant.upsert_desc_page(desc, desc_dense, owner_id=owner_id)
                desc_pages_stored += 1

        # Retain the source PDF so the workbench / MCP can navigate back to
        # the original. Non-fatal: a storage failure must not fail an
        # otherwise-successful ingest, it just leaves the doc source-less.
        source_sha256: str | None = None
        source_bytes: int | None = None
        if self._store_source:
            try:
                stored = await self._source_store.save(doc_id, path)
                source_sha256 = stored.sha256
                source_bytes = stored.bytes
            except Exception as exc:  # noqa: BLE001 — defensive, non-fatal
                logger.warning("source-doc store failed for %s: %s", doc_id, exc)

        await self._meta_db.upsert_document(
            doc_id=doc_id, doc_name=path.name,
            version=None, effective_date=None,
            expiry_date=None, supersedes=None,
            owner_id=owner_id,
            source_sha256=source_sha256, source_bytes=source_bytes,
        )
        # T3 follow-up (P1b review): a successful ingest + deprecate-set
        # operation changes which docs are surfaced by search; drop the
        # cache so the next query re-runs against fresh state.
        from kb.search.cache import invalidate_all as _invalidate_search_cache
        _invalidate_search_cache()
        # Review P1 #6: invalidate_all() clears THIS process's cache only.
        # Bump the cross-process epoch counter so tool_server's cache also
        # misses on the next request — otherwise MCP/REST search returns
        # stale 5-min cached entries that don't reflect the new doc state.
        # Fail-soft: a transient SQLite hiccup must not break the ingest
        # commit (the search cache will TTL-expire within DEFAULT_TTL_S
        # even if the bump didn't land).
        try:
            from kb.search.cache_epoch import CacheEpochStore
            users_db_url = (
                f"sqlite+aiosqlite:///{self._settings.qdrant_path}/kb_metadata.db"
            )
            epoch_store = CacheEpochStore(db_url=users_db_url)
            await epoch_store.init()
            try:
                await epoch_store.bump()
            finally:
                await epoch_store.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cache_epoch bump failed (will rely on TTL expiry): %s", exc,
            )
        for old_id in old_doc_ids:
            await self._meta_db.deprecate_document(old_id)
            try:
                await self._qdrant.delete_by_doc_id(old_id)
            except Exception as exc:
                logger.warning("Failed to delete old vectors for %s: %s", old_id, exc)
            # Also drop the old doc's nav entries so kb_search_nav can never
            # hand the agent a pointer into a deprecated version. Wrapped
            # in try/except for the same reason as the qdrant delete: a
            # stale nav fragment must not block the upgrade path.
            if self._nav_store is not None:
                try:
                    await self._nav_store.delete_entries_for_doc(old_id)
                except Exception as exc:
                    logger.warning(
                        "Failed to delete old nav entries for %s: %s", old_id, exc
                    )
            logger.info("Deprecated old version: %s (%s)", path.name, old_id)

        # Auto-build nav index for this doc. Helper is safe with None
        # stores / disabled config / empty pages; failures are logged
        # but never raised — a broken nav index must not block
        # ingestion success.
        nav_entries_built = await auto_build_nav_for_doc(
            nav_store=self._nav_store,
            nav_builder=self._nav_builder,
            pages=pages,
            enabled=self._settings.nav_auto_build_on_ingest,
        )

        # Best-effort audit write (spec §4.5 / A-3). DB failure is caught
        # inside AuditLog.write and traced; ingest must not fail because
        # audit is down. Skipped paths (unchanged-hash, unprocessable
        # classification) do NOT write doc.ingest — only a real ingest
        # that wrote chunks counts.
        if self._audit_log is not None:
            await self._audit_log.write(
                "doc.ingest",
                user_id=owner_id,   # T3: was None placeholder, fill from arg
                target_kind="document",
                target_id=doc_id,
                payload={
                    "doc_name": path.name,
                    "page_count": len(pages),
                },
            )

        emit(
            "ingest.end",
            duration_ms=round((time.monotonic() - t_total) * 1000, 1),
            doc_id=doc_id,
            skipped=False,
            pages_processed=len(pages),
            chunks_stored=chunks_stored,
            vision_pages_stored=vision_pages_stored,
            desc_pages_stored=desc_pages_stored,
            nav_entries_built=nav_entries_built,
            success=True,
        )
        return {
            "doc_id": doc_id,
            "skipped": False,
            "pages_processed": len(pages),
            "chunks_stored": chunks_stored,
            "nav_entries_built": nav_entries_built,
        }

    @staticmethod
    def _build_description_channel(settings: Settings):
        """Build DescriptionChannel from settings, injecting the vision adapter.

        Description output is a fixed JSON schema; thinking would make the LLM
        emit reasoning text instead of JSON, breaking json.loads. So the
        OpenAI-compat path forces thinking off.
        """
        from kb.adapters.factory import make_chat_adapter
        adapter = make_chat_adapter(
            adapter_type=settings.description_adapter,
            url=settings.description_url,
            api_key=settings.description_api_key,
            model=settings.description_model,
            supports_vision=settings.description_supports_vision,
            disable_thinking=True,    # description outputs fixed JSON schema
            task_type="description",  # Phase 6.4 — kb.llm.call attribution
        )
        return DescriptionChannel(adapter=adapter)
