from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from kb.nav.models import NavEdge, NavEntry


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    """Slice ``items`` into consecutive chunks of at most ``chunk_size``.

    Used by delete_entries_for_doc to stay under SQLite's ~999-parameter
    cap when building ``IN (?, ?, ...)`` clauses for bulk edge / hidden
    cleanup. Returns ``[]`` for an empty input.
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


class NavIndexStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}", poolclass=NullPool
        )

        # Enable WAL for the underlying SQLite connection. Even though
        # NavIndexBuilder is the sole writer, the MCP nav search path reads
        # the same file concurrently — WAL keeps readers from blocking and
        # mirrors the convention used by other kb SQLite stores.
        @event.listens_for(self.engine.sync_engine, "connect")
        def _set_sqlite_wal(dbapi_conn, _record):  # pragma: no cover - trivial
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nav_entries (
                    entry_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    normalized_label TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    doc_name TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    order_index INTEGER NOT NULL,
                    parent_entry_id TEXT,
                    resource_uris_json TEXT NOT NULL,
                    source_ref_ids_json TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nav_edges (
                    from_entry_id TEXT NOT NULL,
                    to_entry_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (from_entry_id, to_entry_id, edge_type)
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nav_hidden_entries (
                    entry_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    hidden_at TEXT NOT NULL
                )
            """))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_nav_doc_pages "
                "ON nav_entries(doc_id, page_start, page_end)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_nav_label "
                "ON nav_entries(normalized_label)"
            ))

    async def _upsert_entries_on(self, conn, entries: list[NavEntry]) -> None:
        """Run entry upserts on an existing connection. Used both by the
        public ``upsert_entries`` (which opens its own transaction) and by
        ``replace_for_doc`` (which runs delete + upsert in ONE transaction
        so an upsert failure rolls the delete back too)."""
        for e in entries:
            await conn.execute(text("""
                INSERT OR REPLACE INTO nav_entries (
                    entry_id, label, normalized_label, entry_type, source,
                    doc_id, doc_name, page_start, page_end, order_index,
                    parent_entry_id, resource_uris_json, source_ref_ids_json,
                    aliases_json, created_at, updated_at
                )
                VALUES (
                    :entry_id, :label, :normalized_label, :entry_type, :source,
                    :doc_id, :doc_name, :page_start, :page_end, :order_index,
                    :parent_entry_id, :resource_uris_json, :source_ref_ids_json,
                    :aliases_json, :created_at, :updated_at
                )
            """), {
                "entry_id": e.entry_id,
                "label": e.label,
                "normalized_label": e.normalized_label,
                "entry_type": e.entry_type,
                "source": e.source,
                "doc_id": e.doc_id,
                "doc_name": e.doc_name,
                "page_start": e.page_start,
                "page_end": e.page_end,
                "order_index": e.order_index,
                "parent_entry_id": e.parent_entry_id,
                "resource_uris_json": json.dumps(e.resource_uris, ensure_ascii=False),
                "source_ref_ids_json": json.dumps(e.source_ref_ids, ensure_ascii=False),
                "aliases_json": json.dumps(e.aliases, ensure_ascii=False),
                "created_at": e.created_at.isoformat() if e.created_at else "",
                "updated_at": e.updated_at.isoformat() if e.updated_at else "",
            })

    async def upsert_entries(self, entries: list[NavEntry]) -> None:
        async with self.engine.begin() as conn:
            await self._upsert_entries_on(conn, entries)

    async def _upsert_edges_on(self, conn, edges: list[NavEdge]) -> None:
        """Run edge upserts on an existing connection. See _upsert_entries_on."""
        for edge in edges:
            await conn.execute(text("""
                INSERT OR REPLACE INTO nav_edges (
                    from_entry_id, to_entry_id, edge_type, weight, source
                )
                VALUES (:f, :t, :type, :w, :s)
            """), {
                "f": edge.from_entry_id,
                "t": edge.to_entry_id,
                "type": edge.edge_type,
                "w": edge.weight,
                "s": edge.source,
            })

    async def upsert_edges(self, edges: list[NavEdge]) -> None:
        """Insert or replace nav edges.

        Uses (from_entry_id, to_entry_id, edge_type) as composite key —
        re-running auto-build for the same doc overwrites the prior edges
        with the same weights rather than inserting duplicates.
        """
        if not edges:
            return
        async with self.engine.begin() as conn:
            await self._upsert_edges_on(conn, edges)

    async def list_edges(self, from_entry_id: str | None = None) -> list[NavEdge]:
        query = "SELECT * FROM nav_edges"
        params: dict[str, str] = {}
        if from_entry_id is not None:
            query += " WHERE from_entry_id=:f"
            params["f"] = from_entry_id
        async with self.engine.connect() as conn:
            rows = (await conn.execute(text(query), params)).mappings().all()
        return [
            NavEdge(
                from_entry_id=row["from_entry_id"],
                to_entry_id=row["to_entry_id"],
                edge_type=row["edge_type"],
                weight=row["weight"],
                source=row["source"],
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_nav_entry(row) -> NavEntry:
        """Build a NavEntry from a SQLAlchemy ``RowMapping`` (or dict).

        Extracted from ``list_entries`` so ``get_entry`` (added in T3-6 for
        the MCP write-tool permission resolution chain) reuses the same
        conversion without duplicating the JSON-decode boilerplate.
        """
        return NavEntry(
            entry_id=row["entry_id"],
            label=row["label"],
            normalized_label=row["normalized_label"],
            entry_type=row["entry_type"],
            source=row["source"],
            doc_id=row["doc_id"],
            doc_name=row["doc_name"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            order_index=row["order_index"],
            parent_entry_id=row["parent_entry_id"],
            resource_uris=json.loads(row["resource_uris_json"]),
            source_ref_ids=json.loads(row["source_ref_ids_json"]),
            aliases=json.loads(row["aliases_json"]),
        )

    async def list_entries(self, doc_id: str | None = None) -> list[NavEntry]:
        query = "SELECT * FROM nav_entries"
        params: dict[str, str] = {}
        if doc_id is not None:
            query += " WHERE doc_id=:doc_id"
            params["doc_id"] = doc_id
        query += " ORDER BY doc_id, order_index"
        async with self.engine.connect() as conn:
            rows = (await conn.execute(text(query), params)).mappings().all()
        return [self._row_to_nav_entry(row) for row in rows]

    async def count_entries(self, doc_ids: Iterable[str] | None = None) -> int:
        ids = list(dict.fromkeys(doc_ids or []))
        if doc_ids is not None and not ids:
            return 0
        if doc_ids is None:
            async with self.engine.connect() as conn:
                return int((await conn.execute(
                    text("SELECT COUNT(*) FROM nav_entries")
                )).scalar_one())

        total = 0
        async with self.engine.connect() as conn:
            for chunk in _chunked(ids, 500):
                placeholders = ",".join(f":id{i}" for i in range(len(chunk)))
                params = {f"id{i}": doc_id for i, doc_id in enumerate(chunk)}
                total += int((await conn.execute(
                    text(f"SELECT COUNT(*) FROM nav_entries WHERE doc_id IN ({placeholders})"),
                    params,
                )).scalar_one())
        return total

    async def count_entries_by_doc(
        self, doc_ids: Iterable[str] | None = None
    ) -> dict[str, int]:
        ids = list(dict.fromkeys(doc_ids or []))
        if doc_ids is not None and not ids:
            return {}

        counts: dict[str, int] = {}
        async with self.engine.connect() as conn:
            if doc_ids is None:
                rows = (await conn.execute(text("""
                    SELECT doc_id, COUNT(*) AS entry_count
                    FROM nav_entries
                    GROUP BY doc_id
                """))).mappings().all()
                return {row["doc_id"]: int(row["entry_count"]) for row in rows}

            for chunk in _chunked(ids, 500):
                placeholders = ",".join(f":id{i}" for i in range(len(chunk)))
                params = {f"id{i}": doc_id for i, doc_id in enumerate(chunk)}
                rows = (await conn.execute(text(f"""
                    SELECT doc_id, COUNT(*) AS entry_count
                    FROM nav_entries
                    WHERE doc_id IN ({placeholders})
                    GROUP BY doc_id
                """), params)).mappings().all()
                for row in rows:
                    counts[row["doc_id"]] = int(row["entry_count"])
        return counts

    async def get_entry(self, entry_id: str) -> NavEntry | None:
        """Single-entry fetch by primary key.

        T3-6: used by MCP write tools that receive ``entry_id``
        (``kb_add_nav_alias``, ``kb_hide_nav_entry``) and need to resolve
        ``doc_id`` so the permission check can target the owning document
        before any mutation runs. Returns ``None`` for unknown entry_id.
        """
        async with self.engine.connect() as conn:
            row = (await conn.execute(
                text("SELECT * FROM nav_entries WHERE entry_id=:eid"),
                {"eid": entry_id},
            )).mappings().first()
        if row is None:
            return None
        return self._row_to_nav_entry(row)

    async def add_alias(self, entry_id: str, alias: str) -> None:
        """Append an alias to an existing entry's aliases_json. Idempotent.

        Raises ValueError if the entry does not exist. Defensive even for
        manual aliases — the invariant of design §6.2 (aliases must
        actually appear in source text) is enforced by the caller, but
        we at least ensure the entry exists before writing.
        """
        async with self.engine.begin() as conn:
            row = (await conn.execute(
                text("SELECT aliases_json FROM nav_entries WHERE entry_id=:eid"),
                {"eid": entry_id},
            )).mappings().first()
            if row is None:
                raise ValueError(f"Nav entry not found: {entry_id}")
            aliases = json.loads(row["aliases_json"])
            if alias not in aliases:
                aliases.append(alias)
            await conn.execute(
                text("UPDATE nav_entries SET aliases_json=:aj WHERE entry_id=:eid"),
                {"aj": json.dumps(aliases, ensure_ascii=False), "eid": entry_id},
            )

    async def hide_entry(self, entry_id: str, reason: str) -> None:
        """Mark a NavEntry as hidden so nav search excludes it.

        Soft-hide: the row in nav_entries is untouched. ``list_entries``
        intentionally returns hidden entries too — filtering happens at the
        search layer (``NavSearchEngine.search``) so debugging tools and
        doctor inspections can still see hidden rows.
        """
        from datetime import datetime, timezone
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                INSERT OR REPLACE INTO nav_hidden_entries (entry_id, reason, hidden_at)
                VALUES (:eid, :reason, :at)
            """), {
                "eid": entry_id,
                "reason": reason,
                "at": datetime.now(timezone.utc).isoformat(),
            })

    async def list_hidden_entry_ids(self) -> set[str]:
        async with self.engine.connect() as conn:
            rows = (await conn.execute(
                text("SELECT entry_id FROM nav_hidden_entries")
            )).mappings().all()
        return {row["entry_id"] for row in rows}

    async def _delete_entries_for_doc_on(self, conn, doc_id: str) -> int:
        """Per-doc delete on an existing connection — same body as
        ``delete_entries_for_doc`` minus the outer ``engine.begin`` call,
        so ``replace_for_doc`` can run delete + upsert in one transaction."""
        rows = (await conn.execute(
            text("SELECT entry_id FROM nav_entries WHERE doc_id=:doc_id"),
            {"doc_id": doc_id},
        )).mappings().all()
        entry_ids = [row["entry_id"] for row in rows]
        if not entry_ids:
            return 0

        for chunk in _chunked(entry_ids, 500):
            placeholders = ",".join(f":id{i}" for i in range(len(chunk)))
            params = {f"id{i}": eid for i, eid in enumerate(chunk)}
            await conn.execute(
                text(
                    f"DELETE FROM nav_edges WHERE from_entry_id IN ({placeholders}) "
                    f"OR to_entry_id IN ({placeholders})"
                ),
                params,
            )

        for chunk in _chunked(entry_ids, 500):
            placeholders = ",".join(f":id{i}" for i in range(len(chunk)))
            params = {f"id{i}": eid for i, eid in enumerate(chunk)}
            await conn.execute(
                text(f"DELETE FROM nav_hidden_entries WHERE entry_id IN ({placeholders})"),
                params,
            )

        await conn.execute(
            text("DELETE FROM nav_entries WHERE doc_id=:doc_id"),
            {"doc_id": doc_id},
        )
        return len(entry_ids)

    async def delete_entries_for_doc(self, doc_id: str) -> int:
        """Remove all nav entries (+ their edges + hidden flags) for a document.

        Returns the count of entries deleted. Safe to call for unknown
        doc_id (returns 0). Cleans up:
          - nav_entries rows where doc_id matches
          - nav_edges rows where either endpoint is one of the deleted entries
          - nav_hidden_entries rows pointing to deleted entries

        SQLite has a default ~999-parameter limit on a single statement,
        so the ``IN (...)`` clauses are chunked at 500 ids per round.
        """
        async with self.engine.begin() as conn:
            return await self._delete_entries_for_doc_on(conn, doc_id)

    async def replace_for_doc(
        self,
        doc_id: str,
        entries: list[NavEntry],
        edges: list[NavEdge],
    ) -> int:
        """Atomic replace of nav data for one doc: delete + upsert in a
        SINGLE transaction. If the upsert half raises, the delete rolls
        back too — so a partial failure can't leave an empty nav for the
        doc (which would silently break navigation).

        Returns count of new entries inserted (== ``len(entries)``).
        Edges are not pre-deleted because they ride on the deleted entries:
        the entry delete clears every edge touching one of the doc's
        entries via the ``IN (...)`` cleanup pass.
        """
        async with self.engine.begin() as conn:
            await self._delete_entries_for_doc_on(conn, doc_id)
            if entries:
                await self._upsert_entries_on(conn, entries)
            if edges:
                await self._upsert_edges_on(conn, edges)
        return len(entries)

    async def list_doc_ids(self) -> set[str]:
        """Return the distinct set of doc_ids currently in the nav index.

        Used by doctor / orphan detection: any doc_id here that is NOT
        in MetadataDB's active set is orphaned (the doc was deleted but
        nav cleanup did not run).
        """
        async with self.engine.connect() as conn:
            rows = (await conn.execute(
                text("SELECT DISTINCT doc_id FROM nav_entries")
            )).mappings().all()
        return {row["doc_id"] for row in rows}

    async def close(self) -> None:
        await self.engine.dispose()
