"""MCP nav tools: kb_search_nav, kb_hybrid_search.

Both tools return pointer-only payloads — agents must fetch original
document/page resources via kb:// URIs before answering. The
``contains_generated_content`` flag on every nav payload is False as a
machine-readable invariant of the navigation-only design.

Wire-format note: FastMCP only emits a proper ``structuredContent`` field
to MCP clients when a tool's return annotation is a Pydantic-compatible
schema (BaseModel, TypedDict, dataclass, …). A bare ``dict`` annotation
short-circuits FastMCP's structured-output detection (see
``mcp/server/fastmcp/utilities/func_metadata.py::_try_create_model_and_schema``
case 4 — ``get_type_hints(dict)`` is empty → ``output_model`` is ``None``)
and ``convert_result`` then JSON-stringifies the whole payload into a
single ``TextContent`` block, dropping ``structuredContent`` entirely on
the wire. To preserve structured output we return
``mcp.types.CallToolResult`` from every tool; FastMCP detects this and
passes the result through verbatim (``convert_result`` short-circuit at
line 114 of the same file). Internal helpers (``nav_hits_payload``,
``_do`` closures, ``evidence_response_to_mcp_payload``) still build
``{"content": ..., "structuredContent": ...}`` dicts so the existing unit
tests, formatters, and audit hooks stay unchanged; ``_to_call_tool_result``
adapts at the outermost MCP boundary.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.types import CallToolResult, TextContent

from kb.auth.context import get_current_user
from kb.auth.permissions import require_owner_or_admin
from kb.tool_server.mcp_security import audit_tool_call


def _to_call_tool_result(payload: dict) -> CallToolResult:
    """Adapt internal ``{"content": str, "structuredContent": dict}`` payloads
    to the MCP wire format that FastMCP propagates without re-serializing.

    See module docstring for why this adapter exists.
    """
    text = payload.get("content", "")
    if not isinstance(text, str):
        text = str(text)
    structured = payload.get("structuredContent")
    if structured is not None and not isinstance(structured, dict):
        # Defensive: structuredContent must be a JSON object per the MCP spec.
        structured = {"value": structured}
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=structured,
    )


def nav_hits_payload(hits) -> dict:
    return {
        "content": "\n".join(
            f"- {hit.label}: {hit.doc_name} pages {hit.page_start}-{hit.page_end} -> {hit.resource_uris[0]}"
            for hit in hits
        ) or "No navigation entries found.",
        "structuredContent": {
            "nav_hits": [asdict(hit) for hit in hits],
            "contains_generated_content": False,
        },
    }


def register_nav_tools(mcp: Any, state: Any) -> None:
    @mcp.tool()
    async def kb_search_nav(query: str, top_k: int = 5) -> CallToolResult:
        """Search navigation index. Returns pointers only, not answer content."""
        async def _do():
            if state.nav_search is None:
                return {
                    "content": "Navigation index is not initialized.",
                    "structuredContent": {"nav_hits": []},
                }
            top_k_clamped = max(
                1,
                min(getattr(state.settings, "mcp_tool_result_limit", 20), top_k),
            )
            hits = await state.nav_search.search(query=query, top_k=top_k_clamped)
            return nav_hits_payload(hits)

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_search_nav",
                input_preview={"query": query[:120], "top_k": top_k},
                is_write_tool=False,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_hybrid_search(query: str, top_k: int = 5) -> CallToolResult:
        """Search evidence and navigation pointers. Fetch originals before
        answering. Each page resource carries prev/next pointers and a
        looks_continued flag — if a fetched page reads as cut off, follow
        them or the section range so you don't answer from half a page."""
        async def _do():
            # Runtime gate: operator can disable hybrid via settings without
            # un-registering the tool (keeps the tool list stable for agent
            # discovery; calls just refuse cleanly).
            if state.settings is not None and not getattr(
                state.settings, "hybrid_nav_enabled", True
            ):
                return {
                    "content": "Hybrid navigation is disabled by configuration.",
                    "structuredContent": {"reason": "hybrid_nav_disabled"},
                }
            if state.hybrid_navigator is None:
                return {
                    "content": "Hybrid navigator is not initialized.",
                    "structuredContent": {},
                }
            top_k_clamped = max(
                1,
                min(getattr(state.settings, "mcp_tool_result_limit", 20), top_k),
            )
            response = await state.hybrid_navigator.search(
                query=query, top_k=top_k_clamped
            )
            return {
                "content": (
                    f"Hybrid search returned {len(response.evidence_hits)} evidence hits, "
                    f"{len(response.nav_hits)} nav pointers, and "
                    f"{len(response.suggested_fetches)} suggested fetches. "
                    f"Fetch originals via kb://documents/.../pages/N before answering."
                ),
                "structuredContent": asdict(response),
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_hybrid_search",
                input_preview={"query": query[:120], "top_k": top_k},
                is_write_tool=False,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_expand_nav_entry(entry_id: str) -> CallToolResult:
        """Return parent/children/siblings nav entries for a given entry. Pointer-only."""
        async def _do():
            if state.nav_store is None:
                return {
                    "content": "Navigation index is not initialized.",
                    "structuredContent": {"neighbors": [], "entry_id": entry_id},
                }
            all_entries = await state.nav_store.list_entries()
            target = next((e for e in all_entries if e.entry_id == entry_id), None)
            if target is None:
                return {
                    "content": f"Nav entry not found: {entry_id}",
                    "structuredContent": {"neighbors": [], "entry_id": entry_id},
                }
            same_doc = [e for e in all_entries if e.doc_id == target.doc_id]
            children = [e for e in same_doc if e.parent_entry_id == target.entry_id]
            siblings = [
                e for e in same_doc
                if e.parent_entry_id == target.parent_entry_id
                and e.entry_id != target.entry_id
            ]
            parent = None
            if target.parent_entry_id:
                parent = next(
                    (e for e in all_entries if e.entry_id == target.parent_entry_id),
                    None,
                )
            neighbors: list[dict] = []
            if parent is not None:
                neighbors.append({
                    "relation": "parent",
                    "entry_id": parent.entry_id,
                    "label": parent.label,
                    "resource_uris": parent.resource_uris,
                })
            for child in children:
                neighbors.append({
                    "relation": "child",
                    "entry_id": child.entry_id,
                    "label": child.label,
                    "resource_uris": child.resource_uris,
                })
            for sib in siblings:
                neighbors.append({
                    "relation": "sibling",
                    "entry_id": sib.entry_id,
                    "label": sib.label,
                    "resource_uris": sib.resource_uris,
                })
            return {
                "content": (
                    f"Found {len(neighbors)} neighbors for entry {target.label!r} "
                    f"(doc={target.doc_name}, pages {target.page_start}-{target.page_end})."
                ),
                "structuredContent": {
                    "entry_id": entry_id,
                    "neighbors": neighbors,
                    "contains_generated_content": False,
                },
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_expand_nav_entry",
                input_preview={"entry_id": entry_id},
                is_write_tool=False,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_list_documents() -> CallToolResult:
        """List all ingested documents with nav-coverage info. Pointer-only."""
        async def _do():
            if state.meta_db is None:
                return {
                    "content": "Metadata DB not initialized.",
                    "structuredContent": {"documents": [], "total": 0},
                }
            active_ids = await state.meta_db.list_active_doc_ids()
            nav_covered_ids: set[str] = set()
            if state.nav_store is not None:
                try:
                    nav_entries = await state.nav_store.list_entries()
                    nav_covered_ids = {e.doc_id for e in nav_entries}
                except Exception:
                    nav_covered_ids = set()
            docs: list[dict] = []
            for doc_id in active_ids:
                doc = await state.meta_db.get_document(doc_id)
                if doc:
                    docs.append({
                        "doc_id": doc_id,
                        "doc_name": doc.get("doc_name", "?"),
                        "version": doc.get("version"),
                        "nav_indexed": doc_id in nav_covered_ids,
                        "resource_uri": f"kb://documents/{doc_id}",
                    })
            return {
                "content": (
                    f"Total {len(docs)} documents "
                    f"({sum(1 for d in docs if d['nav_indexed'])} with nav index)."
                ),
                "structuredContent": {"documents": docs, "total": len(docs)},
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_list_documents",
                input_preview={},
                is_write_tool=False,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_get_document_text(doc_id: str) -> CallToolResult:
        """Return a document's full parsed text in page order — the readable
        whole document (no binary). Carries provenance so you can tell the user
        which document this is and how to reach the original in the KB."""
        async def _do():
            from kb.tool_server.mcp_resources import _assemble_fulltext
            payload = await _assemble_fulltext(state, doc_id)
            if payload.get("status") != "ok":
                return {
                    "content": f"No full text for {doc_id} ({payload.get('status')}).",
                    "structuredContent": payload,
                }
            note = " (truncated)" if payload.get("truncated") else ""
            return {
                "content": (
                    f"{payload['doc_name']}: {payload['page_count']} pages, "
                    f"{payload['chars']} chars{note}."
                ),
                "structuredContent": payload,
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_get_document_text",
                input_preview={"doc_id": doc_id},
                is_write_tool=False,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_get_document_source(doc_id: str) -> CallToolResult:
        """Locate / fetch a document's ORIGINAL PDF. Returns file_path, size,
        sha256, an inline base64 data_url (only when small enough), and an
        http_url for large files — plus provenance. Prefer kb_get_document_text
        for reading; use this for the raw artifact (download / re-processing)."""
        async def _do():
            from kb.tool_server.mcp_resources import _source_envelope
            payload = await _source_envelope(state, doc_id)
            status = payload.get("status")
            if status != "ok":
                return {
                    "content": f"No source PDF for {doc_id} ({status}).",
                    "structuredContent": payload,
                }
            how = "inline base64" if payload.get("data_url") else "by file_path / http_url (too large to inline)"
            return {
                "content": f"{payload['doc_name']}: {payload['size_bytes']} bytes, available {how}.",
                "structuredContent": payload,
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_get_document_source",
                input_preview={"doc_id": doc_id},
                is_write_tool=False,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_get_toc(doc_id: str) -> CallToolResult:
        """Return the ordered TOC for a document. Pointer-only."""
        async def _do():
            if state.nav_store is None:
                return {
                    "content": "Navigation index is not initialized.",
                    "structuredContent": {"doc_id": doc_id, "entries": []},
                }
            # T5-4: DELETE no longer touches nav entries — GC purges them
            # at retention. kb_get_toc reads the live nav index unchanged;
            # re-check doc status here so orphan nav rows don't surface a TOC.
            doc = await state.meta_db.get_document(doc_id)
            if doc is None or doc.get("status") != "active":
                return {
                    "content": f"Document {doc_id} is not active.",
                    "structuredContent": {
                        "doc_id": doc_id,
                        "entries": [],
                        "reason": "doc_not_active",
                    },
                }
            entries = await state.nav_store.list_entries(doc_id=doc_id)
            if not entries:
                return {
                    "content": (
                        f"No nav entries for doc {doc_id}. Either doc doesn't exist "
                        f"or nav index not built."
                    ),
                    "structuredContent": {"doc_id": doc_id, "entries": []},
                }
            toc_entries = [
                {
                    "entry_id": e.entry_id,
                    "label": e.label,
                    "entry_type": e.entry_type,
                    "page_start": e.page_start,
                    "page_end": e.page_end,
                    "resource_uris": e.resource_uris,
                    "parent_entry_id": e.parent_entry_id,
                    "order_index": e.order_index,
                }
                for e in entries
            ]
            return {
                "content": f"Found {len(toc_entries)} TOC entries for doc {doc_id}.",
                "structuredContent": {
                    "doc_id": doc_id,
                    "entries": toc_entries,
                    "contains_generated_content": False,
                },
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_get_toc",
                input_preview={"doc_id": doc_id},
                is_write_tool=False,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_get_status() -> CallToolResult:
        """Server capability/health snapshot. Useful for capability discovery."""
        async def _do():
            evidence_ready = state.engine is not None
            nav_ready = state.nav_store is not None and state.nav_search is not None
            hybrid_ready = state.hybrid_navigator is not None
            meta_db_ready = state.meta_db is not None
            doc_count = 0
            nav_doc_count = 0
            nav_entry_count = 0
            if meta_db_ready:
                try:
                    doc_count = len(await state.meta_db.list_active_doc_ids())
                except Exception:
                    doc_count = -1
            if nav_ready:
                try:
                    entries = await state.nav_store.list_entries()
                    nav_entry_count = len(entries)
                    nav_doc_count = len({e.doc_id for e in entries})
                except Exception:
                    pass
            write_tools = bool(
                getattr(state.settings, "mcp_enable_write_tools", False)
            ) if state.settings is not None else False
            return {
                "content": (
                    f"Evidence layer: {'ready' if evidence_ready else 'not initialized'}; "
                    f"Nav: {'ready' if nav_ready else 'not initialized'} "
                    f"({nav_entry_count} entries across {nav_doc_count} docs); "
                    f"Hybrid: {'ready' if hybrid_ready else 'not initialized'}; "
                    f"Documents: {doc_count}; "
                    f"Write tools: {'enabled' if write_tools else 'disabled'}."
                ),
                "structuredContent": {
                    "evidence_ready": evidence_ready,
                    "nav_ready": nav_ready,
                    "hybrid_ready": hybrid_ready,
                    "meta_db_ready": meta_db_ready,
                    "document_count": doc_count,
                    "nav_entry_count": nav_entry_count,
                    "nav_doc_count": nav_doc_count,
                    "mcp_write_tools_enabled": write_tools,
                },
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_get_status",
                input_preview={},
                is_write_tool=False,
                call=_do,
            )
        )


def register_nav_write_tools(mcp: Any, state: Any) -> None:
    """Register the 3 gated write tools. All check MCP_ENABLE_WRITE_TOOLS first."""

    @mcp.tool()
    async def kb_add_nav_alias(entry_id: str, alias: str) -> CallToolResult:
        """Add a manual alias to an existing nav entry. Gated by mcp_enable_write_tools."""
        async def _do():
            # T5: maintenance gate — MCP tools don't go through FastAPI
            # dependencies, so the check is inline. Placed BEFORE
            # _write_tools_enabled so backup is honored even when MCP
            # writes are otherwise on. getattr guards against test
            # fixtures using SimpleNamespace without a maintenance field.
            _maintenance = getattr(state, "maintenance", None)
            if _maintenance is not None and await _maintenance.is_on():
                return {
                    "content": "server in maintenance mode (backup in progress)",
                    "structuredContent": {"saved": False, "reason": "maintenance_mode"},
                }
            if not _write_tools_enabled(state):
                return {
                    "content": "Write tools disabled. Set MCP_ENABLE_WRITE_TOOLS=true to enable.",
                    "structuredContent": {"saved": False, "reason": "write_tools_disabled"},
                }
            if state.nav_store is None:
                return {
                    "content": "Navigation store not initialized.",
                    "structuredContent": {"saved": False, "reason": "nav_store_not_initialized"},
                }
            # T3-6: ownership check. Resolve entry → doc → owner_id before
            # any mutation. The permission unit is the DOCUMENT (target_id=
            # entry.doc_id, not entry_id) — aliases live on a doc, so the
            # doc owner authorizes the change. ``require_owner_or_admin``
            # raises HTTPException(403) on denial; FastMCP catches it and
            # surfaces an isError tool-call response (audit row already
            # written by _emit_authz_denied before the raise).
            user = get_current_user()
            entry = await state.nav_store.get_entry(entry_id)
            if entry is None:
                return {
                    "content": f"Nav entry {entry_id} not found.",
                    "structuredContent": {"saved": False, "reason": "entry_not_found"},
                }
            doc = await state.meta_db.get_document(entry.doc_id)
            if doc is None:
                return {
                    "content": f"Doc {entry.doc_id} (parent of entry) not found.",
                    "structuredContent": {
                        "saved": False, "reason": "parent_doc_not_found",
                    },
                }
            # T3 follow-up (P2b review): writes must respect the same
            # active-gate as reads. A deleted/deprecated doc must not
            # accept new aliases — its nav entries are tombstones,
            # mutating them re-surfaces orphan content via stale URIs.
            if doc.get("status") != "active":
                return {
                    "content": (
                        f"Document {entry.doc_id} is not active "
                        f"(status={doc.get('status')})."
                    ),
                    "structuredContent": {
                        "saved": False,
                        "reason": "doc_not_active",
                        "doc_status": doc.get("status"),
                    },
                }
            await require_owner_or_admin(
                doc, user,
                audit_log=state.audit_log,
                target_kind="document",
                target_id=entry.doc_id,
            )
            try:
                await state.nav_store.add_alias(entry_id, alias)
            except ValueError as exc:
                return {
                    "content": str(exc),
                    "structuredContent": {"saved": False, "reason": "entry_not_found"},
                }
            return {
                "content": f"Alias '{alias}' added to entry {entry_id}.",
                "structuredContent": {
                    "saved": True,
                    "entry_id": entry_id,
                    "alias": alias,
                },
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_add_nav_alias",
                input_preview={"entry_id": entry_id, "alias": alias[:60]},
                is_write_tool=True,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_hide_nav_entry(entry_id: str, reason: str) -> CallToolResult:
        """Soft-hide a nav entry from search results. Gated by mcp_enable_write_tools."""
        async def _do():
            # T5: maintenance gate — MCP tools don't go through FastAPI
            # dependencies, so the check is inline. Placed BEFORE
            # _write_tools_enabled so backup is honored even when MCP
            # writes are otherwise on. getattr guards against test
            # fixtures using SimpleNamespace without a maintenance field.
            _maintenance = getattr(state, "maintenance", None)
            if _maintenance is not None and await _maintenance.is_on():
                return {
                    "content": "server in maintenance mode (backup in progress)",
                    "structuredContent": {"hidden": False, "reason": "maintenance_mode"},
                }
            if not _write_tools_enabled(state):
                return {
                    "content": "Write tools disabled. Set MCP_ENABLE_WRITE_TOOLS=true to enable.",
                    "structuredContent": {"hidden": False, "reason": "write_tools_disabled"},
                }
            if state.nav_store is None:
                return {
                    "content": "Navigation store not initialized.",
                    "structuredContent": {"hidden": False, "reason": "nav_store_not_initialized"},
                }
            # T3-6: ownership check. Same entry → doc resolution as
            # kb_add_nav_alias — the doc owner authorizes hide on any of
            # its entries. target_id is the doc_id (not entry_id) so audit
            # rows aggregate per-document.
            user = get_current_user()
            entry = await state.nav_store.get_entry(entry_id)
            if entry is None:
                return {
                    "content": f"Nav entry {entry_id} not found.",
                    "structuredContent": {"hidden": False, "reason": "entry_not_found"},
                }
            doc = await state.meta_db.get_document(entry.doc_id)
            if doc is None:
                return {
                    "content": f"Doc {entry.doc_id} (parent of entry) not found.",
                    "structuredContent": {
                        "hidden": False, "reason": "parent_doc_not_found",
                    },
                }
            # T3 follow-up (P2b review): hide is a write — refuse on
            # non-active docs to keep the I-6 invariant symmetric across
            # read and write paths.
            if doc.get("status") != "active":
                return {
                    "content": (
                        f"Document {entry.doc_id} is not active "
                        f"(status={doc.get('status')})."
                    ),
                    "structuredContent": {
                        "hidden": False,
                        "reason": "doc_not_active",
                        "doc_status": doc.get("status"),
                    },
                }
            await require_owner_or_admin(
                doc, user,
                audit_log=state.audit_log,
                target_kind="document",
                target_id=entry.doc_id,
            )
            await state.nav_store.hide_entry(entry_id, reason)
            return {
                "content": f"Entry {entry_id} hidden (reason: {reason[:80]}).",
                "structuredContent": {"hidden": True, "entry_id": entry_id},
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_hide_nav_entry",
                input_preview={"entry_id": entry_id, "reason": reason[:120]},
                is_write_tool=True,
                call=_do,
            )
        )

    @mcp.tool()
    async def kb_rebuild_nav_index(doc_id: str) -> CallToolResult:
        """Rebuild the nav index for one document from its existing qdrant
        chunks. No re-parse — the parser cost is avoided by reading back
        the chunks we already stored. Useful after a parser/heading-path
        upgrade, or to recover after a nav-store hiccup."""
        async def _do():
            # T5: maintenance gate — MCP tools don't go through FastAPI
            # dependencies, so the check is inline. Placed BEFORE
            # _write_tools_enabled so backup is honored even when MCP
            # writes are otherwise on. getattr guards against test
            # fixtures using SimpleNamespace without a maintenance field.
            _maintenance = getattr(state, "maintenance", None)
            if _maintenance is not None and await _maintenance.is_on():
                return {
                    "content": "server in maintenance mode (backup in progress)",
                    "structuredContent": {"started": False, "reason": "maintenance_mode"},
                }
            if not _write_tools_enabled(state):
                return {
                    "content": "Write tools disabled.",
                    "structuredContent": {"started": False, "reason": "write_tools_disabled"},
                }
            if (
                state.qdrant is None
                or state.meta_db is None
                or state.nav_store is None
                or state.nav_builder is None
            ):
                return {
                    "content": "Required services not bound (qdrant/meta_db/nav).",
                    "structuredContent": {
                        "started": False, "reason": "services_not_initialized",
                        "doc_id": doc_id,
                    },
                }
            doc = await state.meta_db.get_document(doc_id)
            if doc is None:
                return {
                    "content": f"Doc not found: {doc_id}",
                    "structuredContent": {
                        "started": False, "reason": "doc_not_found", "doc_id": doc_id,
                    },
                }
            # T3 follow-up (P2b review): rebuild is a write — refuse on
            # non-active docs. Rebuilding nav for a deprecated doc would
            # re-create pointers into stale content (qdrant chunks may
            # still exist from before cleanup).
            if doc.get("status") != "active":
                return {
                    "content": (
                        f"Document {doc_id} is not active (status={doc.get('status')})."
                    ),
                    "structuredContent": {
                        "started": False,
                        "reason": "doc_not_active",
                        "doc_status": doc.get("status"),
                        "doc_id": doc_id,
                    },
                }
            # T3-6: ownership check. Runs after the existing
            # services-bound and doc-not-found gates so an unbound /
            # missing doc never reaches the audit-bracketed permission
            # path. require_owner_or_admin raises HTTPException(403) on
            # denial; the authz.denied audit row is written first.
            user = get_current_user()
            await require_owner_or_admin(
                doc, user,
                audit_log=state.audit_log,
                target_kind="document",
                target_id=doc_id,
            )
            from kb.nav.auto_build import auto_build_nav_for_doc
            from kb.parser.qdrant_to_pages import (  # local import to keep module load light
                synthesize_pages_from_qdrant,
            )
            pages = await synthesize_pages_from_qdrant(
                state.qdrant, doc_id=doc_id, doc_name=doc.get("doc_name", doc_id),
            )
            if not pages:
                return {
                    "content": f"No qdrant chunks for {doc_id} — nothing to rebuild.",
                    "structuredContent": {
                        "started": False, "reason": "no_chunks_found", "doc_id": doc_id,
                    },
                }
            count = await auto_build_nav_for_doc(
                nav_store=state.nav_store,
                nav_builder=state.nav_builder,
                pages=pages,
                enabled=True,
            )
            return {
                "content": f"Rebuilt nav for {doc_id}: {count} entries from {len(pages)} pages.",
                "structuredContent": {
                    "started": True,
                    "completed": True,
                    "doc_id": doc_id,
                    "pages_processed": len(pages),
                    "entries_built": count,
                },
            }

        return _to_call_tool_result(
            await audit_tool_call(
                "kb_rebuild_nav_index",
                input_preview={"doc_id": doc_id},
                is_write_tool=True,
                call=_do,
            )
        )


def _write_tools_enabled(state: Any) -> bool:
    if state.settings is None:
        return False
    return bool(getattr(state.settings, "mcp_enable_write_tools", False))
