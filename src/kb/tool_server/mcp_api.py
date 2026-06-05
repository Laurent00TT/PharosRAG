"""MCP (Model Context Protocol) server endpoint.

Exposes the KB as a set of tools so any MCP-compatible agent (Claude Code,
Continue, Cline, …) can discover and call them.

This module is loaded by app.py and mounted at /mcp via streamable HTTP.
The FastMCP instance holds tool definitions; the live engine + metadata DB
are wired in via ``kb.tool_server.mcp_state.bind_state`` from the FastAPI
lifespan so tools can reach them through the shared ``state`` object.

Only read-only tools are exposed (search, list). kb_add / kb_delete are
intentionally NOT exposed — Agents may not introduce or remove KB content
without explicit human action.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from kb.config import Settings
from kb.tool_server.mcp_formatters import evidence_response_to_mcp_payload
from kb.tool_server.mcp_resources import register_document_resources
from kb.tool_server.mcp_security import audit_tool_call
from kb.tool_server.mcp_state import state
from kb.tool_server.mcp_tools import _to_call_tool_result

# Module-load-time settings for registration gates. Reading Settings() here
# is safe — it just re-parses env/.env, no I/O. The lifespan-bound copy in
# state.settings isn't available yet at import time, so we can't use it
# to decide WHAT to register. Per-request behavior can still rely on
# state.settings (which always reflects the live lifespan binding).
_register_time_settings = Settings()

mcp = FastMCP("kb")
# FastMCP's streamable HTTP transport registers its endpoint at /mcp by
# default; mounting the resulting ASGI app at /mcp in app.py would create
# /mcp/mcp. Move the internal path to "/" so the externally-visible URL
# is exactly /mcp/.
mcp.settings.streamable_http_path = "/"


__all__ = ["mcp", "kb_search", "kb_list"]


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def kb_search(query: str, top_k: int = 5) -> CallToolResult:
    """Search original evidence. Returns text fallback plus structured evidence pointers."""
    async def _do():
        if state.engine is None:
            return {"content": "KB engine not initialized.", "structuredContent": {"results": [], "total": 0}}
        top_k_clamped = max(1, min(20, top_k))
        response, _cache_hit = await state.engine.search(
            query=query,
            top_k=top_k_clamped,
            doc_filter={},
            hyde=False,
            multi_query=False,
            include_deprecated=False,
        )
        return evidence_response_to_mcp_payload(response, text_budget=state.settings.mcp_tool_text_budget)

    return _to_call_tool_result(
        await audit_tool_call(
            "kb_search",
            input_preview={"query": query[:120], "top_k": top_k},
            is_write_tool=False,
            call=_do,
        )
    )


@mcp.tool()
async def kb_list() -> CallToolResult:
    """List all active documents in the local knowledge base.

    Use this to discover what kinds of documents are available before searching,
    or to verify whether a specific document has been ingested. Returns the
    document name and a short doc_id for each one.

    Returns a ``CallToolResult`` so FastMCP propagates ``structuredContent``
    over the wire. The structured payload exposes the active doc list as
    structured data (``documents``, ``total``); the human-readable text
    block stays in ``content[0].text`` for backward-compat consumers.
    """
    async def _do():
        if state.meta_db is None:
            return {
                "content": "Error: metadata DB not initialized.",
                "structuredContent": {"documents": [], "total": 0},
            }

        active_ids = await state.meta_db.list_active_doc_ids()
        if not active_ids:
            return {
                "content": "The KB is empty.",
                "structuredContent": {"documents": [], "total": 0},
            }

        docs: list[dict] = []
        lines = [f"Total: {len(active_ids)} documents\n"]
        for doc_id in active_ids:
            doc = await state.meta_db.get_document(doc_id)
            if doc:
                name = doc.get("doc_name", "?")
                lines.append(f"  - [{doc_id[:12]}] {name}")
                docs.append({
                    "doc_id": doc_id,
                    "doc_name": name,
                    "resource_uri": f"kb://documents/{doc_id}",
                })
        return {
            "content": "\n".join(lines),
            "structuredContent": {"documents": docs, "total": len(docs)},
        }

    return _to_call_tool_result(
        await audit_tool_call(
            "kb_list",
            input_preview={},
            is_write_tool=False,
            call=_do,
        )
    )


# ── Resources ────────────────────────────────────────────────────────────────
# Gated by mcp_enable_resources so an operator can disable kb:// URIs
# without redeploying. If disabled, agents only see tools.

from kb.tool_server.mcp_resources import register_nav_resources  # noqa: E402

if _register_time_settings.mcp_enable_resources:
    register_document_resources(mcp, state)
    register_nav_resources(mcp, state)


# ── Nav tools ────────────────────────────────────────────────────────────────

from kb.tool_server.mcp_tools import (  # noqa: E402
    register_nav_tools,
    register_nav_write_tools,
)

# hybrid_nav_enabled gates kb_hybrid_search at the tool-level — when off,
# the tool's runtime guard returns "hybrid_nav_disabled" rather than
# refusing registration (so the tool list stays stable for any agent
# discovery; the runtime decides whether to honor the call).
register_nav_tools(mcp, state)
# Review P1 #4: write tools are always REGISTERED (so the catalog shape
# stays stable for clients that enumerate at startup), but the tool
# bodies refuse to execute unless settings.mcp_enable_write_tools=True
# at call time. The catalog therefore exposes the schemas; calling them
# on a read-only server returns reason=write_tools_disabled. Docs say
# this explicitly so operators can audit the runtime contract.
register_nav_write_tools(mcp, state)


# ── Prompts ──────────────────────────────────────────────────────────────────
# Gated by mcp_enable_prompts.

from kb.tool_server.mcp_prompts import register_prompts  # noqa: E402

if _register_time_settings.mcp_enable_prompts:
    register_prompts(mcp, state)
