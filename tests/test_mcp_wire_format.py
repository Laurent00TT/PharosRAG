"""Regression test for the FastMCP wire-format bug.

Background
----------
FastMCP only emits ``CallToolResult.structuredContent`` on the wire when a
tool's return annotation maps to a Pydantic-compatible output schema. A
bare ``dict`` annotation does NOT — ``func_metadata._try_create_model_and_schema``
returns ``output_model=None`` for it (``get_type_hints(dict)`` is empty),
which makes ``FuncMetadata.convert_result`` JSON-dump the entire return
value into a single ``TextContent`` block and silently drop the
``structuredContent`` field that downstream agents read.

E2E testing of Package A surfaced this: every nav tool's
``structuredContent`` was missing on the wire even though the in-process
unit tests passed (they call the tool function directly and inspect the
returned dict, which never goes through FastMCP's serialization).

This file pins the fix: every MCP tool registered on ``mcp_api.mcp`` must
return a ``CallToolResult`` so FastMCP's ``convert_result`` short-circuit
(``mcp/server/fastmcp/utilities/func_metadata.py`` line 114) propagates
``structuredContent`` verbatim to the client.

The tests exercise FastMCP's ``call_tool`` end-to-end (the same code path
``tools/call`` JSON-RPC requests take inside the streamable HTTP transport),
so any future regression — e.g. someone adding a new tool annotated
``-> dict`` — will fail the suite instead of slipping into production.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent

from kb.tool_server import mcp_api, mcp_state


# ── Helpers ─────────────────────────────────────────────────────────────────


def _stub_state() -> None:
    """Bind a minimal MCP state so every tool can run without a real engine.

    All collaborators are ``None`` (or a tiny in-memory shim) so each tool
    hits its "not initialized" gentle-error branch. That's enough for the
    wire-format assertion — we only need a well-formed ``CallToolResult``,
    not realistic search hits.
    """
    settings = SimpleNamespace(
        mcp_tool_result_limit=20,
        mcp_tool_text_budget=800,
        mcp_enable_write_tools=False,
    )
    mcp_state.bind_state(
        engine=None,
        meta_db=None,
        settings=settings,
        nav_store=None,
        nav_search=None,
        hybrid_navigator=None,
    )


@pytest.fixture(autouse=True)
def _bind_and_reset_state():
    _stub_state()
    yield
    mcp_state.reset()


# All 11 tools that previously returned bare ``dict``. ``kb_list`` joined the
# group because its old ``str`` return only sneaked structuredContent through
# via FastMCP's wrap-output behavior — the unified ``CallToolResult`` path
# gives it a richer payload (documents list) instead.
ALL_TOOLS_WITH_ARGS: list[tuple[str, dict]] = [
    ("kb_search", {"query": "irrelevant", "top_k": 1}),
    ("kb_list", {}),
    ("kb_search_nav", {"query": "irrelevant", "top_k": 1}),
    ("kb_hybrid_search", {"query": "irrelevant", "top_k": 1}),
    ("kb_expand_nav_entry", {"entry_id": "nonexistent"}),
    ("kb_list_documents", {}),
    ("kb_get_toc", {"doc_id": "nonexistent"}),
    ("kb_get_status", {}),
    ("kb_add_nav_alias", {"entry_id": "x", "alias": "y"}),
    ("kb_hide_nav_entry", {"entry_id": "x", "reason": "noise"}),
    ("kb_rebuild_nav_index", {"doc_id": "x"}),
]


# ── Per-tool wire-format assertion ──────────────────────────────────────────


@pytest.mark.parametrize("tool_name,args", ALL_TOOLS_WITH_ARGS)
async def test_tool_returns_call_tool_result_with_structured_content(
    tool_name: str, args: dict
):
    """The single load-bearing invariant: every registered tool must produce
    a ``CallToolResult`` whose ``structuredContent`` is a dict.

    Going through ``mcp.call_tool`` (instead of the function directly)
    exercises FastMCP's ``FuncMetadata.convert_result``, which is exactly
    where the original bug used to silently strip ``structuredContent``.
    """
    result = await mcp_api.mcp.call_tool(tool_name, args)
    assert isinstance(result, CallToolResult), (
        f"{tool_name}: FastMCP did not preserve CallToolResult — wire bug. "
        f"Got {type(result).__name__}."
    )
    assert result.content, f"{tool_name}: content list is empty"
    assert isinstance(result.content[0], TextContent), (
        f"{tool_name}: first content block is not TextContent"
    )
    assert isinstance(result.structuredContent, dict), (
        f"{tool_name}: structuredContent missing or not a dict — wire-format "
        f"bug regression. Got {type(result.structuredContent).__name__}."
    )


# ── Schema-introspection guard ──────────────────────────────────────────────


def test_no_tool_uses_bare_dict_return_annotation():
    """Compile-time guard: no future tool should sneak in with ``-> dict``.

    ``FuncMetadata.output_schema`` is ``None`` precisely when FastMCP can't
    derive a Pydantic schema from the return annotation — the exact
    condition that triggers the wire-format bug. Tools that return
    ``CallToolResult`` are an explicit allowed exception: FastMCP's
    ``func_metadata`` detects the ``CallToolResult`` subclass at line 295
    and intentionally returns ``output_schema=None`` because the tool will
    short-circuit ``convert_result`` with its own well-formed result.

    The module files use ``from __future__ import annotations``, so
    ``__annotations__`` values are strings; we resolve them via
    ``typing.get_type_hints`` to compare against the real ``CallToolResult``
    class.
    """
    from typing import get_type_hints

    bad: list[str] = []
    for tool_name, tool in mcp_api.mcp._tool_manager._tools.items():
        try:
            hints = get_type_hints(tool.fn)
        except Exception:  # pragma: no cover — defensive
            hints = {}
        return_ann = hints.get("return")
        # Tools that return CallToolResult are allowed to have no output_schema;
        # the convert_result short-circuit handles them.
        if isinstance(return_ann, type) and issubclass(return_ann, CallToolResult):
            continue
        if tool.fn_metadata.output_schema is None:
            bad.append(
                f"{tool_name}: no output_schema (return annotation: {return_ann!r}). "
                f"Use -> CallToolResult or a typed return so structuredContent "
                f"survives the wire."
            )
    assert not bad, "Tools with broken wire format:\n  - " + "\n  - ".join(bad)


# ── Structured payload sanity (one representative tool) ─────────────────────


async def test_kb_get_status_structured_payload_has_expected_keys():
    """Sanity check that structuredContent is the *actual* payload, not a
    JSON-stringified blob shoved into TextContent — which is what the
    original bug produced.
    """
    result = await mcp_api.mcp.call_tool("kb_get_status", {})
    assert isinstance(result, CallToolResult)
    sc = result.structuredContent
    assert isinstance(sc, dict)
    for key in (
        "evidence_ready",
        "nav_ready",
        "hybrid_ready",
        "meta_db_ready",
        "document_count",
    ):
        assert key in sc, f"kb_get_status structuredContent missing {key!r}: {sc!r}"
    # And the human-readable summary still rides along in content[0].text.
    assert "Evidence layer" in result.content[0].text
