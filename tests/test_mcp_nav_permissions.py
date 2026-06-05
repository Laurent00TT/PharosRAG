"""MCP nav write tools require ownership (T3-6).

Three tools are gated by ``require_owner_or_admin``:
  - kb_rebuild_nav_index(doc_id)            — direct doc lookup
  - kb_add_nav_alias(entry_id, alias)       — entry → doc resolution
  - kb_hide_nav_entry(entry_id, reason)     — entry → doc resolution

Permission target is always the document owning the affected nav entry
(or the doc itself for rebuild). ``target_id`` in authz.denied rows is
the ``doc_id``, never ``entry_id`` — so audit aggregation per document
remains consistent with DELETE/cancel handlers.

FastMCP wraps any exception raised inside the tool body (including
``HTTPException(403)`` raised by ``require_owner_or_admin``) into a
``CallToolResult(content=[...], isError=True)`` and returns 200 at the
transport layer. The audit row is written by ``_emit_authz_denied`` BEFORE
the exception is raised, so we can assert on it regardless of how the
wire-level response surfaces the error.
"""
from __future__ import annotations

import pytest

from tests._mcp_helpers import call_mcp_tool


@pytest.fixture(autouse=True)
def _ensure_nav_enabled(monkeypatch):
    """Force ``settings.nav_enabled`` True before the lifespan binds MCPState.

    ``tests/test_lifespan_nav_wiring.py::test_lifespan_disabled_nav_leaves_state_none``
    sets ``NAV_ENABLED=false`` and ``importlib.reload(app_module)``, which
    rebuilds the module-level singleton ``app_module.settings`` with
    ``nav_enabled=False``. monkeypatch reverts the env var but not the
    already-instantiated singleton, so any later test that depends on the
    lifespan wiring nav_store / nav_builder / nav_search (i.e. ours)
    silently gets None state and the MCP tool body returns the early
    ``services_not_initialized`` short-circuit instead of running the
    permission check.

    This fixture runs BEFORE ``test_app_with_users`` (autouse function-
    scoped fixtures run first in definition order) and uses
    ``monkeypatch.setattr`` on the singleton so the value is restored
    after each test — no global leak.
    """
    from kb.tool_server import app as app_module
    monkeypatch.setattr(app_module.settings, "nav_enabled", True)


async def _call_mcp_tool(client, token, tool_name, arguments):
    """Thin wrapper: adapt ``client.app`` to the shared helper."""
    return await call_mcp_tool(client.app, token, tool_name, arguments)


def _enable_write_tools(client):
    """Flip ``mcp_enable_write_tools`` on the live settings object.

    Both ``app.state.settings`` and MCPState's ``state.settings`` point at
    the same Settings instance (bound by lifespan), so a single attribute
    write reaches both. Restored implicitly when the fixture tears down
    (Settings instance is per-pytest-process module-global, so we flip back
    in a finally to keep tests order-independent)."""
    client.app.state.settings.mcp_enable_write_tools = True


def _disable_write_tools(client):
    client.app.state.settings.mcp_enable_write_tools = False


def _seed_entry(nav_store, *, entry_id: str, doc_id: str, label: str, order: int):
    """Insert one NavEntry directly via ``upsert_entries`` so the entry →
    doc resolution chain has a real row to resolve. Keeps fixture setup
    explicit per-test rather than baking nav rows into ``test_app_with_users``
    (most T3 tests don't need them)."""
    from kb.nav.models import NavEntry
    entry = NavEntry(
        entry_id=entry_id,
        label=label,
        normalized_label=label.lower(),
        entry_type="section",
        source="parser_heading",
        doc_id=doc_id,
        doc_name="alice-manual.pdf",
        page_start=order,
        page_end=order,
        order_index=order,
        parent_entry_id=None,
        resource_uris=[f"kb://documents/{doc_id}/pages/{order}"],
        source_ref_ids=[],
        aliases=[],
    )
    return nav_store.upsert_entries([entry])


@pytest.mark.asyncio
async def test_kb_rebuild_nav_index_owner_member_allowed(test_app_with_users):
    """alice owns doc-alice-1 → kb_rebuild_nav_index returns a non-error result.

    The doc has no qdrant chunks in the test fixture, so the tool returns
    structured reason ``no_chunks_found``. That's still a successful
    permission outcome (no HTTPException, no authz.denied audit row);
    the test asserts on the permission path only, not the rebuild outcome.
    """
    setup = test_app_with_users
    client, alice_token = setup["client"], setup["alice_token"]
    audit = setup["audit_log"]
    alice_id = setup["alice_user_id"]
    _enable_write_tools(client)
    try:
        resp = await _call_mcp_tool(
            client, alice_token,
            "kb_rebuild_nav_index", {"doc_id": "doc-alice-1"},
        )
    finally:
        _disable_write_tools(client)
    # JSON-RPC envelope has no transport-level error
    assert resp.get("error") is None, resp
    # Tool result is present and not an isError envelope (owner passed)
    result = resp["result"]
    assert result.get("isError") is not True, result
    # No authz.denied row written for owner path
    denied = await audit.query(
        action="authz.denied", target_id="doc-alice-1", limit=5,
    )
    assert not any(r["user_id"] == alice_id for r in denied), (
        "owner path must not write authz.denied"
    )


@pytest.mark.asyncio
async def test_kb_rebuild_nav_index_non_owner_member_denied(test_app_with_users):
    """bob is a member; doc-alice-1 is alice's — must 403 + authz.denied row.

    HTTPException(403) raised inside the tool body bubbles up to FastMCP,
    which converts it to ``CallToolResult(isError=True)`` at 200 OK on the
    wire (MCP spec: errors are in the result envelope, not transport).
    The audit row is what guarantees forensic visibility.
    """
    setup = test_app_with_users
    client, bob_token = setup["client"], setup["bob_token"]
    audit = setup["audit_log"]
    bob_id = setup["bob_user_id"]
    _enable_write_tools(client)
    try:
        resp = await _call_mcp_tool(
            client, bob_token,
            "kb_rebuild_nav_index", {"doc_id": "doc-alice-1"},
        )
    finally:
        _disable_write_tools(client)
    # FastMCP surfaces tool-body exceptions as isError result; the JSON-RPC
    # envelope may also surface them as an error field depending on path.
    error_surface = (
        resp.get("error") is not None
        or resp.get("result", {}).get("isError") is True
    )
    assert error_surface, f"expected error or isError response, got {resp}"
    rows = await audit.query(
        action="authz.denied", target_id="doc-alice-1", limit=5,
    )
    assert any(r["user_id"] == bob_id for r in rows), (
        f"expected authz.denied row for bob → doc-alice-1, got {rows}"
    )


@pytest.mark.asyncio
async def test_kb_add_nav_alias_admin_allowed(test_app_with_users):
    """admin can add an alias on any doc (role hierarchy bypass).

    Seed a nav entry tied to doc-alice-1 first so the entry → doc
    resolution chain succeeds; admin role passes ``require_owner_or_admin``
    regardless of doc owner. Assert the JSON-RPC result is not an isError
    envelope and the alias survived to the nav store.
    """
    setup = test_app_with_users
    client, admin_token = setup["client"], setup["admin_token"]
    nav_store = client.app.state.nav_store
    await _seed_entry(
        nav_store,
        entry_id="e-alice-1", doc_id="doc-alice-1",
        label="Chapter 1", order=1,
    )
    _enable_write_tools(client)
    try:
        resp = await _call_mcp_tool(
            client, admin_token,
            "kb_add_nav_alias",
            {"entry_id": "e-alice-1", "alias": "approval flow"},
        )
    finally:
        _disable_write_tools(client)
    assert resp.get("error") is None, resp
    result = resp["result"]
    assert result.get("isError") is not True, result
    # Alias was actually written through to nav_store
    entry = await nav_store.get_entry("e-alice-1")
    assert entry is not None
    assert "approval flow" in entry.aliases


@pytest.mark.asyncio
async def test_kb_hide_nav_entry_non_owner_denied(test_app_with_users):
    """bob cannot hide alice's nav entry.

    Mirrors the kb_rebuild_nav_index denied path but exercises the
    entry → doc resolution chain. ``target_id`` in the authz.denied row
    is ``doc-alice-1`` (the parent doc), NOT the entry_id — the document
    is the permission unit.
    """
    setup = test_app_with_users
    client, bob_token = setup["client"], setup["bob_token"]
    audit = setup["audit_log"]
    nav_store = client.app.state.nav_store
    await _seed_entry(
        nav_store,
        entry_id="e-alice-2", doc_id="doc-alice-1",
        label="Section 2", order=2,
    )
    _enable_write_tools(client)
    try:
        resp = await _call_mcp_tool(
            client, bob_token,
            "kb_hide_nav_entry",
            {"entry_id": "e-alice-2", "reason": "irrelevant"},
        )
    finally:
        _disable_write_tools(client)
    error_surface = (
        resp.get("error") is not None
        or resp.get("result", {}).get("isError") is True
    )
    assert error_surface, f"expected error or isError response, got {resp}"
    # authz.denied row targets the DOC (parent of the entry), not entry_id
    rows = await audit.query(
        action="authz.denied", target_id="doc-alice-1", limit=10,
    )
    assert len(rows) >= 1, f"expected authz.denied for doc-alice-1, got {rows}"
    # Entry must NOT have been hidden — denied path returns before
    # state.nav_store.hide_entry runs
    hidden_ids = await nav_store.list_hidden_entry_ids()
    assert "e-alice-2" not in hidden_ids, "denied path must not mutate hide state"
