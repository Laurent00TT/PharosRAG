"""Shared MCP-tool JSON-RPC dance helper.

Extracted from ``tests/test_auth_middleware.py::
test_mcp_tool_body_sees_current_user_via_real_mount`` (T1b) so multiple
test files can exercise mounted FastMCP tools without each duplicating
the initialize → initialized → tools/call sequence.

T1b's inline copy is intentionally NOT removed — it builds its own
minimal app + ``mcp.session_manager.run()`` context, which is a tighter
isolation than the shared helper offers. The helper here targets the
already-running app produced by the ``test_app_with_users`` fixture
(real lifespan runs ``mcp.session_manager.run()`` for the duration of
the fixture).
"""
from __future__ import annotations

import json

from httpx import ASGITransport, AsyncClient


async def call_mcp_tool(
    app, token: str, tool_name: str, arguments: dict,
) -> dict:
    """JSON-RPC initialize → initialized → tools/call dance against a
    mounted FastMCP sub-app.

    Returns the parsed JSON-RPC envelope of the ``tools/call`` response.
    The tool result (with content / structuredContent / isError) is at
    ``envelope["result"]``; transport-level JSON-RPC errors are at
    ``envelope["error"]``.

    Parameters
    ----------
    app
        FastAPI app instance with the /mcp mount. The caller MUST have
        the FastMCP session manager running (production lifespan does
        this automatically; pytest fixtures should use TestClient as a
        context manager so ``__enter__`` triggers lifespan).
    token
        Plaintext API key for the user the MCP request runs as.
    tool_name
        Registered MCP tool name (e.g. ``"kb_search_nav"``).
    arguments
        ``tools/call`` ``params.arguments`` dict.
    """
    transport = ASGITransport(app=app)
    headers = {
        "X-API-Key": token,
        "Accept": "application/json, text/event-stream",
    }
    # FastMCP auto-enables DNS-rebinding protection when host="127.0.0.1"
    # (the default for the streamable HTTP transport). The allowlist is
    # ["127.0.0.1:*", "localhost:*", "[::1]:*"] — note the wildcard port
    # patterns REQUIRE an explicit ":port" suffix on the Host header
    # (see mcp/server/transport_security.py::_validate_host —
    # ``host.startswith(base_host + ":")``). httpx omits the default-port
    # suffix from ``Host`` (e.g. http://localhost → Host: localhost), so
    # we use an explicit non-default port to force ``Host: localhost:80``
    # and pass the wildcard-port match. This bypasses the 421 Invalid-Host
    # rejection without disabling the production security setting.
    async with AsyncClient(transport=transport, base_url="http://localhost:8000") as c:
        init = await c.post("/mcp/", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        })
        if "mcp-session-id" not in init.headers:
            raise RuntimeError(
                f"initialize did not return mcp-session-id; "
                f"status={init.status_code} headers={dict(init.headers)} "
                f"body={init.text[:500]}"
            )
        session_id = init.headers["mcp-session-id"]
        headers_sid = {**headers, "mcp-session-id": session_id}
        try:
            await c.post("/mcp/", headers=headers_sid, json={
                "jsonrpc": "2.0", "method": "notifications/initialized",
                "params": {},
            })
            r = await c.post("/mcp/", headers=headers_sid, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            })
            body = r.text
            # FastMCP returns SSE event-stream OR JSON depending on Accept
            # negotiation. SSE body looks like:  event: message\ndata: {json}
            if body.startswith("event:"):
                line = next(L for L in body.splitlines() if L.startswith("data:"))
                return json.loads(line[len("data:"):].strip())
            return r.json()
        finally:
            # Explicit DELETE terminates the FastMCP streamable-HTTP session
            # so the server-side task group exits cleanly. Without this, the
            # session lingers until the lifespan shutdown tries to cancel it,
            # which deadlocks the TestClient teardown on Windows (observed
            # multi-minute hangs). Best-effort — silently ignore any DELETE
            # error since the AsyncClient is already about to close.
            try:
                await c.delete("/mcp/", headers=headers_sid)
            except Exception:  # noqa: BLE001
                pass
