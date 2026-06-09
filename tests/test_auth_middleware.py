"""T1b — REST + MCP auth contract tests.

Verifies the full request flow:
  1. token in X-API-Key → user resolved via UsersStore.get_by_key_hash
  2. ContextVar.current_user is alice / bob / etc inside the handler
  3. Disabled / unknown / missing token → 401
  4. Cross-request ContextVar does NOT leak (the streaming-response trap)
  5. (Step 7.3) Real FastMCP tool body sees current_user
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Depends
from httpx import ASGITransport, AsyncClient

from kb.auth.context import current_user, get_current_user
from kb.auth.users import UsersStore
from kb.tool_server.security import verify_api_key


@pytest.fixture
async def users_store(tmp_path):
    s = UsersStore(db_url=f"sqlite+aiosqlite:///{tmp_path}/u.db")
    await s.init()
    yield s
    await s.aclose()


@pytest.fixture
async def app_with_auth(users_store):
    """Build a minimal FastAPI app with the same AuthMiddleware as prod
    (Option B per plan §0d): middleware does primary auth + set/reset
    ContextVar; verify_api_key is defense-in-depth marker."""
    from kb.auth.middleware import AuthError, authenticate_token
    from starlette.responses import JSONResponse

    app = FastAPI()
    app.state.users_store = users_store

    _ALLOWLIST = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})

    def _extract_token(request):
        provided = request.headers.get("X-API-Key")
        if provided:
            return provided.strip()
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    @app.middleware("http")
    async def auth_mw(request, call_next):
        path = request.url.path
        if path in _ALLOWLIST or path.startswith("/mcp"):
            return await call_next(request)
        token = _extract_token(request)
        store = request.app.state.users_store
        if store is None:
            return JSONResponse({"detail": "users store not initialized"}, status_code=503)
        try:
            user = await authenticate_token(
                token, store,
                source="rest",
                source_ip=(request.client.host if request.client else None),
            )
        except AuthError as exc:
            return JSONResponse({"detail": exc.detail}, status_code=401)
        ctx_token = current_user.set(user)
        try:
            return await call_next(request)
        finally:
            current_user.reset(ctx_token)

    @app.get("/whoami", dependencies=[Depends(verify_api_key)])
    async def whoami():
        return {"username": get_current_user().username}

    return app


async def test_valid_token_resolves_user(app_with_auth, users_store):
    plaintext, _ = await users_store.create_user(username="alice", role="member")
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/whoami", headers={"X-API-Key": plaintext})
    assert r.status_code == 200
    assert r.json()["username"] == "alice"


async def test_missing_token_returns_401(app_with_auth):
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/whoami")
    assert r.status_code == 401
    assert "missing" in r.json()["detail"].lower()


async def test_unknown_token_returns_401(app_with_auth):
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/whoami", headers={"X-API-Key": "kb_ghost_xxxxx"})
    assert r.status_code == 401
    assert "invalid" in r.json()["detail"].lower()


async def test_disabled_user_returns_401(app_with_auth, users_store):
    plaintext, alice = await users_store.create_user(username="alice", role="member")
    await users_store.disable_user(alice.user_id)
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/whoami", headers={"X-API-Key": plaintext})
    assert r.status_code == 401
    assert "disabled" in r.json()["detail"].lower()


async def test_bearer_auth_header_works(app_with_auth, users_store):
    """Authorization: Bearer <token> is an alternative for generic clients."""
    plaintext, _ = await users_store.create_user(username="bob", role="member")
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/whoami", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    assert r.json()["username"] == "bob"


async def test_two_users_dont_leak_across_requests(app_with_auth, users_store):
    """Spec I-8 anti-regression: alice's ContextVar must NOT leak into
    bob's request in the same Worker. Sequential same-client requests
    are the realistic leak scenario."""
    plaintext_a, _ = await users_store.create_user(username="alice", role="member")
    plaintext_b, _ = await users_store.create_user(username="bob", role="member")
    transport = ASGITransport(app=app_with_auth)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r_a = await client.get("/whoami", headers={"X-API-Key": plaintext_a})
        r_b = await client.get("/whoami", headers={"X-API-Key": plaintext_b})
    assert r_a.json()["username"] == "alice"
    assert r_b.json()["username"] == "bob"  # critical: NOT alice


async def test_mcp_asgi_enforcer_blocks_missing_token(users_store):
    """ASGI-side: APIKeyEnforcer rejects no-token requests with 401."""
    from kb.auth.middleware import APIKeyEnforcer

    async def inner_app(scope, receive, send):
        # If we reach here, that's a bug — enforcer should have rejected
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"leaked"})

    enforcer = APIKeyEnforcer(inner_app, lambda: users_store)
    sent = []
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(m):
        sent.append(m)

    await enforcer(scope, receive, send)
    assert sent[0]["status"] == 401


async def test_mcp_asgi_enforcer_sets_current_user_for_inner(users_store):
    """ASGI-side: on valid token, inner app sees current_user set."""
    from kb.auth.middleware import APIKeyEnforcer

    plaintext, _ = await users_store.create_user(username="carol", role="admin")
    seen = {}

    async def inner_app(scope, receive, send):
        seen["user"] = get_current_user()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    enforcer = APIKeyEnforcer(inner_app, lambda: users_store)
    sent = []
    scope = {
        "type": "http", "method": "GET", "path": "/",
        "headers": [(b"x-api-key", plaintext.encode())],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(m):
        sent.append(m)

    await enforcer(scope, receive, send)
    assert sent[0]["status"] == 200
    assert seen["user"].username == "carol"


async def test_mcp_asgi_enforcer_resets_contextvar_after_request(users_store):
    """Spec I-8: after the inner app returns, current_user must be cleared."""
    from kb.auth.middleware import APIKeyEnforcer

    plaintext, _ = await users_store.create_user(username="dave", role="member")

    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    enforcer = APIKeyEnforcer(inner_app, lambda: users_store)
    scope = {
        "type": "http", "method": "GET", "path": "/",
        "headers": [(b"x-api-key", plaintext.encode())],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_m):
        pass

    assert current_user.get() is None  # Before
    await enforcer(scope, receive, send)
    assert current_user.get() is None  # After: reset back


async def test_mcp_tool_body_sees_current_user_via_real_mount(users_store):
    """End-to-end: alice's token → /mcp/ → FastMCP dispatch → tool body
    reads get_current_user().username == 'alice'. Spec P0-1 critical
    contract — middleware-only tests cannot catch FastMCP-level context drops.
    """
    import json
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from mcp.server.fastmcp import FastMCP

    from kb.auth.middleware import APIKeyEnforcer

    plaintext, _ = await users_store.create_user(username="alice", role="admin")

    from mcp.server.transport_security import TransportSecuritySettings

    mcp = FastMCP("test-contract")
    mcp.settings.streamable_http_path = "/"
    # Disable DNS-rebinding host validation so httpx ASGITransport's
    # synthetic "Host: test" header does not trigger a 421.
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )

    @mcp.tool()
    async def whoami_in_tool() -> dict:
        u = get_current_user()
        return {"username": u.username, "role": u.role}

    app = FastAPI()
    app.state.users_store = users_store
    enforcer = APIKeyEnforcer(mcp.streamable_http_app(), lambda: users_store)
    app.mount("/mcp", enforcer)

    transport = ASGITransport(app=app)
    # The StreamableHTTPSessionManager requires .run() to start its internal
    # task group before any requests can be handled. In production this is done
    # by the FastAPI lifespan. In tests we start it explicitly here.
    async with mcp.session_manager.run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Initialize MCP session
            init_req = {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            }
            init_r = await client.post(
                "/mcp/",
                json=init_req,
                headers={
                    "X-API-Key": plaintext,
                    "Accept": "application/json, text/event-stream",
                },
            )
            assert init_r.status_code == 200, init_r.text
            session_id = init_r.headers.get("mcp-session-id")
            assert session_id, "FastMCP must hand out a session id"

            # initialized notification
            await client.post(
                "/mcp/",
                json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                headers={
                    "X-API-Key": plaintext,
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": session_id,
                },
            )

            # Call the test tool
            call_req = {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "whoami_in_tool", "arguments": {}},
            }
            r = await client.post(
                "/mcp/",
                json=call_req,
                headers={
                    "X-API-Key": plaintext,
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": session_id,
                },
            )
            assert r.status_code == 200, r.text

            # Parse response — FastMCP returns SSE or JSON depending on negotiation
            body = r.text
            if body.startswith("event:"):
                data_line = next(
                    line for line in body.splitlines() if line.startswith("data:")
                )
                payload = json.loads(data_line[len("data:"):].strip())
            else:
                payload = r.json()

            result = payload["result"]
            if "structuredContent" in result and result["structuredContent"]:
                tool_output = result["structuredContent"]
            else:
                tool_output = json.loads(result["content"][0]["text"])
            assert tool_output["username"] == "alice"
            assert tool_output["role"] == "admin"


# ── BruteForceTracker integration ───────────────────────────────────

@pytest.mark.asyncio
async def test_authenticate_token_records_unknown_key_to_tracker(auth_setup):
    from unittest.mock import AsyncMock
    _, _, store = auth_setup
    tracker = AsyncMock()
    tracker.record_failure = AsyncMock()
    from kb.auth.middleware import authenticate_token, AuthError
    with pytest.raises(AuthError):
        await authenticate_token(
            token="garbage-token", users_store=store,
            source="rest", source_ip="1.2.3.4",
            brute_force_tracker=tracker,
        )
    tracker.record_failure.assert_awaited_once_with(
        "1.2.3.4", reason="unknown_key",
    )


@pytest.mark.asyncio
async def test_authenticate_token_records_missing_key_to_tracker(auth_setup):
    from unittest.mock import AsyncMock
    _, _, store = auth_setup
    tracker = AsyncMock()
    tracker.record_failure = AsyncMock()
    from kb.auth.middleware import authenticate_token, AuthError
    with pytest.raises(AuthError):
        await authenticate_token(
            token=None, users_store=store,
            source="rest", source_ip="1.2.3.4",
            brute_force_tracker=tracker,
        )
    tracker.record_failure.assert_awaited_once_with(
        "1.2.3.4", reason="missing_key",
    )


@pytest.mark.asyncio
async def test_authenticate_token_does_NOT_record_disabled(auth_setup):
    """Disabled is a per-user state, not enumeration. Recording would
    pollute the IP-level signal."""
    from unittest.mock import AsyncMock
    _, member_token, store = auth_setup
    member = await store.get_by_username("member0")
    await store.disable_user(member.user_id)
    tracker = AsyncMock()
    tracker.record_failure = AsyncMock()
    from kb.auth.middleware import authenticate_token, AuthError
    with pytest.raises(AuthError, match="disabled"):
        await authenticate_token(
            token=member_token, users_store=store,
            source="rest", source_ip="1.2.3.4",
            brute_force_tracker=tracker,
        )
    tracker.record_failure.assert_not_called()


@pytest.mark.asyncio
async def test_authenticate_token_without_tracker_kwarg_still_works(auth_setup):
    _, member_token, store = auth_setup
    from kb.auth.middleware import authenticate_token
    user = await authenticate_token(
        token=member_token, users_store=store,
        source="rest", source_ip="1.2.3.4",
    )  # no tracker kwarg
    assert user.username == "member0"


@pytest.mark.asyncio
async def test_api_key_enforcer_passes_tracker_through(auth_setup):
    """Verify provider-style wiring: APIKeyEnforcer reads tracker from
    its provider lambda each call and forwards to authenticate_token."""
    from unittest.mock import AsyncMock
    from kb.auth.middleware import APIKeyEnforcer
    _, _, store = auth_setup

    tracker = AsyncMock()
    tracker.record_failure = AsyncMock()

    enforcer = APIKeyEnforcer(
        app=AsyncMock(),
        users_store_provider=lambda: store,
        brute_force_tracker_provider=lambda: tracker,
    )

    sent = []
    async def send(msg): sent.append(msg)
    scope = {
        "type": "http", "method": "POST", "path": "/mcp/",
        "headers": [(b"x-api-key", b"garbage")],
        "client": ("9.9.9.9", 12345),
    }
    async def receive(): return {"type": "http.request"}
    await enforcer(scope, receive, send)

    # 401 returned, AND tracker was called via the provider
    assert sent[0]["status"] == 401
    tracker.record_failure.assert_awaited_once_with(
        "9.9.9.9", reason="unknown_key",
    )
