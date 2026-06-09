"""Shared auth: REST FastAPI middleware + MCP ASGI middleware converge here.

Both paths call ``authenticate_token(token, users_store, source=..., source_ip=...)``.
The function emits ``auth.failed`` trace event on each failure
(NOT a single audit row per I-13). T2's BruteForceTracker will
aggregate these into ``auth.brute_force`` audit entries, keyed by source_ip.

Hard-cutover (spec I-7): no legacy ``tool_server_api_key`` / ``mcp_api_key``
fallback. Every request must present a user token from the users table.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from kb.auth.context import current_user
from kb.auth.key_gen import hash_token
from kb.auth.users import User, UsersStore
from kb.observability.trace import emit

ASGIApp = Callable[[dict, Callable, Callable], Awaitable[None]]


class AuthError(Exception):
    """Internal auth failure marker. Callers map to 401."""
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _extract_token(headers: Headers) -> str | None:
    """X-API-Key takes precedence; Bearer is fallback for generic clients."""
    provided = headers.get("x-api-key")
    if provided:
        return provided.strip()
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


async def authenticate_token(
    token: str | None,
    users_store: UsersStore,
    *,
    source: str = "unknown",      # "rest" | "mcp" — T2 BruteForceTracker groups by this
    source_ip: str | None = None,  # client IP for T2 aggregation; may be None in tests
    brute_force_tracker=None,    # type loose to avoid import cycle
) -> User:
    """Resolve token -> User. Raises AuthError on any failure.

    Emits ``auth.failed`` trace event on each failure (NOT a single audit
    row per I-13). If a ``brute_force_tracker`` is provided, enumeration-
    shaped failures (``missing_key`` / ``unknown_key``) feed its sliding
    window; ``disabled`` deliberately does not (per-user state, would
    pollute IP-level signal).
    """
    if not token:
        emit("auth.failed", reason="missing_key", source=source, source_ip=source_ip)
        if brute_force_tracker is not None:
            await brute_force_tracker.record_failure(source_ip, reason="missing_key")
        raise AuthError("missing API key")
    key_hash = hash_token(token)
    user = await users_store.get_by_key_hash(key_hash)
    if user is None:
        emit("auth.failed", reason="unknown_key", source=source, source_ip=source_ip)
        if brute_force_tracker is not None:
            await brute_force_tracker.record_failure(source_ip, reason="unknown_key")
        raise AuthError("invalid API key")
    if user.is_disabled:
        emit("auth.failed", reason="disabled", user_id=user.user_id,
             source=source, source_ip=source_ip)
        raise AuthError(f"user {user.username} disabled")
    return user


class APIKeyEnforcer:
    """ASGI middleware for the MCP sub-app.

    Wraps the MCP transport so a missing/invalid key returns 401 BEFORE
    reaching FastMCP. On success, sets ``current_user`` ContextVar with
    try/finally reset (I-8 — prevents leaks across streaming responses).

    Non-HTTP scopes (lifespan, websocket if ever added) pass through
    unchanged.

    The store is resolved per-request via ``users_store_provider`` rather
    than captured at construction time. This lets us mount the enforcer
    at module load (single mount, no per-lifespan accumulation) while
    still picking up the store that lifespan binds to ``app.state``. If
    the provider returns None (lifespan hasn't bound the store yet),
    requests fail closed with 503.
    """

    def __init__(
        self,
        app: ASGIApp,
        users_store_provider: Callable[[], UsersStore | None],
        brute_force_tracker_provider: Callable[[], object | None] | None = None,
    ) -> None:
        self.app = app
        self._users_store_provider = users_store_provider
        self._brute_force_tracker_provider = brute_force_tracker_provider

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        users_store = self._users_store_provider()
        if users_store is None:
            response = JSONResponse(
                {"detail": "users store not initialized; server misconfigured"},
                status_code=503,
            )
            await response(scope, receive, send)
            return
        tracker = None
        if self._brute_force_tracker_provider is not None:
            tracker = self._brute_force_tracker_provider()
        token = _extract_token(Headers(scope=scope))
        # ASGI client info: scope["client"] is (host, port) tuple or None
        source_ip = None
        if scope.get("client"):
            source_ip = scope["client"][0]
        try:
            user = await authenticate_token(
                token, users_store,
                source="mcp", source_ip=source_ip,
                brute_force_tracker=tracker,
            )
        except AuthError as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=401)
            await response(scope, receive, send)
            return
        ctx_token = current_user.set(user)
        try:
            await self.app(scope, receive, send)
        finally:
            current_user.reset(ctx_token)
