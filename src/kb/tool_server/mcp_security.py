"""MCP transport security (Origin validation) + tool-call audit trace.

Two concerns grouped here because Task 2 scopes them together:

- ``OriginValidator`` is an ASGI middleware wrapping the mounted MCP sub-app
  so we can reject requests whose ``Origin`` header is not on the configured
  allow-list. Wrapping the sub-app is more explicit than relying on a parent
  FastAPI middleware to intercept mounted routes.
- ``audit_tool_call`` is a thin wrapper invoked from each MCP tool body that
  emits a structured ``mcp.tool`` event via :func:`kb.observability.trace.emit`
  (or :func:`kb.observability.trace.emit_error` on exception) so production
  tool calls show up in ``logs/kb_trace.jsonl`` alongside every other trace
  event.
"""

# NOTE: APIKeyEnforcer (user-token version) lives in kb.auth.middleware as of T1b.
# The single-key version (mcp_api_key config) was deleted in T1b hard cutover.

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from kb.observability.trace import emit, emit_error

ASGIApp = Callable[[dict, Callable, Callable], Awaitable[None]]


def parse_allowed_origins(value: str) -> set[str]:
    return {item.strip().rstrip("/") for item in value.split(",") if item.strip()}


def origin_is_allowed(origin: str | None, allowed_origins: str) -> bool:
    """True iff origin is None/empty OR an exact match (modulo trailing /)
    of one entry in allowed_origins.

    The default allowlist 'http://localhost,http://127.0.0.1' permits the
    common loopback case. To permit a specific localhost port, add it
    explicitly: 'http://localhost:31337,http://localhost,...' — implicit
    'any loopback port' is no longer trusted because a hostile localhost
    service (e.g. a malicious dev tool) could otherwise drive MCP via
    browser fetch.
    """
    if origin is None or origin == "":
        return True
    normalized = origin.rstrip("/")
    return normalized in parse_allowed_origins(allowed_origins)


class OriginValidator:
    """ASGI middleware for the mounted MCP sub-app.

    Wrapping the sub-app is more explicit than relying on a parent FastAPI
    middleware to intercept mounted routes.
    """

    def __init__(self, app: ASGIApp, allowed_origins: list[str] | set[str]) -> None:
        self.app = app
        self.allowed_origins = ",".join(sorted(set(allowed_origins)))

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if not origin_is_allowed(origin, self.allowed_origins):
            response = JSONResponse({"detail": "MCP origin not allowed"}, status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def audit_tool_call(
    name: str,
    input_preview: dict,
    is_write_tool: bool,
    call: Callable[[], Awaitable],
):
    """Wrap an MCP tool call, emit structured ``mcp.tool`` event with timing.

    Reuses :func:`kb.observability.trace.emit` / :func:`emit_error` so events
    land in ``logs/kb_trace.jsonl`` alongside all other project trace events.
    """
    started = time.monotonic()
    try:
        result = await call()
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        emit(
            "mcp.tool",
            tool=name,
            input_preview=input_preview,
            latency_ms=latency_ms,
            is_write_tool=is_write_tool,
            success=True,
        )
        return result
    except BaseException as exc:
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        emit_error(
            "mcp.tool",
            exc,
            tool=name,
            input_preview=input_preview,
            latency_ms=latency_ms,
            is_write_tool=is_write_tool,
        )
        raise
