"""REST auth dependency — defense-in-depth layer.

After T1b, the AuthMiddleware (FastAPI middleware in app.py) is the
PRIMARY auth gate: it parses tokens, calls authenticate_token, sets
current_user ContextVar with try/finally reset.

This dependency keeps its name ``verify_api_key`` (so the 5 existing
routers don't need diff) but its semantics are now:
"assert that AuthMiddleware already authenticated this request".

If middleware was bypassed (e.g. a path mistakenly added to the
allowlist), this dependency raises 401 — belt-and-suspenders.
"""
from __future__ import annotations

from fastapi import HTTPException, status

from kb.auth.context import current_user


async def verify_api_key() -> None:
    """Defense-in-depth: ensure AuthMiddleware ran upstream.

    This dependency relies on the FastAPI AuthMiddleware (mounted in
    app.py) to have set ``current_user`` before any router handler.
    If current_user is None at this point, something is broken:
      - The path was accidentally added to the middleware allowlist
      - A test bypassed middleware via direct app.dependency_overrides
      - A bug in middleware made it skip setting the ContextVar

    All three cases warrant 401 rather than silently granting access.
    """
    if current_user.get() is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )
