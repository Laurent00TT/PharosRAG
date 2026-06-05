"""Per-request current_user propagation.

ASGI middleware set this at request entry; FastAPI middleware does the
same for REST. Tool handlers / MCP tool bodies / trace.emit() call
``get_current_user()`` to read.

**Critical**: every set() MUST be paired with reset() in a try/finally
block. Without reset, streaming responses (SSE / MCP Streamable HTTP /
asyncio.gather background work) can leak one user's identity into
another's request handler. See spec invariant I-8.

ContextVar isolates per asyncio.Task automatically — sibling tasks get
independent context copies — so the "leak" risk is limited to:
  - Same Task crossing request boundaries (long-lived ASGI handler)
  - Code that calls set() without reset() (the bug we guard against)
"""
from __future__ import annotations

import contextvars
from typing import Optional

from kb.auth.users import User

current_user: contextvars.ContextVar[Optional[User]] = contextvars.ContextVar(
    "current_user", default=None,
)


def get_current_user() -> User:
    """Return the user bound to this request. Raises if no auth ran.

    Auth middleware MUST have set ``current_user`` upstream. If this
    raises, it means a request reached a handler without going through
    APIKeyEnforcer / authenticate_token — that's a routing bug, fail
    loud rather than return None and silently grant anonymous access.
    """
    user = current_user.get()
    if user is None:
        raise RuntimeError(
            "current_user not set — auth middleware was bypassed. "
            "Every request handler must run downstream of APIKeyEnforcer / "
            "authenticate_token. This is a routing wiring bug."
        )
    return user
