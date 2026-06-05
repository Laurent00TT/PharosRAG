from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPState:
    engine: Any = None
    meta_db: Any = None
    nav_store: Any = None
    nav_search: Any = None
    nav_builder: Any = None
    hybrid_navigator: Any = None
    settings: Any = None
    qdrant: Any = None
    image_store: Any = None
    source_store: Any = None
    audit_log: Any = None
    maintenance: Any = None    # T5: SQLite-backed cross-process flag


state = MCPState()


def bind_state(
    *,
    engine: Any,
    meta_db: Any,
    settings: Any,
    nav_store: Any = None,
    nav_search: Any = None,
    nav_builder: Any = None,
    hybrid_navigator: Any = None,
    qdrant: Any = None,
    image_store: Any = None,
    source_store: Any = None,
    audit_log: Any = None,
    maintenance: Any = None,    # T5
) -> None:
    state.engine = engine
    state.meta_db = meta_db
    state.settings = settings
    state.nav_store = nav_store
    state.nav_search = nav_search
    state.nav_builder = nav_builder
    state.hybrid_navigator = hybrid_navigator
    state.qdrant = qdrant
    state.image_store = image_store
    state.source_store = source_store
    state.audit_log = audit_log
    state.maintenance = maintenance


def reset() -> None:
    """Clear all bound references. Called from lifespan teardown so the
    next lifespan startup starts from a clean MCPState — important for
    test isolation when multiple TestClient instances are created in
    one pytest session.
    """
    state.engine = None
    state.meta_db = None
    state.settings = None
    state.nav_store = None
    state.nav_search = None
    state.nav_builder = None
    state.hybrid_navigator = None
    state.qdrant = None
    state.image_store = None
    state.source_store = None
    state.audit_log = None
    state.maintenance = None
