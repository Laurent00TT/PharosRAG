"""Verify the tool_server lifespan wires the navigation layer end-to-end.

The lifespan is responsible for:
  - Constructing NavIndexStore, NavSearchEngine, NavIndexBuilder,
    HybridNavigator from settings.
  - Binding them to ``app.state`` so the ingestion pipeline can reach
    them (when wired through a pipeline factory).
  - Binding ``nav_store``, ``nav_search``, ``hybrid_navigator`` to
    MCPState so the MCP nav tools work in live runs (no longer hit the
    "not initialized" branch).
  - NOT binding ``nav_builder`` to MCPState — builder is an
    ingestion-side concern.
"""
from __future__ import annotations

import asyncio
import importlib

from fastapi.testclient import TestClient


def _prepopulate_users_db(qdrant_path: str) -> None:
    """Pre-create an admin user at ``qdrant_path/kb_metadata.db``.

    The lifespan bootstrap check (T1b) raises RuntimeError when the users
    table is empty. Calling this before TestClient.__enter__ satisfies the
    check without touching prod data.
    """
    import pathlib
    from kb.auth.users import UsersStore

    pathlib.Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    db_url = f"sqlite+aiosqlite:///{qdrant_path}/kb_metadata.db"

    async def _init():
        store = UsersStore(db_url=db_url)
        await store.init()
        await store.create_user(username="admin0", role="admin")
        await store.aclose()

    asyncio.run(_init())


def test_lifespan_wires_nav_layer(tmp_path, monkeypatch):
    """Spin up the FastAPI app and assert every nav slot is populated."""
    # Settings reads env vars at construction time. Point the test at
    # tmp_path so the lifespan creates qdrant/nav/etc. inside an isolated
    # directory rather than the developer's project root.
    qdrant_path = str(tmp_path / "qdrant")
    monkeypatch.setenv("QDRANT_PATH", qdrant_path)
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("NAV_DB_PATH", str(tmp_path / "nav.db"))
    monkeypatch.setenv("IMAGE_STORAGE_PATH", str(tmp_path / "images"))

    # T1b: pre-create admin user so the lifespan bootstrap check passes.
    _prepopulate_users_db(qdrant_path)

    import kb.tool_server.app as app_module
    importlib.reload(app_module)

    from kb.nav.builder import NavIndexBuilder
    from kb.nav.hybrid import HybridNavigator
    from kb.nav.search import NavSearchEngine
    from kb.nav.store import NavIndexStore
    from kb.tool_server.mcp_state import state as mcp_state

    with TestClient(app_module.app) as client:
        assert client.get("/health").status_code == 200

        # app.state is populated with concrete instances of every nav class.
        assert isinstance(app_module.app.state.nav_store, NavIndexStore)
        assert isinstance(app_module.app.state.nav_search, NavSearchEngine)
        assert isinstance(app_module.app.state.nav_builder, NavIndexBuilder)
        assert isinstance(app_module.app.state.hybrid_navigator, HybridNavigator)

        # MCPState receives nav_store + nav_search + hybrid_navigator so the
        # MCP tools (kb_search_nav, kb_hybrid_search) no longer hit the
        # "not initialized" branch. nav_builder is intentionally NOT bound
        # — it's an ingestion-side concern, not consumed by MCP read tools.
        assert mcp_state.nav_store is app_module.app.state.nav_store
        assert mcp_state.nav_search is app_module.app.state.nav_search
        assert mcp_state.hybrid_navigator is app_module.app.state.hybrid_navigator


def test_lifespan_disabled_nav_leaves_state_none(tmp_path, monkeypatch):
    """When NAV_ENABLED=false, every nav slot must be None — MCP tools
    then return the "not initialized" message rather than blowing up.
    """
    qdrant_path = str(tmp_path / "qdrant")
    monkeypatch.setenv("QDRANT_PATH", qdrant_path)
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("NAV_DB_PATH", str(tmp_path / "nav.db"))
    monkeypatch.setenv("IMAGE_STORAGE_PATH", str(tmp_path / "images"))
    monkeypatch.setenv("NAV_ENABLED", "false")

    # T1b: pre-create admin user so the lifespan bootstrap check passes.
    _prepopulate_users_db(qdrant_path)

    import kb.tool_server.app as app_module
    importlib.reload(app_module)

    from kb.tool_server.mcp_state import state as mcp_state

    with TestClient(app_module.app) as client:
        assert client.get("/health").status_code == 200
        assert app_module.app.state.nav_store is None
        assert app_module.app.state.nav_search is None
        assert app_module.app.state.nav_builder is None
        assert app_module.app.state.hybrid_navigator is None
        assert mcp_state.nav_store is None
        assert mcp_state.nav_search is None
        assert mcp_state.hybrid_navigator is None
