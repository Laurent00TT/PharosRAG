# src/kb/tool_server/app.py
import logging
import inspect
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles

from kb.config import Settings
from kb.feedback.db import UsageFeedbackDB
from kb.nav.builder import NavIndexBuilder
from kb.nav.hybrid import HybridNavigator
from kb.nav.search import NavSearchEngine
from kb.nav.store import NavIndexStore
from kb.observability import new_trace, set_trace_id, setup_tracing
from kb.search.engine import SearchEngine
from kb.storage.image_store import ImageStore
from kb.storage.kb_store import KnowledgeBaseStore
from kb.storage.metadata_db import MetadataDB
from kb.storage.source_store import SourceDocStore
from kb.storage.qdrant_store import QdrantStore
from kb.jobs.sqlite_store import SQLiteJobStore
from kb.tool_server.ask_api import router as ask_router
from kb.tool_server.audit_api import router as audit_router
from kb.tool_server.documents_api import router as documents_router
from kb.tool_server.feedback_api import router as feedback_router
from kb.tool_server.ingestion_api import router as ingestion_router
from kb.tool_server.mcp_api import mcp as mcp_server
from kb.auth.context import current_user
from kb.auth.middleware import APIKeyEnforcer, AuthError, authenticate_token
from kb.auth.users import UsersStore
from kb.tool_server.mcp_security import OriginValidator
from kb.tool_server.mcp_state import bind_state as bind_mcp_state, reset as reset_mcp_state
from kb.tool_server.search_api import router as search_router
from kb.tool_server.security import verify_api_key
from kb.tool_server.ui_api import router as ui_router
from kb.admin.maintenance import MaintenanceState
from kb.tool_server.admin_api import router as admin_router

logger = logging.getLogger(__name__)

settings = Settings()
qdrant: QdrantStore | None = None
meta_db: MetadataDB | None = None
engine: SearchEngine | None = None
feedback_db: UsageFeedbackDB | None = None
job_store: SQLiteJobStore | None = None
image_store: ImageStore | None = None
kb_store: KnowledgeBaseStore | None = None
nav_store: NavIndexStore | None = None
nav_search: NavSearchEngine | None = None
nav_builder: NavIndexBuilder | None = None
hybrid_navigator: HybridNavigator | None = None


def _reset_session_manager_run_latch(mcp_instance) -> None:
    """Reset the StreamableHTTPSessionManager 'run-once' latch.

    The MCP SDK's StreamableHTTPSessionManager guards against a second
    ``.run()`` call on the same instance via the ``_has_started`` flag
    (see ``mcp/server/streamable_http_manager.py``). That guard is
    correct for production where lifespan runs once, but it blocks
    pytest sessions where multiple ``TestClient(app)`` instances each
    trigger a fresh lifespan against the module-level FastMCP singleton.

    This helper resets the latch (and clears the cached task-group and
    server-instance dict for hygiene) so the next lifespan invocation
    can enter ``run()`` cleanly. It is resilient to upstream attribute
    renames — each reset is guarded by ``hasattr`` so future SDK
    revisions degrade to a no-op rather than crashing.
    """
    sm = getattr(mcp_instance, "_session_manager", None)
    if sm is None:
        # FastMCP lazy-inits ``_session_manager`` on first access. If the
        # cached attribute is absent, no run() was ever started — nothing
        # to reset. (Avoid triggering the lazy ``session_manager``
        # property here, which would needlessly construct an instance.)
        return
    if hasattr(sm, "_has_started"):
        sm._has_started = False
    if hasattr(sm, "_task_group"):
        sm._task_group = None
    server_instances = getattr(sm, "_server_instances", None)
    if isinstance(server_instances, dict):
        server_instances.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant, meta_db, engine, feedback_db, job_store, image_store, kb_store
    global nav_store, nav_search, nav_builder, hybrid_navigator
    app.state.settings = settings
    # Configure the trace file handler at startup (RotatingFileHandler writes logs/kb_trace.jsonl)
    setup_tracing(
        log_dir=settings.log_dir,
        max_mb=settings.log_max_mb,
        backup_count=settings.log_backup_count,
    )
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    qdrant = QdrantStore(settings=settings)
    await qdrant.ensure_collections()
    meta_db = MetadataDB(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    )
    await meta_db.init()
    users_db_url = f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    users_store = UsersStore(db_url=users_db_url)
    await users_store.init()

    # Bootstrap refuse-start (spec I-8 follow-up + user decision review-2):
    # If users table is empty after T1b deployment, no one can authenticate.
    # Fail loud at startup so admin sees the problem immediately, instead
    # of every request returning 401 with no clear cause.
    existing_users = await users_store.list_users(include_disabled=True)
    if not existing_users:
        raise RuntimeError(
            "FATAL: users table empty — no one can authenticate after T1b.\n"
            "Run:  python scripts/manage_users.py create <admin-username> --role admin\n"
            "Then restart the server."
        )
    if not any(u.role == "admin" and not u.is_disabled for u in existing_users):
        raise RuntimeError(
            "FATAL: no active admin user — destructive operations locked.\n"
            "Promote a user via: python scripts/manage_users.py set-role <user> --role admin"
        )

    app.state.users_store = users_store

    # T2: AuditLog + BruteForceTracker (after users_store, before router includes)
    from kb.audit.log import AuditLog
    from kb.audit.brute_force import BruteForceTracker

    audit_log = AuditLog(
        db_url=users_db_url,
        fallback_path=settings.audit_fallback_path,
    )
    await audit_log.init()
    app.state.audit_log = audit_log

    maintenance = MaintenanceState(db_url=users_db_url)
    await maintenance.init()
    app.state.maintenance = maintenance

    # Review P1 #6: cross-process cache invalidation. tool_server reads
    # the epoch on every search; worker bumps it after ingest/deprecate.
    # Shared kb_metadata.db is the rendez-vous.
    from kb.search.cache_epoch import CacheEpochStore
    cache_epoch = CacheEpochStore(db_url=users_db_url)
    await cache_epoch.init()
    app.state.cache_epoch = cache_epoch

    brute_force_tracker = BruteForceTracker(
        audit_log,
        threshold=settings.brute_force_threshold,
        window_s=settings.brute_force_window_s,
    )
    app.state.brute_force_tracker = brute_force_tracker
    feedback_db = UsageFeedbackDB(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/usage_feedback.db"
    )
    await feedback_db.init()
    app.state.feedback_db = feedback_db
    job_store = SQLiteJobStore(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/ingestion_jobs.db",
        maintenance_check=app.state.maintenance.is_on,
    )
    await job_store.init()
    app.state.job_store = job_store
    image_store = ImageStore(settings=settings)
    kb_store = KnowledgeBaseStore(
        qdrant=qdrant,
        meta_db=meta_db,
        image_store=image_store,
    )
    app.state.qdrant = qdrant
    app.state.meta_db = meta_db
    app.state.image_store = image_store
    app.state.source_store = SourceDocStore(settings=settings)
    app.state.kb_store = kb_store
    engine = SearchEngine(
        settings=settings, qdrant=qdrant, meta_db=meta_db,
        cache_epoch=cache_epoch,
    )
    app.state.engine = engine

    # Navigation layer (Phase 2-5 nav-only refactor). Gated by
    # settings.nav_enabled so disabling the feature short-circuits all
    # state binding — MCP nav tools then return "not initialized".
    if settings.nav_enabled:
        nav_store = NavIndexStore(settings.nav_db_path)
        await nav_store.init()
        # Pass an active-doc-id provider so nav search filters out entries
        # for deprecated docs even if the pipeline-level cleanup missed
        # one. Double safety; the lambda is async and cheap (a single
        # indexed SQL query).
        nav_search = NavSearchEngine(
            nav_store,
            active_doc_ids_provider=lambda: meta_db.list_active_doc_ids(),
        )
        nav_builder = NavIndexBuilder()
        hybrid_navigator = HybridNavigator(
            search_engine=engine,
            nav_search=nav_search,
            settings=settings,
        )
        app.state.nav_store = nav_store
        app.state.nav_search = nav_search
        app.state.nav_builder = nav_builder
        app.state.hybrid_navigator = hybrid_navigator
    else:
        nav_store = None
        nav_search = None
        nav_builder = None
        hybrid_navigator = None
        app.state.nav_store = None
        app.state.nav_search = None
        app.state.nav_builder = None
        app.state.hybrid_navigator = None

    # MCP server reads engine + meta_db + nav layer via module-level closure.
    # nav_builder IS now bound to MCPState so kb_rebuild_nav_index can run
    # a per-doc rebuild on demand (re-derives ParsedPage stubs from qdrant
    # text chunks, then auto_build_nav_for_doc with replace semantics).
    # T3-6: ``audit_log`` is bound into MCPState so MCP write-tool bodies
    # (kb_rebuild_nav_index / kb_add_nav_alias / kb_hide_nav_entry) can
    # call ``require_owner_or_admin(..., audit_log=state.audit_log, ...)``
    # without re-reading from request.app.state — those bodies run inside
    # the FastMCP closure and only have the module-level ``state`` handle.
    bind_mcp_state(
        engine=engine,
        meta_db=meta_db,
        settings=settings,
        nav_store=nav_store,
        nav_search=nav_search,
        nav_builder=nav_builder,
        hybrid_navigator=hybrid_navigator,
        qdrant=qdrant,
        image_store=image_store,
        source_store=app.state.source_store,
        audit_log=audit_log,
        maintenance=app.state.maintenance,    # T5
    )
    # FastMCP's streamable HTTP transport has its own lifespan (session
    # manager). When the MCP ASGI sub-app is mounted, FastAPI does NOT
    # automatically trigger its lifespan, so we wrap it here.
    # (The /mcp mount itself is registered at module load — see below the
    # FastAPI() instantiation — so it doesn't accumulate across multi-
    # TestClient pytest runs.)
    try:
        async with mcp_server.session_manager.run():
            yield
    finally:
        await users_store.aclose()
        if engine is not None:
            maybe_close = getattr(engine, "aclose", None)
            if maybe_close is not None:
                result = maybe_close()
                if inspect.isawaitable(result):
                    await result
        # nav_search / nav_builder / hybrid_navigator hold no resources —
        # only nav_store owns the SQLAlchemy engine that needs disposal.
        if nav_store is not None:
            await _maybe_aclose(nav_store, "close")
        if job_store is not None:
            await _maybe_aclose(job_store, "close")
        if feedback_db is not None:
            await _maybe_aclose(feedback_db, "aclose")
        if meta_db is not None:
            await _maybe_aclose(meta_db, "aclose")
        maintenance_inst = getattr(app.state, "maintenance", None)
        if maintenance_inst is not None:
            await maintenance_inst.aclose()
        # audit_log closed AFTER other stores so a future shutdown-time
        # audit emit (e.g. "shutdown" event from any of the above closes)
        # still has a live engine to write through. (Code review of 13121a7
        # flagged the earlier ordering as latent fragility.)
        await audit_log.aclose()
        if qdrant is not None:
            # QdrantClient (local-file mode) holds a file lock and won't
            # release it until close() is called; without this, restarting
            # the server within the same Python process (or running tests
            # with TestClient) can fail with "another instance is already
            # using this storage". For URL mode, this also releases the
            # HTTP connection pool.
            client = getattr(qdrant, "client", None)
            close_fn = getattr(client, "close", None) if client is not None else None
            if close_fn is not None:
                try:
                    close_fn()
                except Exception as exc:  # noqa: BLE001 — defensive teardown
                    logger.warning("QdrantStore close failed: %s", exc)
        # StreamableHTTPSessionManager refuses to be .run() twice on the
        # same instance — fine for production (lifespan runs once), but
        # blocks multi-TestClient pytest sessions where each TestClient
        # triggers a fresh lifespan against the module-level singleton.
        # Reset its run-once latch so the next lifespan can re-enter.
        _reset_session_manager_run_latch(mcp_server)
        # Reset module-level MCP state so a subsequent lifespan starts clean.
        reset_mcp_state()


app = FastAPI(title="Knowledge Base Tool Server", lifespan=lifespan)
app.state.settings = settings
app.include_router(ingestion_router, dependencies=[Depends(verify_api_key)])
app.include_router(search_router)
app.include_router(documents_router)
app.include_router(ask_router)
app.include_router(feedback_router)
app.include_router(audit_router)
app.include_router(admin_router)
app.include_router(ui_router)

# /mcp is mounted ONCE at module load to avoid route accumulation across
# multi-TestClient pytest runs (and the corresponding stale-store bug —
# Starlette routing is first-match, so a duplicate /mcp mount appended in
# a later lifespan would never win against the first one).
#
# Stack outside-in: APIKeyEnforcer (user-token) → OriginValidator → MCP transport.
# Hard cutover (I-7): APIKeyEnforcer is unconditional, no opt-in flag.
#
# The enforcer resolves UsersStore per-request via the provider lambda
# (reads `app.state.users_store` which lifespan binds at startup), so the
# mount works even though users_store doesn't exist at module-load time.
# Before lifespan runs, the provider returns None and the enforcer
# returns 503 — fail-closed.
_mcp_origins = [item.strip() for item in settings.mcp_allowed_origins.split(",") if item.strip()]
_mcp_inner = OriginValidator(mcp_server.streamable_http_app(), _mcp_origins)
_mcp_inner = APIKeyEnforcer(
    _mcp_inner,
    users_store_provider=lambda: getattr(app.state, "users_store", None),
    brute_force_tracker_provider=lambda: getattr(app.state, "brute_force_tracker", None),
)
app.mount("/mcp", _mcp_inner)


# Optional React/Vite workbench SPA. The workbench lives in the separate
# `navikb-workbench` repo and normally runs standalone (its dev server proxies
# /ui/api back here). If you build it and place the output at <repo>/web/dist,
# the core will also serve it at /ui. When absent (the default), the core runs
# headless — /ui/api stays available for any workbench client.
#
# /ui/api stays above this mount and remains protected by the REST auth
# middleware; the static files are public so the SPA can render its token
# login screen before any authenticated API call.
_UI_DIST_DIR = Path(__file__).resolve().parents[3] / "web" / "dist"
if _UI_DIST_DIR.is_dir():
    app.mount(
        "/ui",
        StaticFiles(directory=str(_UI_DIST_DIR), html=True),
        name="ui",
    )


# Paths that bypass auth (plan §0c)
_AUTH_ALLOWLIST = frozenset({
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/ui",
    "/ui/",
})


@app.middleware("http")
async def auth_middleware(request, call_next):
    """REST auth gate (Option B per plan §0d).

    /mcp/* is handled by ASGI APIKeyEnforcer on the mounted sub-app
    and is invisible to this FastAPI middleware (it sees the mount as
    a single endpoint that delegates downstream).
    """
    from starlette.responses import JSONResponse

    path = request.url.path

    # Allowlist: liveness + docs + /mcp sub-app (which has its own
    # APIKeyEnforcer). The /mcp check is exact-or-slash-prefix to avoid
    # accidentally bypassing future /mcp_admin or similar routes.
    if (
        path in _AUTH_ALLOWLIST
        or path == "/mcp"
        or path.startswith("/mcp/")
        or (path.startswith("/ui/") and not path.startswith("/ui/api/"))
    ):
        return await call_next(request)

    # Reuse the canonical extractor from kb.auth.middleware. FastAPI's
    # request.headers IS a starlette Headers instance, so the previous
    # local mirror was pure duplication.
    from kb.auth.middleware import _extract_token
    token = _extract_token(request.headers)
    users_store = getattr(request.app.state, "users_store", None)
    if users_store is None:
        return JSONResponse(
            {"detail": "users store not initialized; server misconfigured"},
            status_code=503,
        )

    tracker = getattr(request.app.state, "brute_force_tracker", None)
    try:
        user = await authenticate_token(
            token, users_store,
            source="rest",
            source_ip=(request.client.host if request.client else None),
            brute_force_tracker=tracker,
        )
    except AuthError as exc:
        return JSONResponse({"detail": exc.detail}, status_code=401)

    ctx_token = current_user.set(user)
    try:
        return await call_next(request)
    finally:
        current_user.reset(ctx_token)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Generate a trace_id per request (or accept upstream X-Trace-Id),
    and echo it back in the response headers."""
    incoming = request.headers.get("X-Trace-Id")
    if incoming:
        set_trace_id(incoming)
        tid = incoming
    else:
        tid = new_trace()
    response = await call_next(request)
    response.headers["X-Trace-Id"] = tid
    return response


async def _maybe_aclose(obj, method_name: str) -> None:
    maybe_close = getattr(obj, method_name, None)
    if maybe_close is None:
        return
    result = maybe_close()
    if inspect.isawaitable(result):
        await result

@app.get("/health")
async def health():
    return {"status": "ok"}
