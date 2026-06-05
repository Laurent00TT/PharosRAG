"""Structured event tracing.

Each HTTP request gets a trace_id from the FastAPI middleware. Every emit()
event automatically carries that trace_id (via contextvars) into
logs/kb_trace.jsonl. asyncio.gather / asyncio.to_thread propagate contextvars
out of the box, so we don't need to forward the id manually.

Never raises — any failure on the observability path is swallowed (see emit/timed).
"""
import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)
_logger = logging.getLogger("kb.trace")
_logger.addHandler(logging.NullHandler())  # default silent so import-only doesn't write
_logger.propagate = False
_initialized = False


def setup_tracing(
    log_dir: str = "./logs",
    max_mb: int = 50,
    backup_count: int = 5,
) -> None:
    """Configure RotatingFileHandler. Idempotent.

    Call from FastAPI lifespan or scripts/ingest.py main(). Tests don't call this —
    they use a custom in-memory capture fixture.
    """
    global _initialized
    if _initialized:
        return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        f"{log_dir}/kb_trace.jsonl",
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    for h in list(_logger.handlers):
        if isinstance(h, logging.NullHandler):
            _logger.removeHandler(h)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _initialized = True


def new_trace() -> str:
    """Generate and set a new trace_id for the current async context."""
    tid = uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_var.get() or "no-trace"


def set_trace_id(tid: str) -> None:
    """Manually set trace_id (e.g. accepting X-Trace-Id from upstream caller)."""
    _trace_id_var.set(tid)


def emit(event, **fields) -> None:
    """Emit one event. trace_id auto-injected.

    Two call forms:
      emit("agent.iteration", round=1, ...)              # legacy
      emit(AgentIterationEvent(round=1, ...))            # typed (preferred)

    The typed form catches field-name typos at edit time. Both produce
    identical JSONL output, so analyze_traces.py is unaffected.

    Failure-safe: catches any exception so logging never affects business path.
    """
    try:
        # Lazy import to avoid circular deps
        from kb.observability.events import TraceEvent
        if isinstance(event, TraceEvent):
            event_name, fields = event.to_emit_args()
        else:
            event_name = event

        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "trace_id": get_trace_id(),
            "event": event_name,
            **fields,
        }
        # T1b I-5: auto-inject user_id from ContextVar when middleware
        # set one. Lazy import keeps trace.py independent of auth at
        # module load (some scripts use trace before importing auth).
        try:
            from kb.auth.context import current_user as _cu
            _user = _cu.get()
            if _user is not None:
                record.setdefault("user_id", _user.user_id)
        except Exception:
            # auth not available (early bootstrap / test isolation) → skip
            pass
        _logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        # Never propagate logging errors
        pass


def emit_error(event: str, exc: Exception, **fields) -> None:
    """Emit error event with exception type+truncated message."""
    emit(
        event,
        level="ERROR",
        error_type=type(exc).__name__,
        error_msg=str(exc)[:500],
        **fields,
    )


@contextmanager
def timed(event: str, **fields):
    """Context manager that emits event with duration_ms on exit.

    Emits even on exception (with success=False). Does NOT swallow the exception —
    re-raises after emit. Use emit_error inside the block if you need richer error info.
    """
    t0 = time.monotonic()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        emit(
            event,
            duration_ms=round((time.monotonic() - t0) * 1000, 1),
            success=success,
            **fields,
        )
