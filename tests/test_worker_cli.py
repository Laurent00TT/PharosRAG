import importlib.util
import inspect
from pathlib import Path


def _load_worker_module():
    script = Path(__file__).parent.parent / "scripts" / "worker.py"
    spec = importlib.util.spec_from_file_location("kb_worker_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_worker_cli_parser_defaults():
    worker = _load_worker_module()

    args = worker.build_parser().parse_args([])

    assert args.queue == "ingestion"
    assert args.poll_interval == 2.0
    assert args.lease_seconds == 60


def test_worker_cli_rejects_unknown_queue():
    worker = _load_worker_module()

    parser = worker.build_parser()

    try:
        parser.parse_args(["--queue", "emails"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parser should reject unsupported queues")


def test_worker_main_wires_nav_into_pipeline_factory():
    # Guard: worker.main MUST construct NavIndexStore + NavIndexBuilder and
    # pass them through to IngestionPipeline. The earlier version of this
    # file constructed `IngestionPipeline(settings)` bare, which silently
    # disabled auto-build for every doc ingested via the worker queue.
    worker = _load_worker_module()
    src = inspect.getsource(worker.main)
    assert "NavIndexStore(" in src, "worker.main must construct a NavIndexStore"
    assert "NavIndexBuilder(" in src, "worker.main must construct a NavIndexBuilder"
    assert "nav_store=nav_store" in src, "pipeline_factory must pass nav_store"
    assert "nav_builder=nav_builder" in src, "pipeline_factory must pass nav_builder"


def test_worker_main_respects_nav_enabled_gate():
    """Guard: worker must NOT initialize NavIndexStore unconditionally.
    Anti-regression for the config-drift bug where the server lifespan
    listened to settings.nav_enabled but the worker didn't — leaving
    users with 'nav disabled' config but a worker that kept writing
    nav db on every ingest."""
    worker = _load_worker_module()
    src = inspect.getsource(worker.main)
    assert "settings.nav_enabled" in src, (
        "worker.main must read settings.nav_enabled before wiring nav"
    )
    # Both flags must be checked — nav_auto_build_on_ingest gates the
    # ingestion-side auto-build separately from server-side feature exposure
    assert "nav_auto_build_on_ingest" in src, (
        "worker.main must also respect nav_auto_build_on_ingest"
    )
