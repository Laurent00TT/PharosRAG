"""C-2 invariant: require_no_maintenance must be wired into every
write path. This test is a grep test — it reads source files and
asserts the dependency is referenced where expected. It does NOT
exercise the actual 503 behavior; the per-endpoint tests do.

If you add a new write endpoint, add it to EXPECTED_REST_WRITES
or EXPECTED_MCP_WRITES below. CI failure here is the canary that
backup consistency may have regressed."""
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


EXPECTED_REST_WRITES = {
    # (file, function name) — REST write endpoints that must be gated
    # by require_no_maintenance.
    ("kb/tool_server/documents_api.py", "delete_document"),
    ("kb/tool_server/documents_api.py", "restore_document"),    # T5-6
    ("kb/tool_server/ingestion_api.py", "create_ingestion_job"),
    ("kb/tool_server/ingestion_api.py", "cancel_ingestion_job"),    # v2 P0 #2
    ("kb/tool_server/ingestion_api.py", "retry_ingestion_job"),     # v2 P0 #2
}

EXPECTED_MCP_WRITES = {
    "kb_rebuild_nav_index",
    "kb_add_nav_alias",
    "kb_hide_nav_entry",
}


def test_rest_write_endpoints_have_require_no_maintenance():
    """Each REST write endpoint's function definition must be
    preceded by a dependencies=[...require_no_maintenance...] in the
    decorator above it."""
    for relpath, func in EXPECTED_REST_WRITES:
        src = (_PROJECT_ROOT / "src" / relpath).read_text(encoding="utf-8")
        idx = src.find(f"async def {func}")
        assert idx != -1, f"function {func} not found in {relpath}"
        prelude = src[max(0, idx - 400): idx]
        assert "require_no_maintenance" in prelude, (
            f"{relpath}::{func} is a write endpoint but the decorator "
            f"above it does NOT reference require_no_maintenance. "
            f"Add: dependencies=[Depends(require_no_maintenance)]"
        )


def test_mcp_write_tools_check_maintenance_inline():
    """Each MCP write tool body must invoke state.maintenance.is_on()
    near the top — MCP tools don't go through FastAPI Depends."""
    mcp = (_PROJECT_ROOT / "src" / "kb" / "tool_server" / "mcp_tools.py").read_text(encoding="utf-8")
    for tool in EXPECTED_MCP_WRITES:
        idx = mcp.find(f"async def {tool}")
        assert idx != -1, f"MCP tool {tool} not found in mcp_tools.py"
        body = mcp[idx: idx + 1000]
        assert "maintenance" in body and "is_on" in body, (
            f"MCP tool {tool} body does not invoke maintenance.is_on. "
            f"Add the inline T5 check at the top of the function."
        )


def test_claim_next_item_checks_maintenance():
    """Worker's hot path also has the gate."""
    src = (_PROJECT_ROOT / "src" / "kb" / "jobs" / "sqlite_store.py").read_text(encoding="utf-8")
    idx = src.find("async def claim_next_item")
    assert idx != -1
    body = src[idx: idx + 1500]
    assert "_maintenance_check" in body and "return None" in body
