"""T5-7 follow-up: backfill_nav_index maintenance probe guard."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx


@pytest.mark.asyncio
async def test_backfill_nav_refuses_when_maintenance_on(monkeypatch):
    """main() returns 3 when the server's maintenance probe returns {on: true}."""
    from scripts.backfill_nav_index import main

    # Build a mock response that looks like {"on": True}
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"on": True})

    # Build a mock AsyncClient whose get() returns the above response
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    # AsyncClient is used as an async context manager: `async with httpx.AsyncClient() as client:`
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client_cls = MagicMock(return_value=mock_client_cm)

    # Patch sys.argv so argparse gets the args we want (admin-key supplied,
    # no --skip-maintenance-check so the probe actually runs)
    monkeypatch.setattr(
        "sys.argv",
        ["backfill_nav_index.py", "--admin-key", "test-key", "--server-url", "http://testserver"],
    )

    with patch.object(httpx, "AsyncClient", mock_client_cls):
        result = await main()

    assert result == 3
    # Verify the probe called the right URL with the right header
    mock_client.get.assert_awaited_once_with(
        "http://testserver/admin/maintenance_mode",
        headers={"X-API-Key": "test-key"},
    )


@pytest.mark.asyncio
async def test_backfill_nav_exits_2_when_no_admin_key(monkeypatch):
    """main() returns 2 when neither --admin-key nor KB_ADMIN_KEY is set."""
    from scripts.backfill_nav_index import main

    monkeypatch.setattr("sys.argv", ["backfill_nav_index.py"])
    monkeypatch.delenv("KB_ADMIN_KEY", raising=False)

    result = await main()
    assert result == 2


@pytest.mark.asyncio
async def test_backfill_nav_skips_probe_when_flag_set(monkeypatch, tmp_path):
    """--skip-maintenance-check bypasses the probe entirely and proceeds."""
    from scripts.backfill_nav_index import main

    monkeypatch.setattr(
        "sys.argv",
        ["backfill_nav_index.py", "--skip-maintenance-check"],
    )

    # Patch downstream KB components so main() returns 0 without real infra
    mock_qdrant = MagicMock()
    mock_qdrant.ensure_collections = AsyncMock()

    mock_meta = MagicMock()
    mock_meta.init = AsyncMock()
    mock_meta.list_active_doc_ids = AsyncMock(return_value=[])
    mock_meta.aclose = AsyncMock()

    mock_nav_store = MagicMock()
    mock_nav_store.init = AsyncMock()
    mock_nav_store.close = AsyncMock()

    with patch("scripts.backfill_nav_index.QdrantStore", return_value=mock_qdrant), \
         patch("scripts.backfill_nav_index.MetadataDB", return_value=mock_meta), \
         patch("scripts.backfill_nav_index.NavIndexStore", return_value=mock_nav_store):
        result = await main()

    assert result == 0
