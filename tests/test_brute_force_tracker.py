"""BruteForceTracker — sliding-window failure counter triggering audit emit."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from kb.audit.brute_force import BruteForceTracker


def _fake_audit():
    log = AsyncMock()
    log.write = AsyncMock()
    return log


@pytest.mark.asyncio
async def test_under_threshold_emits_no_audit():
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=10, window_s=300)
    for _ in range(9):
        await tracker.record_failure("1.2.3.4", reason="unknown_key")
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_threshold_emits_audit_and_clears():
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=10, window_s=300)
    for _ in range(10):
        await tracker.record_failure("1.2.3.4", reason="unknown_key")
    audit.write.assert_called_once()
    args, kwargs = audit.write.call_args
    assert args[0] == "auth.brute_force"
    assert kwargs["user_id"] is None
    assert kwargs["target_kind"] == "system"
    assert kwargs["target_id"] is None
    payload = kwargs["payload"]
    assert payload["source_ip"] == "1.2.3.4"
    assert payload["count"] == 10
    assert "reasons" in payload
    audit.write.reset_mock()
    await tracker.record_failure("1.2.3.4", reason="unknown_key")
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_window_expires_old_failures():
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=3, window_s=1)
    for _ in range(2):
        await tracker.record_failure("1.2.3.4", reason="unknown_key")
    await asyncio.sleep(1.2)
    await tracker.record_failure("1.2.3.4", reason="unknown_key")
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_per_ip_isolation():
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=10, window_s=300)
    for _ in range(9):
        await tracker.record_failure("1.2.3.4", reason="unknown_key")
        await tracker.record_failure("5.6.7.8", reason="unknown_key")
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_record_failure_with_none_ip_is_noop():
    """source_ip=None (stdio MCP / tests) -> no aggregation key -> skip."""
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=2, window_s=300)
    for _ in range(10):
        await tracker.record_failure(None, reason="unknown_key")
    audit.write.assert_not_called()


@pytest.mark.asyncio
async def test_audit_write_failure_does_not_propagate():
    audit = _fake_audit()
    audit.write = AsyncMock(side_effect=RuntimeError("audit dead"))
    tracker = BruteForceTracker(audit, threshold=2, window_s=300)
    await tracker.record_failure("1.2.3.4", reason="unknown_key")
    await tracker.record_failure("1.2.3.4", reason="unknown_key")  # must not raise


@pytest.mark.asyncio
async def test_reasons_summary_in_payload():
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=4, window_s=300)
    for _ in range(2):
        await tracker.record_failure("1.2.3.4", reason="unknown_key")
    for _ in range(2):
        await tracker.record_failure("1.2.3.4", reason="missing_key")
    payload = audit.write.call_args.kwargs["payload"]
    assert payload["reasons"] == {"unknown_key": 2, "missing_key": 2}


@pytest.mark.asyncio
async def test_concurrent_record_failure_emits_once():
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=10, window_s=300)
    await asyncio.gather(*[
        tracker.record_failure("1.2.3.4", reason="unknown_key")
        for _ in range(10)
    ])
    assert audit.write.call_count == 1


# ── P2-5: memory bound ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_max_ips_cap_drops_oldest():
    """When map exceeds ``max_ips``, least-recently-active IPs are
    dropped — defenders care about current attackers, not historical."""
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=100, window_s=300, max_ips=3)
    # Sub-threshold hits from 5 IPs — only the 3 most recent should
    # survive the hard cap.
    await tracker.record_failure("1.0.0.1", reason="unknown_key")
    await tracker.record_failure("1.0.0.2", reason="unknown_key")
    await tracker.record_failure("1.0.0.3", reason="unknown_key")
    await tracker.record_failure("1.0.0.4", reason="unknown_key")
    await tracker.record_failure("1.0.0.5", reason="unknown_key")
    # Internal state inspection — direct attr access is fine in unit tests
    assert len(tracker._events) <= 3
    assert len(tracker._last_seen) <= 3
    # The most recently active IPs survive
    assert "1.0.0.5" in tracker._events
    assert "1.0.0.4" in tracker._events
    # The oldest one is dropped
    assert "1.0.0.1" not in tracker._events


@pytest.mark.asyncio
async def test_stale_ips_pruned_after_window_expires():
    """Once last_seen is past the window the IP key is reclaimed.
    Otherwise long-running servers leak one map entry per unique
    visitor forever."""
    audit = _fake_audit()
    # Tight window so we can verify in <1s. Use threshold=100 so a single
    # hit per IP cannot fire an audit.
    tracker = BruteForceTracker(audit, threshold=100, window_s=1)
    for i in range(70):
        await tracker.record_failure(f"10.0.0.{i}", reason="unknown_key")
    initial = len(tracker._events)
    assert initial == 70

    # Wait past the window
    await asyncio.sleep(1.2)

    # Force the periodic sweep — _SWEEP_EVERY_N_CALLS = 64. Need to cross
    # the next 64-boundary: 70 calls already, sweep at 64 (already past),
    # next at 128 — make 58 more on a fresh IP.
    for _ in range(58):
        await tracker.record_failure("10.0.0.999", reason="unknown_key")

    # The 70 historical IPs should be reclaimed (their last_seen is past
    # the window). Only the fresh "10.0.0.999" should remain.
    historical_survivors = [
        ip for ip in tracker._events
        if ip.startswith("10.0.0.") and ip != "10.0.0.999"
    ]
    assert historical_survivors == [], (
        f"expected 0 historical IPs after sweep, still have: {historical_survivors[:5]}..."
    )
    assert "10.0.0.999" in tracker._events


@pytest.mark.asyncio
async def test_cap_does_not_break_active_attacker_emit():
    """When cap fires it must drop the LEAST-RECENTLY-active IP, not
    a currently-attacking one. If an attacker hammers one IP while
    background traffic from other IPs floods the map, the attacker
    must still trigger their audit."""
    audit = _fake_audit()
    tracker = BruteForceTracker(audit, threshold=3, window_s=300, max_ips=2)
    # 8.8.8.8 is the long-idle visitor; we'll see if cap drops it.
    await tracker.record_failure("8.8.8.8", reason="unknown_key")
    # 9.9.9.9 is the active attacker — last_seen advances on each hit.
    await tracker.record_failure("9.9.9.9", reason="unknown_key")
    await tracker.record_failure("9.9.9.9", reason="unknown_key")
    # 7.7.7.7 arrives — map size = 3 > max_ips = 2 → enforce_cap drops
    # the oldest by last_seen, which is 8.8.8.8 (it was hit before
    # 9.9.9.9's two hits).
    await tracker.record_failure("7.7.7.7", reason="unknown_key")
    assert "9.9.9.9" in tracker._events
    assert len(tracker._events["9.9.9.9"]) == 2
    assert "8.8.8.8" not in tracker._events  # dropped: oldest last_seen
    # The 3rd hit on 9.9.9.9 still fires audit
    await tracker.record_failure("9.9.9.9", reason="unknown_key")
    audit.write.assert_called_once()
