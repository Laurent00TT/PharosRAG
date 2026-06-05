"""BruteForceTracker — in-memory sliding window counting auth failures per IP.

Aggregates ``auth.failed`` events into one ``auth.brute_force`` best-effort
audit row to avoid letting a port scan flood the audit_log table.

Memory only, process-local: a restart clears all counters. Acceptable trade
-- attacker gets (threshold - 1) free attempts before the next window's
audit row fires. Persistence would put every miss through SQLite on the
auth hot path with minimal additional security signal.

source_ip=None is a no-op (stdio MCP, unit tests). The auth.failed trace
is still emitted upstream by ``authenticate_token`` — only the brute-force
*audit* is skipped.

Memory bound (P2-5 review fix): the ``_events`` map keyed by source_ip
grows once per unique IP and was previously never reclaimed — an
attacker rotating through many IPs would slowly leak. Two complementary
defenses now apply:
  1. Each ``record_failure`` opportunistically prunes IPs whose
     newest event has fallen outside the window (their deque is empty
     after eviction). O(1) amortized in the common case (one IP active),
     bounded sweep when many are stale.
  2. ``max_ips`` (default 4096) is a hard cap. When exceeded, the
     least-recently-active IPs are dropped — defenders care most about
     the bursty current-attacker pattern, not historical visitors.
Both defenses preserve the "fire one audit per threshold" contract
because eviction only touches IPs below threshold.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Iterable

from kb.observability.trace import emit_error

logger = logging.getLogger(__name__)


# How often we sweep stale IP entries from the map. Sweeping every call
# would be wasteful; doing it every Nth call amortizes the cost.
_SWEEP_EVERY_N_CALLS = 64


class BruteForceTracker:
    def __init__(
        self,
        audit_log,
        *,
        threshold: int = 10,
        window_s: int = 300,
        max_ips: int = 4096,
    ) -> None:
        self._audit_log = audit_log
        self._threshold = threshold
        self._window_s = window_s
        self._max_ips = max_ips
        self._events: dict[str, deque[tuple[float, str]]] = defaultdict(deque)
        # Tracks the timestamp of each IP's most recent event for LRU eviction
        # when the map exceeds ``max_ips``. Cheaper than scanning the deques.
        self._last_seen: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._call_counter = 0

    async def record_failure(
        self, source_ip: str | None, *, reason: str,
    ) -> None:
        if source_ip is None:
            return  # A-5: stdio MCP / tests — no aggregation key
        now = time.monotonic()
        async with self._lock:
            self._call_counter += 1
            dq = self._events[source_ip]
            self._evict_old_locked(dq, now)
            dq.append((now, reason))
            self._last_seen[source_ip] = now

            # Opportunistic housekeeping every N calls — drops IPs whose
            # deque has aged out entirely. Bounded to ``max_ips`` so it's
            # O(N) worst case but only fires once per 64 calls.
            if self._call_counter % _SWEEP_EVERY_N_CALLS == 0:
                self._sweep_stale_locked(now)

            # Hard cap: if we somehow blow past max_ips (heavy mixed
            # attack), drop the least-recently-active entries until we're
            # back under the cap. This keeps memory bounded regardless of
            # how fast the attacker rotates IPs.
            if len(self._events) > self._max_ips:
                self._enforce_cap_locked()

            if len(dq) < self._threshold:
                return
            reasons = self._reason_counts(dq)
            count = len(dq)
            dq.clear()
            # Clearing the deque leaves the IP key in place; it will be
            # swept on the next housekeeping pass if no further events
            # arrive. Keep last_seen as the latest activity so an
            # immediate follow-up failure restarts cleanly.
        try:
            await self._audit_log.write(
                "auth.brute_force",
                user_id=None,
                target_kind="system",
                target_id=None,
                payload={
                    "source_ip": source_ip,
                    "count": count,
                    "window_s": self._window_s,
                    "reasons": reasons,
                },
            )
        except Exception as exc:
            emit_error(
                "brute_force_tracker.audit_write_failed", exc,
                source_ip=source_ip, count=count,
            )

    def _evict_old_locked(self, dq: deque, now: float) -> None:
        cutoff = now - self._window_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _sweep_stale_locked(self, now: float) -> None:
        """Drop IPs whose last_seen has fallen outside the window. Their
        deque is by definition either empty or only contains entries
        that will be evicted on next access, so reclaiming the key now
        is safe. Called under the lock."""
        cutoff = now - self._window_s
        stale_ips = [
            ip for ip in list(self._events)
            if self._last_seen.get(ip, 0) < cutoff
        ]
        for ip in stale_ips:
            self._events.pop(ip, None)
            self._last_seen.pop(ip, None)

    def _enforce_cap_locked(self) -> None:
        """Hard cap: when the map exceeds ``max_ips``, drop the least-
        recently-active entries. Called under the lock."""
        # Sort by last_seen ascending — oldest first.
        over_by = len(self._events) - self._max_ips
        if over_by <= 0:
            return
        oldest = sorted(self._last_seen.items(), key=lambda kv: kv[1])[:over_by]
        for ip, _ in oldest:
            self._events.pop(ip, None)
            self._last_seen.pop(ip, None)

    @staticmethod
    def _reason_counts(events: Iterable[tuple[float, str]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, reason in events:
            out[reason] = out.get(reason, 0) + 1
        return out
