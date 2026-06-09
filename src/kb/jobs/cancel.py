"""Cooperative cancellation signal for the ingestion pipeline.

Raised by the pipeline when it observes ``job.cancel_requested`` at a
page boundary. The executor must treat this as a *successful* cancel —
i.e., transition the item to ITEM_CANCELLED via ``store.cancel_item``,
NOT call ``store.fail_item`` (which would schedule a retry and pollute
the failed-items doctor view).
"""
from __future__ import annotations


class JobCancelled(Exception):
    """Pipeline observed cancel_requested=True at a page boundary and
    stopped. Carries an optional context message for the trace log."""
