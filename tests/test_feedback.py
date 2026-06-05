# tests/test_feedback.py
import pytest
from kb.feedback.db import UsageFeedbackDB


@pytest.mark.asyncio
async def test_feedback_db_init_and_record(tmp_path):
    db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/fb.db")
    await db.init()
    query_id = await db.record(
        query="approval process",
        retrieved=[("doc1", 0), ("doc1", 1)],
        actually_used=[("doc1", 0)],
    )
    rows = await db.list_recent(limit=10)
    assert query_id
    assert len(rows) == 1
    assert rows[0]["query_id"] == query_id
    assert rows[0]["query"] == "approval process"
    assert rows[0]["retrieved"] == [["doc1", 0], ["doc1", 1]]
    assert rows[0]["actually_used"] == [["doc1", 0]]


@pytest.mark.asyncio
async def test_feedback_db_list_recent_orders_by_time(tmp_path):
    db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/fb2.db")
    await db.init()
    await db.record(query="q1", retrieved=[], actually_used=[])
    await db.record(query="q2", retrieved=[], actually_used=[])
    rows = await db.list_recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["query"] == "q2"
    assert rows[1]["query"] == "q1"


@pytest.mark.asyncio
async def test_feedback_db_empty_on_fresh_init(tmp_path):
    db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/fb3.db")
    await db.init()
    assert await db.list_recent() == []


@pytest.mark.asyncio
async def test_feedback_db_updates_user_feedback(tmp_path):
    db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/fb-feedback.db")
    await db.init()
    query_id = await db.record(query="q", retrieved=[], actually_used=[])

    updated = await db.update_feedback(
        query_id=query_id,
        was_helpful=False,
        comment="wrong section",
        expected_doc="manual.pdf",
    )

    rows = await db.list_recent(limit=10)
    assert updated is True
    assert rows[0]["was_helpful"] is False
    assert rows[0]["comment"] == "wrong section"
    assert rows[0]["expected_doc"] == "manual.pdf"


from datetime import datetime, timedelta
from kb.feedback.analyzer import (
    compute_weekly_stats,
    feedback_to_wiki_review_kind,
    format_report,
)


@pytest.mark.asyncio
async def test_weekly_stats_counts_citations(tmp_path):
    db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/fb4.db")
    await db.init()
    # 5 retrievals of (doc1, 0), 3 actually used
    for i in range(5):
        await db.record(
            query=f"q{i}",
            retrieved=[("doc1", 0), ("doc1", 5)],
            actually_used=[("doc1", 0)] if i < 3 else [],
        )
    stats = await compute_weekly_stats(db)
    assert stats["total_queries"] == 5
    most = {(d["doc_id"], d["page_num"]): d["count"] for d in stats["most_used_pages"]}
    assert most[("doc1", 0)] == 3


@pytest.mark.asyncio
async def test_weekly_stats_identifies_underused(tmp_path):
    db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{tmp_path}/fb5.db")
    await db.init()
    # 6 retrievals of (doc2, 9), 0 cited → underused
    for i in range(6):
        await db.record(query=f"q{i}", retrieved=[("doc2", 9)], actually_used=[])
    stats = await compute_weekly_stats(db)
    underused = {(d["doc_id"], d["page_num"]) for d in stats["underused_pages"]}
    assert ("doc2", 9) in underused


def test_format_report_includes_sections():
    stats = {
        "total_queries": 17,
        "most_used_pages": [{"doc_id": "doc1", "page_num": 0, "count": 5}],
        "underused_pages": [{"doc_id": "doc2", "page_num": 9, "retrieved": 6}],
        "low_satisfaction_queries": [
            {
                "query_id": "q1",
                "query": "approval?",
                "comment": "bad answer",
                "expected_doc": "policy.pdf",
            }
        ],
    }
    report = format_report(stats)
    assert "Total queries: 17" in report
    assert "Most-cited pages" in report
    assert "doc1" in report
    assert "doc2" in report
    assert "Low-satisfaction queries" in report
    assert "approval?" in report


def test_feedback_to_wiki_review_kind_unhelpful_returns_user_feedback():
    assert feedback_to_wiki_review_kind(False, "any comment") == "user_feedback"


def test_feedback_to_wiki_review_kind_helpful_but_low_coverage_chinese():
    assert feedback_to_wiki_review_kind(True, "回答里缺少来源") == "low_source_coverage"
    assert feedback_to_wiki_review_kind(True, "你这个回答没有引用") == "low_source_coverage"


def test_feedback_to_wiki_review_kind_neutral_returns_none():
    assert feedback_to_wiki_review_kind(True, "thanks") is None
    assert feedback_to_wiki_review_kind(None, "thanks") is None


def test_feedback_to_wiki_review_kind_unhelpful_takes_priority_over_substring():
    # Even if the comment would match the low-coverage substring, was_helpful=False
    # wins because the explicit thumbs-down is a stronger signal than substring hint.
    assert feedback_to_wiki_review_kind(False, "缺少来源") == "user_feedback"
