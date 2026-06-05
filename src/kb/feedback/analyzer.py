# src/kb/feedback/analyzer.py
from collections import Counter
from datetime import datetime, timedelta, timezone

from kb.feedback.db import UsageFeedbackDB


_UNDERUSED_THRESHOLD = 5  # min retrievals before flagging as underused


async def compute_weekly_stats(
    db: UsageFeedbackDB,
    as_of: datetime | None = None,
) -> dict:
    """Aggregate citations from the past 7 days."""
    now = as_of or datetime.now(timezone.utc).replace(tzinfo=None)
    week_ago = now - timedelta(days=7)
    recent = await db.list_recent_since(since=week_ago)

    retrieved_counter: Counter = Counter()
    used_counter: Counter = Counter()
    for r in recent:
        for entry in r["retrieved"]:
            retrieved_counter[(entry[0], entry[1])] += 1
        for entry in r["actually_used"]:
            used_counter[(entry[0], entry[1])] += 1

    underused = [
        {"doc_id": d, "page_num": p, "retrieved": retrieved_counter[(d, p)]}
        for (d, p), _ in retrieved_counter.most_common()
        if used_counter.get((d, p), 0) == 0
        and retrieved_counter[(d, p)] >= _UNDERUSED_THRESHOLD
    ]
    low_satisfaction = [
        {
            "query_id": r["query_id"],
            "query": r["query"],
            "comment": r.get("comment", ""),
            "expected_doc": r.get("expected_doc", ""),
        }
        for r in recent
        if r.get("was_helpful") is False
    ]

    return {
        "total_queries": len(recent),
        "most_used_pages": [
            {"doc_id": d, "page_num": p, "count": c}
            for (d, p), c in used_counter.most_common(10)
        ],
        "underused_pages": underused[:10],
        "low_satisfaction_queries": low_satisfaction[:10],
    }


def format_report(stats: dict) -> str:
    lines = [
        "# Weekly Knowledge Base Usage Report",
        "",
        f"Total queries: {stats['total_queries']}",
        "",
        "## Most-cited pages",
    ]
    if stats["most_used_pages"]:
        for entry in stats["most_used_pages"]:
            lines.append(
                f"- {entry['doc_id']} p{entry['page_num']}: {entry['count']} citations"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append(f"## Pages retrieved but never cited (>= {_UNDERUSED_THRESHOLD} retrievals)")
    if stats["underused_pages"]:
        for entry in stats["underused_pages"]:
            lines.append(
                f"- {entry['doc_id']} p{entry['page_num']}: "
                f"{entry['retrieved']} retrievals, 0 citations"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Low-satisfaction queries")
    if stats.get("low_satisfaction_queries"):
        for entry in stats["low_satisfaction_queries"]:
            detail = f"- {entry['query']}"
            if entry.get("comment"):
                detail += f" | comment: {entry['comment']}"
            if entry.get("expected_doc"):
                detail += f" | expected: {entry['expected_doc']}"
            lines.append(detail)
    else:
        lines.append("(none)")
    return "\n".join(lines)


def feedback_to_wiki_review_kind(was_helpful: bool | None, comment: str) -> str | None:
    if was_helpful is False:
        return "user_feedback"
    if "缺少来源" in comment or "没有引用" in comment:
        return "low_source_coverage"
    return None
