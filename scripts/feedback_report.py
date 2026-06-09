# scripts/feedback_report.py
"""Generate weekly usage feedback report.

Usage:
  python scripts/feedback_report.py
  python scripts/feedback_report.py --output report.md
  python scripts/feedback_report.py --db-path ./feedback.db
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.config import Settings
from kb.feedback.analyzer import compute_weekly_stats, format_report
from kb.feedback.db import UsageFeedbackDB


async def main() -> int:
    parser = argparse.ArgumentParser(description="KB Weekly Usage Report")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to feedback SQLite file. Defaults to <qdrant_path>/usage_feedback.db",
    )
    parser.add_argument("--output", help="Path to write the report markdown")
    args = parser.parse_args()

    settings = Settings()
    db_path = args.db_path or f"{settings.qdrant_path}/usage_feedback.db"
    db = UsageFeedbackDB(db_url=f"sqlite+aiosqlite:///{db_path}")
    await db.init()

    stats = await compute_weekly_stats(db)
    report = format_report(stats)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
