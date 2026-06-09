"""Offline trace log analyzer.

Usage:
  python scripts/analyze_traces.py
  python scripts/analyze_traces.py --since 2026-05-09
  python scripts/analyze_traces.py --type search
  python scripts/analyze_traces.py --type ingest
  python scripts/analyze_traces.py --type errors
  python scripts/analyze_traces.py --log-dir /custom/path
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_traces(log_dir: Path, since: datetime | None = None) -> list[dict]:
    """Read kb_trace.jsonl and rotated backups, filter by `since` (UTC)."""
    main = log_dir / "kb_trace.jsonl"
    backups = sorted(
        [p for p in log_dir.glob("kb_trace.jsonl.*") if p.suffix.lstrip(".").isdigit()],
        key=lambda p: int(p.suffix.lstrip(".")),
    )
    files = ([main] if main.exists() else []) + backups

    events: list[dict] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since:
                    ts = _parse_ts(rec.get("ts", ""))
                    if ts and ts < since:
                        continue
                events.append(rec)
    return events


def _parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def group_by_trace(events: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        g[e.get("trace_id", "no-trace")].append(e)
    return g


def _p(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    idx = max(0, min(len(sorted_xs) - 1, int(len(sorted_xs) * q)))
    return sorted_xs[idx]


def _has(evs: list[dict], event: str) -> bool:
    return any(e.get("event") == event for e in evs)


def report_search(traces: dict[str, list[dict]]) -> None:
    search_traces = {tid: evs for tid, evs in traces.items() if _has(evs, "search.start")}
    if not search_traces:
        print("(no search traces)")
        return

    total_latencies: list[float] = []
    cache_hits = 0
    failed = 0
    stage_durations: dict[str, list[float]] = defaultdict(list)

    for evs in search_traces.values():
        for e in evs:
            ev = e.get("event")
            if ev == "search.end":
                total_latencies.append(e.get("total_duration_ms", 0))
                if e.get("cache_hit"):
                    cache_hits += 1
                if not e.get("success", True):
                    failed += 1
            dur = e.get("duration_ms")
            if dur is not None and ev in (
                "hyde.call", "multi_query.call", "rerank.call",
                "enrich.end", "rrf.fused",
            ):
                stage_durations[ev].append(dur)
            # retrieve.variant has no duration_ms (broken into embed_ms + search_ms); count separately
            if ev == "retrieve.variant":
                t = (e.get("embed_ms", 0) or 0) + (e.get("search_ms", 0) or 0)
                stage_durations["retrieve.variant"].append(t)

    n = len(search_traces)
    print(f"=== Search traces: {n} ===")
    if total_latencies:
        print(f"  Total latency:   median={statistics.median(total_latencies):.0f}ms  "
              f"p95={_p(total_latencies, 0.95):.0f}ms  "
              f"max={max(total_latencies):.0f}ms")
    print(f"  Cache hit rate:  {cache_hits}/{n} ({100*cache_hits/max(n,1):.0f}%)")
    if failed:
        print(f"  Failed:          {failed}")
    print()
    print("  Stage breakdown (median ms):")
    for stage in ("hyde.call", "multi_query.call", "retrieve.variant", "rrf.fused", "enrich.end", "rerank.call"):
        durs = stage_durations.get(stage, [])
        if durs:
            print(f"    {stage:22s}  median={statistics.median(durs):8.1f}  "
                  f"p95={_p(durs, 0.95):.0f}  n={len(durs)}")


def report_agent(traces: dict[str, list[dict]]) -> None:
    agent_traces = {tid: evs for tid, evs in traces.items() if _has(evs, "agent.start")}
    if not agent_traces:
        print("(no agent traces)")
        return

    iterations: list[int] = []
    citations: list[int] = []
    images: list[int] = []
    thinking_chars: list[int] = []
    content_chars: list[int] = []
    iter_durations: list[float] = []

    for evs in agent_traces.values():
        for e in evs:
            ev = e.get("event")
            if ev == "agent.iteration":
                iter_durations.append(e.get("duration_ms", 0))
                if e.get("thinking_chars") is not None:
                    thinking_chars.append(e["thinking_chars"])
                if e.get("content_chars") is not None:
                    content_chars.append(e["content_chars"])
            if ev == "agent.final":
                iterations.append(e.get("iterations", 0))
                citations.append(e.get("citations_count", 0))
            if ev == "agent.multimodal_attached":
                images.append(e.get("images_count", 0))

    n = len(agent_traces)
    print(f"=== Agent traces: {n} ===")
    if iterations:
        print(f"  Iterations:      median={statistics.median(iterations):.1f}  "
              f"max={max(iterations)}")
    if citations:
        print(f"  Citations:       median={statistics.median(citations):.1f}  "
              f"max={max(citations)}")
    if iter_durations:
        print(f"  Per-iter latency: median={statistics.median(iter_durations):.0f}ms  "
              f"p95={_p(iter_durations, 0.95):.0f}ms")
    if thinking_chars and content_chars:
        ratio = sum(thinking_chars) / max(sum(content_chars), 1)
        print(f"  Thinking/content char ratio:  {ratio:.2f}x  "
              f"(agent should be ~3x; HyDE/MQ/Description should be 0)")
    if images:
        print(f"  Images attached: total={sum(images)}, "
              f"median per turn={statistics.median(images):.0f}")


def report_ingest(traces: dict[str, list[dict]]) -> None:
    ingest_traces = {tid: evs for tid, evs in traces.items() if _has(evs, "ingest.start")}
    if not ingest_traces:
        print("(no ingest traces)")
        return

    docs_processed = 0
    docs_skipped = 0
    pages_total = 0
    parse_durations: list[float] = []
    embed_text_durations: list[float] = []
    embed_vision_durations: list[float] = []
    describe_durations: list[float] = []
    json_parse_failures = 0

    for evs in ingest_traces.values():
        for e in evs:
            ev = e.get("event")
            if ev == "ingest.end":
                if e.get("skipped"):
                    docs_skipped += 1
                else:
                    docs_processed += 1
                pages_total += e.get("pages_processed", 0)
            if ev == "parse.mineru":
                parse_durations.append(e.get("duration_ms", 0))
            if ev == "embed.text":
                embed_text_durations.append(e.get("duration_ms", 0))
            if ev == "embed.vision":
                embed_vision_durations.append(e.get("duration_ms", 0))
            if ev == "describe.page":
                describe_durations.append(e.get("duration_ms", 0))
                if not e.get("json_parse_ok", True):
                    json_parse_failures += 1

    print(f"=== Ingest traces: processed={docs_processed}, skipped={docs_skipped}, total pages={pages_total} ===")
    for label, durs in (
        ("Parse (MinerU)",   parse_durations),
        ("Embed text",       embed_text_durations),
        ("Embed vision",     embed_vision_durations),
        ("Describe page",    describe_durations),
    ):
        if durs:
            print(f"  {label:18s}  median={statistics.median(durs):.0f}ms  "
                  f"p95={_p(durs, 0.95):.0f}ms  n={len(durs)}")
    if json_parse_failures:
        print(f"  Description JSON parse failures: {json_parse_failures}")


def report_errors(events: list[dict]) -> None:
    errors = [e for e in events if e.get("level") == "ERROR"]
    if not errors:
        print("(no errors)")
        return
    print(f"=== Errors: {len(errors)} ===")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        by_type[e.get("error_type", "?")].append(e)
    for et, lst in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {et:30s} × {len(lst)}")
        sample = lst[0]
        print(f"    sample event={sample.get('event')}  msg={sample.get('error_msg', '')[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--since", help="ISO date or datetime (UTC)")
    parser.add_argument(
        "--type",
        choices=["all", "search", "agent", "ingest", "errors"],
        default="all",
    )
    args = parser.parse_args()

    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Bad --since value: {args.since}")
            sys.exit(1)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log dir not found: {log_dir}")
        sys.exit(1)

    events = load_traces(log_dir, since=since)
    print(f"Loaded {len(events)} events from {log_dir}\n")

    traces = group_by_trace(events)
    if args.type in ("all", "search"):
        report_search(traces)
        print()
    if args.type in ("all", "agent"):
        report_agent(traces)
        print()
    if args.type in ("all", "ingest"):
        report_ingest(traces)
        print()
    if args.type in ("all", "errors"):
        report_errors(events)
        print()


if __name__ == "__main__":
    main()
