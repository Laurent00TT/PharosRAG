"""Group-by analysis on an eval_search.py snapshot.

Aggregates Recall@K / NDCG@K / MRR per `metadata.<field>` value so we can
see, e.g., whether desc channel helps "figure" queries more than "text"
queries.

Usage:
  python scripts/analyze_eval_groups.py snapshot1.json [snapshot2.json ...] \
      --group-by metadata.type_bucket
  python scripts/analyze_eval_groups.py s1.json s2.json --group-by metadata.domain

When given multiple snapshots, the script prints a side-by-side delta table.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _get_path(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def aggregate(snapshot: dict, group_field: str) -> dict[str, dict]:
    """Return {group_value: {recall, ndcg, mrr, n}}."""
    bucket = defaultdict(list)
    for q in snapshot.get("per_query", []):
        g = _get_path(q, group_field)
        bucket[g if g is not None else "<unknown>"].append(q)
    out = {}
    for g, qs in bucket.items():
        out[str(g)] = {
            "n": len(qs),
            "recall": statistics.mean(q["recall"] for q in qs),
            "ndcg": statistics.mean(q["ndcg"] for q in qs),
            "mrr": statistics.mean(q["mrr"] for q in qs),
        }
    return out


def overall(snapshot: dict) -> dict:
    qs = snapshot.get("per_query", [])
    return {
        "n": len(qs),
        "recall": statistics.mean(q["recall"] for q in qs) if qs else 0.0,
        "ndcg": statistics.mean(q["ndcg"] for q in qs) if qs else 0.0,
        "mrr": statistics.mean(q["mrr"] for q in qs) if qs else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots", nargs="+", type=Path)
    ap.add_argument("--group-by", default="metadata.type_bucket")
    args = ap.parse_args()

    snaps = []
    for p in args.snapshots:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_label"] = p.stem
        snaps.append(d)

    print(f"Snapshots: {[s['_label'] for s in snaps]}\n")

    # Overall summary
    print("=" * 80)
    print("OVERALL")
    print("=" * 80)
    print(f"  {'snapshot':<30s}  {'n':>5}  {'R@K':>6}  {'NDCG':>6}  {'MRR':>6}")
    for s in snaps:
        ov = overall(s)
        print(f"  {s['_label']:<30s}  {ov['n']:>5}  "
              f"{ov['recall']:>6.3f}  {ov['ndcg']:>6.3f}  {ov['mrr']:>6.3f}")

    # Group-by table
    print()
    print("=" * 80)
    print(f"GROUP BY {args.group_by}")
    print("=" * 80)
    aggs = [aggregate(s, args.group_by) for s in snaps]
    all_groups = sorted({g for a in aggs for g in a})
    header = f"  {'group':<28s}"
    for s in snaps:
        header += f"  {s['_label'][:14]:>14s}"
    if len(snaps) >= 2:
        header += f"  {'Δ NDCG':>8s}"
    print(header)

    for g in all_groups:
        n_per_snap = [a.get(g, {}).get("n", 0) for a in aggs]
        ndcg_per_snap = [a.get(g, {}).get("ndcg") for a in aggs]
        recall_per_snap = [a.get(g, {}).get("recall") for a in aggs]
        # Print n once (should be same across snapshots if same testset)
        n = max(n_per_snap)
        row = f"  {g[:28]:<28s}"
        for s, agg in zip(snaps, aggs):
            entry = agg.get(g, {})
            if entry:
                row += f"  R={entry['recall']:.3f}/N={entry['ndcg']:.3f}".rjust(15)
            else:
                row += "  --".rjust(15)
        if len(snaps) >= 2 and ndcg_per_snap[0] is not None and ndcg_per_snap[-1] is not None:
            delta = ndcg_per_snap[-1] - ndcg_per_snap[0]
            row += f"  {delta:+.3f}".rjust(8)
        row += f"   (n={n})"
        print(row)


if __name__ == "__main__":
    main()
