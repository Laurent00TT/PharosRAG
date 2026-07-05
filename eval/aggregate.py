"""合并【程序化指标(results_*.json)】+【Claude 双独立判分(verdicts.json)】-> 最终报表。

verdicts.json:{"pass1":[{id,faithful,correct}], "pass2":[...]}。id=f"{mode}#{i:03d}"(results 行序)。
自动纳入存在的模式(single/agentic/decompose)。双层归因在【两模式都判过的公共题】上做 **paired** 差(R5.H2)。

R5 修复:
- 指纹校验(M1):比对 _judge/fingerprint.json 与当前 results 的 query 指纹,不一致即报错拒绝出数(防"重跑 results 未重判"张冠李戴)。
- 缺判 loud(H2/MED):每模式/每 hop 打印 n_judged;判过的题为空标 "n/a" 而非 nan 混入。
- 双层归因用公共判过题集,分母相同才相减。
- 字段缺失(L)按 None 排除,不 bool(None)=False 静默判假。

跑:python eval/aggregate.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["single", "agentic", "decompose"]


def qhash(q):
    return hashlib.sha1((q or "").encode("utf-8")).hexdigest()[:12]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else math.nan


def fmt(x):
    return "  n/a " if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"


def load_verdicts():
    v = json.load(open(os.path.join(HERE, "verdicts.json"), encoding="utf-8"))
    def idx(p):
        return {x["id"]: x for x in v.get(p, [])}
    return idx("pass1"), idx("pass2")


def andflag(p1, p2, rid, key):
    """两 pass 都判过且都 true -> True;都判过但非都 true -> False;任一未判/字段缺 -> None(排除)。"""
    a, b = p1.get(rid), p2.get(rid)
    if a is None or b is None or a.get(key) is None or b.get(key) is None:
        return None
    return bool(a[key]) and bool(b[key])


def main():
    p1, p2 = load_verdicts()
    fp_path = os.path.join(HERE, "_judge", "fingerprint.json")
    fp = json.load(open(fp_path, encoding="utf-8")) if os.path.exists(fp_path) else None
    if fp is None:
        print("⚠ 无 _judge/fingerprint.json,跳过 results↔verdicts 一致性校验(建议重跑 dump_judge_units)")

    modes = {}
    for m in MODES:
        path = os.path.join(HERE, f"results_{m}.json")
        if not os.path.exists(path):
            continue
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        # 指纹校验:results 的 query 与判分时的一致才对齐
        if fp is not None:
            for i, r in enumerate(rows):
                rid = f"{m}#{i:03d}"
                if rid in fp and fp[rid] != qhash(r["query"]):
                    raise SystemExit(f"✗ 对齐失败:{rid} 的 query 指纹与判分时不符 —— results 已重跑但未重判,拒绝出数。请重跑 dump_judge_units + 重判。")
        for i, r in enumerate(rows):
            rid = f"{m}#{i:03d}"
            r["_id"] = rid
            r["_correct"] = andflag(p1, p2, rid, "correct")
            r["_faith"] = andflag(p1, p2, rid, "faithful")
        modes[m] = rows

    def agg(rows):
        n = len(rows)
        judged = [r for r in rows if r["_correct"] is not None]
        return {"n": n, "n_judged": len(judged),
                "retrieval_recall": mean(r["retrieval_hit_frac"] for r in rows),
                "retrieval_full": mean(1.0 if r["retrieval_full"] else 0.0 for r in rows),
                "mrr": mean((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows),
                "citation_recall": mean(r["citation_recall"] for r in rows),
                "avg_rounds": mean(r["n_rounds"] for r in rows),
                "correctness": mean(r["_correct"] for r in judged) if judged else None,
                "faithfulness": mean(r["_faith"] for r in judged) if judged else None}

    print("=" * 74)
    for m, rows in modes.items():
        a = agg(rows)
        miss = a["n"] - a["n_judged"]
        print(f"\n### {m}  (n={a['n']}, judged={a['n_judged']}" + (f", ⚠缺判 {miss}" if miss else "") + ")")
        print(f"  检索召回 {fmt(a['retrieval_recall'])}  全召回 {fmt(a['retrieval_full'])}  MRR {fmt(a['mrr'])}  引用召回 {fmt(a['citation_recall'])}  平均轮 {fmt(a['avg_rounds'])}")
        print(f"  忠实度 {fmt(a['faithfulness'])}   正确性 {fmt(a['correctness'])}   (judged {a['n_judged']}/{a['n']})")

    # 按 hop 拆(每模式各自 n_judged;空判标 n/a)
    print("\n" + "=" * 74 + "\n按 hop 拆(正确性AND / 检索召回 / n_judged):")
    hops = ["single", "multi_intra", "multi_cross"]
    head = "  ".join(f"{m:>26}" for m in modes)
    print(f"  {'hop':12} | {head}")
    for hop in hops:
        cells = []
        for m, rows in modes.items():
            rs = [r for r in rows if r.get("hop") == hop]
            judged = [r for r in rs if r["_correct"] is not None]
            corr = mean(r["_correct"] for r in judged) if judged else None
            rec = mean(r["retrieval_hit_frac"] for r in rs)
            cells.append(f"{fmt(corr)}/{fmt(rec)}/j{len(judged)}of{len(rs)}".rjust(26))
        print(f"  {hop:12} | " + "  ".join(cells))

    # 双层归因:paired,只在 single 与 other 都判过的【公共 index】上相减
    if "single" in modes:
        base = {int(r["_id"].split("#")[1]): r["_correct"] for r in modes["single"] if r["_correct"] is not None}
        print("\n" + "=" * 74 + "\n双层归因(paired,公共判过题集):")
        for m in ("agentic", "decompose"):
            if m not in modes:
                continue
            other = {int(r["_id"].split("#")[1]): r["_correct"] for r in modes[m] if r["_correct"] is not None}
            common = sorted(set(base) & set(other))
            if not common:
                print(f"  single vs {m}: 无公共判过题,跳过")
                continue
            sc = mean(base[i] for i in common)
            oc = mean(other[i] for i in common)
            print(f"  single vs {m:9}: 正确性 {sc:.3f} -> {oc:.3f}  (Δ{oc - sc:+.3f}, 公共 n={len(common)})")


if __name__ == "__main__":
    main()
