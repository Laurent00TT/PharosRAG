"""7.A 端到端 + 7.B 双层归因。在评估库副本上跑真系统(真 Retriever + 真 DeepSeek 生成),对 gold 每条算:

  程序化指标(不需裁判,本脚本直接算):
    retrieval_recall  golden chunk(可多)被召回的比例 / retrieval_full 是否全召回 / MRR(最佳名次)
    citation_recall   答案 [cite:n] 实际引用到 golden chunk 的比例
  主观指标(--judge 决定谁判):
    faithfulness / correctness —— `--judge deepseek` 用 DeepSeek 自判(同厂,有循环偏差);
    `--judge none` 不判,只产出 rows(含 answer + 喂进的 context + golden_answer),交给【Claude 子agent 裁判】去偏。

多跳:gold 用 golden_chunk_ids(列表);单跳=长度 1。retrieval/citation recall 按"命中比例"算。
模式 --mode single|agentic|both:single 闭管道单跳;agentic DeepSeek 驱动 query 改写多跳(=被测 agent,非裁判)。

用法(去偏全流程):
  PHAROS_EVAL_SRC=~/rag_eval_big PHAROS_EVAL_COLLECTION=evalbig \
    python eval/run_eval.py --mode both --judge none --gold eval/gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

from _common import JsonLLM, TextLLM, demo_user, load_env, make_retriever

HERE = os.path.dirname(os.path.abspath(__file__))

FAITH_SYS = (
    "你是严格的 RAG 忠实度裁判。判断【系统答案】中的每个事实论断是否都能由【提供的上下文】支持。"
    "只要存在任一论断在上下文里找不到依据(引入外部知识或编造)=> faithful=false。"
    "若系统答案是'信息不足'这类合理拒答,视为 faithful=true。只输出 JSON。"
)
CORRECT_SYS = (
    "你是严格的问答正确性裁判。判断【系统答案】是否在事实上正确回答了【问题】,以【参考答案】为准。"
    "允许措辞不同、更详尽或更简略;核心事实与参考答案一致且无矛盾 = correct=true。"
    "系统答'信息不足'但参考答案有实质内容 = correct=false。只输出 JSON。"
)
SUFFICIENCY_SYS = (
    "你在驱动一个检索 agent。给定【问题】和【已检索上下文】,判断上下文是否足以完整回答问题。"
    "若不足,给一个更可能命中缺失信息的【改写检索查询】(换措辞/补关键词/拆子问题),与问题同语言。只输出 JSON。"
)
DECOMPOSE_SYS = (
    "你在驱动一个检索 agent。把【问题】拆成 1-4 个【可独立检索的子问题】,每个针对一个实体/方面/文档。"
    "跨文档对比类问题必须拆成对每个对象各一个子问题(如'比较A和B的X'->['A的X','B的X']);"
    "单一事实问题无需拆,直接返回原问题作为唯一子问题。子问题与原问题同语言。只输出 JSON。"
)


def _ctx_block(contexts):
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(contexts, 1)) or "(无上下文)"


def _dedup(seq):
    """保序去重(R5.L):agentic/decompose 的 union_ids 是逐轮 append 含重复,去重后 best_rank/MRR 才与 single 的有序 hits 可比。"""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def judge_faithful(jllm, ctx_text, answer):
    return jllm.ask(FAITH_SYS, f"上下文:\n{ctx_text}\n\n系统答案:\n{answer}\n\n"
                               '输出 JSON:{"faithful": true/false, "reason": "..."}')


def judge_correct(jllm, query, golden, answer):
    return jllm.ask(CORRECT_SYS, f"问题:{query}\n参考答案:{golden}\n系统答案:{answer}\n\n"
                                 '输出 JSON:{"correct": true/false, "reason": "..."}')


def best_rank(gcids, retrieved):
    ranks = [i for i, c in enumerate(retrieved, 1) if c in set(gcids)]
    return min(ranks) if ranks else None


def run_single(retriever, gen, query, user, k, rerank, smart_tables=False):
    """--smart-tables:**失败驱动**表格补检(与 pharos smart-ask 同一策略与参数,generator.signals 同源)
    ——第一轮纯净;数值题拒答/部分拒答时带 kind=table 腿重问一轮(硬上限 1 次)。
    ⚠ 前置腿版本已被 88 题实测否决(表格 +0.25 但误伤 5 道散文题,散文 0.861->0.792),勿改回。
    重试时检索指标口径 = 主检索 ∪ 腿检索(去重保序)——系统实际交付给 LLM 的就是这个并集。"""
    hits = retriever.search(query, user, top_k=k, rerank=rerank)
    ids = [h.chunk_id for h in hits]
    ans = gen.answer(query, user, top_k=k, rerank=rerank)
    retried = retry_kept = False
    if smart_tables:
        from generator import DEFAULT_TABLE_LEG, is_refusal, looks_numeric
        if looks_numeric(query) and is_refusal(ans.text):
            retried = True
            leg = dict(DEFAULT_TABLE_LEG)
            lkw = {kk: leg[kk] for kk in ("doc_ids", "doc_type", "kind", "strategy", "rerank_top_n")
                   if leg.get(kk) is not None}
            ans2 = gen.answer(query, user, top_k=k, rerank=rerank, extra_legs=[leg])
            # 与 pharos service.py 同策略(改一处必须同步另一处):重试只采用**完整答出**的结果——
            # 部分回答会夹带错误缺失声明(忠实度 1.0->0.93 实测),保守保留第一轮诚实拒答。
            if not is_refusal(ans2.text):
                retry_kept = True
                ans = ans2
                ids += [h.chunk_id for h in retriever.search(query, user, top_k=leg.get("top_k"),
                                                             rerank=bool(leg.get("rerank")), **lkw)]
                ids = _dedup(ids)
    ctx_text = next((m.content for m in ans.raw_messages if m.role == "user"), "")
    return ans.text, [c.chunk_id for c in ans.citations], ids, ctx_text, retried, retry_kept


def run_agentic(retriever, tllm, jllm, query, user, k, rounds):
    from generator import PromptBuilder
    collected, seen, union_ids, queries = [], set(), [], [query]
    cur_q, n_rounds = query, 0
    for rnd in range(rounds):
        n_rounds = rnd + 1
        for r in retriever.search_with_context(cur_q, user, top_k=k):
            hit, ctx = r["hit"], r["context"]
            text = (ctx.text if ctx is not None else hit.text) or ""
            union_ids.append(hit.chunk_id)
            if hit.chunk_id in seen or not text.strip():
                continue
            seen.add(hit.chunk_id)
            dm = (hit.payload or {}).get("doc_meta") or {}
            collected.append({"chunk_id": hit.chunk_id, "text": text, "source": dm.get("title") or hit.doc_id})
        if rnd == rounds - 1:
            break
        dec = jllm.ask(SUFFICIENCY_SYS, f"问题:{query}\n已检索上下文:\n{_ctx_block(collected)}\n\n"
                                        '输出 JSON:{"sufficient": true/false, "refined_query": "..."}')
        if dec.get("sufficient") or not (dec.get("refined_query") or "").strip():
            break
        cur_q = dec["refined_query"].strip()
        queries.append(cur_q)
    messages = PromptBuilder().build(query, [{"text": c["text"], "source": c["source"]} for c in collected])
    answer = tllm.complete(messages)
    cited = sorted({int(n) for n in re.findall(r"\[cite:(\d+)\]", answer)})
    cited_ids = [collected[n - 1]["chunk_id"] for n in cited if 1 <= n <= len(collected)]
    ctx_text = next((m.content for m in messages if m.role == "user"), "")
    return answer, cited_ids, _dedup(union_ids), ctx_text, n_rounds, queries


def run_decompose(retriever, tllm, jllm, query, user, k, max_subs=4, cap=14):
    """B2+B3:把 query 拆成子问题 -> 各自检索 -> **并集**(不替换、不丢任一跳)-> 合成。
    区别于 run_agentic 的'改写替换 query'(多跳会窄化丢另一跳)。跨文档对比尤其受益(各对象各检索一次)。"""
    from generator import PromptBuilder
    dec = jllm.ask(DECOMPOSE_SYS, f"问题:{query}\n输出 JSON:{{\"sub_queries\": [\"...\"]}}")
    subs = [s for s in (dec.get("sub_queries") or []) if (s or "").strip()][:max_subs] or [query]
    collected, seen, union_ids = [], set(), []
    for sq in subs:
        for r in retriever.search_with_context(sq, user, top_k=k):
            hit, ctx = r["hit"], r["context"]
            text = (ctx.text if ctx is not None else hit.text) or ""
            union_ids.append(hit.chunk_id)
            if hit.chunk_id in seen or not text.strip():
                continue
            seen.add(hit.chunk_id)
            dm = (hit.payload or {}).get("doc_meta") or {}
            collected.append({"chunk_id": hit.chunk_id, "text": text, "source": dm.get("title") or hit.doc_id})
    collected = collected[:cap]                              # 封顶,防多子问题并集撑爆 context
    messages = PromptBuilder().build(query, [{"text": c["text"], "source": c["source"]} for c in collected])
    answer = tllm.complete(messages)
    cited = sorted({int(n) for n in re.findall(r"\[cite:(\d+)\]", answer)})
    cited_ids = [collected[n - 1]["chunk_id"] for n in cited if 1 <= n <= len(collected)]
    ctx_text = next((m.content for m in messages if m.role == "user"), "")
    return answer, cited_ids, _dedup(union_ids), ctx_text, len(subs), subs


def evaluate(mode, retriever, gen, tllm, jllm, gold, user, k, rerank, rounds, judge, smart_tables=False):
    rows = []
    for i, g in enumerate(gold, 1):
        gcids = g.get("golden_chunk_ids") or ([g["golden_chunk_id"]] if g.get("golden_chunk_id") else [])
        q, hop = g["query"], g.get("hop", "single")
        if mode == "single":
            ans, cited, retrieved, ctx_text, retried, retry_kept = run_single(
                retriever, gen, q, user, k, rerank, smart_tables=smart_tables)
            n_rounds, queries = 1, [q]
        elif mode == "decompose":
            ans, cited, retrieved, ctx_text, n_rounds, queries = run_decompose(retriever, tllm, jllm, q, user, k)
            retried = retry_kept = False
        else:
            ans, cited, retrieved, ctx_text, n_rounds, queries = run_agentic(retriever, tllm, jllm, q, user, k, rounds)
            retried = retry_kept = False
        rset, cset, gset = set(retrieved), set(cited), set(gcids)
        hit_frac = len(gset & rset) / len(gset) if gset else 0.0
        cit_frac = len(gset & cset) / len(gset) if gset else 0.0
        faith = correct = None
        fr = cr = ""
        if judge == "deepseek":
            jf, jc = judge_faithful(jllm, ctx_text, ans), judge_correct(jllm, q, g["golden_answer"], ans)
            faith, correct, fr, cr = bool(jf.get("faithful")), bool(jc.get("correct")), jf.get("reason", ""), jc.get("reason", "")
        rows.append({"query": q, "hop": hop, "golden_chunk_ids": gcids, "golden_doc_ids": g.get("golden_doc_ids", []),
                     "golden_answer": g["golden_answer"], "answer": ans, "ctx_text": ctx_text,
                     "retrieval_hit_frac": hit_frac, "retrieval_full": gset <= rset and bool(gset),
                     "rank": best_rank(gcids, retrieved), "citation_recall": cit_frac, "n_citations": len(cited),
                     "n_rounds": n_rounds, "queries": queries, "faithful": faith, "correct": correct,
                     "faith_reason": fr, "correct_reason": cr,
                     "retried": retried, "retry_kept": retry_kept})
        mk = lambda v: ("✓" if v else "·")
        jm = "" if judge == "none" else f" {('✓' if faith else '✗')}忠 {('✓' if correct else '✗')}对"
        print(f"  [{i:2d}/{len(gold)}] {hop[:5]:5s} 检索{hit_frac:.2f} 引用{cit_frac:.2f}{jm} r{n_rounds}  {q[:40]}", flush=True)
    return rows


def agg(rows, judge):
    n = len(rows)
    out = {"n": n,
           "retrieval_recall": sum(r["retrieval_hit_frac"] for r in rows) / n,
           "retrieval_full": sum(1 for r in rows if r["retrieval_full"]) / n,
           "mrr": sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / n,
           "citation_recall": sum(r["citation_recall"] for r in rows) / n,
           "avg_rounds": sum(r["n_rounds"] for r in rows) / n}
    if judge != "none":
        out["faithfulness"] = sum(1 for r in rows if r["faithful"]) / n
        out["correctness"] = sum(1 for r in rows if r["correct"]) / n
    return out


def show(tag, a):
    print(f"\n=== {tag} (n={a['n']}) ===")
    print(f"  检索召回 {a['retrieval_recall']:.3f}  全召回 {a['retrieval_full']:.3f}  MRR {a['mrr']:.3f}")
    print(f"  引用召回 {a['citation_recall']:.3f}   平均轮数 {a['avg_rounds']:.2f}")
    if "faithfulness" in a:
        print(f"  忠实度 {a['faithfulness']:.3f}   正确性 {a['correctness']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "agentic", "decompose", "both"], default="both")
    ap.add_argument("--judge", choices=["deepseek", "none"], default="deepseek",
                    help="none=只算程序化指标+产出rows给Claude子agent裁判(去偏);deepseek=DeepSeek自判(有循环偏差)")
    ap.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--smart-tables", action="store_true", dest="smart_tables",
                    help="数值题自动加表格补检腿(与 pharos smart-ask 同源:generator.signals)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_env()
    if not os.path.exists(args.gold):
        raise SystemExit(f"缺 gold:{args.gold}")
    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    n0 = len(gold)
    gold = [g for g in gold if g.get("golden_chunk_ids") or g.get("golden_chunk_id")]   # R5.L:空 golden 会污染检索/引用召回分母
    if len(gold) < n0:
        print(f"⚠ 跳过 {n0 - len(gold)} 条空 golden_chunk_ids 的 gold", flush=True)
    if args.limit:
        gold = gold[:args.limit]
    print(f"gold {len(gold)} 条;mode={args.mode} judge={args.judge} top_k={args.top_k} rerank={args.rerank} rounds={args.rounds}", flush=True)

    from generator import Generator
    retriever, _ = make_retriever(os.path.join(tempfile.gettempdir(), "rag_eval_run"))
    tllm, jllm = TextLLM(), JsonLLM()
    gen = Generator(retriever, tllm)
    user = demo_user()

    aggs = {}
    for m in (["single", "agentic"] if args.mode == "both" else [args.mode]):
        print(f"\n----- 跑 {m} -----", flush=True)
        rows = evaluate(m, retriever, gen, tllm, jllm, gold, user, args.top_k, args.rerank, args.rounds,
                        args.judge, smart_tables=args.smart_tables)
        aggs[m] = agg(rows, args.judge)
        show(m, aggs[m])
        with open(os.path.join(HERE, f"results_{m}.json"), "w", encoding="utf-8") as f:
            json.dump({"agg": aggs[m], "rows": rows}, f, ensure_ascii=False, indent=2)

    if args.mode == "both" and args.judge != "none":
        s, a = aggs["single"], aggs["agentic"]
        print("\n=== 双层归因(7.B):agentic − single ===")
        print(f"  正确性 {s['correctness']:.3f} -> {a['correctness']:.3f} (Δ{a['correctness']-s['correctness']:+.3f})")
        print(f"  检索召回 {s['retrieval_recall']:.3f} -> {a['retrieval_recall']:.3f} (Δ{a['retrieval_recall']-s['retrieval_recall']:+.3f})")
    print(f"\n结果 -> eval/results_*.json" + ("(judge=none:忠实/正确待 Claude 子agent 判)" if args.judge == "none" else ""), flush=True)


if __name__ == "__main__":
    main()
