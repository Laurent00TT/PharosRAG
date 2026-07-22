"""7.0 全自动 gold 构造:从 demo 索引(4 篇)采样文本 chunk,DeepSeek 据【每个 chunk】生成
(问题, golden 答案) -> gold.jsonl。**纯 CPU + API,无需 GPU**(只 scroll payload,不编码)。

为何"问题从单 chunk 生成":让 golden_chunk_id 无歧义 —— 问题就该被这段回答,citation/retrieval recall 可程序化判定,
免人工标注。代价是问题偏单跳事实型(测不出多跳),这正是 run_eval --mode agentic 双层归因要补的盲区。

用法:conda activate pharos && python eval/gen_gold.py [--per-doc 6]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from _common import JUDGE_MODEL, JsonLLM, copy_demo, load_env, scroll_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold.jsonl")

SKIP_KINDS = {"table", "image", "figure", "banner", "caption", "formula", "equation"}
MIN_CHARS = 150          # 太短的段落问不出有意义的事实题
MAX_SRC_CHARS = 1600     # 喂给生成器的原文上限(省 token,够提一个问题)

SYS = (
    "你是 RAG 评估数据构造器。给定一段文档原文,生成一个【该段落能明确、完整回答的具体事实型问题】"
    "和【仅依据该段落得出的简短准确答案(1-3 句)】。要求:问题自然、像真实用户会问的,不得出现"
    "'根据该段落/这段文字'之类指代原文的字样,也不要宽泛到该段落答不全;问题与答案的语言必须与原文一致"
    "(中文段落出中文,英文段落出英文)。只输出一个 JSON 对象。"
)


def pick_evenly(items: list, k: int) -> list:
    """从已排序 items 里等距挑 k 个(确定性,无随机),覆盖文档首中尾。"""
    n = len(items)
    if n <= k:
        return items
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=6, help="每篇采样多少 chunk(默认 6)")
    args = ap.parse_args()

    load_env()
    work = os.path.join(tempfile.gettempdir(), "rag_eval_gold")
    qpath, _ = copy_demo(work)
    payloads = scroll_chunks(qpath)
    print(f"扫到 {len(payloads)} 个 point", flush=True)

    by_doc: dict[str, list[dict]] = {}
    for p in payloads:
        kind = (p.get("kind") or "text").lower()
        text = (p.get("text") or "").strip()
        if kind in SKIP_KINDS or len(text) < MIN_CHARS:
            continue
        by_doc.setdefault(p.get("doc_id", ""), []).append(p)

    llm = JsonLLM(model=JUDGE_MODEL)
    rows = []
    for doc_id, chunks in sorted(by_doc.items()):
        chunks.sort(key=lambda p: (p.get("page_start", 0), p.get("chunk_id", "")))
        sample = pick_evenly(chunks, args.per_doc)
        title = ((sample[0].get("doc_meta") or {}).get("title")) or doc_id
        print(f"\n[{doc_id}] 合格 chunk {len(chunks)} -> 采样 {len(sample)}", flush=True)
        for p in sample:
            src = (p.get("text") or "").strip()[:MAX_SRC_CHARS]
            try:
                qa = llm.ask(SYS, f"文档标题:{title}\n原文段落:\n{src}\n\n"
                                  '输出 JSON:{"question": "...", "answer": "..."}')
            except Exception as e:
                print(f"  跳过 {p.get('chunk_id')}: {e}", flush=True)
                continue
            q, a = (qa.get("question") or "").strip(), (qa.get("answer") or "").strip()
            if len(q) < 5 or len(a) < 2:
                print(f"  跳过 {p.get('chunk_id')}: 生成的 Q/A 太短", flush=True)
                continue
            rows.append({"query": q, "golden_answer": a,
                         "golden_chunk_id": p.get("chunk_id"), "golden_doc_id": doc_id,
                         "doc_title": title, "source_text": src})
            print(f"  + {q[:60]}", flush=True)

    with open(GOLD, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n写出 {len(rows)} 条 -> {GOLD}", flush=True)


if __name__ == "__main__":
    main()
