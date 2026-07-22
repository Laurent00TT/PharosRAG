"""gold 补表格题:从 eval 索引定向采样 kind=table 的块,DeepSeek 据 content_raw(表体)出数值型题。

**为什么单独一个脚本**:gen_gold.py 当年显式 SKIP table——那时表格块 text 只有 caption,问不出题。
表格检索文本增强(chunker 2bd97a5)后 content_raw 才真正可用作出题素材;而 gold 全散文的偏置
已实测造成"表格向改动只显示成本不显示收益"的测量盲区(pharos TESTING §3)。

**程序化 QC(防出题幻觉)**:golden_answer 里的关键数值必须逐字出现在表体里(逗号规范化后子串匹配),
不满足直接丢弃——出题人自己编的数,不配进考卷。

用法:conda activate pharos && python eval/gen_gold_tables.py [--per-doc 2 --cap 16]
产出:gold_tables.jsonl(单独留档)+ 追加进 gold.jsonl(原 72 题先备份 gold_backup_prose72.jsonl)。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile

from _common import EVAL_COLLECTION, JUDGE_MODEL, JsonLLM, copy_demo, load_env, scroll_chunks
from gen_gold import pick_evenly

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold.jsonl")
GOLD_TABLES = os.path.join(HERE, "gold_tables.jsonl")
BACKUP = os.path.join(HERE, "gold_backup_prose72.jsonl")

MIN_RAW = 120            # 太小的表问不出数值题
MAX_RAW = 3000           # 喂给出题器的表体上限

SYS = (
    "你是 RAG 评估数据构造器。给定一张表格(HTML)及其所在文档/小节,生成一个【必须依据表内数值才能"
    "回答的具体问题】和【简短准确答案】。要求:1) 答案必须包含表内的具体数值(含单位),且数值逐字来自"
    "表格;2) 问题里点明范围(哪一年/哪个分部/哪个指标),避免歧义;3) 问题自然、像真实用户会问的,"
    "不得出现'根据该表格'这类字样;4) 问题与答案的语言跟文档语言一致(中文研报出中文,英文财报出英文)。"
    "只输出一个 JSON 对象。"
)

_NUM_RE = re.compile(r"\d[\d,\.]*\d|\d")


def _norm_nums(s: str) -> str:
    return (s or "").replace(",", "").replace(" ", "")


def answer_grounded(answer: str, raw: str) -> bool:
    """答案中至少一个 >=2 位的数值(去逗号后)在表体中出现 —— 出题幻觉的程序化闸门。"""
    raw_n = _norm_nums(raw)
    nums = [_norm_nums(m) for m in _NUM_RE.findall(answer or "")]
    nums = [n for n in nums if len(n) >= 2]
    return any(n in raw_n for n in nums) if nums else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--cap", type=int, default=16)
    args = ap.parse_args()

    load_env()
    work = os.path.join(tempfile.gettempdir(), "rag_eval_gold_tables")
    qpath, _ = copy_demo(work)
    payloads = scroll_chunks(qpath, collection=EVAL_COLLECTION)

    by_doc: dict[str, list[dict]] = {}
    for p in payloads:
        if (p.get("kind") or "").lower() != "table":
            continue
        raw = (p.get("content_raw") or "").strip()
        if len(raw) < MIN_RAW:
            continue
        by_doc.setdefault(p.get("doc_id", ""), []).append(p)
    print(f"含合格表格的文档 {len(by_doc)} 篇 / 表格块 {sum(len(v) for v in by_doc.values())} 个", flush=True)

    llm = JsonLLM(model=JUDGE_MODEL)
    rows = []
    for doc_id, chunks in sorted(by_doc.items()):
        if len(rows) >= args.cap:
            break
        chunks.sort(key=lambda p: (p.get("page_start", 0), p.get("chunk_id", "")))
        title = ((chunks[0].get("doc_meta") or {}).get("title")) or doc_id
        for p in pick_evenly(chunks, args.per_doc):
            if len(rows) >= args.cap:
                break
            raw = (p.get("content_raw") or "").strip()[:MAX_RAW]
            sec = p.get("section_path") or ""
            try:
                qa = llm.ask(SYS, f"文档标题:{title}\n所在小节:{sec}\n表格 HTML:\n{raw}\n\n"
                                  '输出 JSON:{"question": "...", "answer": "..."}')
            except Exception as e:
                print(f"  跳过 {p.get('chunk_id')}: {e}", flush=True)
                continue
            q, a = (qa.get("question") or "").strip(), (qa.get("answer") or "").strip()
            if len(q) < 5 or len(a) < 2:
                print(f"  跳过 {p.get('chunk_id')}: Q/A 太短", flush=True)
                continue
            if not answer_grounded(a, raw):
                print(f"  ✗ QC 拒绝(答案数值不在表内){p.get('chunk_id')}: {a[:50]}", flush=True)
                continue
            rows.append({"query": q, "golden_answer": a,
                         "golden_chunk_ids": [p.get("chunk_id")], "golden_doc_ids": [doc_id],
                         "hop": "single", "doc_type": p.get("doc_type") or "", "asset": "table"})
            print(f"  + [{doc_id[:36]}] {q[:56]}", flush=True)

    with open(GOLD_TABLES, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not os.path.exists(BACKUP):
        shutil.copy(GOLD, BACKUP)                          # 原 72 题留档,只备份一次
    with open(GOLD, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n写出 {len(rows)} 条表格题 -> {GOLD_TABLES} 并追加进 {GOLD}(原件备份 {BACKUP})", flush=True)


if __name__ == "__main__":
    main()
