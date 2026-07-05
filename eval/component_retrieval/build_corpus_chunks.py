"""评估基准 step1(CPU):选语料 -> chunk -> dump chunks.jsonl + 程序化精确词查询 queries_exact.jsonl。
精确词查询:从 chunk 挖稀有精确串(法条/编号/金额/型号),df==1 的串作 query、含它的 chunk 作 golden
—— 这测 sparse 的精确命中(BM25 强项)。语义查询另由 agent 生成。"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir

CJK = re.compile(r"[一-鿿]")
PICK_TYPES = {"academic_paper": 3, "law": 3, "financial_report_en": 3,
              "financial_research_zh": 3, "government": 2}

# 精确串候选(法条/条款/金额/型号/长编号),都是"一字不差才对"的串
PATTERNS = [
    re.compile(r"\bSection\s+\d+(?:\.\d+)+\b"),
    re.compile(r"\bArticle\s+\d+\b"),
    re.compile(r"第\s*\d+\s*条"),
    re.compile(r"\b[A-Z][A-Za-z]{1,}-\d+(?:\.\d+)?\b"),       # GPT-4, BERT-3, v1.2 类
    re.compile(r"\$\s?\d[\d,]{3,}(?:\.\d+)?"),                # 金额 $1,234
    re.compile(r"\b\d{6,}\b"),                                 # 长编号/大数
]


def is_zh(t):
    return len(CJK.findall(t)) / max(len(t), 1) > 0.2


# 1) 选语料 + chunk
docs = sorted(glob.glob(os.path.join(REPO, "parsed", "*")))
picked, counts = [], {}
for d in docs:
    if not os.path.isdir(d):
        continue
    dtype = os.path.basename(d).split("__")[0]
    if dtype in PICK_TYPES and counts.get(dtype, 0) < PICK_TYPES[dtype]:
        picked.append(d); counts[dtype] = counts.get(dtype, 0) + 1

chunks = []
for d in picked:
    name = os.path.basename(d)
    try:
        els = from_mineru_dir(d)
        res = Chunker().chunk(els, doc_id=name, doc_type=name.split("__")[0],
                              lang="zh" if "_zh" in name else "en")
    except Exception as e:
        print("skip", name, repr(e)[:60]); continue
    for c in res.chunks:
        t = (c.text or "").strip()
        if len(t) < 30:
            continue
        chunks.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_type": c.doc_type or name.split("__")[0],
                       "lang": "zh" if is_zh(t) else "en", "section_id": c.section_id, "text": t})

with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for ch in chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")
print(f"语料: {len(picked)} 文档 -> {len(chunks)} chunk (中文 {sum(c['lang']=='zh' for c in chunks)})")

# 2) 精确词查询:挖 df==1 的稀有精确串
df = {}                                   # 串 -> set(chunk_id)
for ch in chunks:
    seen = set()
    for pat in PATTERNS:
        for m in pat.findall(ch["text"]):
            s = m if isinstance(m, str) else m[0]
            s = s.strip()
            if len(s) >= 3:
                seen.add(s)
    for s in seen:
        df.setdefault(s, set()).add(ch["chunk_id"])

exact = [{"query": s, "golden": list(ids)[0], "kind": "exact"}
         for s, ids in df.items() if len(ids) == 1]
# 每个 golden chunk 最多取 1 条精确查询(避免同 chunk 刷屏),并限总量
by_golden, dedup = {}, []
for q in exact:
    if by_golden.get(q["golden"], 0) < 1:
        dedup.append(q); by_golden[q["golden"]] = 1
exact = dedup[:80]

with open(os.path.join(OUT, "queries_exact.jsonl"), "w", encoding="utf-8") as f:
    for q in exact:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"精确词查询: {len(exact)} 条(df==1 稀有串)")
print("样例:", [q["query"] for q in exact[:12]])
