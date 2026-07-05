"""从 chunks.jsonl 选语义查询候选 chunk(实质散文段落,跨文档,中英混),dump candidates.jsonl 供 agent 生成问题。"""
import json
import os
import re

OUT = os.path.dirname(os.path.abspath(__file__))
CJK = re.compile(r"[一-鿿]")

chunks = [json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl"), encoding="utf-8")]

# 选实质散文段:150-900 char、不是纯数字/表格碎片(字母或汉字占比够)、每文档限量、中英都要
def is_prose(t):
    alpha = sum(1 for c in t if c.isalpha() or CJK.match(c))
    return alpha / max(len(t), 1) > 0.6

cand, per_doc = [], {}
for ch in chunks:
    t = ch["text"]
    if not (150 <= len(t) <= 900) or not is_prose(t):
        continue
    if per_doc.get(ch["doc_id"], 0) >= 4:
        continue
    cand.append(ch); per_doc[ch["doc_id"]] = per_doc.get(ch["doc_id"], 0) + 1

# 保证中文有份额
zh = [c for c in cand if c["lang"] == "zh"]
en = [c for c in cand if c["lang"] == "en"]
sel = en[:40] + zh[:15]
with open(os.path.join(OUT, "candidates.jsonl"), "w", encoding="utf-8") as f:
    for c in sel:
        f.write(json.dumps({"chunk_id": c["chunk_id"], "lang": c["lang"], "text": c["text"]}, ensure_ascii=False) + "\n")
print(f"语义候选: {len(sel)} (en {len([c for c in sel if c['lang']=='en'])}, zh {len([c for c in sel if c['lang']=='zh'])})")
