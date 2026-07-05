"""#3 tokenizer 重标:用 Qwen3-VL 真 tokenizer 测中/英 chunk 的真实 char/token 比,
对比 est_tokens 当前启发式(中 1.7 / 英 4.0),决定是否重标 BUDGETS。纯 CPU(只用 tokenizer)。"""
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir

MODEL = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
CJK = re.compile(r"[一-鿿]")


def is_zh(t):
    return len(CJK.findall(t)) / max(len(t), 1) > 0.2


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

docs = sorted(glob.glob(os.path.join(REPO, "parsed", "*")))
pick = [d for d in docs if os.path.isdir(d) and any(
    k in os.path.basename(d) for k in
    ["academic_paper", "financial_research_zh", "law__", "financial_report_en", "government", "policy"])]

agg = {"zh": [0, 0, 0], "en": [0, 0, 0]}   # [n_chunk, total_char, total_token]
per_type = {}
for d in pick:
    name = os.path.basename(d)
    dtype = name.split("__")[0]
    try:
        els = from_mineru_dir(d)
        res = Chunker().chunk(els, doc_id=name, lang="zh" if "_zh" in name else "en")
    except Exception as e:
        print("skip", name, repr(e)[:60]); continue
    for c in res.chunks:
        t = (c.text or "").strip()
        if len(t) < 20:
            continue
        n = len(tok.encode(t, add_special_tokens=False))
        lang = "zh" if is_zh(t) else "en"
        agg[lang][0] += 1; agg[lang][1] += len(t); agg[lang][2] += n
        pt = per_type.setdefault(dtype, {"zh": [0, 0, 0], "en": [0, 0, 0]})
        pt[lang][0] += 1; pt[lang][1] += len(t); pt[lang][2] += n

print("\n=== char/token 比(= est_tokens 的 divisor)===")
for lang, cur in (("zh", 1.7), ("en", 4.0)):
    n, ch, tk = agg[lang]
    if tk:
        r = ch / tk
        print(f"{lang}: n={n:5d} chunk  实测 char/token={r:.3f}  当前启发式={cur}  "
              f"偏差={'低估' if r > cur else '高估'} {abs(r-cur)/cur*100:.0f}%")

print("\n=== 按文档类型(char/token)===")
for dtype, d in sorted(per_type.items()):
    parts = []
    for lang in ("zh", "en"):
        n, ch, tk = d[lang]
        if tk and n >= 3:
            parts.append(f"{lang} {ch/tk:.2f}(n={n})")
    if parts:
        print(f"  {dtype:24} {'  '.join(parts)}")
