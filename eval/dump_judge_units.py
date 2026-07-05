"""把 run_eval --judge none 的 results_{mode}.json 批量化成【Claude 子agent 裁判】的单元文件
eval/_judge/judge_NN.json(每文件一批,一个 agent 判一批)。每个判分项含 query/golden_answer/answer/contexts。

R5.H1:忠实度裁判必须看到生成器实际用到的【完整】context —— 旧版 CTX_CAP=5000 头截断使裁判只见中位 40% 段落,
把"依据在后段"的论断误判 unfaithful,系统性拉低忠实度。现默认不截(CTX_CAP 设很大,仅防病态超长)。
R5.M1:写 _judge/fingerprint.json {id: sha1(query)},aggregate 消费时校验 rows 指纹一致 -> 防"重跑 results 未重判"静默张冠李戴。

跑(需先 run_eval --judge none):python eval/dump_judge_units.py [single agentic decompose]
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE = os.path.join(HERE, "_judge")
BATCH = 8
CTX_CAP = 200000     # R5.H1:实为"不截断"(ctx_text 中位 ~16k),仅对病态超长兜底,防裁判漏看被引证据


def qhash(q: str) -> str:
    return hashlib.sha1((q or "").encode("utf-8")).hexdigest()[:12]


def main():
    modes = sys.argv[1:] or ["single", "agentic"]   # 可指定模式,如 `dump_judge_units.py single agentic decompose`
    items, fingerprint = [], {}
    for mode in modes:
        path = os.path.join(HERE, f"results_{mode}.json")
        if not os.path.exists(path):
            raise SystemExit(f"缺 {path};先跑 run_eval --judge none")
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        for i, r in enumerate(rows):
            rid = f"{mode}#{i:03d}"
            ctx = (r.get("ctx_text") or "")
            if len(ctx) > CTX_CAP:
                ctx = ctx[:CTX_CAP]     # 仅病态超长兜底(正常不触发)
            items.append({"id": rid, "mode": mode, "hop": r.get("hop", ""),
                          "query": r["query"], "golden_answer": r["golden_answer"],
                          "answer": r["answer"], "contexts": ctx})
            fingerprint[rid] = qhash(r["query"])

    if os.path.exists(JUDGE):
        shutil.rmtree(JUDGE)
    os.makedirs(JUDGE)
    manifest = []
    for i in range(0, len(items), BATCH):
        path = os.path.join(JUDGE, f"judge_{i // BATCH:02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"items": items[i:i + BATCH]}, f, ensure_ascii=False, indent=1)
        manifest.append(path)
    json.dump(manifest, open(os.path.join(JUDGE, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(fingerprint, open(os.path.join(JUDGE, "fingerprint.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(items)} 判分项({len(modes)} 模式)-> {len(manifest)} 批 -> {JUDGE}(含 fingerprint 防错位)")


if __name__ == "__main__":
    main()
