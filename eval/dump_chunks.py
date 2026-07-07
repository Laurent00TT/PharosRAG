"""扫评估库,产出给【Claude 子 agent 造 gold】用的"单元文件"(eval/_units/*.json)+ manifest。
每个单元文件 = 一个子 agent 的输入,1:1 映射避免索引切片歧义。三类:

  single_NN.json   单跳:一批(~10)独立 chunk,每个让 agent 出 (问题,golden答案),golden_chunk_ids=[该块]
  doc_NN.json      单篇多跳:一篇文档的 ~8 个 chunk,让 agent 造需≥2块联合回答的题
  type_N.json      跨文档多跳:同类型 2 篇各 ~3 chunk,让 agent 造需跨篇对比/综合的题

chunk 原文只落到这些文件(子 agent 用 Read 读),不进编排者上下文。CPU,无需 GPU。
跑(需先 index_eval_corpus.py):
  PHAROS_EVAL_SRC=~/rag_eval_big PHAROS_EVAL_COLLECTION=evalbig python eval/dump_chunks.py
"""
from __future__ import annotations

import json
import os
import tempfile

from _common import EVAL_COLLECTION, EVAL_SRC, copy_demo, scroll_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(HERE, "_units")

SKIP_KINDS = {"table", "image", "figure", "banner", "caption", "formula", "equation"}
MIN_CHARS = 150
MAX_SRC = 1200
SINGLE_PER_DOC = 4          # 单跳每篇采样
SINGLE_BATCH = 10           # 单跳每个 agent 文件装几个
DOC_CHUNKS = 8              # 单篇多跳每篇给 agent 几个 chunk
CROSS_DOCS_PER_TYPE = 2     # 跨文档每类型选几篇
CROSS_CHUNKS = 3            # 跨文档每篇给几个 chunk


def pick_evenly(items: list, k: int) -> list:
    n = len(items)
    if n <= k:
        return items
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def main():
    # 与 gen_gold/run_eval 同款锁规避:copytree 到临时副本再开,不抢 live 锁、不污染原库
    work = os.path.join(tempfile.gettempdir(), "rag_eval_dump")
    qpath, _ = copy_demo(work)
    payloads = scroll_chunks(qpath, EVAL_COLLECTION)
    print(f"扫到 {len(payloads)} 个 point(库={EVAL_SRC} coll={EVAL_COLLECTION})", flush=True)

    by_doc: dict[str, list[dict]] = {}
    for p in payloads:
        kind = (p.get("kind") or "text").lower()
        text = (p.get("text") or "").strip()
        if kind in SKIP_KINDS or len(text) < MIN_CHARS:
            continue
        by_doc.setdefault(p.get("doc_id", ""), []).append(p)
    for d in by_doc.values():
        d.sort(key=lambda p: (p.get("page_start", 0), p.get("chunk_id", "")))

    def meta(p):
        return {"chunk_id": p.get("chunk_id"), "doc_id": p.get("doc_id"),
                "doc_type": p.get("doc_type"), "title": (p.get("doc_meta") or {}).get("title") or p.get("doc_id"),
                "text": (p.get("text") or "").strip()[:MAX_SRC], "page": p.get("page_start", 0)}

    if os.path.exists(UNITS):
        import shutil
        shutil.rmtree(UNITS)
    os.makedirs(UNITS)
    manifest = {"singles": [], "docs": [], "types": []}

    # --- 单跳:每篇采样 -> 拉平 -> 分批 ---
    singles = []
    for doc_id, chunks in sorted(by_doc.items()):
        for p in pick_evenly(chunks, SINGLE_PER_DOC):
            singles.append(meta(p))
    for i in range(0, len(singles), SINGLE_BATCH):
        path = os.path.join(UNITS, f"single_{i // SINGLE_BATCH:02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "single", "items": singles[i:i + SINGLE_BATCH]}, f, ensure_ascii=False, indent=1)
        manifest["singles"].append(path)

    # --- 单篇多跳:每篇一个文件 ---
    for doc_id, chunks in sorted(by_doc.items()):
        if len(chunks) < 3:
            continue
        sel = pick_evenly(chunks, DOC_CHUNKS)
        m = meta(sel[0])
        path = os.path.join(UNITS, f"doc_{len(manifest['docs']):02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "multi_intra", "doc_id": doc_id, "doc_type": m["doc_type"], "title": m["title"],
                       "chunks": [{"id": meta(p)["chunk_id"], "text": meta(p)["text"]} for p in sel]},
                      f, ensure_ascii=False, indent=1)
        manifest["docs"].append(path)

    # --- 跨文档多跳:同类型 2 篇 ---
    by_type: dict[str, list[str]] = {}
    for doc_id, chunks in by_doc.items():
        dt = meta(chunks[0])["doc_type"]
        by_type.setdefault(dt, []).append(doc_id)
    for dt, doc_ids in sorted(by_type.items()):
        doc_ids = sorted(doc_ids)[:CROSS_DOCS_PER_TYPE]
        if len(doc_ids) < 2:
            continue
        docs = []
        for doc_id in doc_ids:
            sel = pick_evenly(by_doc[doc_id], CROSS_CHUNKS)
            m = meta(sel[0])
            docs.append({"doc_id": doc_id, "title": m["title"],
                         "chunks": [{"id": meta(p)["chunk_id"], "text": meta(p)["text"]} for p in sel]})
        path = os.path.join(UNITS, f"type_{len(manifest['types'])}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "multi_cross", "doc_type": dt, "docs": docs}, f, ensure_ascii=False, indent=1)
        manifest["types"].append(path)

    with open(os.path.join(UNITS, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"单跳 {len(singles)} 候选 -> {len(manifest['singles'])} 批文件;"
          f"单篇多跳 {len(manifest['docs'])} 篇;跨文档 {len(manifest['types'])} 组", flush=True)
    print(f"manifest -> {os.path.join(UNITS, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
