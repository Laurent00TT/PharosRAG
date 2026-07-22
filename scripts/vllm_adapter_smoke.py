#!/usr/bin/env python3
"""Phase 1 端到端冒烟:pharos(remote 后端)→ vLLM 适配器 → vLLM serve,打真库出真命中。
证明"应用层一字不改、PHAROS_INFERENCE_URL 指向适配器即从 FastAPI 切到 vLLM"(docs/VLLM_PLAN.md Phase 1)。

前置(同一 WSL invocation 内已起):vllm serve :8000 + 适配器 :8900。本脚本(pharos,0 GPU—remote 模式)
构造 Retriever(inference_url=适配器)打嵌入式真库。跑:
  PHAROS_INFERENCE_URL=http://localhost:8900 python scripts/vllm_adapter_smoke.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from embedder.config import EmbedConfig      # noqa: E402
from embedder.retrieve import Retriever      # noqa: E402
from embedder.types import User              # noqa: E402

ADAPTER = os.environ.get("PHAROS_INFERENCE_URL", "http://localhost:8900")
EMBED_DB = os.path.expanduser("~/rag_real")
QUERIES = ["What was IBM's total revenue growth?", "Netflix 2015 营收和净利润",
           "unsupervised cross-lingual transfer"]


def main() -> None:
    cfg = EmbedConfig(qdrant_path=os.path.join(EMBED_DB, "qdrant"),
                      sidecar_dir=os.path.join(EMBED_DB, "sidecar"),
                      collection="real", dense_dim=1024,
                      inference_url=ADAPTER)          # 非空 -> RemoteDense(不加载本地 GPU 模型)
    r = Retriever(cfg)
    assert type(r.dense).__name__ == "RemoteDense", f"期望 RemoteDense,得 {type(r.dense).__name__}"
    print(f"[smoke] Retriever.dense = {type(r.dense).__name__}  (inference_url={ADAPTER})", flush=True)
    user = User(tenant="demo", principals=[])
    ok = 0
    for q in QUERIES:
        hits = r.store.hybrid_search(r.dense.encode_query(q).tolist(),
                                     __import__("embedder.sparse", fromlist=["query_sparse"]).query_sparse(q, cfg.stopwords),
                                     user, top_k=3)
        cids = [h.chunk_id for h in hits]
        real = bool(cids and any(c for c in cids))
        ok += real
        print(f"[smoke] «{q[:40]}» -> {len(cids)} hits: {cids[:2]}{' …' if len(cids)>2 else ''}", flush=True)
    print(f"\n{'✅' if ok==len(QUERIES) else '❌'} Phase 1 端到端:{ok}/{len(QUERIES)} 查询经 vLLM 适配器出真命中"
          f"({'应用层未改,vLLM 后端可用' if ok==len(QUERIES) else '有查询空召回,查适配器/模板'})", flush=True)


if __name__ == "__main__":
    main()
