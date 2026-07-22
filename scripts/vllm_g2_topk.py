#!/usr/bin/env python3
"""G2 混建混查:vLLM 编码查询 vs 官方编码查询,打**同一个真库**(官方建),比 top-k 是否翻转
(docs/VLLM_PLAN.md §4 G2)。这是决定 vLLM 能否当 Plan A 的**生产门** —— G1 的逐元素微漂(cosine 0.9997)
不蕴含 top-k 不变(HNSW 近似 + RRF rank 融合会放大近重复 chunk 的微差)。

**隔离设计**:vLLM 只出**查询向量**(它唯一的活);检索管道(dense+BM25+RRF+ACL)**只在 pharos 跑一次**,
两侧唯一差别 = dense 向量来源(官方 encode_query vs vLLM 预算)。同 sparse、同库、同 RRF → 纯隔离出编码器差异。

**分时**(避免 8B 双载 OOM,同探针):
  docker compose --env-file .env.compose stop inference          # 腾 GPU
  # 1) vLLM 出 88 条查询向量(vllm env;截维 4096→1024+renorm,对齐库维度)
  conda activate vllm && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python scripts/vllm_g2_topk.py --step vllm --out /tmp/vllm_g2
  # 2) pharos 出官方向量 + 两侧检索 + 比对(pharos;开嵌入式库 ~/rag_real)
  conda activate pharos && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python scripts/vllm_g2_topk.py --step retrieve --out /tmp/vllm_g2
  docker compose --env-file .env.compose start inference          # 还原栈
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_MODEL = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
EMBED_DB = os.path.expanduser("~/rag_real")            # 嵌入式库(官方建,7652 点;server 由它迁来)
COLLECTION = "real"
TENANT = "demo"                                         # 库 ACL:tenant=demo/visibility=public
DENSE_DIM = 1024                                        # 生产 MRL 截维;库就是 1024,查询向量必须同维
QUERY_INSTRUCTION = "Retrieve relevant documents for the query."
TOPK = 10
GOLD = os.path.join(REPO, "eval", "gold.jsonl")


def _load_queries() -> list[str]:
    qs = []
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line)["query"])
    return qs


def _mrl_np(full: np.ndarray, d: int = DENSE_DIM) -> np.ndarray:
    """4096 全维 → 前 d 维 + L2 renorm(复刻 embedder.remote.RemoteDense._mrl_np,对齐库维度)。"""
    v = full[:d]
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def step_vllm(out_dir: str) -> None:
    """vllm env:88 条查询 → vLLM pooling 向量 → 截 1024 + renorm → 存 npz。只依赖 vllm+numpy。"""
    import unicodedata

    from vllm import LLM
    queries = _load_queries()

    def sys_instr(instr: str) -> str:
        instr = instr.strip()
        return instr if unicodedata.category(instr[-1]).startswith("P") else instr + "."

    print(f"[vllm] 加载 {EMB_MODEL} (pooling, max_model_len=8192)…", flush=True)
    llm = LLM(model=EMB_MODEL, runner="pooling", dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, max_model_len=8192, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    prompts = []
    for q in queries:
        conv = [{"role": "system", "content": [{"type": "text", "text": sys_instr(QUERY_INSTRUCTION)}]},
                {"role": "user", "content": [{"type": "text", "text": q}]}]
        prompts.append({"prompt": tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)})
    outs = llm.embed(prompts)
    vecs = np.stack([_mrl_np(np.asarray(o.outputs.embedding, dtype=np.float32)) for o in outs])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "vllm_qvecs.npz"), vecs=vecs)   # 只存数值,读侧无 pickle
    print(f"[vllm] 存 {vecs.shape} 查询向量(1024 维,已 renorm)-> {out_dir}/vllm_qvecs.npz", flush=True)


def step_retrieve(out_dir: str) -> None:
    """pharos:官方向量 + vLLM 向量各打同一嵌入式库,比 top-k。检索管道只此一处,纯隔离编码器差异。"""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from embedder.config import EmbedConfig
    from embedder.dense import Dense
    from embedder.sparse import query_sparse
    from embedder.store import Store
    from embedder.types import User

    p = os.path.join(out_dir, "vllm_qvecs.npz")
    if not os.path.exists(p):
        raise SystemExit(f"未找到 vLLM 查询向量 {p};先在 vllm env 跑 --step vllm。")
    vllm_vecs = np.load(p)["vecs"]

    queries = _load_queries()
    if len(queries) != len(vllm_vecs):
        raise SystemExit(f"查询数 {len(queries)} != vLLM 向量数 {len(vllm_vecs)};重跑 --step vllm。")

    cfg = EmbedConfig(qdrant_path=os.path.join(EMBED_DB, "qdrant"),
                      sidecar_dir=os.path.join(EMBED_DB, "sidecar"),
                      collection=COLLECTION, dense_dim=DENSE_DIM)
    print(f"[retrieve] 开嵌入式库 {cfg.qdrant_path} + 官方编码器…", flush=True)
    store = Store(cfg)
    dense = Dense(cfg)                                  # local 官方 Qwen3VLEmbedder(GPU)
    user = User(tenant=TENANT, principals=[])

    def topk(dvec: np.ndarray, q: str) -> list[str]:
        sparse = query_sparse(q, cfg.stopwords)         # 两侧同一 sparse(CPU,jieba),隔离 dense 差异
        hits = store.hybrid_search(dvec.tolist(), sparse, user, top_k=TOPK)
        return [h.chunk_id for h in hits]

    exact_seq = exact_set = top1 = 0
    jaccard_sum = 0.0
    diverged = []
    for i, q in enumerate(queries):
        off = topk(dense.encode_query(q), q)            # 官方(库就是它建的,权威基准)
        vll = topk(vllm_vecs[i], q)                     # vLLM 预算向量
        if not off:                                     # 空召回(不该发生)跳过统计
            continue
        if off == vll:
            exact_seq += 1
        if set(off) == set(vll):
            exact_set += 1
        if off[0] == (vll[0] if vll else None):
            top1 += 1
        inter = len(set(off) & set(vll))
        jaccard_sum += inter / len(set(off) | set(vll))
        if off != vll:
            # 记首个分歧位次(top-k 中第一个不同的 rank)
            fd = next((r for r in range(min(len(off), len(vll))) if off[r] != vll[r]), min(len(off), len(vll)))
            diverged.append((i, fd, q[:44]))

    n = len(queries)
    print("\n=== G2 混建混查 top-%d(官方建库,对比 官方 vs vLLM 查询编码)===" % TOPK)
    print(f"  样本数            : {n}")
    print(f"  top-1 一致        : {top1}/{n}  ({100*top1/n:.1f}%)")
    print(f"  top-{TOPK} 集合一致 : {exact_set}/{n}  ({100*exact_set/n:.1f}%)   (顺序可不同)")
    print(f"  top-{TOPK} 序完全一致: {exact_seq}/{n}  ({100*exact_seq/n:.1f}%)")
    print(f"  平均 Jaccard@{TOPK} : {jaccard_sum/n:.4f}")
    print(f"\n  分歧样本数        : {len(diverged)};首分歧位次分布(rank: 数量):")
    from collections import Counter
    for rank, cnt in sorted(Counter(fd for _, fd, _ in diverged).items()):
        print(f"    rank {rank}: {cnt}")
    print("\n  分歧样本(前 8,含首分歧位次):")
    for i, fd, qt in diverged[:8]:
        print(f"    q#{i} 首分歧@rank{fd}  «{qt}»")

    # 判据(诚实,不拍死):top-1 稳 + 高 Jaccard = vLLM 可当 Plan A 的强证据;顺序完全一致才是最严
    j = jaccard_sum / n
    print("\n--- 判读 ---")
    if top1 == n and j >= 0.95:
        print("✅ G2 强通过:top-1 全一致 + Jaccard≥0.95 —— 微漂不翻转有效排序,vLLM 可当 Plan A(顺序小抖动多在尾部近重复,无害)。")
    elif top1 >= 0.95 * n and j >= 0.85:
        print("⚠ G2 基本通过:top-1 高度一致但有尾部抖动 —— 生产可接受(top-k 作证据池,尾序抖动不改答案);"
              "若要严格可对齐 transformers 版本或全库 vLLM 重建。")
    else:
        print("❌ G2 未过:top-1 或 Jaccard 掉太多 —— 微漂真翻转召回。现有库不宜被 vLLM 直接查,"
              "须对齐 transformers 版本重测,或全库用 vLLM 重建(见 VLLM_PLAN §8)。")


def main() -> None:
    ap = argparse.ArgumentParser(description="G2 混建混查 top-k 稳定性(vLLM vs 官方查询编码)")
    ap.add_argument("--step", required=True, choices=["vllm", "retrieve"])
    ap.add_argument("--out", default="/tmp/vllm_g2")
    args = ap.parse_args()
    (step_vllm if args.step == "vllm" else step_retrieve)(args.out)


if __name__ == "__main__":
    main()
