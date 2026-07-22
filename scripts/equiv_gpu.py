"""GPU 等价性端到端实测(阶段B go/no-go,docs/SCALE_OUT.md §5-B)。证明 local 建库向量 == remote 查询向量。

**分时跑**(避 OOM:inference 2×8B 占 4090 ~33GB,local Dense 再要 16GB,双加载爆显存):
  1) 起推理服务:   conda activate pharos && python -m embedder.inference_server   # 127.0.0.1:8900
  2) remote 存档:  python scripts/equiv_gpu.py --step remote   # remote encode/query/rerank -> /tmp/equiv_remote.npz
  3) 停推理服务(释放 GPU)
  4) local 对比:   python scripts/equiv_gpu.py --step local    # local vs 存档 -> E1/E3 GO/NO-GO

判据:encode/query 归一化 cosine>0.9999(容忍 bf16 local vs fp32 remote 微舍入,同 vLLM 迁移 gate 口径);rerank maxdiff<1e-3。
实测(2026-07):encode cosine=1.0000000 / maxdiff 2.98e-08 / query cosine=1.0 / rerank maxdiff 0.00 -> E1/E3 GO。
注:E2(混建混查 top-k)是**独立未做项**——E1 逐元素等价不蕴含 hybrid RRF+HNSW 下 top-k 序不变(近重复段分差可 <maxdiff),需另跑真实 query 建库/查询对比,见 §5-B。"""
import argparse

import numpy as np

from embedder.config import EmbedConfig

URL = "http://127.0.0.1:8900"
ARCHIVE = "/tmp/equiv_remote.npz"
D = 1024
TEXTS = [
    "Netflix 2015 年的营收是多少",
    "The company reported strong quarterly revenue growth in Q3.",
    "表格中第三季度的净利润与去年同期对比数据",
    "mixed 中英文 test 123 !@#$% special chars",
    "a",
]
QUERY = "Netflix revenue 2015"
DOCS = [
    "Netflix 2015 full-year revenue was $6.78 billion, up from prior year.",
    "cats are cute animals that sleep a lot",
    "The quarterly report shows steady subscriber growth.",
]


def remote_step():
    from embedder.remote import RemoteDense, RemoteReranker
    rd = RemoteDense(EmbedConfig(inference_url=URL, dense_dim=D))
    v = rd.encode_text(TEXTS)
    q = rd.encode_query(QUERY)
    rr = RemoteReranker(EmbedConfig(inference_url=URL))
    s = np.array(rr.score(QUERY, DOCS), dtype=np.float64)
    np.savez(ARCHIVE, v=v, q=q, s=s)
    print(f"remote 存档 -> {ARCHIVE}: encode{v.shape} query{q.shape} rerank={s.tolist()}")


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def local_step():
    from embedder.dense import Dense
    from embedder.rerank import Reranker
    d = Dense(EmbedConfig(dense_dim=D))
    v_local, q_local = d.encode_text(TEXTS), d.encode_query(QUERY)
    s_local = np.array(Reranker(EmbedConfig()).score(QUERY, DOCS), dtype=np.float64)
    a = np.load(ARCHIVE)
    v_r, q_r, s_r = a["v"], a["q"], a["s"]

    e1 = [_cos(v_local[i], v_r[i]) for i in range(len(TEXTS))]
    print("[norm] local :", [f"{float(np.linalg.norm(v_local[i])):.5f}" for i in range(len(TEXTS))])
    print("[norm] remote:", [f"{float(np.linalg.norm(v_r[i])):.5f}" for i in range(len(TEXTS))])
    print("[E1 encode] cosine:", [f"{c:.7f}" for c in e1], " maxdiff=", f"{float(np.abs(v_local - v_r).max()):.2e}")
    print(f"[E1 query] cosine={_cos(q_local, q_r):.7f}")
    e3_max = float(np.abs(s_local - s_r).max())
    print(f"[E3 rerank] local={s_local.tolist()} remote={s_r.tolist()} maxdiff={e3_max:.2e}")
    E1 = all(c > 0.9999 for c in e1) and _cos(q_local, q_r) > 0.9999
    print(f"\nE1 {'✅ GO' if E1 else '❌ NO-GO'}(encode 等价)   E3 {'✅ GO' if e3_max < 1e-3 else '❌ NO-GO'}(rerank 等价)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", choices=["remote", "local"], required=True)
    args = ap.parse_args()
    (remote_step if args.step == "remote" else local_step)()
