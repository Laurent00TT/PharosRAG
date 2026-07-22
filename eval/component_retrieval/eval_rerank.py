"""验证 rerank:hybrid 召回 top-N → Qwen3-VL-Reranker 重排 → 对比 MRR(分 exact/semantic)。
复用已 index 的 eval qdrant + chunks。main() 保护(GPU 习惯)。"""
import json
import os

EVAL = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from qdrant_client import QdrantClient, models

    from embedder.config import EmbedConfig
    from embedder.dense import Dense
    from embedder.rerank import Reranker
    from embedder.sparse import query_sparse

    chunks = {c["chunk_id"]: c["text"]
              for c in (json.loads(l) for l in open(os.path.join(EVAL, "chunks.jsonl"), encoding="utf-8"))}
    queries = [json.loads(l) for fn in ("queries_exact.jsonl", "queries_semantic.jsonl")
               for l in open(os.path.join(EVAL, fn), encoding="utf-8")]
    print(f"queries: {len(queries)}", flush=True)

    client = QdrantClient(path=os.path.join(EVAL, "qdrant"))
    dense = Dense(EmbedConfig(dense_dim=1024))
    rer = Reranker()
    N = 50

    def hybrid_ids(qtext, n):
        qd = dense.encode_query(qtext)
        qs = query_sparse(qtext)
        pf = [models.Prefetch(query=qd.tolist(), using="dense", limit=n)]
        if qs is not None:
            pf.append(models.Prefetch(query=qs, using="bm25", limit=n))
        res = client.query_points("eval", prefetch=pf, query=models.FusionQuery(fusion=models.Fusion.RRF),
                                  limit=n, with_payload=True).points
        return [p.payload["chunk_id"] for p in res]

    def rank_of(g, ids):
        for i, c in enumerate(ids, 1):
            if c == g:
                return i
        return None

    rows = []
    for k, q in enumerate(queries):
        ids = hybrid_ids(q["query"], N)
        scores = rer.score(q["query"], [chunks.get(cid, "") for cid in ids])
        rids = [ids[i] for i in sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)]
        rows.append({"kind": q["kind"], "hybrid": rank_of(q["golden"], ids), "rerank": rank_of(q["golden"], rids)})
        if k % 10 == 0:
            print(f"  {k}/{len(queries)}", flush=True)

    def mrr(key, kind=None):
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        return sum((1.0 / r[key] if r[key] else 0.0) for r in sel) / len(sel)

    def rec(key, kk, kind=None):
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        return sum(1.0 for r in sel if r[key] and r[key] <= kk) / len(sel)

    print("\n=== rerank 验证(hybrid 召回 top-50 重排)===", flush=True)
    print(f"{'':12}{'exact MRR':>11}{'sem MRR':>10}{'all MRR':>10}{'all R@5':>9}")
    for key in ("hybrid", "rerank"):
        print(f"{key:12}{mrr(key,'exact'):>11.3f}{mrr(key,'semantic'):>10.3f}{mrr(key):>10.3f}{rec(key,5):>9.2f}",
              flush=True)


if __name__ == "__main__":
    main()
