"""评估 step3(GPU):#1 BM25 vs BGE-M3 vs dense 召回(分 exact/semantic),#2 dense+bm25 RRF 权重扫描。
指标 MRR + Recall@10。同样必须 main()+if __name__ 保护(BGE-M3 多进程重入坑)+ bge-m3 锁单卡。"""
import json
import os

EVAL = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from qdrant_client import QdrantClient, models

    from embedder.config import EmbedConfig
    from embedder.dense import Dense
    from embedder.sparse import query_sparse

    COLL = "eval"
    client = QdrantClient(path=os.path.join(EVAL, "qdrant"))

    queries = []
    for fn in ("queries_exact.jsonl", "queries_semantic.jsonl"):
        for l in open(os.path.join(EVAL, fn), encoding="utf-8"):
            queries.append(json.loads(l))
    print(f"queries: {len(queries)} (exact {sum(q['kind']=='exact' for q in queries)}, "
          f"semantic {sum(q['kind']=='semantic' for q in queries)})", flush=True)
    qtexts = [q["query"] for q in queries]

    # query 端三种编码
    dense = Dense(EmbedConfig(dense_dim=1024))
    qd = [dense.encode_query(t) for t in qtexts]
    qb = [query_sparse(t) for t in qtexts]
    from FlagEmbedding import BGEM3FlagModel
    m3 = BGEM3FlagModel(os.path.expanduser("~/models/bge-m3"), use_fp16=True, devices="cuda:0")
    m3out = m3.encode(qtexts, batch_size=16, max_length=512,
                      return_dense=False, return_sparse=True, return_colbert_vecs=False)

    def to_sparse(d):
        items = [(int(t), float(w)) for t, w in (d or {}).items() if float(w) > 0]
        return models.SparseVector(indices=[i for i, _ in items], values=[w for _, w in items]) if items else None

    qg = [to_sparse(d) for d in m3out["lexical_weights"]]
    print("query 编码完成\n", flush=True)

    LIM = 50

    def results(using, vec):
        if vec is None:
            return []
        r = client.query_points(COLL, query=vec, using=using, limit=LIM, with_payload=True).points
        return [p.payload["chunk_id"] for p in r]

    def rank_of(golden, res):
        for i, cid in enumerate(res, 1):
            if cid == golden:
                return i
        return None

    rows = []
    for i, q in enumerate(queries):
        rows.append({"kind": q["kind"], "lang": q.get("lang", ""), "golden": q["golden"],
                     "dense": results("dense", qd[i].tolist()),
                     "bm25": results("bm25", qb[i]),
                     "bgem3": results("bgem3", qg[i])})

    def mrr_recall(route, k=10, kind=None):
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        rr = [(1.0 / rank_of(r["golden"], r[route])) if rank_of(r["golden"], r[route]) else 0.0 for r in sel]
        rec = [1.0 if (rank_of(r["golden"], r[route]) and rank_of(r["golden"], r[route]) <= k) else 0.0 for r in sel]
        return sum(rr) / len(rr), sum(rec) / len(rec), len(sel)

    print("=== #1 单路召回(MRR / Recall@10)===", flush=True)
    print(f"{'route':8} {'exact':>22} {'semantic':>22} {'all':>22}", flush=True)
    for route in ("bm25", "bgem3", "dense"):
        cells = []
        for kind in ("exact", "semantic", None):
            mrr, rec, n = mrr_recall(route, 10, kind)
            cells.append(f"MRR={mrr:.3f} R@10={rec:.2f}")
        print(f"{route:8} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}", flush=True)

    def wrrf_rank(golden, dres, sres, w_d, k0=60):
        score = {}
        for r, cid in enumerate(dres, 1):
            score[cid] = score.get(cid, 0) + w_d / (k0 + r)
        for r, cid in enumerate(sres, 1):
            score[cid] = score.get(cid, 0) + (1 - w_d) / (k0 + r)
        return rank_of(golden, sorted(score, key=lambda c: score[c], reverse=True))

    print("\n=== #2 RRF 权重(dense + bm25)===", flush=True)
    print(f"{'w_dense':10} {'all MRR':>9} {'exact':>9} {'semantic':>9}", flush=True)
    for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        out = {}
        for kind in ("all", "exact", "semantic"):
            sel = [r for r in rows if kind == "all" or r["kind"] == kind]
            rr = [(1.0 / wrrf_rank(r["golden"], r["dense"], r["bm25"], w)) if wrrf_rank(r["golden"], r["dense"], r["bm25"], w) else 0.0 for r in sel]
            out[kind] = sum(rr) / len(rr)
        tag = " (纯sparse)" if w == 0 else " (纯dense)" if w == 1 else " (等权)" if w == 0.5 else ""
        print(f"{w:<10.2f} {out['all']:>9.3f} {out['exact']:>9.3f} {out['semantic']:>9.3f}{tag}", flush=True)


if __name__ == "__main__":
    main()
