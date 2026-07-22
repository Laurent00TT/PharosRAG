"""评估 step2(GPU):index 1067 chunk 到 Qdrant,同库三向量:dense(Qwen3-VL 1024)+ bm25 + bge-m3。
关键(踩坑):所有逻辑必须放进 main()+if __name__=='__main__' —— 否则 BGE-M3 多进程 spawn 的子进程会
re-import 本模块当 __main__ 重跑整脚本(无限重入 + RuntimeError);并锁 bge-m3 单卡 devices='cuda:0'(单卡
不起多进程 pool)。"""
import json
import os
import uuid

EVAL = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from qdrant_client import QdrantClient, models

    from embedder.config import EmbedConfig
    from embedder.dense import Dense
    from embedder.sparse import doc_sparse

    chunks = [json.loads(l) for l in open(os.path.join(EVAL, "chunks.jsonl"), encoding="utf-8")]
    texts = [c["text"] for c in chunks]
    print(f"{len(chunks)} chunk", flush=True)

    # 1) dense (Qwen3-VL 1024 MRL) 分批
    dense = Dense(EmbedConfig(dense_dim=1024))
    dvecs = []
    B = 16
    for i in range(0, len(texts), B):
        dvecs.extend(dense.encode_text(texts[i:i + B]))
        if (i // B) % 15 == 0:
            print(f"  dense {min(i + B, len(texts))}/{len(texts)}", flush=True)
    print("dense done", flush=True)

    # 2) bm25(我们的 doc_sparse)
    bm25 = [doc_sparse(t) for t in texts]

    # 3) bge-m3 sparse(lexical_weights),锁单卡 -> 不起多进程
    from FlagEmbedding import BGEM3FlagModel
    m3 = BGEM3FlagModel(os.path.expanduser("~/models/bge-m3"), use_fp16=True, devices="cuda:0")
    out = m3.encode(texts, batch_size=16, max_length=2048,
                    return_dense=False, return_sparse=True, return_colbert_vecs=False)

    def to_sparse(d):
        items = [(int(t), float(w)) for t, w in (d or {}).items() if float(w) > 0]
        return models.SparseVector(indices=[i for i, _ in items], values=[w for _, w in items]) if items else None

    bgem3 = [to_sparse(d) for d in out["lexical_weights"]]
    print("bge-m3 done", flush=True)

    # 4) qdrant 同库三向量
    client = QdrantClient(path=os.path.join(EVAL, "qdrant"))
    COLL = "eval"
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    client.create_collection(
        COLL,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
                               "bgem3": models.SparseVectorParams()})
    points = []
    for i, c in enumerate(chunks):
        vec = {"dense": dvecs[i].tolist()}
        if bm25[i] is not None:
            vec["bm25"] = bm25[i]
        if bgem3[i] is not None:
            vec["bgem3"] = bgem3[i]
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),
            vector=vec, payload={"chunk_id": c["chunk_id"], "lang": c["lang"], "doc_type": c["doc_type"]}))
    client.upsert(COLL, points=points)
    print(f"indexed {len(points)} -> {EVAL}/qdrant", flush=True)


if __name__ == "__main__":
    main()
