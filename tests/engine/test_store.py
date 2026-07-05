"""Qdrant store + ACL 硬过滤单测(:memory: 嵌入式,假 dense + 真 BM25 sparse,纯 CPU)。
重点:ACL fail-closed —— 跨租户/无权/unset 文档绝不被检索到。"""
import os
import random
import sys


from qdrant_client import models

from embedder.config import EmbedConfig
from embedder.sparse import doc_sparse, query_sparse
from embedder.store import Store
from embedder.types import User

DIM = 8


def _vec():
    return [random.random() for _ in range(DIM)]


def _pt(i, text, acl):
    """acl = 嵌套 {tenant, allow, visibility, unset};payload 同时带 split 字段(filter 用)+ 嵌套 acl
    (Batch2.A 出口 acl_admits 复核用,贴合真实 embed payload)。"""
    split = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
             "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(
        id=i, vector={"dense": _vec(), "sparse": doc_sparse(text)},
        payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": "text", "text": text, "acl": acl, **split})


_A = lambda tenant, allow, vis, unset=False: {"tenant": tenant, "allow": allow, "visibility": vis, "unset": unset}


def test_acl_hard_filter():
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()
    s.upsert([
        _pt(1, "公开 营收 报告", _A("t1", [], "public")),                # ✓ public
        _pt(2, "t1 hr 营收 报告", _A("t1", ["g_hr"], "restricted")),     # ✓ t1 + g_hr
        _pt(3, "t1 finance 营收", _A("t1", ["g_fin"], "restricted")),    # ✗ 无权(g_fin)
        _pt(4, "t2 hr 营收", _A("t2", ["g_hr"], "restricted")),         # ✗ 跨租户
        _pt(5, "未授权 营收", _A("t1", [], "restricted", unset=True)),   # ✗ fail-closed(unset)
    ])
    user = User(tenant="t1", principals=["g_hr"])
    hits = s.hybrid_search(_vec(), query_sparse("营收 报告"), user, top_k=10)
    ids = {h.chunk_id for h in hits}
    assert "c1" in ids, "public 文档应召回"
    assert "c2" in ids, "同租户 + 有权(g_hr)应召回"
    assert "c3" not in ids, "无权(g_fin)泄漏!"
    assert "c4" not in ids, "跨租户(t2)泄漏!"
    assert "c5" not in ids, "fail-closed 失效:unset 文档泄漏!"


def test_hybrid_search_doc_ids_acl_and():
    # Batch2.B:doc_ids 过滤与 ACL 必 AND —— 点名无权 doc 仍 fail-closed 召回 0
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()
    s.upsert([
        _pt(1, "公开 营收 报告", _A("t1", [], "public")),                # d1 public 有权
        _pt(2, "t1 finance 营收", _A("t1", ["g_fin"], "restricted")),    # d2 g_hr user 无权
    ])
    user = User(tenant="t1", principals=["g_hr"])
    only_d1 = s.hybrid_search(_vec(), query_sparse("营收 报告"), user, top_k=10, doc_ids=["d1"])
    assert {h.doc_id for h in only_d1} == {"d1"}, "doc_ids=[d1] 应只回有权的 d1"
    none = s.hybrid_search(_vec(), query_sparse("营收"), user, top_k=10, doc_ids=["d2"])
    assert none == [], "点名无权 doc_id(d2)必须召回 0(ACL AND doc_id,不能借 doc_id 绕过 ACL)"


def _pt_doc(i, doc, title, acl):
    """带嵌套 acl(出口/list_documents 复核用)+ 拆分字段(filter 用)的 point,贴合真实 embed payload。"""
    split = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
             "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(id=i, vector={"dense": _vec()},
        payload={"chunk_id": f"c{i}", "doc_id": doc, "doc_meta": {"title": title}, "acl": acl, **split})


def test_list_documents_acl_scoped():
    # list_documents 按 ACL 作用域:无权/跨租户/unset 文档不出现在清单(fail-closed,agentic RAG 工具用)
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()
    A = lambda tenant, allow, vis, unset=False: {"tenant": tenant, "allow": allow,
                                                 "visibility": vis, "unset": unset}
    s.upsert([
        _pt_doc(1, "dPub", "公开报告", A("t1", [], "public")),
        _pt_doc(2, "dHr", "HR文档", A("t1", ["g_hr"], "restricted")),
        _pt_doc(3, "dFin", "财务文档", A("t1", ["g_fin"], "restricted")),     # t1 无 g_hr 权
        _pt_doc(4, "dT2", "他司文档", A("t2", ["g_hr"], "restricted")),       # 跨租户
        _pt_doc(5, "dUnset", "未授权", A("t1", [], "restricted", unset=True)),  # fail-closed
    ])
    ids = lambda u: {d["doc_id"] for d in s.list_documents(u)}
    assert ids(User("t1", ["g_hr"])) == {"dPub", "dHr"}, "t1/g_hr 只该见 public + 有权"
    assert ids(User("t1", ["g_fin"])) == {"dPub", "dFin"}, "t1/g_fin 见 public + 财务"
    assert ids(User("t2", ["g_hr"])) == {"dT2"}, "t2 只见自己租户"
    docs = s.list_documents(User("t1", ["g_hr"]))
    assert {d["doc_id"]: d["title"] for d in docs}["dHr"] == "HR文档", "title 应回填"


def test_doc_type_kind_filter():
    # 3.F:doc_type/kind 过滤与 ACL AND 收窄
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()

    def pt(i, dt, kind):
        return models.PointStruct(id=i, vector={"dense": _vec(), "sparse": doc_sparse("营收 报告")},
            payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": kind, "doc_type": dt, "text": "营收 报告",
                     "acl": _A("t1", [], "public"),
                     "acl_tenant": "t1", "acl_allow": [], "acl_visibility": "public", "acl_unset": False})
    s.upsert([pt(1, "financial_research_zh", "text"), pt(2, "academic_paper", "text"), pt(3, "financial_research_zh", "table")])
    u = User("t1", [])
    q = query_sparse("营收 报告")
    assert {h.doc_id for h in s.hybrid_search(_vec(), q, u, top_k=10, doc_type="financial_research_zh")} == {"d1", "d3"}
    assert {h.doc_id for h in s.hybrid_search(_vec(), q, u, top_k=10, kind="table")} == {"d3"}
    assert {h.doc_id for h in s.hybrid_search(_vec(), q, u, top_k=10, doc_type="financial_research_zh", kind="table")} == {"d3"}


def test_get_by_chunk_id_acl():
    # 3.C 前置:get_by_chunk_id 凭 chunk_id(uuid5)直取 payload,**retrieve-by-id 绕过 filter 故必 acl_admits 复核**
    import uuid as _uuid
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()

    def pt(cid, acl):
        return models.PointStruct(id=str(_uuid.uuid5(_uuid.NAMESPACE_URL, cid)),
            vector={"dense": _vec()},
            payload={"chunk_id": cid, "doc_id": "d1", "kind": "text", "text": "x", "acl": acl,
                     "acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
                     "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)})
    s.upsert([pt("pub#1", _A("t1", [], "public")), pt("fin#1", _A("t1", ["g_fin"], "restricted"))])
    u = User("t1", ["g_hr"])
    assert s.get_by_chunk_id("pub#1", u)["chunk_id"] == "pub#1", "public 有权 -> 拿到 payload"
    assert s.get_by_chunk_id("fin#1", u) is None, "无权 chunk -> None(retrieve-by-id 不绕过 ACL)"
    assert s.get_by_chunk_id("nonexist#9", u) is None, "不存在 -> None(与无权一致)"


def test_strategy_modes_and_score_kind():
    # 4.A:strategy 选路 + 原生 score_kind(rrf/cosine/bm25)
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()
    s.upsert([_pt(1, "公开 营收 报告", _A("t1", [], "public"))])
    u = User("t1", [])
    q = query_sparse("营收 报告")
    assert s.hybrid_search(_vec(), q, u, top_k=5)[0].score_kind == "rrf"
    assert s.hybrid_search(_vec(), q, u, top_k=5, strategy="dense")[0].score_kind == "cosine"
    assert s.hybrid_search(_vec(), q, u, top_k=5, strategy="sparse")[0].score_kind == "bm25"
    assert s.hybrid_search(_vec(), None, u, top_k=5, strategy="sparse") == [], "sparse 但无 sparse query -> 空"


def test_delete_by_doc():
    # 重索引前删旧 doc 的 point,只删该 doc、不误删其他(lazy-tree review#4)
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()

    def pt(i, doc):
        return models.PointStruct(id=i, vector={"dense": _vec()}, payload={"chunk_id": f"c{i}", "doc_id": doc})

    s.upsert([pt(1, "dA"), pt(2, "dA"), pt(3, "dB")])
    s.delete_by_doc("dA")
    assert s.client.count("t").count == 1                       # dA 的 2 个删、dB 的 1 个留
    remain = s.client.scroll("t", limit=10)[0]
    assert all(p.payload["doc_id"] == "dB" for p in remain)


if __name__ == "__main__":
    test_acl_hard_filter()
    test_delete_by_doc()
    print("store tests OK")
