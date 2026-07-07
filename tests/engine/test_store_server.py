"""Qdrant **server 模式**回归(阶段D go/no-go)。需 Qdrant server @ localhost:6333,CI/无 server 自动 skip。
手动:起 server(~/qdrant/qdrant)后 pytest tests/engine/test_store_server.py -v
强制不许静默 skip(CI compose job 用):PHAROS_REQUIRE_QDRANT_SERVER=1 pytest ...

覆盖(经阶段D对抗审查修订):
  Q2 ACL:嵌入式 QdrantLocal 的 RRF fusion 丢顶层 should(铁律->越权泄漏),store.py 靠 ACL 下推每个 prefetch 绕开。
    - 管道级:hybrid/dense/sparse 三路最终交付 fail-closed(含出口 acl_admits 复核);
    - **fusion 层(M4):绕过出口 acl_admits 直接断言 server RRF 原始输出不含越权** —— 这才是"铁律未在 server 复发"的真命题
      (否则出口复核会掩盖 fusion 泄漏);
    - list_documents(scroll)的 should 语义 + 原始 scroll 探针(S1)。
  多副本锁:嵌入式单进程独占 -> server 两 client 同开(注:同进程,验的是文件锁天花板已在 client 层移除;真多进程副本应答留阶段E)。"""
import os
import random
import shutil
import tempfile

import pytest
from qdrant_client import QdrantClient, models

from embedder.config import EmbedConfig
from embedder.sparse import doc_sparse, query_sparse
from embedder.store import Store
from embedder.types import User

URL = "http://localhost:6333"
DIM = 8
random.seed(0)   # S4:确定性(假向量排序不 flaky)


def _server_up() -> bool:
    try:
        QdrantClient(url=URL, timeout=2).get_collections()   # 真连(比裸 HTTP 贴近测试实际用的 client)
        return True
    except Exception:
        return False


_UP = _server_up()
if os.getenv("PHAROS_REQUIRE_QDRANT_SERVER") == "1" and not _UP:   # M5:CI 强制不许静默 skip
    raise RuntimeError("PHAROS_REQUIRE_QDRANT_SERVER=1 但 Qdrant server 不可达 —— D go/no-go 不许静默跳过")
pytestmark = pytest.mark.skipif(not _UP, reason=f"需 Qdrant server @ {URL}(设 PHAROS_REQUIRE_QDRANT_SERVER=1 强制不 skip)")


def _vec():
    return [random.random() for _ in range(DIM)]


def _A(t, a, v, unset=False):
    return {"tenant": t, "allow": a, "visibility": v, "unset": unset}


def _pt(i, text, acl):
    split = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
             "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(id=i, vector={"dense": _vec(), "sparse": doc_sparse(text)},
        payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": "text", "text": text, "acl": acl, **split})


def _seed_acl_data(s):
    s.upsert([
        _pt(1, "公开 营收 报告", _A("t1", [], "public")),               # ✓ public
        _pt(2, "t1 hr 营收 报告", _A("t1", ["g_hr"], "restricted")),    # ✓ 同租户 + g_hr
        _pt(3, "t1 finance 营收", _A("t1", ["g_fin"], "restricted")),   # ✗ 无权
        _pt(4, "t2 hr 营收", _A("t2", ["g_hr"], "restricted")),        # ✗ 跨租户
        _pt(5, "未授权 营收", _A("t1", [], "restricted", unset=True)),  # ✗ fail-closed(unset)
    ])


def _fresh(coll):
    s = Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection=coll, prefetch_limit=20))
    try:
        s.client.delete_collection(coll)
    except Exception:
        pass
    s.ensure_collection()
    return s


def test_server_mode_acl_pipeline_fail_closed_all_strategies():
    """管道级:server 模式 hybrid(RRF)/dense/sparse 三路最终交付都 fail-closed(含出口 acl_admits)。"""
    s = _fresh("test_acl_server")
    try:
        _seed_acl_data(s)
        user = User(tenant="t1", principals=["g_hr"])
        q = query_sparse("营收 报告")
        for strat in ("hybrid", "dense", "sparse"):
            ids = {h.chunk_id for h in s.hybrid_search(_vec(), q, user, top_k=10, strategy=strat)}
            assert "c1" in ids and "c2" in ids, f"{strat}: public + 同租户有权应召回"
            assert not (ids & {"c3", "c4", "c5"}), f"{strat}: 越权泄漏! 召回={sorted(ids)}"
    finally:
        s.client.delete_collection("test_acl_server")


def test_server_fusion_no_should_leak_raw():
    """M4(真命题):**绕过出口 acl_admits**,直接断言 server RRF fusion 的原始 query_points 输出不含越权点。
    出口复核是 mode 无关硬兜底,会掩盖 fusion 层泄漏;这里直接观测 server filter 是否真 fail-closed。"""
    s = _fresh("test_acl_raw")
    try:
        _seed_acl_data(s)
        user = User(tenant="t1", principals=["g_hr"])
        acl = s.acl_filter(user)
        prefetch = [models.Prefetch(query=_vec(), using="dense", filter=acl, limit=20),
                    models.Prefetch(query=query_sparse("营收 报告"), using="sparse", filter=acl, limit=20)]
        raw = s.client.query_points("test_acl_raw", prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=acl, limit=10, with_payload=True).points
        raw_ids = {p.payload["chunk_id"] for p in raw}                # 未过 acl_admits
        assert not (raw_ids & {"c3", "c4", "c5"}), \
            f"server RRF fusion 层泄漏 should(prefetch filter 未生效)! 原始召回={sorted(raw_ids)}"
        assert "c1" in raw_ids and "c2" in raw_ids, "有权点应在 fusion 原始输出里"
    finally:
        s.client.delete_collection("test_acl_raw")


def test_server_list_documents_acl_scoped():
    """S1:list_documents(scroll)在 server 模式 ACL 作用域,且绕过 acl_admits 的原始 scroll 探针(should 语义 + 防分页漏列)。"""
    s = _fresh("test_list_server")
    try:
        _seed_acl_data(s)
        user = User(tenant="t1", principals=["g_hr"])
        docs, truncated = s.list_documents(user)
        ids = {d["doc_id"] for d in docs}
        assert ids == {"d1", "d2"}, f"t1/g_hr 只该见 public(d1)+有权(d2),实得 {sorted(ids)}"
        assert truncated is False, "个位数 point 不该报截断"
        # 原始 scroll 探针:绕过出口 acl_admits,断言 server scroll filter 层不放行越权
        raw, _ = s.client.scroll("test_list_server", scroll_filter=s.acl_filter(user), limit=100, with_payload=True)
        raw_ids = {p.payload["chunk_id"] for p in raw}
        assert not (raw_ids & {"c3", "c4", "c5"}), f"server scroll 层泄漏 should! 原始={sorted(raw_ids)}"
    finally:
        s.client.delete_collection("test_list_server")


def test_server_ensure_collection_heals_missing_payload_indexes():
    """评审修(修复4):手工 create_collection(不建索引)模拟"建库中途崩溃的半初始化 collection"——
    ensure_collection 存在分支须幂等补齐全部 7 个 payload index(否则 acl_*/doc_* 过滤在 server 退化全扫且永不自愈)。"""
    coll = "test_index_heal"
    raw = QdrantClient(url=URL)
    try:
        raw.delete_collection(coll)
    except Exception:
        pass
    raw.create_collection(coll,
        vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    try:
        assert not raw.get_collection(coll).payload_schema, "前置:半初始化 collection 应无任何索引"
        Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection=coll)).ensure_collection()
        schema = set(raw.get_collection(coll).payload_schema)
        expected = {"acl_tenant", "acl_visibility", "acl_allow", "acl_unset", "doc_id", "doc_type", "kind"}
        assert expected <= schema, f"存在分支未补建索引,缺 {expected - schema}"
    finally:
        raw.delete_collection(coll)


def test_server_unlocks_multi_client():
    """多副本锁:嵌入式单进程独占(第二个 client 报错) -> server 两 client 同开一库。
    ⚠ S2 局限:同进程两个 Store,验的是"嵌入式文件锁天花板已在 client 层移除",非"两个独立 pharos serve 进程"
    的真多副本(那含 sidecar 共享等,留阶段E compose 端到端)。反证的 already accessed 是 QdrantLocal 进程内注册表检查。"""
    tmp = tempfile.mkdtemp()
    s1 = Store(EmbedConfig(qdrant_path=tmp, dense_dim=DIM, collection="c"))
    s1.ensure_collection()
    with pytest.raises(RuntimeError, match="already accessed"):
        Store(EmbedConfig(qdrant_path=tmp, dense_dim=DIM, collection="c"))
    del s1
    shutil.rmtree(tmp, ignore_errors=True)

    sa = _fresh("test_multi_replica")
    try:
        sb = Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection="test_multi_replica"))
        assert sa.client.get_collection("test_multi_replica").points_count == 0
        assert sb.client.get_collection("test_multi_replica").points_count == 0   # 第二个 client 读到同一库
    finally:
        sa.client.delete_collection("test_multi_replica")
