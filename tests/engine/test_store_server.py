"""Qdrant **server 模式**回归(阶段D go/no-go)。需 Qdrant server @ localhost:6333,CI/无 server 自动 skip。
手动跑:起 server(~/qdrant/qdrant)后 pytest tests/engine/test_store_server.py -v

覆盖两条 D 的 go/no-go:
  Q2 ACL 越权重测:嵌入式 QdrantLocal 的 RRF fusion 丢顶层 should(记忆铁律->越权泄漏),store.py 靠"ACL 下推每个
    prefetch"绕开;server 模式 fusion 语义不同,这里实测 hybrid/dense/sparse 三路都 fail-closed(越权不可召回)。
  多副本锁:嵌入式单进程文件锁(第二个 client 报错)->迁 server 后两个 client 同开一库(多副本前置)。"""
import random
import shutil
import tempfile

import pytest
from qdrant_client import models

from embedder.config import EmbedConfig
from embedder.sparse import doc_sparse, query_sparse
from embedder.store import Store
from embedder.types import User

URL = "http://localhost:6333"
DIM = 8


def _server_up() -> bool:
    try:
        import httpx
        return httpx.get(URL, timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"需 Qdrant server @ {URL};CI/无 server 跳过")


def _vec():
    return [random.random() for _ in range(DIM)]


def _A(t, a, v, unset=False):
    return {"tenant": t, "allow": a, "visibility": v, "unset": unset}


def _pt(i, text, acl):
    split = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
             "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(id=i, vector={"dense": _vec(), "sparse": doc_sparse(text)},
        payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": "text", "text": text, "acl": acl, **split})


def _fresh(coll):
    s = Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection=coll, prefetch_limit=20))
    try:
        s.client.delete_collection(coll)
    except Exception:
        pass
    s.ensure_collection()
    return s


def test_server_mode_acl_fail_closed_all_strategies():
    """Q2:server 模式 hybrid(RRF)/dense/sparse 三路都 fail-closed —— 无权/跨租户/unset 绝不召回。"""
    s = _fresh("test_acl_server")
    try:
        s.upsert([
            _pt(1, "公开 营收 报告", _A("t1", [], "public")),               # ✓ public
            _pt(2, "t1 hr 营收 报告", _A("t1", ["g_hr"], "restricted")),    # ✓ 同租户 + g_hr
            _pt(3, "t1 finance 营收", _A("t1", ["g_fin"], "restricted")),   # ✗ 无权
            _pt(4, "t2 hr 营收", _A("t2", ["g_hr"], "restricted")),        # ✗ 跨租户
            _pt(5, "未授权 营收", _A("t1", [], "restricted", unset=True)),  # ✗ fail-closed(unset)
        ])
        user = User(tenant="t1", principals=["g_hr"])
        q = query_sparse("营收 报告")
        for strat in ("hybrid", "dense", "sparse"):
            ids = {h.chunk_id for h in s.hybrid_search(_vec(), q, user, top_k=10, strategy=strat)}
            assert "c1" in ids, f"{strat}: public 应召回"
            assert "c2" in ids, f"{strat}: 同租户有权应召回"
            assert not (ids & {"c3", "c4", "c5"}), f"{strat}: 越权泄漏! 召回={sorted(ids)}"
    finally:
        s.client.delete_collection("test_acl_server")


def test_server_unlocks_multi_client():
    """多副本前置:嵌入式单进程文件锁(第二个 client 报错) -> server 模式两个 client 同开一库都成功。"""
    # 反证:嵌入式同 path,第二个报锁
    tmp = tempfile.mkdtemp()
    s1 = Store(EmbedConfig(qdrant_path=tmp, dense_dim=DIM, collection="c"))
    s1.ensure_collection()
    with pytest.raises(RuntimeError, match="already accessed"):
        Store(EmbedConfig(qdrant_path=tmp, dense_dim=DIM, collection="c"))
    del s1
    shutil.rmtree(tmp, ignore_errors=True)

    # 正证:server 两个 client 同开一库
    sa = _fresh("test_multi_replica")
    try:
        sb = Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection="test_multi_replica"))
        assert sa.client.get_collection("test_multi_replica").points_count == 0
        assert sb.client.get_collection("test_multi_replica").points_count == 0   # 第二个 client 读到同一库
    finally:
        sa.client.delete_collection("test_multi_replica")
