"""Qdrant 存储 + 检索:dense+sparse hybrid(RRF)+ ACL 硬过滤。
ACL 兑现 chunker INTEGRATION §6 契约:fail-closed + 租户隔离 + (allow ANY OR public),嵌套保安全。

并发(M1 锁下沉):嵌入式 QdrantLocal 单 client **非线程安全**(读读安全、读写不安全:upsert 走 np.append 重绑
数组/_update_point 原地写)。删掉 LockedRetriever 大锁后,碰 self.client 的方法各自套 `self._lock` 串行。
`acl_filter` 是纯构造(不碰 client),不加锁。换 Qdrant server 模式后此锁可整体删除(阶段 F/Q3)。
注:`self._lock` 只保证**单方法**原子;`embed.py:index_document` 的 delete_by_doc+upsert 跨两次锁,重索引期间
有该 doc 短暂召回不到的窗口(既有性质,非 M1 引入 —— 旧大锁也不裹建库路径)。同进程并发建库+查询不在当前契约内。"""
from __future__ import annotations

import threading
import uuid

from qdrant_client import QdrantClient, models

from .acl import acl_admits
from .config import EmbedConfig
from .types import Hit, User


class Store:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        # 三分支(优先级 url > :memory: > path):server(阶段D 多副本)> 内存(测试,须在 path 前——大量测试依赖 :memory:)> 嵌入式(默认单进程)
        if cfg.qdrant_url:
            self.client = QdrantClient(url=cfg.qdrant_url)
        elif cfg.qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=cfg.qdrant_path)
        self._lock = threading.Lock()   # 嵌入式单 client 串行(M1);server 模式 Qdrant 并发安全,此锁可放宽(留阶段F/Q3)

    def ensure_collection(self) -> None:
        with self._lock:
            c = self.cfg
            if self.client.collection_exists(c.collection):
                # 不能静默早返回:dense_dim 改了但 collection 维度固化在建库时 -> 维度漂移会让 upsert 崩
                # (维度变大)或检索语义错(变小,COSINE 在不同子空间);config 注释本就规划调 dense_dim(seal#4)。
                size = self.client.get_collection(c.collection).config.params.vectors["dense"].size
                if size != c.dense_dim:
                    raise ValueError(
                        f"collection {c.collection!r} 已存在且 dense size={size} != cfg.dense_dim={c.dense_dim};"
                        f"改维度需重建 collection(删 {c.qdrant_path})或把 dense_dim 对齐回 {size}。")
                return
            self.client.create_collection(
                c.collection,
                vectors_config={"dense": models.VectorParams(size=c.dense_dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            for fld, schema in [("acl_tenant", "keyword"), ("acl_visibility", "keyword"),
                                ("acl_allow", "keyword"), ("acl_unset", "bool"), ("doc_id", "keyword"),
                                ("doc_type", "keyword"), ("kind", "keyword")]:   # 3.F:doc_type/kind 过滤索引(local 无效但 server 需要)
                self.client.create_payload_index(c.collection, field_name=fld, field_schema=schema)

    def acl_filter(self, user: User, doc_ids: list[str] | None = None,
                   doc_type: str | None = None, kind: str | None = None) -> models.Filter:
        """fail-closed:acl_unset==False AND acl_tenant==user.tenant AND (acl_allow∩principals 非空 OR public)。
        (allow OR public) 用**嵌套 Filter** 表达,不靠顶层 must+should 语义——杜绝权限漂移(seal-review 同款)。
        可选过滤(Batch2.B/3.A/3.F):doc_ids / doc_type / kind 一律 FieldCondition 进 **must**(与 ACL 是 AND,
        **非顶层 should**,不踩嵌入式 fusion 丢 should 的坑);任何过滤都只收窄、绝不放宽,无权内容仍 fail-closed。
        纯构造 models.Filter,不碰 client -> **不加锁**。"""
        must = [
            models.FieldCondition(key="acl_unset", match=models.MatchValue(value=False)),
            models.FieldCondition(key="acl_tenant", match=models.MatchValue(value=user.tenant)),
            models.Filter(should=[
                models.FieldCondition(key="acl_allow", match=models.MatchAny(any=user.principals or [])),
                models.FieldCondition(key="acl_visibility", match=models.MatchValue(value="public")),
            ]),
        ]
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))))
        if doc_type:
            must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)))
        if kind:
            must.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
        return models.Filter(must=must)

    def upsert(self, points: list[models.PointStruct]) -> None:
        with self._lock:
            self.client.upsert(self.cfg.collection, points=points)

    def delete_by_doc(self, doc_id: str) -> None:
        """删该 doc 的所有 point。重索引前清旧:否则新版 chunk 变少时旧高编号 point 残留成孤儿向量,
        仍可被召回但其 anchor/source_indices 指向新 sidecar 的错 idx -> 取错段 + ACL 错配(lazy-tree review#4)。
        doc_id 已建 payload index,filter 删高效。"""
        with self._lock:
            self.client.delete(self.cfg.collection, points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))])))

    def hybrid_search(self, dense_vec, sparse_vec, user: User, top_k: int | None = None,
                      doc_ids: list[str] | None = None, doc_type: str | None = None,
                      kind: str | None = None, strategy: str = "hybrid") -> list[Hit]:
        c = self.cfg
        acl = self.acl_filter(user, doc_ids, doc_type, kind)
        limit = c.top_k if top_k is None else max(1, top_k)   # Batch1.C:防 0-falsy(0 静默变默认)/负数
        # 4.A strategy 让 agent 选路:单路走直查拿**原生**分(cosine/bm25,可解释)、hybrid 走 RRF 融合。
        # ACL filter 必须下推到每个 prefetch(hybrid):嵌入式 QdrantLocal 在 fusion 模式下会丢弃顶层 query_filter 的
        # should(实测 diag_acl),只 prefetch-level filter 才让 should 生效;顶层 query_filter 双保险。单路直查无 fusion
        # 不踩此坑,且出口 acl_admits 复核兜底。
        with self._lock:                                 # M1:Qdrant client 前向串行(dense_vec 是入参,encode 已在锁外算完)
            if strategy == "dense":
                res = self.client.query_points(c.collection, query=dense_vec, using="dense",
                                               query_filter=acl, limit=limit, with_payload=True).points
                score_kind = "cosine"
            elif strategy == "sparse":
                if sparse_vec is None:                   # query 无有效精确词 token -> sparse 路无结果
                    return []
                res = self.client.query_points(c.collection, query=sparse_vec, using="sparse",
                                               query_filter=acl, limit=limit, with_payload=True).points
                score_kind = "bm25"
            else:                                        # hybrid(默认)
                prefetch = [models.Prefetch(query=dense_vec, using="dense", filter=acl, limit=c.prefetch_limit)]
                if sparse_vec is not None:               # image_only chunk 无 sparse;query 也可能无
                    prefetch.append(models.Prefetch(query=sparse_vec, using="sparse", filter=acl, limit=c.prefetch_limit))
                res = self.client.query_points(
                    c.collection, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=acl, limit=limit, with_payload=True).points
                score_kind = "rrf"
        # Batch2.A 出口逐命中复核:与 list_documents 对称,裸 hit 交付路径补第二道 ACL 闸(嵌套 acl 缺失/不可见即丢)。
        return [Hit(chunk_id=p.payload.get("chunk_id", ""), doc_id=p.payload.get("doc_id", ""),
                    kind=p.payload.get("kind", ""), text=p.payload.get("text", ""),
                    score=p.score, payload=dict(p.payload), score_kind=score_kind)
                for p in res if acl_admits(p.payload.get("acl") or {}, user)]

    def list_documents(self, user: User, limit: int = 10000) -> list[dict]:
        """列出 user 可见的文档(doc_id + title),ACL 作用域 —— 给 agentic RAG 的 list_documents 工具用。
        scroll 用 acl_filter 下推,且**逐条 acl_admits 复核**(双保险:嵌入式 QdrantLocal 对嵌套 should 的处理
        在 fusion 下已知会丢[见 hybrid_search],scroll 下未单独验证,fail-closed 不赌——宁可多校验一次)。"""
        docs: dict[str, tuple] = {}
        flt = self.acl_filter(user)
        with self._lock:                                 # M1:整段 scroll 循环共用一个 client,串行
            offset, seen = None, 0
            while seen < limit:
                points, offset = self.client.scroll(
                    self.cfg.collection, scroll_filter=flt, limit=min(256, limit - seen),
                    offset=offset, with_payload=True)
                if not points:
                    break
                for p in points:
                    if not acl_admits(p.payload.get("acl") or {}, user):   # 双保险:逐条复核 ACL
                        continue
                    did = p.payload.get("doc_id", "")
                    if did and did not in docs:                            # 带 doc_type(B6.B:给路由层算覆盖摘要)
                        docs[did] = ((p.payload.get("doc_meta") or {}).get("title") or did, p.payload.get("doc_type") or "")
                seen += len(points)
                if offset is None:
                    break
        return [{"doc_id": k, "title": t, "doc_type": dt} for k, (t, dt) in sorted(docs.items())]

    def get_by_chunk_id(self, chunk_id: str, user: User) -> dict | None:
        """凭 chunk_id 取回该 point 的 payload(给 expand 用)。point_id=uuid5(chunk_id) 确定性正算 -> O(1) 直取,
        无需 chunk_id 索引。**retrieve-by-id 绕过 acl_filter,故必须 acl_admits 复核**(无 point/无权一律 None,
        fail-closed,且不区分'不存在'与'无权')。"""
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
        with self._lock:
            pts = self.client.retrieve(self.cfg.collection, ids=[pid], with_payload=True)
        if not pts:
            return None
        payload = pts[0].payload or {}
        if not acl_admits(payload.get("acl") or {}, user):
            return None
        return payload

    def get_doc_lang(self, doc_id: str) -> str:
        """取该 doc 的权威 lang(任一 chunk 的 payload lang;doc 级一致)。get_document 估 token 量纲用,默认 en。
        非敏感元数据 + 调用方已过 doc 级 ACL 预检,故不另设 ACL。"""
        with self._lock:
            pts, _ = self.client.scroll(self.cfg.collection, limit=1, with_payload=True,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))]))
        return (pts[0].payload.get("lang") if pts else None) or "en"
