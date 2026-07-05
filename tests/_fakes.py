"""测试替身:fake retriever / user / app 构造器(纯 CPU,不碰 Qdrant/GPU/网络)。
形状对齐引擎 mcp_server/tests/test_tools.py 的 _MockRet(同一 duck-typing 契约)。"""
from __future__ import annotations

from types import SimpleNamespace

from pharos import config as pconfig
from pharos.service import create_app


def make_hit(cid="c1", doc="d1", title="标题A", section="第一章 > 1.1", score=0.83,
             text="命中块原文", kind="text", **payload_extra):
    payload = {"doc_meta": {"title": title}, "section_path": section,
               "page_start": 4, "page_end": 4, "kind": kind,
               "acl": {"tenant": "t1", "allow": [], "visibility": "public", "unset": False}}
    payload.update(payload_extra)
    return SimpleNamespace(chunk_id=cid, doc_id=doc, text=text, score=score, kind=kind, payload=payload)


def make_res(hit, ctx_text=None, status="full_section", anchor=None, n_tokens=42):
    ctx = None
    if ctx_text is not None:
        ctx = SimpleNamespace(text=ctx_text, anchor=anchor, resolved_section="sec#x",
                              n_tokens=n_tokens, climbed=0)
    return {"hit": hit, "context": ctx, "context_status": status}


class FakeRetriever:
    """duck-typing 对齐 embedder.Retriever 被 toolcore/Generator 消费的方法面。
    results_factory 每次调用产新结果(toolcore._demote 原地改 dict,复用同一对象会串)。"""

    def __init__(self, results_factory=None, docs=None, document=None, outline=None,
                 expand_big=None, grouped=None):
        self._rf = results_factory or (lambda: [])
        self._document, self._outline = document, outline
        self._expand, self._grouped = expand_big, grouped
        self.calls: list = []
        self.store = SimpleNamespace(list_documents=lambda user: list(docs or []))

    def search_with_context(self, query, user, top_k=None, rerank=False, doc_ids=None,
                            doc_type=None, kind=None, assemble=True, strategy="hybrid", rerank_top_n=None):
        self.calls.append({"query": query, "top_k": top_k, "rerank": rerank, "assemble": assemble,
                           "doc_ids": doc_ids, "doc_type": doc_type, "kind": kind, "strategy": strategy,
                           "user_tenant": getattr(user, "tenant", None),        # 多身份:证明身份流到引擎
                           "user_principals": list(getattr(user, "principals", []) or [])})
        return self._rf()

    def get_document(self, doc_id, user, max_tokens=6000):
        if self._document is None:
            raise PermissionError
        return dict(self._document)

    def get_outline(self, doc_id, user):
        if self._outline is None:
            raise PermissionError
        return self._outline

    def expand(self, chunk_id, user, target_tokens=1500):
        return self._expand

    def search_grouped(self, query, user, doc_ids, top_k=3, rerank=False):
        return self._grouped or {}


def make_user(tenant="t1", principals=("g",)):
    return SimpleNamespace(tenant=tenant, principals=list(principals))


def make_cfg(**kw):
    kw.setdefault("tenant", "t1")
    return pconfig.PharosConfig(**kw)


def make_app(retriever=None, user=None, cfg=None, generator_factory=None, keys=None):
    return create_app(cfg=cfg or make_cfg(), retriever=retriever or FakeRetriever(),
                      user=user or make_user(), generator_factory=generator_factory, keys=keys)
