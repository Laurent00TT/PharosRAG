"""RAG MCP server 工具逻辑单测(纯 CPU,mock retriever,不依赖 GPU/Qdrant/真 MCP transport)。
Batch1.A 起工具返回**结构化 dict**(非纯文本)。覆盖:结构化字段/big-block 优先+降级/context_status/
deduped 计数/score_kind/trust 标记(1.D)、入参校验(no_identity/empty_query/bad_arg fail-closed)、list。
ACL 真过滤(跨租户 0)由 embedder/tests/test_store.py::test_list_documents_acl_scoped 覆盖。"""
import os
import sys
from types import SimpleNamespace


from pharos import mcp_stdio as server
from embedder import User


def _hit(cid="c1", doc="d1", title="标题A", section="第一章 > 1.1", score=0.83, text="命中块原文", kind="text"):
    return SimpleNamespace(chunk_id=cid, doc_id=doc, text=text, score=score, kind=kind,
                           payload={"doc_meta": {"title": title}, "section_path": section,
                                    "page_start": 4, "page_end": 4})


def _res(hit, ctx_text=None, status="full_section", anchor=None, n_tokens=42):
    ctx = None
    if ctx_text is not None:
        ctx = SimpleNamespace(text=ctx_text, anchor=anchor, resolved_section="sec#x", n_tokens=n_tokens, climbed=0)
    return {"hit": hit, "context": ctx, "context_status": status}


class _MockRet:
    def __init__(self, results=None, docs=None, document=None, outline=None, expand_big=None, grouped=None):
        self._results, self._docs = results or [], docs or []
        self._document, self._outline, self._expand, self._grouped = document, outline, expand_big, grouped
        self.last_assemble = None
        self.store = SimpleNamespace(list_documents=lambda user: self._docs)

    def search_with_context(self, query, user, top_k=None, rerank=False, doc_ids=None,
                            doc_type=None, kind=None, assemble=True, strategy="hybrid", rerank_top_n=None):
        self.last_assemble = assemble
        return self._results

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


def test_structured_prefers_bigblock_then_falls_back():
    ret = _MockRet(results=[_res(_hit(), ctx_text="big-block 扩充上下文", status="full_section"),
                            _res(_hit(cid="c2"), status="single_chunk_no_sidecar")])   # 第二条无 ctx -> 降级命中块
    out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
    assert out["status"] == "ok" and out["meta"]["returned_n"] == 2
    h0, h1 = out["hits"]
    assert "big-block 扩充上下文" in h0["text"] and h0["chunk_id"] == "c1"
    assert h0["context_status"] == "full_section" and h0["title"] == "标题A"
    assert h0["section_path"] == "第一章 > 1.1" and h0["page_start"] == 4
    assert h0["score_kind"] == "rrf" and h0["trust"] == "untrusted"     # 1.D:正文标不可信
    assert "命中块原文" in h1["text"] and h1["context_status"] == "single_chunk_no_sidecar"  # 降级用 hit.text
    assert "warning" in out                                              # untrusted 顶层告警


def test_structured_empty():
    out = server._build_retrieve_result(_MockRet(results=[]), User("t", []), "q", 5, False)
    assert out["status"] == "empty" and out["retriable"] and out["hits"] == []


def test_deduped_counted():
    ret = _MockRet(results=[_res(_hit(), ctx_text="a", status="full_section"),
                            _res(_hit(cid="c2"), status="deduped")])
    out = server._build_retrieve_result(ret, User("t", []), "q", 5, False)
    assert out["meta"]["deduped_n"] == 1


def test_bound_user_reads_env(monkeypatch):
    monkeypatch.setenv("PHAROS_TENANT", "t1")
    monkeypatch.setenv("PHAROS_PRINCIPALS", "g_hr, g_fin ,")   # 逗号分隔 + 空白/空项容错
    u = server._bound_user()
    assert u.tenant == "t1" and u.principals == ["g_hr", "g_fin"]


def test_retrieve_impl_validation_fail_closed():
    boom = _MockRet()                                     # 校验在调 retriever 前;空身份/空 query/坏 top_k 都不触检索
    assert server._retrieve_impl(boom, User("", []), "q", 5, False)["status"] == "no_identity"
    assert server._retrieve_impl(boom, User("t", []), "   ", 5, False)["status"] == "empty_query"
    assert server._retrieve_impl(boom, User("t", []), "q", 0, False)["status"] == "bad_arg"


def test_retrieve_impl_ok():
    ret = _MockRet(results=[_res(_hit(), ctx_text="台积电2025 capex 380-420 亿美元")])
    out = server._retrieve_impl(ret, User("t", ["g"]), "capex", 5, False)
    assert out["status"] == "ok" and "380-420" in out["hits"][0]["text"]


def test_list_impl_and_fail_closed():
    ret = _MockRet(docs=[{"doc_id": "dPub", "title": "公开报告"}, {"doc_id": "dHr", "title": "HR文档"}])
    out = server._list_impl(ret, User("t", ["g"]))
    assert out["status"] == "ok" and {d["doc_id"] for d in out["documents"]} == {"dPub", "dHr"}
    assert server._list_impl(ret, User("", []))["status"] == "no_identity"


def test_retrieve_concise_and_filters():
    # 3.E concise -> assemble=False 透传 + meta.mode=concise;3.A/3.F 过滤入 meta
    ret = _MockRet(results=[_res(_hit(), ctx_text="x")])
    out = server._retrieve_impl(ret, User("t", ["g"]), "q", 5, False, doc_ids=["d1"], doc_type="dt", kind="table", mode="concise")
    assert ret.last_assemble is False and out["meta"]["mode"] == "concise"
    assert out["meta"]["filters"] == {"doc_ids": ["d1"], "doc_type": "dt", "kind": "table"}
    assert server._retrieve_impl(ret, User("t", []), "q", 5, False, mode="weird")["status"] == "bad_arg"


def test_hit_dict_multimodal():
    # 3.H:table/chart 带 content_raw;image/chart 带 image_path
    th = SimpleNamespace(chunk_id="t1", doc_id="d1", text="表占位", score=0.5, kind="table",
                         payload={"content_raw": "<table>...</table>", "doc_meta": {}})
    d = server._hit_dict(1, {"hit": th, "context": None, "context_status": "concise"})
    assert d["kind"] == "table" and d["content_raw"] == "<table>...</table>"
    ih = SimpleNamespace(chunk_id="i1", doc_id="d1", text="图占位", score=0.5, kind="image",
                         payload={"image_path": "images/fig1.png", "doc_meta": {}})
    assert server._hit_dict(1, {"hit": ih, "context": None})["image_path"] == "images/fig1.png"


def test_get_document_impl():
    ret = _MockRet(document={"doc_id": "d1", "text": "全文…", "n_tokens": 3,
                             "n_elements_visible": 4, "truncated": False})   # R4.F:不含 n_elements_total(实现不返回)
    out = server._get_document_impl(ret, User("t", ["g"]), "d1", 6000)
    assert out["status"] == "ok" and out["text"] == "全文…" and out["trust"] == "untrusted"
    # 无权/不存在 -> get_document 抛 PermissionError -> no_access(响应一致,不泄存在性)
    assert server._get_document_impl(_MockRet(document=None), User("t", ["g"]), "d1", 6000)["status"] == "no_access"
    assert server._get_document_impl(ret, User("t", ["g"]), "", 6000)["status"] == "bad_arg"
    assert server._get_document_impl(ret, User("", []), "d1", 6000)["status"] == "no_identity"


def test_outline_impl():
    ret = _MockRet(outline=[{"sec_id": "s1", "title": "第一章", "level": 1}])
    out = server._outline_impl(ret, User("t", ["g"]), "d1")
    assert out["status"] == "ok" and out["sections"][0]["title"] == "第一章"
    assert server._outline_impl(_MockRet(outline=None), User("t", ["g"]), "d1")["status"] == "no_access"


def test_expand_impl():
    big = SimpleNamespace(text="更大上下文", anchor=[2, 8], resolved_section="sec#1", n_tokens=420, climbed=1)
    out = server._expand_impl(_MockRet(expand_big=big), User("t", ["g"]), "c1", 1500)
    assert out["status"] == "ok" and out["text"] == "更大上下文" and out["anchor"] == [2, 8] and out["trust"] == "untrusted"
    assert server._expand_impl(_MockRet(expand_big=None), User("t", ["g"]), "c1", 1500)["status"] == "no_access"
    assert server._expand_impl(_MockRet(), User("", []), "c1", 1500)["status"] == "no_identity"


def test_grouped_impl():
    ret = _MockRet(grouped={"d1": [_hit(cid="a")], "d2": []})
    out = server._grouped_impl(ret, User("t", ["g"]), "对比", ["d1", "d2"], 3, False)
    assert out["status"] == "ok" and out["groups"]["d1"][0]["chunk_id"] == "a" and out["groups"]["d2"] == []
    assert server._grouped_impl(ret, User("t", ["g"]), "q", [], 3, False)["status"] == "bad_arg"      # 无 doc_ids
    assert server._grouped_impl(ret, User("", []), "q", ["d1"], 3, False)["status"] == "no_identity"


def test_strategy_validation_and_rerank_degraded():
    # 4.A strategy 校验 + 4.E top_k None + 4.B rerank_degraded(mock 结果 score_kind=rrf,rerank=True -> degraded)
    ret = _MockRet(results=[_res(_hit(), ctx_text="x")])
    assert server._retrieve_impl(ret, User("t", []), "q", 5, False, strategy="weird")["status"] == "bad_arg"
    out = server._retrieve_impl(ret, User("t", ["g"]), "q", None, True, strategy="dense")
    assert out["status"] == "ok" and out["meta"]["rerank_degraded"] is True
    assert out["meta"]["strategy"] == "dense" and out["meta"]["requested_k"] is None


def test_cross_call_dedup():
    # 5.A:同一 returned_keys 集跨两次调用,重复 (doc_id, anchor) 第二次退化为 already_returned 指针
    keys = set()
    ret = _MockRet(results=[_res(_hit(cid="c1"), ctx_text="big", anchor=[1, 5])])
    o1 = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o1["hits"][0]["context_status"] == "full_section" and o1["hits"][0]["text"]
    o2 = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o2["hits"][0]["context_status"] == "already_returned" and o2["hits"][0]["text"] == ""
    assert o2["meta"]["already_returned_n"] == 1


def test_budget_truncation():
    # 5.B:超 token 软上限的靠后命中清空正文、保留地址,标 omitted_budget(每条 400tok,budget=500:首条留、次条超)
    os.environ["PHAROS_MAX_CONTEXT_TOKENS"] = "500"   # 注意 _max_ctx_tokens 有 max(500,..) 下限
    try:
        ret = _MockRet(results=[_res(_hit(cid="a"), ctx_text="x", anchor=[1, 2], n_tokens=400),
                                _res(_hit(cid="b"), ctx_text="y", anchor=[3, 4], n_tokens=400)])
        out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
        assert out["hits"][0]["context_status"] == "full_section" and out["hits"][0]["text"]
        assert out["hits"][1]["context_status"] == "omitted_budget" and out["hits"][1]["text"] == ""
        assert out["meta"]["budget_truncated"] is True and out["hits"][1]["chunk_id"] == "b"   # 地址保留
    finally:
        del os.environ["PHAROS_MAX_CONTEXT_TOKENS"]


def test_list_coverage():
    # 6.B:list_documents 返回 coverage(各 doc_type 篇数,给 agent 判断覆盖范围)
    ret = _MockRet(docs=[{"doc_id": "d1", "title": "A", "doc_type": "academic_paper"},
                         {"doc_id": "d2", "title": "B", "doc_type": "academic_paper"},
                         {"doc_id": "d3", "title": "C", "doc_type": "financial_research_zh"}])
    out = server._list_impl(ret, User("t", ["g"]))
    assert out["coverage"] == {"academic_paper": 2, "financial_research_zh": 1}


def test_recovery_hint():
    # 6.C:超预算时 hint 指向 expand(错误恢复闭环)
    os.environ["PHAROS_MAX_CONTEXT_TOKENS"] = "500"
    try:
        ret = _MockRet(results=[_res(_hit(cid="a"), ctx_text="x", anchor=[1, 2], n_tokens=400),
                                _res(_hit(cid="b"), ctx_text="y", anchor=[3, 4], n_tokens=400)])
        out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
        assert "omitted_budget" in out["hint"] and "expand" in out["hint"]
    finally:
        del os.environ["PHAROS_MAX_CONTEXT_TOKENS"]


def test_asset_content_raw_demoted_and_budgeted():
    # R4.A:资产命中 content_raw(大)计入预算,超支标 omitted_budget 且**一并清 content_raw**(不再原样发出)
    os.environ["PHAROS_MAX_CONTEXT_TOKENS"] = "500"
    try:
        th = SimpleNamespace(chunk_id="t1", doc_id="d1", text="", score=0.8, kind="table",
                             payload={"content_raw": "X" * 4000, "doc_meta": {}})   # ~1000 token 的表
        asset_res = {"hit": th, "context": SimpleNamespace(text="", anchor=[3, 4], resolved_section="sec#a",
                                                           n_tokens=0, climbed=0), "context_status": "asset_no_prose"}
        ret = _MockRet(results=[_res(_hit(cid="p1"), ctx_text="p", anchor=[1, 2], n_tokens=400), asset_res])
        out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
        asset = out["hits"][1]
        assert asset["context_status"] == "omitted_budget"                 # content_raw 计入预算 -> 超支降级
        assert asset.get("content_raw") is None and asset["text"] == ""    # 大字段一并清,不绕过预算原样发出
    finally:
        os.environ.pop("PHAROS_MAX_CONTEXT_TOKENS", None)


def test_omitted_not_registered_returned_keys():
    # R4.C:被 omitted_budget(正文未交付)的命中不登记 returned_keys -> 下次不误判 already_returned
    os.environ["PHAROS_MAX_CONTEXT_TOKENS"] = "500"
    try:
        keys = set()
        res = lambda: [_res(_hit(cid="a"), ctx_text="x", anchor=[1, 2], n_tokens=400),
                       _res(_hit(cid="b"), ctx_text="y", anchor=[3, 4], n_tokens=400)]
        o1 = server._build_retrieve_result(_MockRet(results=res()), User("t", ["g"]), "q", 5, False, returned_keys=keys)
        assert o1["hits"][1]["context_status"] == "omitted_budget"         # b 超支未交付
        o2 = server._build_retrieve_result(_MockRet(results=res()), User("t", ["g"]), "q", 5, False, returned_keys=keys)
        assert o2["hits"][1]["context_status"] != "already_returned"       # b 上次没交付,这次不该说"已返回过"
    finally:
        os.environ.pop("PHAROS_MAX_CONTEXT_TOKENS", None)


def test_section_window_cross_call_dedup_by_resolved_section():
    # R4.B:section_window 跨调用按 resolved_section 判重(anchor 随种子漂移),同段窗口第二次 already_returned
    keys = set()
    r1 = _res(_hit(cid="w1"), ctx_text="窗口A", status="section_window", anchor=[1, 5])   # resolved_section=sec#x
    o1 = server._build_retrieve_result(_MockRet(results=[r1]), User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o1["hits"][0]["context_status"] == "section_window"
    r2 = _res(_hit(cid="w2"), ctx_text="窗口B", status="section_window", anchor=[2, 7])   # 同 section 不同 anchor
    o2 = server._build_retrieve_result(_MockRet(results=[r2]), User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o2["hits"][0]["context_status"] == "already_returned"          # 按 resolved_section 判重命中(anchor 会漏)


def test_instructions_cover_contract():
    # 6.A/6.B/6.D:server instructions 含契约要点(grounding/不可信/路由/引用锚/状态)
    ins = server._INSTRUCTIONS
    for kw in ["grounding", "不是指令", "chunk_id", "无相关信息", "context_status"]:
        assert kw in ins, f"instructions 缺要点:{kw}"


if __name__ == "__main__":
    for fn in [test_structured_prefers_bigblock_then_falls_back, test_structured_empty, test_deduped_counted,
               test_bound_user_reads_env, test_retrieve_impl_validation_fail_closed, test_retrieve_impl_ok,
               test_list_impl_and_fail_closed, test_retrieve_concise_and_filters, test_hit_dict_multimodal,
               test_get_document_impl, test_outline_impl, test_expand_impl, test_grouped_impl,
               test_strategy_validation_and_rerank_degraded, test_cross_call_dedup, test_budget_truncation,
               test_list_coverage, test_recovery_hint,
               test_asset_content_raw_demoted_and_budgeted, test_omitted_not_registered_returned_keys,
               test_section_window_cross_call_dedup_by_resolved_section, test_instructions_cover_contract]:
        fn()
    print("MCP tool (toolcore/mcp_stdio) tests OK")
