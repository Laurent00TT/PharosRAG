"""Pharos HTTP 服务单测(纯 CPU:fake retriever + MockLLM,不碰 Qdrant/GPU/网络)。

覆盖:healthz / 六个检索端点 wiring(语义本体在引擎 toolcore 已测,这里测 HTTP 绑定)/
ACL fail-closed(no_identity)/ API key 门槛 / per-session 去重隔离(本服务新增的行为)/
/v1/ask 闭管道(引用映射、include_contexts、空 query、llm_unconfigured、ask_failed 降级)。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from _fakes import FakeRetriever, make_app, make_cfg, make_hit, make_res, make_user
from generator import Generator, MockLLM


# ---------- 健康 / 基本 wiring ----------
def test_healthz():
    with TestClient(make_app()) as c:
        r = c.get("/healthz").json()
    assert r["status"] == "ok" and r["service"] == "pharos" and r["tenant_bound"] is True


def test_retrieve_ok():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="big-block 上下文")])
    with TestClient(make_app(retriever=ret)) as c:
        r = c.post("/v1/retrieve", json={"query": "capex"}).json()
    assert r["status"] == "ok" and r["hits"][0]["text"] == "big-block 上下文"
    assert r["hits"][0]["trust"] == "untrusted" and "warning" in r


def test_retrieve_no_identity_fail_closed():
    with TestClient(make_app(user=make_user(tenant=""), cfg=make_cfg(tenant=""))) as c:
        r = c.post("/v1/retrieve", json={"query": "q"}).json()
    assert r["status"] == "no_identity" and r["hits"] == []


def test_retrieve_bad_arg_structured_not_422():
    # mode/strategy 校验留给 toolcore -> 结构化 bad_arg(agent 可程序化恢复),不是 FastAPI 422
    with TestClient(make_app()) as c:
        r = c.post("/v1/retrieve", json={"query": "q", "mode": "weird"})
    assert r.status_code == 200 and r.json()["status"] == "bad_arg"


def test_documents_document_outline_expand_grouped_wiring():
    from types import SimpleNamespace
    big = SimpleNamespace(text="更大上下文", anchor=[2, 8], resolved_section="sec#1", n_tokens=420, climbed=1)
    ret = FakeRetriever(docs=[{"doc_id": "d1", "title": "A", "doc_type": "law"}],
                        document={"doc_id": "d1", "text": "全文", "n_tokens": 2,
                                  "n_elements_visible": 3, "truncated": False},
                        outline=[{"sec_id": "s1", "title": "第一章", "level": 1}],
                        expand_big=big, grouped={"d1": [make_hit(cid="a")]})
    with TestClient(make_app(retriever=ret)) as c:
        assert c.get("/v1/documents").json()["coverage"] == {"law": 1}
        assert c.get("/v1/documents/d1").json()["text"] == "全文"
        assert c.get("/v1/documents/d1/outline").json()["sections"][0]["title"] == "第一章"
        assert c.post("/v1/expand", json={"chunk_id": "c1"}).json()["text"] == "更大上下文"
        g = c.post("/v1/retrieve_grouped", json={"query": "q", "doc_ids": ["d1"]}).json()
        assert g["groups"]["d1"][0]["chunk_id"] == "a"
        cap = c.post("/v1/retrieve_grouped", json={"query": "q", "doc_ids": [f"d{i}" for i in range(21)]}).json()
        assert cap["status"] == "bad_arg"


def test_document_no_access():
    with TestClient(make_app(retriever=FakeRetriever(document=None))) as c:
        assert c.get("/v1/documents/dX").json()["status"] == "no_access"


# ---------- API key 门槛 ----------
def test_api_key_gate():
    app = make_app(cfg=make_cfg(api_key="sek"))
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200                       # healthz 豁免
        assert c.get("/v1/documents").status_code == 401                  # 无 key 拒
        assert c.get("/v1/documents", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/documents", headers={"X-API-Key": "sek"}).status_code == 200


# ---------- per-session 去重(守护进程多会话的核心新行为)----------
def _fresh():
    return [make_res(make_hit(cid="c1"), ctx_text="big", anchor=[1, 5])]


def test_session_dedup_and_isolation():
    ret = FakeRetriever(results_factory=_fresh)
    with TestClient(make_app(retriever=ret)) as c:
        a1 = c.post("/v1/retrieve", json={"query": "q"}, headers={"X-Pharos-Session": "A"}).json()
        a2 = c.post("/v1/retrieve", json={"query": "q"}, headers={"X-Pharos-Session": "A"}).json()
        b1 = c.post("/v1/retrieve", json={"query": "q"}, headers={"X-Pharos-Session": "B"}).json()
    assert a1["hits"][0]["context_status"] == "full_section" and a1["hits"][0]["text"]
    assert a2["hits"][0]["context_status"] == "already_returned" and a2["hits"][0]["text"] == ""
    assert b1["hits"][0]["context_status"] == "full_section" and b1["hits"][0]["text"]   # B 会话不受 A 影响


def test_no_session_header_no_dedup():
    ret = FakeRetriever(results_factory=_fresh)
    with TestClient(make_app(retriever=ret)) as c:
        r1 = c.post("/v1/retrieve", json={"query": "q"}).json()
        r2 = c.post("/v1/retrieve", json={"query": "q"}).json()
    assert r1["hits"][0]["context_status"] == "full_section"
    assert r2["hits"][0]["context_status"] == "full_section"              # 去重 opt-in:无头不去重


# ---------- /v1/ask 闭管道 ----------
def _mock_gen_factory(retriever, cfg):
    return Generator(retriever, MockLLM())


def test_ask_citations_mapping():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="台积电 2025 capex 380-420 亿美元")])
    with TestClient(make_app(retriever=ret, generator_factory=_mock_gen_factory)) as c:
        r = c.post("/v1/ask", json={"query": "capex?"}).json()
    assert r["status"] == "ok" and "[cite:1]" in r["answer"] and r["n_contexts"] == 1
    cit = r["citations"][0]
    assert cit["marker"] == 1 and cit["chunk_id"] == "c1" and cit["title"] == "标题A"
    assert "text" not in cit                                              # 默认不带原文(省载荷)


def test_ask_include_contexts():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="上下文原文")])
    with TestClient(make_app(retriever=ret, generator_factory=_mock_gen_factory)) as c:
        r = c.post("/v1/ask", json={"query": "q", "include_contexts": True}).json()
    assert r["citations"][0]["text"] == "上下文原文"


def test_ask_empty_query_and_no_identity():
    with TestClient(make_app(generator_factory=_mock_gen_factory)) as c:
        assert c.post("/v1/ask", json={"query": "  "}).json()["status"] == "empty_query"
    with TestClient(make_app(user=make_user(tenant=""), cfg=make_cfg(tenant=""),
                             generator_factory=_mock_gen_factory)) as c:
        assert c.post("/v1/ask", json={"query": "q"}).json()["status"] == "no_identity"


def test_ask_llm_unconfigured():
    def boom_factory(retriever, cfg):
        raise ValueError("缺 key")
    with TestClient(make_app(generator_factory=boom_factory)) as c:
        r = c.post("/v1/ask", json={"query": "q"}).json()
    assert r["status"] == "llm_unconfigured" and "DEEPSEEK_API_KEY" in r["hint"]


def test_ask_failure_degrades_structured():
    class BoomGen:
        llm = None

        def answer(self, *a, **kw):
            raise RuntimeError("LLM 上游炸了(内部细节不该外泄)")
    with TestClient(make_app(generator_factory=lambda r, c: BoomGen())) as c:
        r = c.post("/v1/ask", json={"query": "q"}).json()
    assert r["status"] == "ask_failed" and r["retriable"] is True
    assert "炸了" not in r["hint"]                                        # 内部细节不外泄
