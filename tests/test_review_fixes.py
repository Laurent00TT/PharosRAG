"""对抗评审 P1(2026-07-02)修复的回归测试:每条 confirmed / 自核实属实的发现钉一个测试。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import pytest
from fastapi.testclient import TestClient

from _fakes import FakeRetriever, make_app, make_cfg, make_user
from pharos import config as pconfig
from pharos import mcp_adapter as A
from pharos.indexer import run_index


# ---------- C2:no_identity hint 必须指向 PHAROS_TENANT(不是引擎的 RAG_TENANT)----------
def test_no_identity_hint_names_pharos_vars():
    app = make_app(user=make_user(tenant=""), cfg=make_cfg(tenant=""))
    with TestClient(app) as c:
        r1 = c.post("/v1/retrieve", json={"query": "q"}).json()
        r2 = c.post("/v1/ask", json={"query": "q"}).json()
    for r in (r1, r2):
        assert r["status"] == "no_identity"
        assert "PHAROS_TENANT" in r["hint"] and "RAG_TENANT" not in r["hint"]


# ---------- C1:indexer 拒绝 restricted + 空 allow(静默不可见陷阱)----------
def test_indexer_rejects_restricted_without_allow():
    with pytest.raises(SystemExit, match="allow"):
        run_index(make_cfg(), visibility="restricted", allow="")


def test_indexer_restricted_with_allow_passes_guard():
    # 有 principals 时守卫放行(随后因语料目录不存在退出 —— 证明卡的是 ACL 守卫而非别的)
    with pytest.raises(SystemExit, match="语料目录不存在"):
        run_index(make_cfg(), corpus="Z:/definitely/not/exist", visibility="restricted", allow="g_hr")


# ---------- ask:工厂非 ValueError 异常不再裸抛 500 ----------
def test_ask_factory_runtime_error_degrades():
    def boom(retriever, cfg):
        raise ModuleNotFoundError("openai")
    with TestClient(make_app(generator_factory=boom)) as c:
        r = c.post("/v1/ask", json={"query": "q"})
    assert r.status_code == 200 and r.json()["status"] == "ask_failed"


# ---------- 适配器:doc_id URL 编码 / 空 doc_id 本地拒 / 3xx 结构化 ----------
class _StubClient:
    def __init__(self, responder):
        self.responder = responder
        self.base_url = "http://stub:8787"
        self.calls: list = []

    def request(self, method, path, json=None, params=None):
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        return self.responder(method, path, json, params)


def test_adapter_quotes_doc_id(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(200, json={"status": "ok"}))
    monkeypatch.setattr(A, "_client", sc)
    A.get_document("Q1#report/v2")
    assert sc.calls[0]["path"] == "/v1/documents/Q1%23report%2Fv2"   # #、/ 不再截断/改路由
    A.get_outline("研报 2026")
    assert "%" in sc.calls[1]["path"] and sc.calls[1]["path"].endswith("/outline")


def test_adapter_empty_doc_id_local_bad_arg(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(200, json={"status": "ok"}))
    monkeypatch.setattr(A, "_client", sc)
    assert A.get_document("")["status"] == "bad_arg"
    assert A.get_outline("")["status"] == "bad_arg"
    assert sc.calls == []                    # 未发 HTTP(否则空 doc_id 打到列表/307)


def test_adapter_3xx_structured(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(307, headers={"location": "/v1/documents"}))
    monkeypatch.setattr(A, "_client", sc)
    out = A.get_document("d1")
    assert out["status"] == "backend_unavailable" and out["retriable"] is True


# ---------- config:.env 行内注释/引号、int 指名报错、覆盖路径 expanduser ----------
def test_parse_env_value():
    assert pconfig._parse_env_value("8787  # 服务端口") == "8787"
    assert pconfig._parse_env_value('"sk-abc # not comment"') == "sk-abc # not comment"
    assert pconfig._parse_env_value("'  spaced  '") == "  spaced  "
    assert pconfig._parse_env_value("plain") == "plain"


def test_int_env_named_error(monkeypatch):
    monkeypatch.setenv("PHAROS_PORT", "8787  x")
    with pytest.raises(SystemExit, match="PHAROS_PORT"):
        pconfig._int_env("PHAROS_PORT", 8787)


def test_override_paths_expanduser(monkeypatch):
    monkeypatch.setenv("PHAROS_QDRANT_PATH", "~/qtest")
    monkeypatch.setenv("PHAROS_SIDECAR_DIR", "~/stest")
    cfg = pconfig.from_env()
    assert cfg.qdrant_path == os.path.expanduser("~/qtest")     # 不再落在字面 "./~" 目录
    assert cfg.sidecar_dir == os.path.expanduser("~/stest")


# ---------- 契约同文:适配器六工具 docstring 与引擎 server.py 逐一相同 ----------
def test_adapter_docstrings_match_engine():
    import importlib.util
    eng = os.path.join(make_cfg().engine, "mcp_server")
    sys.path.insert(0, os.path.join(make_cfg().engine, "embedder", "src"))
    sys.path.insert(0, eng)
    spec = importlib.util.spec_from_file_location("engine_server_for_test", os.path.join(eng, "server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    for name in ["retrieve", "list_documents", "get_document", "get_outline", "expand", "retrieve_grouped"]:
        eng_doc = getattr(srv, name).__doc__ or ""
        adp_doc = getattr(A, name).__doc__ or ""
        assert adp_doc.strip() == eng_doc.strip(), f"{name} docstring 与引擎不同文"


# ---------- N3:/v1/ask 检索过滤透传 ----------
from generator import Generator, MockLLM
from _fakes import make_res, make_hit


def test_ask_forwards_filters_to_retriever():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="表数据 6,779,511")])
    app = make_app(retriever=ret, generator_factory=lambda r, c: Generator(r, MockLLM()))
    with TestClient(app) as c:
        r = c.post("/v1/ask", json={"query": "营收?", "kind": "table",
                                    "doc_ids": ["d1"], "strategy": "sparse"}).json()
    assert r["status"] == "ok"
    call = ret.calls[0]
    assert call["kind"] == "table" and call["doc_ids"] == ["d1"] and call["strategy"] == "sparse"


def test_ask_bad_strategy_structured():
    app = make_app(generator_factory=lambda r, c: Generator(r, MockLLM()))
    with TestClient(app) as c:
        r = c.post("/v1/ask", json={"query": "q", "strategy": "weird"}).json()
    assert r["status"] == "bad_arg" and "strategy" in r["hint"]
