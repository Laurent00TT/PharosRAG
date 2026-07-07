"""MCP 薄适配器单测:HTTP 转发映射 + 各类失败的结构化降级(不裸抛给 agent)。
用 stub client 替换 mcp_adapter._client(不起真服务;适配器×真守护进程见 GPU 冒烟)。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import pytest

from pharos import mcp_adapter as A


class StubClient:
    """记录调用并按 responder 出响应(或抛异常)。"""

    def __init__(self, responder):
        self.responder = responder
        self.base_url = "http://stub:8787"
        self.calls: list = []

    def request(self, method, path, json=None, params=None):
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        return self.responder(method, path, json, params)


@pytest.fixture
def stub(monkeypatch):
    holder = {}

    def install(responder):
        sc = StubClient(responder)
        monkeypatch.setattr(A, "_client", sc)
        holder["c"] = sc
        return sc
    return install


def _ok(payload):
    return lambda m, p, j, q: httpx.Response(200, json=payload)


def test_retrieve_forwards_args(stub):
    sc = stub(_ok({"status": "ok", "hits": []}))
    out = A.retrieve("capex", top_k=5, rerank=True, doc_type="law", strategy="sparse")
    assert out["status"] == "ok"
    call = sc.calls[0]
    assert call["method"] == "POST" and call["path"] == "/v1/retrieve"
    assert call["json"]["query"] == "capex" and call["json"]["top_k"] == 5
    assert call["json"]["rerank"] is True and call["json"]["strategy"] == "sparse"


def test_get_document_uses_params(stub):
    sc = stub(_ok({"status": "ok", "text": "x"}))
    A.get_document("d1", max_tokens=1234)
    call = sc.calls[0]
    assert call["method"] == "GET" and call["path"] == "/v1/documents/d1"
    assert call["params"] == {"max_tokens": 1234}


def test_backend_down_structured(stub):
    def refuse(m, p, j, q):
        raise httpx.ConnectError("connection refused")
    stub(refuse)
    out = A.retrieve("q")
    assert out["status"] == "backend_unavailable" and out["retriable"] is True
    assert "pharos serve" in out["hint"]                     # hint 指向恢复动作


def test_401_maps_unauthorized(stub):
    stub(lambda m, p, j, q: httpx.Response(401, json={"status": "unauthorized"}))
    assert A.list_documents()["status"] == "unauthorized"


def test_5xx_maps_backend_unavailable(stub):
    stub(lambda m, p, j, q: httpx.Response(503, text="boom"))
    out = A.expand("c1")
    assert out["status"] == "backend_unavailable" and out["retriable"] is True


def test_422_maps_contract_mismatch_not_retriable(stub):
    # 修复:非 401 的 4xx(422 版本漂移致字段契约不匹配)是永久性错误 —— 不可标 retriable=True,
    # 否则 toolcore 契约会教 agent 对同一请求无效循环重试;hint 指向版本/URL 排障而非"稍后重试"
    stub(lambda m, p, j, q: httpx.Response(422, json={"detail": "field type error"}))
    out = A.retrieve("q")
    assert out["status"] == "contract_mismatch" and out["retriable"] is False
    assert "PHAROS_URL" in out["hint"] and "版本" in out["hint"]
    assert "稍后重试" not in out["hint"]


def test_404_maps_contract_mismatch_not_retriable(stub):
    # PHAROS_URL 误指他服的 404 同样不可重试(3xx 分支评审后已指向 PHAROS_URL,404 此前漏了同等处理)
    stub(lambda m, p, j, q: httpx.Response(404, text="not found"))
    out = A.get_document("d1")
    assert out["status"] == "contract_mismatch" and out["retriable"] is False


def test_non_json_maps_backend_unavailable(stub):
    stub(lambda m, p, j, q: httpx.Response(200, text="<html>not json</html>"))
    assert A.get_outline("d1")["status"] == "backend_unavailable"


def test_session_header_present_on_real_client():
    # 适配器进程级 uuid 会话头:守护进程按它做 per-session 去重隔离
    keys = {k.lower() for k in A._client.headers}
    assert "x-pharos-session" in keys


def test_instructions_same_source_as_engine():
    # 契约同源:适配器下发的 instructions 与引擎 toolcore 同文(不漂移)
    assert "grounding" in A._tc._INSTRUCTIONS and "chunk_id" in A._tc._INSTRUCTIONS
