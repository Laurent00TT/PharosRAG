"""远程推理后端单测(纯 CPU,mock HTTP,不碰 GPU/真服务)。覆盖:
 工厂选路 / _load no-op / _mrl_np↔_mrl 等价 / encode_text + query 缓存 / reranker 契约;
 P0-1 重试:503 瞬态重试成功 / 4xx 立即抛零重试 / 耗尽抛 InferenceUnavailable / ConnectError 重试;
 P0-1 toolcore 分流:InferenceUnavailable -> inference_unavailable,普通异常 -> backend_unavailable;
 P0-2 _readiness 优先级:err 优先于 loading;
 P2-1 dense/reranker 共享连接池;非对称:InferenceUnavailable 是 Exception 子类(会被 retrieve rerank 降级吞)。

注:测试自身可用 CPU torch 做 _mrl(torch)↔_mrl_np(numpy)对照(被测应用层代码路径 remote.py 不 import torch)。"""
import httpx
import numpy as np
import pytest

from embedder import remote as R
from embedder.config import EmbedConfig
from embedder.dense import Dense
from embedder.errors import InferenceUnavailable
from embedder.remote import RemoteDense, RemoteReranker, make_dense, make_reranker


class _Resp:
    """假 httpx.Response:带 status_code;raise_for_status 在 >=400 时抛 httpx.HTTPStatusError(模拟 4xx)。"""
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._d = data if data is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}",
                                        request=httpx.Request("POST", "http://x"),
                                        response=httpx.Response(self.status_code))

    def json(self):
        return self._d


class _ScriptClient:
    """按脚本逐次响应:每项是 _Resp(返回)或 Exception 实例(抛出)。用尽后重复最后一项。记录 calls。"""
    def __init__(self, *steps):
        self.steps = list(steps) or [_Resp(200, {})]
        self.calls = []

    def post(self, path, json=None):
        self.calls.append({"path": path, "json": json})
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def _fast_cfg(**kw):
    """remote cfg + backoff=0(重试不实际 sleep,测试快)。"""
    return EmbedConfig(inference_url="http://x", inference_backoff=0.0, **kw)


# ---------- 骨架契约(原 4 测试) ----------
def test_factory_selects_backend():
    # 唯一决定 local vs remote 的地方:inference_url 空=Local,非空=Remote
    assert type(make_dense(EmbedConfig())).__name__ == "Dense"
    assert type(make_dense(EmbedConfig(inference_url="http://x:8900"))).__name__ == "RemoteDense"
    assert type(make_reranker(EmbedConfig())).__name__ == "Reranker"
    assert type(make_reranker(EmbedConfig(inference_url="http://x:8900"))).__name__ == "RemoteReranker"


def test_remote_dense_load_noop_and_mrl_equivalent_to_local():
    # RemoteDense._load 不加载模型;客户端 numpy 截维 == Dense._mrl 的 torch 截维(数学等价 => local 建/remote 查不错位)
    import torch
    rd = RemoteDense(EmbedConfig(inference_url="http://x", dense_dim=1024))
    rd._load()
    assert rd._model is None
    rng = np.random.default_rng(0)
    full = rng.standard_normal((3, 4096)).astype(np.float32)
    full = full / np.linalg.norm(full, axis=-1, keepdims=True)
    remote_out = rd._mrl_np(full)
    local_out = Dense(EmbedConfig(dense_dim=1024))._mrl(torch.from_numpy(full))
    assert remote_out.shape == (3, 1024)
    assert np.allclose(remote_out, local_out, atol=1e-5)


def test_encode_text_and_query_cache():
    rd = RemoteDense(_fast_cfg(dense_dim=1024, query_instruction="Q-INSTR"))
    full = np.random.default_rng(1).standard_normal((1, 4096)).astype(np.float32)
    rd._client = _ScriptClient(_Resp(200, {"vectors": full.tolist()}))
    out = rd.encode_text(["hello"], instruction="I")
    c0 = rd._client.calls[0]
    assert c0["path"] == "/embed" and c0["json"]["texts"] == ["hello"] and c0["json"]["instruction"] == "I"
    assert out.shape == (1, 1024)                                       # 服务端全维 -> 客户端截到 dense_dim
    rd.encode_query("qq")
    assert rd._client.calls[1]["json"]["instruction"] == "Q-INSTR"      # encode_query 继承:自动带 query_instruction
    rd.encode_query("qq")
    assert len(rd._client.calls) == 2                                   # LRU 缓存命中,未新增 HTTP


def test_reranker_score_contract():
    rr = RemoteReranker(_fast_cfg())
    rr._client = _ScriptClient(_Resp(200, {"scores": [0.1, 0.9]}))
    assert rr.score("q", ["a", "b"]) == [0.1, 0.9]
    c = rr._client.calls[0]
    assert c["path"] == "/rerank" and c["json"]["query"] == "q" and c["json"]["documents"] == ["a", "b"]
    rr2 = RemoteReranker(_fast_cfg())                                   # 契约:分数数≠文档数 -> fail-loud
    rr2._client = _ScriptClient(_Resp(200, {"scores": [0.5]}))
    with pytest.raises(RuntimeError, match="契约破坏"):
        rr2.score("q", ["a", "b"])


# ---------- P0-1 客户端重试 ----------
def test_retry_succeeds_after_transient_503():
    rd = RemoteDense(_fast_cfg(dense_dim=8, inference_retries=2))
    rd._client = _ScriptClient(_Resp(503), _Resp(503), _Resp(200, {"vectors": [[0.0] * 8]}))
    out = rd.encode_text(["x"])
    assert out.shape == (1, 8)
    assert len(rd._client.calls) == 3                                  # 恰好 2 次重试后第 3 次成功


def test_retry_4xx_no_retry():
    rd = RemoteDense(_fast_cfg(inference_retries=3))
    rd._client = _ScriptClient(_Resp(400))
    with pytest.raises(httpx.HTTPStatusError):                         # 4xx 客户端错立即抛,不转 InferenceUnavailable
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 1                                  # 零重试


def test_retry_exhausts_raises_inference_unavailable():
    rd = RemoteDense(_fast_cfg(inference_retries=2))
    rd._client = _ScriptClient(_Resp(503))                            # 持续 503
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 3                                  # retries+1 次都试过


def test_retry_connect_error_then_exhaust():
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(httpx.ConnectError("refused"))         # 连不上(宕机)
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2


def test_inference_unavailable_is_exception_with_marker():
    e = InferenceUnavailable("x")
    assert getattr(e, "inference_unavailable", False) is True          # toolcore duck-typing 依据
    assert isinstance(e, Exception)                                    # 保证会被 retrieve.py rerank 的 except Exception 吞(降级)


# ---------- P0-1 toolcore 分流(dense loud) ----------
def test_toolcore_dispatches_inference_unavailable():
    from types import SimpleNamespace

    from embedder.types import User
    from pharos import toolcore as tc

    def boom_inf(*a, **k):
        raise InferenceUnavailable("503 预热中")

    def boom_other(*a, **k):
        raise RuntimeError("qdrant down")

    u = User("t", [])
    r_inf = SimpleNamespace(search_with_context=boom_inf, search_grouped=boom_inf)
    assert tc._retrieve_impl(r_inf, u, "q", None, False)["status"] == "inference_unavailable"
    assert tc._grouped_impl(r_inf, u, "q", ["d1"], None, False)["status"] == "inference_unavailable"
    # 反面:普通后端异常仍归通用 backend_unavailable(不误判成 inference)
    r_other = SimpleNamespace(search_with_context=boom_other, search_grouped=boom_other)
    assert tc._retrieve_impl(r_other, u, "q", None, False)["status"] == "backend_unavailable"
    assert tc._grouped_impl(r_other, u, "q", ["d1"], None, False)["status"] == "backend_unavailable"


# ---------- P0-2 readiness 优先级(err 优先于 loading) ----------
def test_readiness_error_precedence():
    from embedder.inference_server import _readiness
    body_err, code_err = _readiness(False, "CUDA error")
    assert code_err == 503 and body_err["status"] == "error"          # 加载失败 -> error,不报成 loading
    assert _readiness(False, None)[0]["status"] == "loading"          # 预热中 -> loading
    assert _readiness(True, None) == ({"status": "ready"}, 200)       # 就绪 -> ready


# ---------- P2-1 共享连接池 ----------
def test_get_client_shared_by_url():
    R._CLIENTS.clear()
    try:
        cfg = EmbedConfig(inference_url="http://shared:8900")
        d = RemoteDense(cfg)
        rr = RemoteReranker(cfg)
        assert d._client is rr._client                                # 同 url 的 dense+reranker 共用一个连接池
    finally:
        R._close_clients()                                            # 关掉本测试建的真 client


# ---------- 审查修:M2/M3/S1/S6b/M4 ----------
def test_retry_5xx_gateway_retried():                                 # M3:502/504 网关瞬态(被 kill 的后端)也该重试
    rd = RemoteDense(_fast_cfg(dense_dim=8, inference_retries=2))
    rd._client = _ScriptClient(_Resp(502), _Resp(504), _Resp(200, {"vectors": [[0.0] * 8]}))
    out = rd.encode_text(["x"])
    assert out.shape == (1, 8) and len(rd._client.calls) == 3


def test_retry_500_exhausts_to_inference_unavailable():              # M3:500 也瞬态(不当客户端错裸抛)
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(_Resp(500))
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2


def test_retry_connect_timeout_retried():                            # M2:ConnectTimeout 是 TimeoutException,非 ConnectError
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(httpx.ConnectTimeout("timeout"))      # 修 M2 前会裸抛、绕过重试
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2                                # 被捕获 -> 重试到耗尽


def test_negative_retries_no_crash():                                # S1:负配置 clamp 到 0,不 raise None(TypeError)
    rd = RemoteDense(_fast_cfg(inference_retries=-1))
    rd._client = _ScriptClient(_Resp(503))
    with pytest.raises(InferenceUnavailable):                        # 是语义异常,不是 TypeError
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 1                                # clamp 到 0 -> 试 1 次


def test_backoff_exponential_sequence(monkeypatch):                  # S6b:退避确实是指数(2^attempt),非线性
    slept = []
    monkeypatch.setattr(R.time, "sleep", lambda s: slept.append(s))
    rd = RemoteDense(EmbedConfig(inference_url="http://x", inference_retries=2, inference_backoff=0.5))
    rd._client = _ScriptClient(_Resp(503))                          # 持续 503 -> 退避 2 次后耗尽
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert slept == [0.5, 1.0]                                       # backoff*2^0, backoff*2^1


def test_rerank_degrades_on_inference_unavailable():                 # M4:非对称另半边——rerank 抛 InferenceUnavailable 被降级吞
    from types import SimpleNamespace

    from embedder.retrieve import Retriever
    from embedder.types import Hit, User

    hits = [Hit(f"c{i}", "d", "text", "t", 0.1, {}, "rrf") for i in range(3)]
    r = Retriever.__new__(Retriever)                                 # 不走 __init__(不建 Store/Dense)
    r.cfg = EmbedConfig()
    r.dense = SimpleNamespace(encode_query=lambda q: SimpleNamespace(tolist=lambda: [0.0] * 8))
    r.store = SimpleNamespace(hybrid_search=lambda *a, **k: list(hits))

    def boom(*a, **k):
        raise InferenceUnavailable("推理服务 503")

    r._get_reranker = lambda: SimpleNamespace(rerank=boom)
    out = r.search("x", User("t", []), top_k=2, rerank=True)
    assert [h.chunk_id for h in out] == ["c0", "c1"]                 # rerank 失败降级返回 hybrid top-2,不 loud、不崩
    assert all(h.score_kind == "rrf" for h in out)                   # 仍 hybrid 量纲(dense loud / rerank 降级 的非对称)
