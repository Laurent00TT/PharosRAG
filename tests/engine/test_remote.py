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
    """按脚本逐次响应:每项是 _Resp(返回)或 Exception 实例(抛出)。用尽后重复最后一项。记录 calls。
    get(健康握手 fix3):health 缺省返回 200 空 dict(字段缺失=不阻断);health_exc 模拟探活不可达。"""
    def __init__(self, *steps, health=None, health_exc=None):
        self.steps = list(steps) or [_Resp(200, {})]
        self.calls = []
        self.health, self.health_exc = health, health_exc
        self.get_calls = 0

    def post(self, path, json=None):
        self.calls.append({"path": path, "json": json})
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step

    def get(self, path):
        self.get_calls += 1
        if self.health_exc is not None:
            raise self.health_exc
        return _Resp(200, self.health or {})


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


def test_mrl_fp32_normalize_on_bf16_input():
    """阶段B审查 M1:真实建库入口是 **bf16 张量**进 Dense._mrl(model.process 返回 bf16)。fp32 修复(先 .float()
    再截+normalize)-> norm=1.0;反向守卫证明 bf16 上 normalize 会坏(norm≈1.002)。删 dense.py:_mrl 的 .float()
    本测试即红——守住 P1-2 修复。纯 CPU(torch CPU),进 CI,不需 GPU。"""
    import torch
    import torch.nn.functional as F

    from embedder.dense import Dense
    torch.manual_seed(0)                                             # 固定随机(bf16 舍入偏差依赖具体值,否则 flaky)
    emb_bf16 = torch.randn(32, 4096).bfloat16()                     # 模拟 model.process 的 bf16 输出(生产建库入口)
    v = Dense(EmbedConfig(dense_dim=1024))._mrl(emb_bf16)            # 生产路径:bf16 -> _mrl(.float() 后截+normalize)
    assert v.dtype == np.float32
    v_dev = float(np.abs(np.linalg.norm(v, axis=-1) - 1.0).max())   # 偏离 1(双向)
    assert v_dev < 1e-5, f"fp32 修复应让 norm=1.0(偏离<1e-5),实测偏离 {v_dev:.2e}"
    bad = F.normalize(emb_bf16[:, :1024], p=2, dim=-1).float().cpu().numpy()   # 旧路径:bf16 上 normalize
    bad_dev = float(np.abs(np.linalg.norm(bad, axis=-1) - 1.0).max())
    assert bad_dev > 5e-4, \
        f"bf16 上 normalize 应产生明显 norm 偏差(证明 fp32 修复必要 + 本测试能抓回归),实测偏离 {bad_dev:.2e}"


# ---------- P1-1 下界断言(纯 CPU;从 test_equivalence_gpu.py 移来,避免被 GPU skipif 吞成假绿) ----------
def test_mrl_np_lower_bound_assertion():
    """P1-1:客户端 dense_dim > 服务端全维 -> fail-loud(不静默返回过短向量灌进库)。纯 CPU,常驻回归网。"""
    rd = RemoteDense.__new__(RemoteDense)                     # 不 __init__(不建 httpx client)
    rd.cfg = EmbedConfig(dense_dim=8192)                      # > 4096 全维
    with pytest.raises(RuntimeError, match="配置错位"):
        rd._mrl_np(np.zeros((2, 4096), dtype=np.float32))


# ---------- 阶段F 审查:TransportError 谱扩宽(docker kill/restart inference 的断连形态) ----------
def test_retry_read_error_retried():
    """ReadError(对端 RST)是 TransportError 但**非** ConnectError/TimeoutException —— 旧的窄捕获会绕过重试、
    绕过 InferenceUnavailable 语义,被吞成无重试的 backend_unavailable。改 `except httpx.TransportError` 后应被吸收。"""
    rd = RemoteDense(_fast_cfg(dense_dim=8, inference_retries=2))
    rd._client = _ScriptClient(httpx.ReadError("connection reset"),
                               _Resp(200, {"vectors": [[0.0] * 8]}))
    out = rd.encode_text(["x"])
    assert out.shape == (1, 8) and len(rd._client.calls) == 2      # 第一次 ReadError 被重试,第二次成功


def test_retry_remote_protocol_error_then_exhaust():
    """RemoteProtocolError("Server disconnected without sending a response")是 keep-alive 死连接被 docker
    kill/restart 后复用的最典型异常。旧窄捕获漏它 -> "kill 无感"链条断。改 TransportError 后应重试到耗尽抛语义异常。"""
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(httpx.RemoteProtocolError("Server disconnected"))
    with pytest.raises(InferenceUnavailable):                     # 非裸 RemoteProtocolError 冒泡
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2                             # retries+1 都试过


# ---------- 修复3:首查握手(healthz 的 model_dense/full_dim 校验,防换模型后向量空间静默错位) ----------
_OK_VEC = _Resp(200, {"vectors": [[0.0] * 8]})


def test_handshake_model_mismatch_fail_loud():
    """服务端换了模型(model_dense 与客户端 dense_model_path basename 不符)-> 首查 RuntimeError,
    不发业务 POST(否则错空间向量已经灌出去了)。此前仅 _mrl_np 下界断言,同维不同模型完全静默。"""
    rd = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health={"model_dense": "OtherModel", "full_dim": 4096})
    with pytest.raises(RuntimeError, match="模型身份错配"):
        rd.encode_text(["x"])
    assert rd._client.calls == []                                 # 校验在业务请求之前


def test_handshake_full_dim_below_dense_dim_fail_loud():
    rd = RemoteDense(_fast_cfg(dense_dim=1024, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health={"model_dense": "MyEmb", "full_dim": 512})
    with pytest.raises(RuntimeError, match="full_dim"):
        rd.encode_text(["x"])


def test_handshake_ok_verifies_once():
    """两字段都校验通过 -> 置标志位,后续查询零 GET 开销。"""
    rd = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health={"model_dense": "MyEmb", "full_dim": 4096})
    rd.encode_text(["a"])
    rd.encode_text(["b"])
    assert rd._client.get_calls == 1                              # 只握手一次
    assert len(rd._client.calls) == 2                             # 业务照常


def test_handshake_unreachable_or_warming_not_blocking():
    """预热边界:healthz 不可达 / full_dim 尚为 None(预热中)都不阻断业务(瞬态留给 _post_retry 重试链),
    且不算握手完成 —— 下次查询继续尝试,直到验上为止。"""
    rd = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health_exc=httpx.ConnectError("refused"))
    assert rd.encode_text(["a"]).shape == (1, 8)                  # 探活失败不挡业务
    rd2 = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd2._client = _ScriptClient(_OK_VEC, health={"model_dense": "MyEmb", "full_dim": None})   # 预热中
    rd2.encode_text(["a"])
    rd2.encode_text(["b"])
    assert rd2._client.get_calls == 2                             # 没验上 -> 每查重试握手(验上即停,见上一测试)


# ---------- 修复5:RemoteReranker.score 签名与基类对齐(instruction 形参 + 透传 payload) ----------
def test_remote_reranker_score_instruction_passthrough():
    """显式 instruction 进 payload(基类契约:per-call 覆盖);不传时回落 cfg.rerank_instruction(旧行为不变)。
    删 remote.py score 的 instruction 形参即红 —— LSP 收窄输入域回归。"""
    rr = RemoteReranker(_fast_cfg(rerank_instruction="CFG_DEFAULT"))
    rr._client = _ScriptClient(_Resp(200, {"scores": [0.5]}))
    rr.score("q", ["a"], instruction="PER_CALL")
    assert rr._client.calls[0]["json"]["instruction"] == "PER_CALL"
    rr.score("q", ["a"])
    assert rr._client.calls[1]["json"]["instruction"] == "CFG_DEFAULT"
