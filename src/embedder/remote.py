"""远程推理后端(生产拆分):dense/rerank 的 GPU 前向走 HTTP 调独立推理服务,本进程不加载模型。

**为什么**:模型在进程内加载 = 应用层绑死 GPU、无法水平扩展。把 GPU 前向拆成独立服务(inference_server.py)
后,应用层(pharos serve)变无状态、不吃 GPU,能多副本 + 上编排。这是"上 K8s 弹性"的前置(见 docs/SCALE_OUT.md)。

**等价性保证(关键)**:推理服务用"全维"配置返回未截维的 normalized 向量;客户端(RemoteDense)拿回后用
自己真实的 dense_dim 做 MRL 截维。数学上 local 与 remote 产出**同一个最终向量**——同一个库能 local 建、
remote 查,不会错位。RemoteDense/RemoteReranker **继承** Local 基类,只 override "前向"方法(走 HTTP),
MRL 截维 / query LRU 缓存 / 排序写回 Hit 这些**业务逻辑全部继承复用**,零重复、行为一致。

**失败模式(SCALE_OUT.md §5-A / P0-1)**:推理服务预热(1-2 分钟)、副本滚动重启是正常运维态。`_post_retry`
对 503/连接错/读超时做有限重试 + 指数退避吸收瞬态;耗尽抛语义异常 `InferenceUnavailable`(errors.py),
上层(toolcore duck-typing)据此返回可重试的 `inference_unavailable`,而非无重试的通用 `backend_unavailable`。
4xx(客户端错)立即抛 httpx.HTTPStatusError,不重试。dense/reranker 共享一个连接池(按 url 缓存),atexit 统一关。

后端选择由 EmbedConfig.inference_url 决定(空=local),经 make_dense/make_reranker 工厂(下方)。
"""
from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time

import numpy as np

from .config import EmbedConfig
from .dense import Dense
from .errors import InferenceUnavailable
from .rerank import Reranker

# 模块级 client 缓存:同一 inference_url 的 dense+reranker **共用一个连接池**(避免双连接池浪费)。
# 进程级单例,atexit 统一 close。审查 S3/P3:并发构造 Remote* 时 check-then-set 有竞态
# (输家 client 脱挂、atexit 关不到 -> 连接泄漏),故 get-or-create 与 close 同一把锁串行。
_CLIENTS: dict[str, "object"] = {}
_CLIENTS_LOCK = threading.Lock()


def _get_client(cfg: EmbedConfig):
    """按 inference_url 复用 httpx.Client(dense/reranker 共享);拆分超时:connect 短、read 长。
    connect 短 -> 服务端 hang/宕机时快速失败进重试,不套用 read 的 120s;read 长 -> 容忍长文本前向。
    锁内 get-or-create(构造 httpx.Client 很轻,锁开销可忽略):消除并发下重复建池 + 脱挂泄漏。"""
    import httpx
    url = cfg.inference_url.rstrip("/")
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(url)
        if c is None:
            c = httpx.Client(base_url=url, timeout=httpx.Timeout(
                connect=cfg.inference_connect_timeout, read=cfg.inference_timeout, write=10.0, pool=5.0))
            _CLIENTS[url] = c
        return c


@atexit.register
def _close_clients() -> None:
    """进程退出统一关连接池(避免连接泄漏);幂等、吞异常(退出路径不应因关连接报错)。"""
    with _CLIENTS_LOCK:
        for c in list(_CLIENTS.values()):
            try:
                c.close()
            except Exception:
                pass
        _CLIENTS.clear()


def _post_retry(client, cfg: EmbedConfig, path: str, payload: dict) -> dict:
    """POST + 有限重试(P0-1)。可重试(指数退避):
      - **任意 5xx**(503 未就绪 / 502·504 网关无健康后端 / 500 推理崩一次自愈)—— 都是滚动重启期的瞬态;
      - **任意 httpx 传输层异常**(`TransportError`:含 Connect/Read/Write/Close/RemoteProtocol + 全部 Timeout)。
    不重试:4xx 客户端错 —— `raise_for_status()` 立即抛 httpx.HTTPStatusError(它**非** TransportError,继续冒泡,重试无益)。
    重试耗尽 -> 抛 InferenceUnavailable(语义异常,上层据此返回可重试 inference_unavailable)。

    刀刃说明:①`>= 500` 而非硬编码 `== 503` —— 审查 M3:K8s/Nginx 对被 kill 的后端返 502/504,硬编码 503 会
    把它们当客户端错裸抛;②`except httpx.TransportError` 而非只捕 `ConnectError+TimeoutException` —— 阶段F审查(高):
    **docker kill/restart inference 的最常见断连形态是 `RemoteProtocolError`("Server disconnected without sending a
    response",keep-alive 死连接复用)/ `ReadError`(对端 RST)**,二者既非 ConnectError 也非 TimeoutException,漏了它们
    则"kill 无感"链条断——被吞成无重试的通用 backend_unavailable。TransportError 是它们与超时/连接错的共同父类,
    且不含 `HTTPStatusError`(4xx 仍立即冒泡),一网打尽又不误吞客户端错;③`max(0, retries)` clamp —— 审查 S1:
    负配置不至于空 range 后 `raise None` 掩盖真因。"""
    import httpx
    retries = max(0, cfg.inference_retries)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = client.post(path, json=payload)
            if r.status_code >= 500:                       # 5xx 皆瞬态 -> 可重试
                last = InferenceUnavailable(f"推理服务 {r.status_code} @ {path}")
            else:
                r.raise_for_status()                       # 4xx 客户端错 -> 立即抛,不重试
                return r.json()
        except httpx.TransportError as e:                  # 连接被拒/断连(RST·keep-alive 复用死连接)+ 全部超时
            last = InferenceUnavailable(f"推理服务 {type(e).__name__} @ {path}")
        if attempt < retries:                              # 还有重试机会 -> 指数退避
            time.sleep(cfg.inference_backoff * (2 ** attempt))
    raise last                                             # 重试耗尽(last 必为 InferenceUnavailable)


class RemoteDense(Dense):
    """dense 前向走 HTTP。继承 Dense 复用 _mrl(客户端 dense_dim 截维)+ encode_query 缓存。
    只 override:_load(不加载模型)、encode_text/encode_image(调推理服务,拿全维向量后本地 _mrl)。"""

    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        self._client = _get_client(cfg)         # 共享连接池(按 url);测试可替换 self._client 注入 mock
        # M1:remote 走 HTTP,无本地 GPU 前向/加载 -> 前向&加载锁置空,退避/HTTP 不持任何锁(_cache_lock 仍继承生效)
        self._fwd_lock = contextlib.nullcontext()
        self._load_lock = contextlib.nullcontext()
        self._handshake_done = False            # 一次性模型/维度握手(_verify_server);成功校验后不再发 GET
        self._handshake_lock = threading.Lock()

    def _load(self) -> None:                    # 远程后端本进程不加载模型、不碰 GPU
        return

    def _verify_server(self) -> None:
        """首查前的一次性握手:GET /healthz,比对服务端模型身份与全维(评审修:换服务端模型不重建库时,
        _mrl_np 只有 full_dim<dense_dim 的下界断言,同维不同语义空间会静默错位,召回崩塌只能靠 eval 事后发现)。
        边界:探活不可达/字段缺失或 None(预热中 full_dim=None,见 inference_server.py)**不阻断**——瞬态留给
        _post_retry 既有重试链;只有拿到字段且不符才 RuntimeError fail-loud(配置错配非瞬态,不该被重试吸收)。
        两个字段都校验过才置 _handshake_done,预热期每查重试直到验上。"""
        if self._handshake_done:
            return
        with self._handshake_lock:              # single-flight:并发首查只发一次 GET(重复也幂等,锁只省流量)
            if self._handshake_done:
                return
            try:
                r = self._client.get("/healthz")
                if r.status_code != 200:
                    return
                data = r.json()
            except Exception:                   # 不可达/非 JSON:不阻断,交给业务请求的重试链
                return
            model, full_dim = data.get("model_dense"), data.get("full_dim")
            if model is not None and model != os.path.basename(self.cfg.dense_model_path):
                raise RuntimeError(
                    f"推理服务模型身份错配:服务端 model_dense={model!r} != 客户端 "
                    f"{os.path.basename(self.cfg.dense_model_path)!r}(PHAROS_DENSE_MODEL_PATH)。"
                    f"向量空间会静默错位 —— 对齐两端模型或重建库。")
            if full_dim is not None and int(full_dim) < self.cfg.dense_dim:
                raise RuntimeError(
                    f"推理服务全维 full_dim={full_dim} < 客户端 dense_dim={self.cfg.dense_dim},配置错位 —— "
                    f"检查 PHAROS_DENSE_DIM 与推理服务模型是否匹配。")
            if model is not None and full_dim is not None:
                self._handshake_done = True

    def _post_vectors(self, path: str, payload: dict) -> np.ndarray:
        self._verify_server()                                       # lazy 握手:模型/维度错配 fail-loud(仅首次真发 GET)
        data = _post_retry(self._client, self.cfg, path, payload)   # 带重试;耗尽抛 InferenceUnavailable
        return np.asarray(data["vectors"], dtype=np.float32)        # 推理服务返回**全维** normalized 向量

    def encode_text(self, texts: list[str], instruction: str | None = None) -> np.ndarray:
        full = self._post_vectors("/embed", {"texts": texts, "instruction": instruction})
        return self._mrl_np(full)               # 客户端按真实 dense_dim 截维 + renorm(与 local 等价)

    def encode_image(self, image_paths: list[str], instruction: str | None = None) -> np.ndarray:
        # image_paths 需是推理服务可读到的路径(同机/共享卷)。跨机应改 base64——见 inference_server.py TODO。
        full = self._post_vectors("/embed_image", {"image_paths": image_paths, "instruction": instruction})
        return self._mrl_np(full)

    def _mrl_np(self, full: np.ndarray) -> np.ndarray:
        """对全维 numpy 向量做 MRL 截维 + L2 renorm(纯 numpy,与 Dense._mrl 的 torch 版数学等价)。
        用 numpy 而非 torch:远程后端不该为了截维引入 torch 依赖(应用层要能无 GPU/无 torch 跑)。"""
        d = self.cfg.dense_dim
        if full.shape[-1] < d:              # P1-1:客户端 dense_dim > 服务端全维 = 配置错位,fail-loud(不静默返回过短向量灌进库)
            raise RuntimeError(
                f"dense_dim={d} > 推理服务返回的全维={full.shape[-1]},配置错位 —— "
                f"检查 PHAROS_DENSE_DIM 与推理服务模型是否匹配。")
        if full.shape[-1] > d:
            v = full[:, :d]
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            v = v / np.clip(norm, 1e-12, None)
            return v.astype(np.float32)
        return full.astype(np.float32)      # == d:全维即目标维,已 normalized,直接返回


class RemoteReranker(Reranker):
    """rerank 走 HTTP。继承 Reranker 复用 rerank()(排序+写回 Hit.score/score_kind);只 override
    _load(不加载)、score(调推理服务)。score 失败(耗尽抛 InferenceUnavailable)会被 retrieve.py 的
    rerank try/except 吞掉 -> 降级 hybrid(rerank 是增强信号,降级安全;与 dense loud 的非对称是有意的)。"""

    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        self._client = _get_client(cfg)
        self._fwd_lock = contextlib.nullcontext()      # M1:remote rerank 走 HTTP,前向&加载锁置空
        self._load_lock = contextlib.nullcontext()

    def _load(self) -> None:
        return

    def score(self, query: str, docs_text: list[str], instruction: str | None = None) -> list[float]:
        # instruction 形参与基类 Reranker.score 对齐(评审修:此前缺参,按基类契约传 instruction 直接 TypeError,
        # 且 remote 永远只能用 cfg 默认 —— LSP 子类收窄输入域);None 回落 cfg 与旧行为完全一致。
        data = _post_retry(self._client, self.cfg, "/rerank",
                           {"query": query, "documents": list(docs_text),
                            "instruction": instruction or self.cfg.rerank_instruction})
        scores = data["scores"]
        if len(scores) != len(docs_text):       # 与 local 同款契约校验:分数与文档一一对应,破坏则 fail-loud
            raise RuntimeError(f"remote reranker 返回 {len(scores)} 分 != {len(docs_text)} 文档,API 契约破坏")
        return [float(s) for s in scores]


# ---------- 后端工厂:唯一决定 local vs remote 的地方 ----------
def make_dense(cfg: EmbedConfig | None = None) -> Dense:
    """按 cfg.inference_url 选 dense 后端:空 -> Local Dense(进程内 GPU,默认,向后兼容);非空 -> RemoteDense。"""
    cfg = cfg or EmbedConfig()
    return RemoteDense(cfg) if cfg.inference_url else Dense(cfg)


def make_reranker(cfg: EmbedConfig | None = None) -> Reranker:
    cfg = cfg or EmbedConfig()
    return RemoteReranker(cfg) if cfg.inference_url else Reranker(cfg)
