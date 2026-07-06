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
import time

import numpy as np

from .config import EmbedConfig
from .dense import Dense
from .errors import InferenceUnavailable
from .rerank import Reranker

# 模块级 client 缓存:同一 inference_url 的 dense+reranker **共用一个连接池**(避免双连接池浪费)。
# 进程级单例,atexit 统一 close。
_CLIENTS: dict[str, "object"] = {}


def _get_client(cfg: EmbedConfig):
    """按 inference_url 复用 httpx.Client(dense/reranker 共享);拆分超时:connect 短、read 长。
    connect 短 -> 服务端 hang/宕机时快速失败进重试,不套用 read 的 120s;read 长 -> 容忍长文本前向。"""
    import httpx
    url = cfg.inference_url.rstrip("/")
    c = _CLIENTS.get(url)
    if c is None:
        c = httpx.Client(base_url=url, timeout=httpx.Timeout(
            connect=cfg.inference_connect_timeout, read=cfg.inference_timeout, write=10.0, pool=5.0))
        _CLIENTS[url] = c
    return c


@atexit.register
def _close_clients() -> None:
    """进程退出统一关连接池(避免连接泄漏);幂等、吞异常(退出路径不应因关连接报错)。"""
    for c in list(_CLIENTS.values()):
        try:
            c.close()
        except Exception:
            pass
    _CLIENTS.clear()


def _post_retry(client, cfg: EmbedConfig, path: str, payload: dict) -> dict:
    """POST + 有限重试(P0-1)。可重试(指数退避):
      - **任意 5xx**(503 未就绪 / 502·504 网关无健康后端 / 500 推理崩一次自愈)—— 都是滚动重启期的瞬态;
      - **任意 httpx 超时**(TimeoutException:connect/read/write/pool)+ ConnectError(连接被拒)。
    不重试:4xx 客户端错 —— `raise_for_status()` 立即抛 httpx.HTTPStatusError(重试无益)。
    重试耗尽 -> 抛 InferenceUnavailable(语义异常,上层据此返回可重试 inference_unavailable)。

    刀刃说明:①`>= 500` 而非硬编码 `== 503` —— 审查 M3:K8s/Nginx 对被 kill 的后端返 502/504,硬编码 503 会
    把它们当客户端错裸抛;②`httpx.TimeoutException` 而非 ReadTimeout/PoolTimeout —— 审查 M2:`ConnectTimeout`
    是 TimeoutException 子类**不是 ConnectError 子类**,漏了它 connect 超时(config 明设 connect=3s)会绕过重试;
    ③`max(0, retries)` clamp —— 审查 S1:负配置不至于空 range 后 `raise None` 掩盖真因。"""
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
        except (httpx.ConnectError, httpx.TimeoutException) as e:   # 连接被拒 + 全部超时(connect/read/write/pool)
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

    def _load(self) -> None:                    # 远程后端本进程不加载模型、不碰 GPU
        return

    def _post_vectors(self, path: str, payload: dict) -> np.ndarray:
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
        用 numpy 而非 torch:远程后端不该为了截维引入 torch 依赖(应用层要能无 GPU/无 torch 跑)。
        (P1-1 下界断言留阶段 B 补 —— SCALE_OUT.md §5-B。)"""
        d = self.cfg.dense_dim
        if full.shape[-1] > d:
            v = full[:, :d]
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            v = v / np.clip(norm, 1e-12, None)
            return v.astype(np.float32)
        return full.astype(np.float32)


class RemoteReranker(Reranker):
    """rerank 走 HTTP。继承 Reranker 复用 rerank()(排序+写回 Hit.score/score_kind);只 override
    _load(不加载)、score(调推理服务)。score 失败(耗尽抛 InferenceUnavailable)会被 retrieve.py 的
    rerank try/except 吞掉 -> 降级 hybrid(rerank 是增强信号,降级安全;与 dense loud 的非对称是有意的)。"""

    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        self._client = _get_client(cfg)

    def _load(self) -> None:
        return

    def score(self, query: str, docs_text: list[str]) -> list[float]:
        data = _post_retry(self._client, self.cfg, "/rerank",
                           {"query": query, "documents": list(docs_text),
                            "instruction": self.cfg.rerank_instruction})
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
