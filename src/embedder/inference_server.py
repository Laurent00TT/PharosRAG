"""GPU 模型推理服务(生产拆分):把 embedder 里唯一吃 GPU 的部分(Qwen3-VL embedding + reranker)
独立成一个 HTTP 服务。应用层(pharos serve)配 EmbedConfig.inference_url 指向本服务后即不再加载模型、
不吃 GPU -> 无状态可多副本。客户端见 remote.py。

**纯 GPU 前向,无业务逻辑**:只暴露 model.process 的原始输出(全维 normalized 向量 / 原始 rerank 分数);
MRL 截维、query 缓存、排序写回 Hit 全部留在客户端(remote.py)。这样本服务纯粹到将来能换成 TEI/vLLM/Triton。

**全维返回**:用 dense_dim=极大值构造 Dense,让其 _mrl 永不截维 -> 返回模型原始全维向量;客户端按真实
dense_dim 截。保证 local 与 remote 最终向量等价(remote.py 说明)。

**真 readiness**:startup 后台线程预热两个 8B 模型(各 1-2 分钟);/readyz 在热完前返 503 —— 编排系统
(compose/K8s)据此在模型没热前不导流量。这正是"liveness(进程活着) vs readiness(能服务了)"的教学点。

**GPU 前向串行化**:单卡上并发 model.process 非线程安全,用一把锁串行(同 pharos LockedRetriever 思路)。

跑:conda activate navikb && python -m embedder.inference_server   # 默认 0.0.0.0:8900
需 pip install fastapi uvicorn。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import replace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import EmbedConfig
from .dense import Dense
from .rerank import Reranker

_FULL_DIM = 10 ** 9        # 极大 dense_dim -> _mrl 永不截,返回模型原始全维(客户端再按真实维度截)
# 预热自愈:瞬态失败(非永久配置错)重试次数 + 退避基;耗尽仍失败则 os._exit(1) 让 restart 拉起(阶段F 审查)。
_WARMUP_RETRIES = int(os.environ.get("INFERENCE_WARMUP_RETRIES", "3"))
_WARMUP_BACKOFF = float(os.environ.get("INFERENCE_WARMUP_BACKOFF", "5.0"))
# 背压:同时在途(执行 + 等 GPU 锁)上限。GPU 前向串行,深队列纯浪费 + 拥塞正反馈;满则 503 overloaded 快速失败。
_MAX_INFLIGHT = int(os.environ.get("INFERENCE_MAX_INFLIGHT", "16"))


def _readiness(ready: bool, err: str | None) -> tuple[dict, int]:
    """就绪判定(纯函数,单测友好)。**加载失败优先于未就绪**(P0-2 核心):
    - err(加载永久失败:错卡/缺模型)-> error 503。别报成 loading,否则运维看"永远预热中"而非"加载失败",误导排障;
    - 未 ready(预热中)-> loading 503;
    - 就绪 -> ready 200。
    readyz 与各端点 _guard 共用这一处,避免"三处各判一遍"的漂移(v1 的 _guard 就漏了 err 这支)。"""
    if err:
        return {"status": "error", "detail": err}, 503
    if not ready:
        return {"status": "loading", "hint": "模型预热中,请稍后重试"}, 503
    return {"status": "ready"}, 200


class EmbedReq(BaseModel):
    texts: list[str] = []
    instruction: str | None = None


class EmbedImageReq(BaseModel):
    image_paths: list[str] = []          # 推理服务可读到的路径(同机/共享卷);跨机应改 base64(TODO)
    instruction: str | None = None


class RerankReq(BaseModel):
    query: str = ""
    documents: list[str] = []
    instruction: str | None = None       # P2-3 后被消费(客户端唯一真相):非 None 用客户端值,None 回落服务端 cfg 默认


def create_app(cfg: EmbedConfig | None = None) -> FastAPI:
    cfg = cfg or EmbedConfig()
    full = replace(cfg, dense_dim=_FULL_DIM, inference_url="")   # 全维 + 强制 local(不递归成 remote)
    dense = Dense(full)
    reranker = Reranker(full)

    app = FastAPI(title="pharos-inference")
    state = app.state
    state.ready = False
    state.err: str | None = None
    state.gpu_lock = threading.Lock()    # 单卡 GPU 前向串行化
    state.full_dim: int | None = None    # 模型全维(预热探针取);healthz 暴露:供排障 + 客户端首查握手校验(remote._verify_server)
    # 有界准入(阶段F 审查·背压):GPU 前向串行,过载时无上限排队 = 客户端 read-timeout 放弃后请求仍在队里
    # 烧 GPU + 重试再入队 = 3× 无效功 + 拥塞正反馈。信号量把在途(执行中 + 等锁)钉在 _MAX_INFLIGHT,
    # 满则立即 503 overloaded;客户端把 5xx 当 InferenceUnavailable 退避重试,天然兼容,队列长度有界。
    state.inflight = threading.BoundedSemaphore(_MAX_INFLIGHT)

    @app.on_event("startup")
    def _warmup():
        def load():
            for attempt in range(_WARMUP_RETRIES + 1):
                try:
                    dense._load()            # 各 1-2 分钟(8B),后台预热;热完前 /readyz 返 503
                    reranker._load()
                    probe = dense._model.process([{"text": "dim-probe", "instruction": None}], normalize=True)
                    state.full_dim = int(probe.shape[-1])   # 全维(如 4096);此处 dense_dim=_FULL_DIM 故 _mrl 不截
                    state.ready = True
                    state.err = None
                    return
                except Exception as e:       # 加载失败:区分永久配置错 vs 瞬态
                    state.err = f"{type(e).__name__}: {e}"
                    # 永久配置错(错卡/缺 CUDA,Dense 已 _gpu_error 缓存)重试无益 -> 直接停,别空转烧日志。
                    if getattr(dense, "_gpu_error", None) is not None:
                        break
                    if attempt < _WARMUP_RETRIES:   # 瞬态(如共享卷抖动/传递依赖偶发)-> 指数退避重试
                        time.sleep(_WARMUP_BACKOFF * (2 ** attempt))
            # 阶段F 审书·自愈:瞬态耗尽仍失败 -> 进程自杀,让 restart:unless-stopped 拉起重来
            # (start_period 兜住重启预热窗口)。永久配置错不自杀(crashloop 无意义,留 err 供 /healthz 排障)。
            if not state.ready and getattr(dense, "_gpu_error", None) is None:
                import sys
                print(f"FATAL inference 预热重试耗尽,退出让编排重启:{state.err}", file=sys.stderr, flush=True)
                os._exit(1)
        threading.Thread(target=load, daemon=True).start()

    @app.get("/healthz")                 # liveness:进程活着(不代表模型热)。**加载永久失败也仍报 ok** ——
    async def healthz():                 # async:纯内存读,不进 40-线程池 -> 高负载下探针不饥饿(阶段F 审查)。
        return {"status": "ok", "service": "inference", "ready": state.ready, "error": state.err,
                "full_dim": state.full_dim, "model_dense": os.path.basename(cfg.dense_model_path)}
        # ↑ err 供人工排障;full_dim/model_dense 被 remote._verify_server 首查握手消费(模型身份/维度错配 fail-loud,
        #   预热期 full_dim=None 时客户端不阻断、下查重试)。liveness 判定不变,导流量用 /readyz。

    @app.get("/readyz")                  # readiness:模型热了、能服务了 —— 编排据此导流量
    async def readyz():                  # async:纯内存读 state.ready/err,不进线程池(阶段F 审查:探针与 GPU 业务共池会饥饿假 unhealthy)
        body, code = _readiness(state.ready, state.err)
        return body if code == 200 else JSONResponse(body, status_code=code)

    def _guard():                        # 端点守卫:未就绪(含加载失败)一律 503,且区分 error(P0-2)/loading
        body, code = _readiness(state.ready, state.err)
        return None if code == 200 else JSONResponse(body, status_code=code)

    def _admit():                        # 背压守卫:在途满则立即 503 overloaded(非阻塞尝试);返回 token(release 用)或 None+503 响应
        if not state.inflight.acquire(blocking=False):
            return None, JSONResponse({"status": "overloaded",
                                       "hint": "推理服务在途已满,请退避重试。"}, status_code=503)
        return state.inflight, None

    @app.post("/embed")
    def embed(q: EmbedReq):
        g = _guard()
        if g is not None:
            return g
        tok, over = _admit()
        if over is not None:
            return over
        try:
            with state.gpu_lock:
                vecs = dense.encode_text(q.texts, instruction=q.instruction)   # 全维(dense_dim=极大不截)
            return {"vectors": vecs.tolist()}
        finally:
            tok.release()

    @app.post("/embed_image")
    def embed_image(q: EmbedImageReq):
        g = _guard()
        if g is not None:
            return g
        tok, over = _admit()
        if over is not None:
            return over
        try:
            with state.gpu_lock:
                vecs = dense.encode_image(q.image_paths, instruction=q.instruction)
            return {"vectors": vecs.tolist()}
        finally:
            tok.release()

    @app.post("/rerank")
    def rerank(q: RerankReq):
        g = _guard()
        if g is not None:
            return g
        tok, over = _admit()
        if over is not None:
            return over
        try:
            with state.gpu_lock:
                # P2-3:消费客户端 payload 的 instruction(客户端唯一真相);None 时 score 内回落服务端 cfg 默认。
                scores = reranker.score(q.query, q.documents, instruction=q.instruction)
            return {"scores": scores}
        finally:
            tok.release()

    return app


def main() -> None:
    import uvicorn
    host = os.environ.get("INFERENCE_HOST", "0.0.0.0")
    port = int(os.environ.get("INFERENCE_PORT", "8900"))
    print(f"pharos-inference  http://{host}:{port}  (模型后台预热中,/readyz 热完转 ready)", flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
