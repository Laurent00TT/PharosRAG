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
from dataclasses import replace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import EmbedConfig
from .dense import Dense
from .rerank import Reranker

_FULL_DIM = 10 ** 9        # 极大 dense_dim -> _mrl 永不截,返回模型原始全维(客户端再按真实维度截)


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
    instruction: str | None = None       # 冗余:rerank instruction 由服务端 cfg 定(约定两端一致,默认天然一致)


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
    state.full_dim: int | None = None    # 模型全维(预热探针取);healthz 暴露供排障(客户端自动校验待做,当前靠 remote._mrl_np 运行时下界断言兜底)

    @app.on_event("startup")
    def _warmup():
        def load():
            try:
                dense._load()            # 各 1-2 分钟(8B),后台预热;热完前 /readyz 返 503
                reranker._load()
                probe = dense._model.process([{"text": "dim-probe", "instruction": None}], normalize=True)
                state.full_dim = int(probe.shape[-1])   # 全维(如 4096);此处 dense_dim=_FULL_DIM 故 _mrl 不截
                state.ready = True
            except Exception as e:       # 加载失败(缺卡/错卡/模型缺):readyz 暴露,编排不导流量
                state.err = f"{type(e).__name__}: {e}"
        threading.Thread(target=load, daemon=True).start()

    @app.get("/healthz")                 # liveness:进程活着(不代表模型热)。**加载永久失败也仍报 ok** ——
    def healthz():                       # liveness=进程活着;配置错(错卡/缺模型)重启也修不好,报不健康只会 crashloop。
        return {"status": "ok", "service": "inference", "ready": state.ready, "error": state.err,
                "full_dim": state.full_dim, "model_dense": os.path.basename(cfg.dense_model_path)}
        # ↑ err/full_dim 仅**暴露供人工排障**(暂无客户端自动读取校验;dense_dim 错配靠 remote._mrl_np 运行时下界断言);liveness 判定不变,导流量用 /readyz。

    @app.get("/readyz")                  # readiness:模型热了、能服务了 —— 编排据此导流量
    def readyz():
        body, code = _readiness(state.ready, state.err)
        return body if code == 200 else JSONResponse(body, status_code=code)

    def _guard():                        # 端点守卫:未就绪(含加载失败)一律 503,且区分 error(P0-2)/loading
        body, code = _readiness(state.ready, state.err)
        return None if code == 200 else JSONResponse(body, status_code=code)

    @app.post("/embed")
    def embed(q: EmbedReq):
        g = _guard()
        if g is not None:
            return g
        with state.gpu_lock:
            vecs = dense.encode_text(q.texts, instruction=q.instruction)   # 全维(dense_dim=极大不截)
        return {"vectors": vecs.tolist()}

    @app.post("/embed_image")
    def embed_image(q: EmbedImageReq):
        g = _guard()
        if g is not None:
            return g
        with state.gpu_lock:
            vecs = dense.encode_image(q.image_paths, instruction=q.instruction)
        return {"vectors": vecs.tolist()}

    @app.post("/rerank")
    def rerank(q: RerankReq):
        g = _guard()
        if g is not None:
            return g
        with state.gpu_lock:
            scores = reranker.score(q.query, q.documents)    # 原始 0~1 分,与文档一一对应
        return {"scores": scores}

    return app


def main() -> None:
    import uvicorn
    host = os.environ.get("INFERENCE_HOST", "0.0.0.0")
    port = int(os.environ.get("INFERENCE_PORT", "8900"))
    print(f"pharos-inference  http://{host}:{port}  (模型后台预热中,/readyz 热完转 ready)", flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
