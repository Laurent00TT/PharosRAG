"""vLLM 推理后端适配器(Plan B/A,docs/VLLM_PLAN.md Phase 1)。

**定位**:呈现与 `inference_server.py` **完全相同**的契约(`/embed`/`/embed_image`/`/rerank`/`/readyz`/`/healthz`),
背后由 `vllm serve --runner pooling`(continuous batching)出 embedding。应用层 `remote.py` 一字不改,
`PHAROS_INFERENCE_URL` 指向本适配器即从 FastAPI 后端切到 vLLM 后端。**吞吐 ~12×(G4 实测 297 vs 24 QPS)**。

**为什么要适配器而不是让 pharos 直连 vLLM /v1/embeddings**:
  1. 契约边界:pharos 契约是 `/embed {texts,instruction}→{vectors}`(全维归一,客户端截维);vLLM 是 OpenAI
     `/v1/embeddings {input}→{data:[{embedding}]}`。适配器翻译,应用层不碰。
  2. **chat 模板是等价性命门**:官方 Qwen3VLEmbedder 对 query 用 `[{system:instruction},{user:text}]` +
     `apply_chat_template(add_generation_prompt=True)`(实读官方 wrapper)。vLLM `/v1/embeddings` 收裸字符串**不套模板**
     → 必须适配器**照抄这个 recipe** 预套模板再发 vLLM,否则 last-token pooling 错位、向量漂。tokenizer 在适配器(CPU)。

**Phase 1 边界(诚实)**:
  - `/embed`:走 vLLM(查询路径的 QPS 命门,先迁它拿大头)。
  - `/embed_image`:501(建库专用、走 local torch,不经 vLLM;查询路径不调)。
  - `/rerank`:配了 `RERANK_PROXY_URL` 则代理到 torch reranker 容器;否则 503 —— pharos rerank 降级安全
    (retrieve.py rerank except → hybrid),Phase 1 无 rerank 不崩。reranker 迁 vLLM 是 Phase 3(G3 未验)。

环境变量:
  INFERENCE_VLLM_URL   vLLM serve 地址(默认 http://localhost:8000)
  INFERENCE_MODEL_PATH  模型路径(取 tokenizer 套 chat 模板;默认 ~/models/Qwen3-VL-Embedding-8B)
  RERANK_PROXY_URL     可选:torch reranker 的 /rerank 地址(不配则 /rerank 返 503 降级)
  ADAPTER_HOST/ADAPTER_PORT  绑定(默认 0.0.0.0:8900,与 inference_server 同端口 -> 平替)

**为什么放 src/ 根而非 embedder/ 包内**:适配器要在 **vllm 环境**跑(那里没装 pharos/qdrant_client),故必须
不触发 `embedder/__init__`(它 import qdrant_client);且放包内直接 `python 文件` 跑会让 `src/embedder/` 进
sys.path[0]、stdlib `import types` 被 `embedder/types.py` 遮蔽(两坑均实测)。本适配器零 embedder 依赖,独立成
top-level 模块最干净(容器 CMD 同理 `python src/inference_vllm_adapter.py`)。

跑(bare-metal 验证,同阶段E pivot):
  conda activate vllm && vllm serve ~/models/Qwen3-VL-Embedding-8B --runner pooling --max-model-len 8192 --port 8000 &
  python src/inference_vllm_adapter.py    # :8900,pharos 指 PHAROS_INFERENCE_URL=http://localhost:8900
"""
from __future__ import annotations

import os
import unicodedata

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class EmbedReq(BaseModel):
    texts: list[str] = []
    instruction: str | None = None


class EmbedImageReq(BaseModel):
    image_paths: list[str] = []
    instruction: str | None = None


class RerankReq(BaseModel):
    query: str = ""
    documents: list[str] = []
    instruction: str | None = None


def _sys_instruction(instr: str | None, default: str) -> str:
    """复刻官方 Qwen3VLEmbedder.format_model_input:instruction 末尾非标点(Unicode P*)则补 '.'。"""
    instr = (instr or default).strip()
    if instr and not unicodedata.category(instr[-1]).startswith("P"):
        instr = instr + "."
    return instr


def create_app(cfg=None) -> FastAPI:
    vllm_url = os.environ.get("INFERENCE_VLLM_URL", "http://localhost:8000").rstrip("/")
    model_path = os.environ.get("INFERENCE_MODEL_PATH", os.path.expanduser("~/models/Qwen3-VL-Embedding-8B"))
    rerank_proxy = os.environ.get("RERANK_PROXY_URL", "").rstrip("/")
    # 默认 instruction 对齐官方 wrapper 的 default_instruction(query 路径 pharos 会显式传 query_instruction)
    default_instr = os.environ.get("INFERENCE_DEFAULT_INSTRUCTION", "Represent the user's input.")

    app = FastAPI(title="pharos-inference-vllm-adapter")
    state = app.state
    state.tok = None           # 延迟加载 tokenizer(启动快;首个 /embed 才需要)
    state.err = None

    def _tokenizer():
        if state.tok is None:
            from transformers import AutoProcessor
            state.tok = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return state.tok

    # vLLM serve 打 OpenAI /v1/embeddings;continuous batching 在 vLLM 侧,适配器纯 async 转发(无 gpu_lock)
    state.client = httpx.Client(base_url=vllm_url, timeout=httpx.Timeout(120.0, connect=3.0))

    def _templated(texts: list[str], instruction: str | None) -> list[str]:
        """把 {texts,instruction} 按官方 recipe 套成 chat 模板字符串(等价性命门)。"""
        tok = _tokenizer()
        sys_txt = _sys_instruction(instruction, default_instr)
        out = []
        for t in texts:
            conv = [{"role": "system", "content": [{"type": "text", "text": sys_txt}]},
                    {"role": "user", "content": [{"type": "text", "text": t}]}]
            out.append(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
        return out

    @app.get("/healthz")
    async def healthz():                       # liveness:适配器进程活着(async,纯内存)
        return {"status": "ok", "service": "inference-vllm-adapter", "vllm_url": vllm_url,
                "model_dense": os.path.basename(model_path), "error": state.err}

    @app.get("/readyz")
    async def readyz():                        # readiness:探下游 vLLM /health(async,不阻塞事件循环)
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(vllm_url + "/health", timeout=httpx.Timeout(1.5, connect=1.0))
            if r.status_code == 200:
                return {"status": "ready"}
            return JSONResponse({"status": "vllm_not_ready"}, status_code=503)
        except Exception:
            return JSONResponse({"status": "vllm_unavailable"}, status_code=503)   # 不回内网 host:port(sec 同纪律)

    @app.post("/embed")
    def embed(q: EmbedReq):
        prompts = _templated(q.texts, q.instruction)
        r = state.client.post("/v1/embeddings",
                              json={"input": prompts, "model": model_path, "encoding_format": "float"})
        r.raise_for_status()
        data = r.json()["data"]
        vecs = [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]   # 按 index 复原顺序
        return {"vectors": vecs}               # vLLM 出全维归一;客户端 remote._mrl_np 截维(契约同 inference_server)

    @app.post("/embed_image")
    def embed_image(q: EmbedImageReq):         # Phase 1:image 编码建库专用、走 local torch,不经 vLLM
        return JSONResponse({"status": "not_implemented",
                             "detail": "image 编码建库专用(local torch);查询路径不调 /embed_image。"}, status_code=501)

    @app.post("/rerank")
    def rerank(q: RerankReq):
        if not rerank_proxy:                   # Phase 1 无 reranker:503 -> pharos 降级 hybrid(rerank 降级安全)
            return JSONResponse({"status": "rerank_unavailable",
                                 "detail": "Phase 1 未接 reranker(降级安全,pharos 退回 hybrid)。"}, status_code=503)
        r = state.client.post(rerank_proxy + "/rerank", json=q.model_dump())   # 代理到 torch reranker 容器
        return JSONResponse(r.json(), status_code=r.status_code)

    return app


def main() -> None:
    import uvicorn
    host = os.environ.get("ADAPTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ADAPTER_PORT", "8900"))
    print(f"pharos-inference-vllm-adapter  http://{host}:{port}  -> vLLM "
          f"{os.environ.get('INFERENCE_VLLM_URL', 'http://localhost:8000')}", flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
