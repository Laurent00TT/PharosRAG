"""Pharos HTTP 守护进程(FastAPI):系统里**唯一**碰嵌入式 Qdrant 与 GPU 模型的进程。

为什么是守护进程(而不是每客户端各自打开索引)—— 两个硬约束(见 docs/DESIGN.md §2):
  1. 嵌入式 Qdrant 单客户端独占锁:第二个进程打开同一路径直接报错;
  2. dense 模型(Qwen3-VL 8B)加载 1-2 分钟:stdio MCP 每会话一进程,每开一个会话都重付这笔钱。
守护进程独占索引 + 常驻模型,HTTP 出口给所有消费方共享:curl/脚本(闭管道 /v1/ask)、
MCP 薄适配器(agentic,每会话秒连)。

安全模型:与引擎 stdio server 一致 —— **ACL 身份启动时绑定**(PHAROS_TENANT/PHAROS_PRINCIPALS),
HTTP 客户端不能经参数改身份;tenant 未设则一切 fail-closed 返回空。可选 PHAROS_API_KEY 做接入门槛
(设了则 /v1/* 需 X-API-Key;/healthz 豁免)。**部署本服务 = 把该身份可见的内容授权给能连上端口的人**,
默认只绑 127.0.0.1。

工具语义(校验/结构化结果/去重/预算/错误映射)全部来自引擎 toolcore —— 本文件只做 HTTP 绑定:
路由 + 身份注入 + per-session 去重(X-Pharos-Session,见 sessions.py)+ 闭管道 /v1/ask。
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__, config, engine, smart
from .sessions import SessionRegistry

log = logging.getLogger("pharos")


# ---------- 请求模型(mode/strategy 等枚举校验留给 toolcore,返回结构化 bad_arg 而非 422)----------
class RetrieveReq(BaseModel):
    query: str = ""
    top_k: int | None = None
    rerank: bool = False
    doc_ids: list[str] | None = None
    doc_type: str | None = None
    kind: str | None = None
    mode: str = "full"
    strategy: str = "hybrid"
    rerank_top_n: int | None = None


class AskReq(BaseModel):
    query: str = ""
    top_k: int | None = None
    rerank: bool = False
    include_contexts: bool = False   # true 时 citations 带被引段原文(大;默认只回溯源元数据)
    # N3:检索过滤/选路(与 /v1/retrieve 同语义)。实证场景:"数字埋在表里"的题,
    # 通用问法下表格块被散文挤出 top-k,kind='table' 一击命中。
    doc_ids: list[str] | None = None
    doc_type: str | None = None
    kind: str | None = None
    strategy: str | None = None      # hybrid|dense|sparse;None=引擎默认(hybrid)


class ExpandReq(BaseModel):
    chunk_id: str = ""
    target_tokens: int = 1500


class GroupedReq(BaseModel):
    query: str = ""
    doc_ids: list[str] = []
    top_k: int = 3
    rerank: bool = False


def create_app(cfg: config.PharosConfig | None = None, retriever=None, user=None,
               generator_factory=None) -> FastAPI:
    """app 工厂。生产:全默认(从 env 建配置,启动时打开真索引)。测试:注入 fake retriever/user/generator。"""
    cfg = cfg or config.from_env()
    # toolcore 的交付预算走 RAG_MAX_CONTEXT_TOKENS(引擎契约);Pharos 配置在此兑现
    os.environ["RAG_MAX_CONTEXT_TOKENS"] = str(cfg.max_context_tokens)
    tc = engine.load_toolcore(cfg.engine)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        if state.user is None:
            state.user = engine.build_user(cfg)
        if state.retriever is None:
            log.info("打开索引 %s (collection=%s)…", cfg.qdrant_path, cfg.collection)
            state.retriever = engine.build_retriever(cfg)   # 此刻取得嵌入式 Qdrant 独占锁
        if not cfg.tenant:
            log.warning("PHAROS_TENANT 未设 —— 一切检索将 fail-closed 返回空(no_identity)。")
        yield

    app = FastAPI(title="Pharos", version=__version__, lifespan=_lifespan)
    state = app.state
    state.cfg, state.tc = cfg, tc
    state.retriever = retriever            # None -> lifespan 时真建(独占 Qdrant 锁)
    state.user = user
    state.gen_local = threading.local()    # 评审修:Generator/LLM per-thread —— 共享单例的
    state.generator_factory = generator_factory or engine.build_generator   # last_finish_reason 并发下会跨请求串味
    state.sessions = SessionRegistry()

    # 评审修(C2):toolcore 的 no_identity hint 指示设 RAG_TENANT,但 Pharos 只读 PHAROS_TENANT
    # —— 照原 hint 操作后依然 fail-closed,形成死循环误导。绑定层负责把契约文本翻译成本产品的配置名。
    no_id_hint = tc._NO_IDENTITY_HINT.replace("RAG_TENANT", "PHAROS_TENANT").replace(
        "RAG_PRINCIPALS", "PHAROS_PRINCIPALS")

    def _adapt(d: dict) -> dict:
        if isinstance(d, dict) and d.get("status") == "no_identity":
            d["hint"] = no_id_hint
        return d

    # ---------- 可选 API key(/healthz 豁免)----------
    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if cfg.api_key and request.url.path != "/healthz":
            if request.headers.get("x-api-key") != cfg.api_key:
                return JSONResponse({"status": "unauthorized",
                                     "hint": "缺少或错误的 X-API-Key(服务端设置了 PHAROS_API_KEY)。"},
                                    status_code=401)
        return await call_next(request)

    def _session_keys(request: Request):
        """去重 opt-in:带 X-Pharos-Session 头才启用跨调用去重(MCP 适配器每进程一个 uuid)。"""
        sid = request.headers.get("x-pharos-session")
        return state.sessions.get(sid) if sid else None

    # ---------- 健康 ----------
    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "service": "pharos", "version": __version__,
                "collection": cfg.collection, "tenant_bound": bool(cfg.tenant),
                "llm_model": cfg.llm_model, "auth": bool(cfg.api_key)}

    @app.get("/v1/instructions")
    def instructions():
        """agent 使用契约(与引擎 stdio server 的 FastMCP instructions 同文)。"""
        return {"status": "ok", "instructions": tc._INSTRUCTIONS}

    # ---------- 检索工具面(六个,与 MCP 工具一一对应,语义同 toolcore)----------
    @app.post("/v1/retrieve")
    def retrieve(q: RetrieveReq, request: Request):
        return _adapt(tc._retrieve_impl(state.retriever, state.user, q.query, q.top_k, q.rerank,
                                        q.doc_ids, q.doc_type, q.kind, q.mode, q.strategy,
                                        q.rerank_top_n, returned_keys=_session_keys(request)))

    @app.get("/v1/documents")
    def list_documents():
        return _adapt(tc._list_impl(state.retriever, state.user))

    @app.get("/v1/documents/{doc_id}")
    def get_document(doc_id: str, max_tokens: int = 6000):
        return _adapt(tc._get_document_impl(state.retriever, state.user, doc_id, max_tokens))

    @app.get("/v1/documents/{doc_id}/outline")
    def get_outline(doc_id: str):
        return _adapt(tc._outline_impl(state.retriever, state.user, doc_id))

    @app.post("/v1/expand")
    def expand(q: ExpandReq):
        return _adapt(tc._expand_impl(state.retriever, state.user, q.chunk_id, q.target_tokens))

    @app.post("/v1/retrieve_grouped")
    def retrieve_grouped(q: GroupedReq):
        return _adapt(tc._grouped_impl(state.retriever, state.user, q.query, q.doc_ids, q.top_k, q.rerank))

    # ---------- 闭管道问答(generator:检索 + grounding prompt + DeepSeek + 引用解析)----------
    def _get_generator():
        """per-thread 惰性构建(线程池有界,实例数有界):同线程内 answer→读 finish_reason 无并发窗口。"""
        gen = getattr(state.gen_local, "gen", None)
        if gen is None:
            gen = state.generator_factory(state.retriever, cfg)
            state.gen_local.gen = gen
        return gen

    @app.post("/v1/ask")
    def ask(q: AskReq):
        if not state.user or not state.user.tenant:
            return {"status": "no_identity", "retriable": False, "hint": no_id_hint}
        if not (q.query or "").strip():
            return {"status": "empty_query", "retriable": False, "hint": "query 为空,请提供具体问题。"}
        if q.strategy is not None and q.strategy not in ("hybrid", "dense", "sparse"):
            return {"status": "bad_arg", "retriable": False,
                    "hint": f"strategy 必须 hybrid|dense|sparse(收到 {q.strategy})。"}
        try:
            gen = _get_generator()
        except ValueError:
            return {"status": "llm_unconfigured", "retriable": False,
                    "hint": f"缺 LLM API key(环境变量 {cfg.llm_api_key_env},放 .env)。"}
        except Exception:                  # 评审修:openai 包缺失/引擎路径断等不再裸抛 500
            log.exception("Generator 构建失败")
            return {"status": "ask_failed", "retriable": False,
                    "hint": "生成器初始化失败(依赖或引擎配置问题),详见服务端日志。"}
        # smart-ask 第 2 层(D9):**失败驱动**表格补检——第一轮纯净;数值题拒答/部分拒答时
        # 带 kind=table 腿重问一轮(硬上限 1 次重试;auto 留痕;用户显式给 kind 则尊重用户)。
        # ⚠ 前置腿方案已被 88 题实测否决(误伤 5 道散文题),留档 TESTING §3——别改回去。
        auto: list[str] = []
        numeric = False
        if cfg.smart_ask:
            from generator import looks_numeric
            numeric = looks_numeric(q.query)
        try:
            # 检索在 LockedRetriever 锁内、LLM 网络调用在锁外(不阻塞其他检索请求)
            ans = gen.answer(q.query, state.user, top_k=q.top_k, rerank=q.rerank,
                             doc_ids=q.doc_ids, doc_type=q.doc_type, kind=q.kind, strategy=q.strategy)
            if cfg.smart_ask and numeric and q.kind is None and smart.is_refusal(ans.text):
                from generator import DEFAULT_TABLE_LEG
                ans2 = gen.answer(q.query, state.user, top_k=q.top_k, rerank=q.rerank,
                                  doc_ids=q.doc_ids, doc_type=q.doc_type, strategy=q.strategy,
                                  extra_legs=[dict(DEFAULT_TABLE_LEG)])
                # 只采用**完整答出**的重试(不再含拒答/缺失声明)。88 题实测:部分回答会夹带
                # "未提供 X"的错误缺失声明(X 其实在 context 里),忠实度 1.0->0.93——宁可保留
                # 第一轮的诚实拒答+hints,不说错话。忠实度是本系统头牌,排序在"多答一点"之前。
                if not smart.is_refusal(ans2.text):
                    ans = ans2
                    auto.append("table_leg_retry")
                else:
                    auto.append("table_leg_retry_discarded")   # 留痕:重试过但按守则弃用
        except Exception:
            log.exception("ask 失败")   # 细节只进服务端日志,不外泄给客户端
            return {"status": "ask_failed", "retriable": True,
                    "hint": "生成失败(检索后端或 LLM 上游异常),请稍后重试。"}
        citations = []
        for c in ans.citations:
            d = {"marker": c.marker, "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                 "title": c.title, "section": c.section, "page": c.page}
            if q.include_contexts:
                d["text"] = c.text
            citations.append(d)
        # smart-ask 第 1 层:拒答/部分拒答时给可操作 hints(正常答案不打扰)
        hints = (smart.build_hints(q.query, auto=auto, req_kind=q.kind, req_rerank=q.rerank,
                                   numeric=numeric)
                 if cfg.smart_ask and smart.is_refusal(ans.text) else [])
        return {"status": "ok", "answer": ans.text, "citations": citations,
                "n_contexts": ans.n_contexts, "model": cfg.llm_model,
                "finish_reason": getattr(gen.llm, "last_finish_reason", None),
                "auto": auto, "hints": hints}

    return app
