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
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__, config, engine, identity as identity_mod, smart, toolcore
from .obs import RequestLog, Stats
from .sessions import SessionRegistry
from embedder import User
from generator import DEFAULT_TABLE_LEG, looks_numeric

log = logging.getLogger("pharos")

# 探针专用线程池(阶段F 审查):/readyz 的 qdrant 阻塞调用走这里,与 FastAPI 默认 40 线程业务池隔离
# —— 满负载下业务打满默认池也不会让健康探针排队饥饿(否则 nginx 会摘掉正常干活的健康副本 = 全局 outage)。
_PROBE_LIMITER = anyio.CapacityLimiter(8)


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
               generator_factory=None, keys=None) -> FastAPI:
    """app 工厂。生产:全默认(从 env 建配置,启动时打开真索引)。测试:注入 fake retriever/user/generator/keys。"""
    cfg = cfg or config.from_env()
    # toolcore 的交付预算读环境变量;此处把 Pharos 配置兑现到产品命名空间(cfg 为单一真源)
    os.environ["PHAROS_MAX_CONTEXT_TOKENS"] = str(cfg.max_context_tokens)
    tc = toolcore

    # ---------- 身份模式(D10):keys(团队,默认)/ legacy(单密钥)/ open(仅回环)----------
    if keys is None and cfg.keys_file:
        keys = identity_mod.load_keys(cfg.keys_file)     # 格式错 → SystemExit,拒绝启动
    mode = "keys" if keys else ("legacy" if cfg.api_key else "open")
    if not identity_mod.is_loopback(cfg.host) and mode != "keys":
        # fail-closed 启动守卫:部署即授权 —— 绑非回环地址必须多身份鉴权,不允许把整库裸奔到局域网
        raise SystemExit(f"PHAROS_HOST={cfg.host} 非回环地址,必须配置 PHAROS_KEYS_FILE(keys 模式)才能启动。")
    if mode == "keys" and cfg.api_key:
        log.warning("keys 模式下 PHAROS_API_KEY 被忽略(身份以 keys 文件为准)。")

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
        # 优雅停机(阶段F):drain 期把后台写线程队列里的请求日志落盘,别丢最后一批(观测完整性)。
        # 带超时:磁盘挂起(bind-mount IO 卡死)时放弃而非冻死事件循环 —— 无限 join 会吃满
        # stop_grace_period 30s 被 SIGKILL(uvicorn 25s drain + 5s flush ≤ 30s,预算自洽)。
        try:
            state.reqlog.flush(timeout=5.0)
        except Exception:
            log.warning("shutdown: reqlog flush 失败", exc_info=True)

    app = FastAPI(title="Pharos", version=__version__, lifespan=_lifespan)
    state = app.state
    state.cfg, state.tc = cfg, tc
    state.retriever = retriever            # None -> lifespan 时真建(独占 Qdrant 锁)
    state.user = user
    state.gen_local = threading.local()    # 评审修:Generator/LLM per-thread —— 共享单例的
    state.generator_factory = generator_factory or engine.build_generator   # last_finish_reason 并发下会跨请求串味
    state.sessions = SessionRegistry()
    state.stats = Stats()                  # D11:进程内指标 + JSONL 请求日志
    state.reqlog = RequestLog(cfg.log_dir, log_queries=cfg.log_queries)

    def _current_user(request: Request):
        """本次请求的引擎 User。keys 模式按解析出的身份现建(多身份核心);legacy/open 用启动绑定的。"""
        iden = getattr(request.state, "identity", None)
        if mode == "keys" and iden is not None:
            return User(tenant=iden.tenant, principals=list(iden.principals))
        return state.user

    def _iden_name(request: Request) -> str:
        iden = getattr(request.state, "identity", None)
        return iden.name if iden is not None else ("local" if mode == "open" else "default")

    # toolcore 的 no_identity hint 已在源头用 PHAROS_* 命名(W3 命名空间统一),绑定层无需再翻译。
    no_id_hint = tc._NO_IDENTITY_HINT

    def _adapt(d: dict) -> dict:
        if isinstance(d, dict) and d.get("status") == "no_identity":
            d["hint"] = no_id_hint
        return d

    # ---------- 鉴权 + 身份解析(D10;/healthz 豁免)----------
    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if request.url.path not in ("/healthz", "/readyz"):   # readyz 同 healthz 豁免:编排/nginx 探针无 API key
            k = request.headers.get("x-api-key", "")
            if mode == "keys":
                iden = keys.get(k)
                if iden is None:                        # 未知/缺失一律 401,不泄"key 是否存在过"
                    return JSONResponse({"status": "unauthorized",
                                         "hint": "缺少或无效的 X-API-Key(本服务为多身份 keys 模式)。"},
                                        status_code=401)
                request.state.identity = iden
            elif mode == "legacy":
                if k != cfg.api_key:
                    return JSONResponse({"status": "unauthorized",
                                         "hint": "缺少或错误的 X-API-Key(服务端设置了 PHAROS_API_KEY)。"},
                                        status_code=401)
        return await call_next(request)

    # ---------- 观测(D11):计时 + 请求日志(不落 key 本体;截断在 obs 层)----------
    @app.middleware("http")
    async def _observe(request: Request, call_next):
        t0 = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            # try/finally:处理器崩溃(未捕获异常)时仍记录,否则严重错误在观测里隐形(评审)
            ms = (time.time() - t0) * 1000
            ep = request.url.path
            if ep.startswith("/v1/") or ep == "/healthz":
                code = response.status_code if response is not None else 500
                extra = dict(getattr(request.state, "log_extra", None) or {})
                biz = extra.get("status")
                # errors 计入 http 4xx/5xx **或**结构化业务失败(no_access/bad_arg/ask_failed…);
                # ok/empty 算成功。此前只看 http>=400,漏计一切返回 200 的结构化失败(评审)
                err = code >= 400 or (biz is not None and biz not in ("ok", "empty"))
                # stats 键用**路由模板**(/v1/documents/{doc_id})而非原始路径:原始路径下枚举 doc_id 会让
                # 键基数无界(Stats 的 defaultdict 每新键建一个 deque,长期守护进程=慢性泄漏/低速 DoS)。
                # 无 route(404 未匹配路由、被 _auth 401 短路 —— _observe 在最外层,二者都会到这)归并固定桶,
                # 键集合 = 已注册路由数 + 1,天然有界。JSONL 的 rec["ep"] 保留原始路径(调试价值,在磁盘不在内存)。
                route = request.scope.get("route")
                stat_ep = getattr(route, "path", None) or ("/v1/_unmatched" if ep.startswith("/v1/") else ep)
                state.stats.record(stat_ep, ms, err)
                rec = {"ts": round(time.time(), 3), "ep": ep, "user": _iden_name(request),
                       "http": code, "ms": round(ms, 1)}
                if response is None:
                    rec["crashed"] = True
                rec.update(extra)
                state.reqlog.write(rec, state.stats)

    def _log(request: Request, out: dict, **extra):
        """统一登记 log_extra(带业务 status,供 _observe 计 errors + 落盘)。返回 out 供直接 return。"""
        request.state.log_extra = {"status": out.get("status") if isinstance(out, dict) else None, **extra}
        return out

    def _session_keys(request: Request):
        """去重 opt-in:带 X-Pharos-Session 头才启用。登记键带身份名前缀 —— 多用户下即便
        伪造相同会话 id 也互不可见(单身份下"会话碰撞无害"的前提在多用户下不再成立)。"""
        sid = request.headers.get("x-pharos-session")
        return state.sessions.get(f"{_iden_name(request)}|{sid}") if sid else None

    # ---------- 健康 / 指标 ----------
    @app.get("/healthz")
    async def healthz():
        """liveness:进程活着。**async**(纯内存读)—— 阶段F 审查:探针若与 /v1/ask(LLM 数十秒)、
        /v1/retrieve(inference hang 时单请求最长 ~361s 重试)共享 anyio 默认 40 线程池,高负载/下游故障下
        探针在池里排队饥饿 → docker healthcheck / K8s liveness 假 unhealthy → crashloop。async 端点在事件
        循环里直接跑、不进线程池,彻底隔离。
        信息边界与 /readyz 对齐(sec):本端点未鉴权,不回 collection/llm_model/identity_mode 这类
        侦察信息(readyz 为同一评审结论刻意不回集合名,healthz 回了等于架空它)—— 详细字段挪到
        admin-gated 的 /v1/stats。"""
        return {"status": "ok", "service": "pharos", "version": __version__,
                "tenant_bound": bool(cfg.tenant) or mode == "keys",
                "uptime_s": round(time.time() - state.stats.started, 1)}

    @app.get("/readyz")
    async def readyz():
        """readiness:探下游可达【且集合已就绪】。F 的 nginx 据此导流量;与 /healthz(liveness)分离——
        下游抖动只让本副本暂时摘流量,不触发 crashloop(healthz 仍 200)。豁免鉴权(同 healthz),供编排探针无 key 调用。
        **async + 专用 limiter**(阶段F 审查):端点本身 async(不占业务 40 池);qdrant 阻塞调用 offload 到
        _PROBE_LIMITER(8 线程,与业务隔离)、inference 探活用 AsyncClient —— 二者都不与 /v1/* 争默认池,
        故满负载下探针仍准时。否则探针饥饿会让 nginx 把正在正常干活的健康副本全摘掉(全局 outage)。
        安全(评审 sec-2):异常只记服务端日志,响应体**不回 str(e)**——否则未鉴权探针能读到内网 qdrant/inference host:port
        (同 /v1/ask 的"细节只进服务端日志"纪律)。不进 _observe 计量白名单(service.py)——高频探针不污染业务指标。"""
        if state.retriever is None:                          # lifespan 未建完(极早期):未就绪
            return JSONResponse({"status": "starting"}, status_code=503)
        try:
            exists = await anyio.to_thread.run_sync(         # qdrant 阻塞 HTTP -> 专用 limiter 线程,不占业务池
                lambda: state.retriever.store.client.collection_exists(cfg.collection),
                limiter=_PROBE_LIMITER)
        except Exception:
            log.warning("readyz: qdrant 探活失败", exc_info=True)   # 细节只进服务端日志,不外泄内网拓扑
            return JSONResponse({"status": "qdrant_unavailable"}, status_code=503)
        if not exists:                                       # 评审 compose-B:collection 不存在(新起 server 未迁移)= 未就绪,别绿着查空库
            return JSONResponse({"status": "collection_missing"}, status_code=503)  # sec:不回集合名(未鉴权探针)
        if cfg.inference_url:                                 # remote 模式才探 inference /readyz
            try:
                import httpx
                # 显式短超时(评审 sec-3):httpx.Timeout(3) 每阶段各 3s,最坏 ~9s > healthcheck 5s → 会把本副本误判 unhealthy 重启
                async with httpx.AsyncClient() as client:
                    r = await client.get(cfg.inference_url.rstrip("/") + "/readyz",
                                          timeout=httpx.Timeout(1.5, connect=1.0))
                if r.status_code != 200:
                    return JSONResponse({"status": "inference_not_ready"}, status_code=503)
            except Exception:
                log.warning("readyz: inference 探活失败", exc_info=True)
                return JSONResponse({"status": "inference_unavailable"}, status_code=503)
        return {"status": "ready"}

    @app.get("/v1/stats")
    def stats(request: Request):
        """进程内指标快照。keys 模式下 admin key 才可读(聚合查询模式也是信息)。"""
        iden = getattr(request.state, "identity", None)
        if mode == "keys" and not (iden is not None and iden.admin):
            return JSONResponse({"status": "forbidden", "hint": "stats 需要 admin key。"}, status_code=403)
        snap = state.stats.snapshot()
        snap.update({"status": "ok", "identity_mode": mode, "sessions": len(state.sessions),
                     "collection": cfg.collection, "llm_model": cfg.llm_model,   # 从 /healthz 挪入(未鉴权探针不该见)
                     "log_path": state.reqlog.path if state.reqlog.enabled else ""})
        return snap

    @app.get("/v1/instructions")
    def instructions():
        """agent 使用契约(与引擎 stdio server 的 FastMCP instructions 同文)。"""
        return {"status": "ok", "instructions": tc._INSTRUCTIONS}

    # ---------- 检索工具面(六个,与 MCP 工具一一对应,语义同 toolcore)----------
    @app.post("/v1/retrieve")
    def retrieve(q: RetrieveReq, request: Request):
        out = _adapt(tc._retrieve_impl(state.retriever, _current_user(request), q.query, q.top_k, q.rerank,
                                       q.doc_ids, q.doc_type, q.kind, q.mode, q.strategy,
                                       q.rerank_top_n, returned_keys=_session_keys(request)))
        return _log(request, out, query=q.query, n=out.get("meta", {}).get("returned_n"))

    @app.get("/v1/documents")
    def list_documents(request: Request):
        return _log(request, _adapt(tc._list_impl(state.retriever, _current_user(request))))

    @app.get("/v1/documents/{doc_id}")
    def get_document(doc_id: str, request: Request, max_tokens: int = 6000):
        return _log(request, _adapt(tc._get_document_impl(state.retriever, _current_user(request), doc_id, max_tokens)))

    @app.get("/v1/documents/{doc_id}/outline")
    def get_outline(doc_id: str, request: Request):
        return _log(request, _adapt(tc._outline_impl(state.retriever, _current_user(request), doc_id)))

    @app.post("/v1/expand")
    def expand(q: ExpandReq, request: Request):
        return _log(request, _adapt(tc._expand_impl(state.retriever, _current_user(request), q.chunk_id, q.target_tokens)))

    @app.post("/v1/retrieve_grouped")
    def retrieve_grouped(q: GroupedReq, request: Request):
        return _log(request, _adapt(tc._grouped_impl(state.retriever, _current_user(request), q.query, q.doc_ids, q.top_k, q.rerank)))

    # ---------- 闭管道问答(generator:检索 + grounding prompt + DeepSeek + 引用解析)----------
    def _get_generator():
        """per-thread 惰性构建(线程池有界,实例数有界):同线程内 answer→读 finish_reason 无并发窗口。"""
        gen = getattr(state.gen_local, "gen", None)
        if gen is None:
            gen = state.generator_factory(state.retriever, cfg)
            state.gen_local.gen = gen
        return gen

    @app.post("/v1/ask")
    def ask(q: AskReq, request: Request):
        req_user = _current_user(request)
        if not req_user or not req_user.tenant:
            return _log(request, {"status": "no_identity", "retriable": False, "hint": no_id_hint})
        if not (q.query or "").strip():
            return _log(request, {"status": "empty_query", "retriable": False, "hint": "query 为空,请提供具体问题。"})
        if q.strategy is not None and q.strategy not in ("hybrid", "dense", "sparse"):
            return _log(request, {"status": "bad_arg", "retriable": False,
                                  "hint": f"strategy 必须 hybrid|dense|sparse(收到 {q.strategy})。"})
        try:
            gen = _get_generator()
        except ValueError:
            return _log(request, {"status": "llm_unconfigured", "retriable": False,
                                  "hint": f"缺 LLM API key(环境变量 {cfg.llm_api_key_env},放 .env)。"})
        except Exception:                  # 评审修:openai 包缺失/引擎路径断等不再裸抛 500
            log.exception("Generator 构建失败")
            return _log(request, {"status": "ask_failed", "retriable": False,
                                  "hint": "生成器初始化失败(依赖或引擎配置问题),详见服务端日志。"})
        # smart-ask 第 2 层(D9):**失败驱动**表格补检——第一轮纯净;数值题拒答/部分拒答时
        # 带 kind=table 腿重问一轮(硬上限 1 次重试;auto 留痕;用户显式给 kind 则尊重用户)。
        # ⚠ 前置腿方案已被 88 题实测否决(误伤 5 道散文题),留档 TESTING §3——别改回去。
        auto: list[str] = []
        numeric = False
        if cfg.smart_ask:
            numeric = looks_numeric(q.query)
        try:
            # 检索的 Qdrant/GPU 段在资源类锁内(Store._lock / Dense._fwd_lock,M1 锁下沉)、LLM 网络调用在锁外(不阻塞其他检索请求)
            ans = gen.answer(q.query, req_user, top_k=q.top_k, rerank=q.rerank,
                             doc_ids=q.doc_ids, doc_type=q.doc_type, kind=q.kind, strategy=q.strategy)
            if cfg.smart_ask and numeric and q.kind is None and smart.is_refusal(ans.text):
                ans2 = gen.answer(q.query, req_user, top_k=q.top_k, rerank=q.rerank,
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
            return _log(request, {"status": "ask_failed", "retriable": True,
                                  "hint": "生成失败(检索后端或 LLM 上游异常),请稍后重试。"}, query=q.query)
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
        # finish_reason 读 Answer 快照而非 gen.llm 实例属性:重试被弃用时实例上残留第二轮的值(与返回的
        # 第一轮答案错位),零召回时残留同线程上一请求的值 —— 快照随答案走,天然对齐。
        return _log(request, {"status": "ok", "answer": ans.text, "citations": citations,
                              "n_contexts": ans.n_contexts, "model": cfg.llm_model,
                              "finish_reason": ans.finish_reason,
                              "auto": auto, "hints": hints},
                    query=q.query, auto=auto or None, n_citations=len(citations), refusal=bool(hints))

    return app
