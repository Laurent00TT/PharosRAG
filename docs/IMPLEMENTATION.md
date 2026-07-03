# Pharos 实施文档

> 对应版本 v0.1。设计动机见 [DESIGN.md](DESIGN.md);本文讲"代码怎么长的"。

## 1. 模块地图

```
pharos/
  config.py       env → 不可变 PharosConfig;.env 双源加载(本仓 → 引擎仓兜底,setdefault 语义)
  engine.py       引擎装配:bootstrap(sys.path 插三个组件 src)/ load_toolcore(按文件路径 importlib)
                  / LockedRetriever(线程锁代理)/ build_retriever / build_user / build_generator
  sessions.py     SessionRegistry:per-session returned_keys,有界 LRU(64 会话)
  service.py      FastAPI app 工厂 create_app(cfg, retriever, user, generator_factory 全可注入)
  smart.py        smart-ask 产品层(D9):拒答检测 + hints;数值判定/补检腿参数来自引擎 signals
  mcp_adapter.py  MCP 薄适配器:六工具 → httpx 转发;结构化降级;进程级 uuid 会话头
  indexer.py      pharos index:MinerU 解析目录 → chunker → embedder(index_real.py 的参数化版)
  cli.py          serve / mcp / index / ask / health
tests/            25 项 CPU 单测(_fakes.py 提供 FakeRetriever/make_app)
```

## 2. 请求路径

**/v1/retrieve(及其余 5 个检索端点)**:

```
HTTP 请求 ─► API key 中间件 ─► pydantic 请求模型(只定形,不做枚举校验)
  ─► X-Pharos-Session? → SessionRegistry.get(sid) / None
  ─► toolcore._retrieve_impl(LockedRetriever, bound_user, …, returned_keys)
        · 校验(no_identity/empty_query/bad_arg)在锁外、检索在锁内
        · already_returned / omitted_budget / 预算含资产 —— 全部 toolcore 原语义
  ─► _adapt():no_identity 的 hint 翻译成 PHAROS_TENANT 措辞(评审 C2:引擎 hint 指 RAG_TENANT 会死循环误导)
  ─► 结构化 dict 原样出(HTTP 200 + status 字段)
```

**/v1/ask(闭管道)**:

```
校验(no_identity/empty_query)
  ─► _get_generator():**per-thread** 惰性建 Generator(LockedRetriever, OpenAICompatibleLLM, acl_check=acl_admits)
        · threading.local:共享单例的 llm.last_finish_reason 在并发 ask 下会跨请求串味(评审修);
          线程池有界 → 实例数有界,同线程内 answer→读 finish_reason 无并发窗口
        · ValueError(缺 key)→ status=llm_unconfigured;其余构建异常 → ask_failed(不裸抛 500)
  ─► 第一轮 gen.answer(query, user, top_k, rerank, filters…)——**纯净,与无 smart 同路径**
        · 检索段:LockedRetriever 锁内;DeepSeek 网络调用:锁外(D8)
        · 零召回 → generator R3.E 确定性"信息不足"(不调 LLM)
        · 异常 → status=ask_failed(retriable),细节只进服务端日志
  ─► smart-ask(D9,PHAROS_SMART_ASK=on):looks_numeric(query) 且未显式给 kind 且
        is_refusal(第一轮答案) → 重问一轮 gen.answer(…, extra_legs=[DEFAULT_TABLE_LEG])
        (kind=table top_k=5 rerank_top_n=50;腿命中按 chunk_id 去重并集在主命中之后),
        auto+=["table_leg_retry"];硬上限 1 次重试
  ─► 拒答/部分拒答 → smart.build_hints(≤3 条,不重复建议已自动做过的动作)
  ─► Answer → {answer, citations[](默认不带原文,include_contexts=true 才带), n_contexts,
               finish_reason(=length 表示被 max_tokens 截断), model, auto[], hints[]}
```

**MCP 适配器**:六工具 = 纯转发函数 + `_call()` 统一错误映射
(RequestError→backend_unavailable+`pharos serve` hint;401→unauthorized;3xx→backend_unavailable
(httpx 不跟随重定向,否则误报"非 JSON");≥400→backend_unavailable;非 JSON→backend_unavailable)。
路径参数 doc_id 一律 `quote(…, safe="")`(#、/ 不再截断/改路由),空 doc_id 本地即拒(bad_arg)。
instructions 与引擎同源(toolcore._INSTRUCTIONS);六工具 docstring 与引擎逐字同文(回归测试钉住)。

## 3. 并发与锁

- FastAPI sync 端点跑在线程池 → 可能并发进 retriever。
- `LockedRetriever` 对 search_with_context / get_document / get_outline / expand / search_grouped /
  store.list_documents 全部加同一把 `threading.Lock`(嵌入式 Qdrant 与 GPU 前向都按串行对待)。
- SessionRegistry 自带锁;generator 为 per-thread 实例(threading.local,无共享可变状态)。
- toolcore 的 returned_keys set 操作发生在 impl 内(检索锁外)——单会话内的两次并发调用
  理论上可交错,个人场景(一个 agent 会话串行调工具)不构成实际问题,记入 TODO 观察项。

## 4. 与引擎的接缝(全部收在 engine.py)

| 接缝 | 方式 | 断裂时表现 |
|---|---|---|
| chunker/embedder/generator | sys.path 插 `<engine>/{pkg}/src` | bootstrap 抛 FileNotFoundError,报错含修复提示 |
| toolcore | importlib 按文件路径加载(模块名 pharos_engine_toolcore) | load_toolcore 抛 FileNotFoundError(要求引擎 ≥ 7fbf709) |
| 交付预算 | create_app 把 PHAROS_MAX_CONTEXT_TOKENS 写进 RAG_MAX_CONTEXT_TOKENS | —(toolcore 契约) |
| .env | 本仓 .env → 引擎 .env 兜底(DEEPSEEK_API_KEY 历史在引擎仓) | ask 返回 llm_unconfigured |

## 5. 配置兑现

`config.from_env()` 启动时读一次 → frozen dataclass;运行中不热更。
`PHAROS_INDEX_DIR` 展开后派生 qdrant_path/sidecar_dir(可分别覆盖)。
适配器进程只用 `adapter_base_url()`(PHAROS_URL)+ PHAROS_API_KEY,不建 PharosConfig。

## 6. 实现顺序(实际发生)

1. 引擎重构:toolcore 拆分(引擎 commit `7fbf709`,原测试不改跑绿);
2. 骨架 + config/engine/sessions;
3. service(FastAPI)+ mcp_adapter + indexer + cli;
4. 25 项 CPU 测试(一把全绿)→ on_event→lifespan 清告警;
5. GPU 冒烟(healthz/documents/retrieve/ask/去重/适配器/CLI 全过,见 TESTING.md);
6. 对抗评审 P1(39 agent,三视角×双反驳)→ 15 项修复 + 11 项回归测试 + 3 项证伪留档
   (清单见 COMPONENT_NOTES §对抗评审 P1);修复后全量回归 36+22+7 全绿。
