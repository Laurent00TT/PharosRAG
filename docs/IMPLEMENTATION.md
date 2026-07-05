# Pharos 实施文档

> 设计动机见 [DESIGN.md](DESIGN.md);本文讲"代码怎么长的"。

## 1. 模块地图

```
src/pharos/
  config.py       env → 不可变 PharosConfig;单一 .env(仓根 pharos/.env)加载
  engine.py       LockedRetriever(线程锁代理)+ build_retriever / build_user / build_generator
                  工厂,普通 in-repo 导入(from chunker/embedder/generator import …)
  toolcore.py     transport 无关的工具语义层(六工具原语 + _INSTRUCTIONS 单一来源)
  mcp_stdio.py    stdio-direct MCP server(pharos mcp --direct 的无守护回退)
  sessions.py     SessionRegistry:per-session returned_keys,有界 LRU(64 会话)
  service.py      FastAPI app 工厂 create_app(cfg, retriever, user, generator_factory 全可注入)
  smart.py        smart-ask 产品层(D9):拒答检测 + hints;数值判定/补检腿参数来自 signals
  mcp_adapter.py  MCP 薄适配器:六工具 → httpx 转发;结构化降级;进程级 uuid 会话头
  indexer.py      pharos index:MinerU 解析目录 → chunker → embedder(index_real.py 的参数化版)
  cli.py          serve / mcp / index / ask / health
  identity.py     多身份(D10):keys 文件解析/生成(fail-closed 校验)/身份 dataclass
  obs.py          可观测(D11):Stats(端点计数+延迟分位)+ RequestLog(JSONL,截断在本层)
src/{chunker,embedder,generator}/  三个组件包(pip install -e .[dev],src-layout 可编辑安装)
tests/            179 项(产品 59 + 引擎 120 在 tests/engine/);_fakes.py 提供 FakeRetriever/make_app
```

## 2. 请求路径

**/v1/retrieve(及其余 5 个检索端点)**:

```
HTTP 请求 ─► API key 中间件 ─► pydantic 请求模型(只定形,不做枚举校验)
  ─► X-Pharos-Session? → SessionRegistry.get(sid) / None
  ─► toolcore._retrieve_impl(LockedRetriever, bound_user, …, returned_keys)
        · 校验(no_identity/empty_query/bad_arg)在锁外、检索在锁内
        · already_returned / omitted_budget / 预算含资产 —— 全部 toolcore 原语义
  ─► _adapt():no_identity 的 hint 统一成 PHAROS_TENANT 措辞(评审 C2:toolcore 底层 hint 若指旧 RAG_TENANT 别名会死循环误导)
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
instructions 单一来源于 toolcore._INSTRUCTIONS;适配器与 mcp_stdio 的六工具 docstring 逐字同文
(in-repo 结构化回归测试钉住:adapter vs mcp_stdio docstring 相等 + _INSTRUCTIONS 单源)。

## 3. 并发与锁

- FastAPI sync 端点跑在线程池 → 可能并发进 retriever。
- `LockedRetriever` 对 search_with_context / get_document / get_outline / expand / search_grouped /
  store.list_documents 全部加同一把 `threading.Lock`(嵌入式 Qdrant 与 GPU 前向都按串行对待)。
- SessionRegistry 自带锁;generator 为 per-thread 实例(threading.local,无共享可变状态)。
- toolcore 的 returned_keys set 操作发生在 impl 内(检索锁外)——单会话内的两次并发调用
  理论上可交错,常规用法(一个 agent 会话串行调工具)不构成实际问题,记入 TODO 观察项。

## 4. 组件装配(全部收在 engine.py)

组件(chunker/embedder/generator)与 toolcore 都是同仓包,`pip install -e .[dev]` 后普通 import;
engine.py 只做工厂装配,不再有跨仓接缝。

| 装配点 | 方式 | 断裂时表现 |
|---|---|---|
| chunker/embedder/generator | in-repo 包(`from chunker/embedder/generator import …`),src-layout 可编辑安装 | ImportError(未 `pip install -e .`) |
| toolcore | 同仓模块 `from pharos import toolcore`(六工具原语 + _INSTRUCTIONS) | ImportError |
| 交付预算 | create_app 把 PHAROS_MAX_CONTEXT_TOKENS 传给 toolcore | —(toolcore 契约) |
| .env | 单一 .env(仓根 pharos/.env;DEEPSEEK_API_KEY 在此) | ask 返回 llm_unconfigured |

## 5. 配置兑现

`config.from_env()` 启动时读一次 → frozen dataclass;运行中不热更。
`PHAROS_INDEX_DIR` 展开后派生 qdrant_path/sidecar_dir(可分别覆盖)。
适配器进程只用 `adapter_base_url()`(PHAROS_URL)+ PHAROS_API_KEY,不建 PharosConfig。

## 6. 构建脉络

自底向上、每层带测试:

- **toolcore 先行**:从 MCP server 拆出 toolcore(transport 无关的工具语义层),让 stdio / HTTP 两条
  transport 共用同一套契约(单一来源)。组件测试一行未改跑绿 = 拆分无回归的证据。
- **产品层分层搭建**:config/engine/sessions → service(FastAPI)+ mcp_adapter + indexer + cli →
  身份(identity)+ 观测(obs)。每加一层补 CPU 单测(fake retriever + MockLLM,不碰 GPU/网络)。
- **每层两道把关**:CPU 单测(逻辑,产品 59 + 引擎 120 = 179 项)+ GPU 冒烟/压测/演练(真库真行为);
  行为质量(smart-ask)与服务面(多身份)各经一轮对抗评审 + 自核实修复。全部数字与实录见 [TESTING.md](TESTING.md)。
