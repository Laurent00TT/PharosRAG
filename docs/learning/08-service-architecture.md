# 08 服务化与工程架构

> **本篇导读**:一个 RAG 从"能跑的检索脚本"到"多人共用的常驻服务",要跨过进程形态、并发锁、身份鉴权、会话状态、错误契约、可观测、健康探针七道关口。本篇讲 Pharos 的服务层怎么过这七关,以及每个决策背后被否决的备选和实测数据。
> **面试权重:高**——这是后端/平台向岗位区分"调包侠"与"工程师"的主战场;RAG 算法可以背,服务化的取舍只能真做过才讲得出。
> **前置阅读**:[../DESIGN.md](../DESIGN.md)(D1-D12 决策原文);数据口径见 [07 评估方法论](07-evaluation.md)。

---

## 1. 概念底座:为什么 RAG 需要一个"服务层"

教程里的 RAG 是一段脚本:加载模型 → 打开向量库 → 查询 → 生成。它默认了三件事:**单用户、单进程、进程活多久无所谓**。生产环境三条全破:

1. **资源独占性**。嵌入式向量库(Qdrant local / Chroma persistent / LanceDB)普遍是**单进程独占**的——第二个进程打开同一路径直接报错或损坏数据。嵌入模型(尤其 GPU 上的大模型)加载要几十秒到几分钟。谁持有这些资源、别人怎么共享,是服务化的第一问。
2. **多消费方**。同一套检索栈会被脚本(curl)、闭管道问答、agent 工具调用(MCP)同时消费。每个消费方各自实现一遍"校验/去重/预算/错误处理",必然漂移。
3. **程序化消费方的错误语义**。调用方是 agent 而非人:它需要靠结构化状态机决定"重试还是换策略",不能靠解析自然语言报错。

主流方案光谱大致三档:

| 形态 | 代表 | 优点 | 代价 |
|---|---|---|---|
| 每进程各自加载(脚本式) | LangChain/LlamaIndex 单机 demo | 零运维 | 独占锁互踩、模型重复加载、无共享 |
| **常驻守护进程 + 轻客户端** | 本项目;Ollama 式形态 | 资源加载一次、多消费方共享热后端 | 要自己解决并发/鉴权/探活 |
| 全托管拆分(向量库 server + 无状态应用 + 推理服务) | 云上标准架构 | 每层独立扩缩 | 运维面数倍、小规模不划算 |

Pharos 从第二档起步,规模需求一出现就往第三档走(拆 GPU 推理层 + Qdrant server + nginx 多副本,见 [../SCALE_OUT.md](../SCALE_OUT.md))——这条演进弧线本身就是本篇最有面试价值的素材。

---

## 2. Pharos 怎么做

### 2.0 数据流总览:一个守护进程,三个入口

```
curl / 脚本 ──────────────┐
pharos mcp(薄适配器,stdio↔HTTP)──┤→ pharos serve(FastAPI 守护进程)
                                │     ├─ toolcore(工具语义单一来源)
pharos mcp --direct(stdio 直连,│     ├─ Retriever(Qdrant 嵌入式/server + GPU local/remote)
  降级路径,不经守护进程)────────┘     └─ Generator(检索 + grounding prompt + DeepSeek)
```

**D1:守护进程独占资源**。`pharos serve` 是系统里唯一打开嵌入式 Qdrant、加载 GPU 模型的进程,原因是两个硬约束([src/pharos/service.py:1-16](../../src/pharos/service.py#L1) 文件级 docstring 写明):

- 嵌入式 Qdrant **单客户端独占锁**——第二个进程打开同一路径直接报错([src/embedder/store.py:25-32](../../src/embedder/store.py#L25) 的三分支与 `_lock` 注释);
- dense 模型(Qwen3-VL 8B)**加载 1-2 分钟**——"每会话一进程"意味着每开一个 agent 会话重付这笔钱。

独占锁在 lifespan 启动时才真正取得([src/pharos/service.py:105](../../src/pharos/service.py#L105)),retriever/user/generator 全部是注入点——测试用 fake 完整替换,这是整个服务层能纯 CPU 回归的前提。三个入口:HTTP 直打;`pharos mcp` 薄适配器(零 GPU 依赖,毫秒启动);`pharos mcp --direct` 在守护进程没跑时用 stdio 直连引擎,作降级路径([src/pharos/cli.py:40-47](../../src/pharos/cli.py#L40)),但与 serve 不能同开同一索引。

冒烟实测数据:适配器毫秒级连上,首查 19s(模型热缓存中),后续秒级(DESIGN D1)。

### 2.1 toolcore:工具语义单一来源(D3)

六个工具(retrieve / list_documents / get_document / get_outline / expand / retrieve_grouped)的入参校验、结构化结果、跨调用去重、token 预算、错误映射、agent 使用契约 `_INSTRUCTIONS`,全部收在 [src/pharos/toolcore.py:1-13](../../src/pharos/toolcore.py#L1)——纯 stdlib,不 import FastMCP/embedder/GPU,retriever 与 user 靠 duck typing 注入。HTTP 端点与两个 MCP 绑定都只做 transport 绑定,语义零复制。

这层契约里有三条经对抗评审打磨出来的细节链,值得背下来:

1. **预算必须计入资产 `content_raw`**([src/pharos/toolcore.py:105-109](../../src/pharos/toolcore.py#L105)):表格命中的散文 text 近乎为空、数据全在 content_raw(可数千 token),漏算它就等于让最大的载荷绕过软上限;
2. **section_window 去重键用 `(doc_id, resolved_section)` 而非 anchor**([src/pharos/toolcore.py:112-117](../../src/pharos/toolcore.py#L112)):窗口 anchor 随命中种子漂移,用它永远判不了重;
3. **returned_keys 登记推迟到预算之后、只登记真正交付了正文的命中**([src/pharos/toolcore.py:151-158](../../src/pharos/toolcore.py#L151)):否则 `omitted_budget` 的块下次会被误判 `already_returned`——agent 明明从没收到过。

还有一条分层纪律:运行期异常里"推理服务暂不可用"要给 agent 更准的 retriable 语义,但 toolcore 不许 import embedder.errors(保持 stdlib-only),于是用 duck-typing marker 属性分流:`getattr(e, "inference_unavailable", False)`([src/pharos/toolcore.py:230-233](../../src/pharos/toolcore.py#L230))。**依赖方向本身是设计资产,宁可用 getattr 也不破分层。**

### 2.2 身份:三种模式 + fail-closed 铁律(D10)

- **keys 模式**(团队):`PHAROS_KEYS_FILE` 指 JSON,每请求 X-API-Key 解析成 `Identity{name, tenant, principals, admin}`,未知/缺失一律 401 且不泄"key 是否存在过"([src/pharos/service.py:148-164](../../src/pharos/service.py#L148));
- **legacy**(单人):单 `PHAROS_API_KEY` 门槛;**open**:都不设,仅回环可用。

fail-closed 有三级,全部**启动期响亮失败**而非运行期静默放行:

1. tenant 未设 → toolcore 每个工具入口先查 `user.tenant`,空则返回 `no_identity` 空结果([src/pharos/toolcore.py:215-216](../../src/pharos/toolcore.py#L215));
2. 绑非回环地址而非 keys 模式 → `create_app` 直接 `SystemExit`([src/pharos/service.py:93-95](../../src/pharos/service.py#L93))——"**部署即授权**",不允许把整库裸奔到局域网;
3. keys 文件任何格式错(短 key、缺 name/tenant、name 重复、name 含 `|`、key 重复)→ `SystemExit`([src/pharos/identity.py:29-64](../../src/pharos/identity.py#L29)),绝不静默降级。

keys 模式下 `_current_user` 按解析出的身份**逐请求现建**引擎 User([src/pharos/service.py:128-133](../../src/pharos/service.py#L128))——"谁在问"(服务层 identity)与"能看什么"(embedder ACL 硬过滤)正交分层(D12),身份不越界替 ACL 决策。

### 2.3 会话去重:opt-in + 身份|会话双重隔离(D6)

stdio 直连里"进程 = 会话",一个进程级 set 就够([src/pharos/mcp_stdio.py:83-85](../../src/pharos/mcp_stdio.py#L83),其注释当年就预告了"换 HTTP 多会话前必须 per-session 隔离")。守护进程是多会话共享的,所以:

- 请求带 `X-Pharos-Session` 头才启用去重(**不带 = 不去重**,curl 一次性调用不该有跨调用状态);
- `SessionRegistry` 是有界 LRU(64 会话,逐出丢去重不影响正确性,[src/pharos/sessions.py:16-32](../../src/pharos/sessions.py#L16));
- 登记键是 `f"{身份名}|{会话id}"`([src/pharos/service.py:204-208](../../src/pharos/service.py#L204))——多用户下即使伪造相同会话 id 也互不可见。

这个登记键设计**反向推导出了输入校验规则**:身份 name 禁止含 `|`(否则 `'a' + 'b|c'` 与 `'a|b' + 'c'` 同键,命名空间碰撞)且必须唯一(重名两身份共享去重命名空间 = 串味),见 [src/pharos/identity.py:51-58](../../src/pharos/identity.py#L51)。MCP 适配器每进程生成一个 uuid 作会话头([src/pharos/mcp_adapter.py:26](../../src/pharos/mcp_adapter.py#L26)),stdio 下天然获得会话语义。

### 2.4 错误契约:领域结果一律 HTTP 200 + status 字段(D7)

`no_identity / empty_query / bad_arg / no_access / config_error / backend_unavailable / inference_unavailable / llm_unconfigured / ask_failed` 都是**领域结果**——agent 要程序化决策的状态机——统一 200 + JSON `{status, retriable, hint}`([src/pharos/toolcore.py:64-66](../../src/pharos/toolcore.py#L64))。HTTP 状态码只留给传输层:401 鉴权、403 stats 非 admin、422 请求体不是合法 JSON、5xx 崩溃。

两个反直觉细节:

- mode/strategy 等枚举校验**故意不放 pydantic 层**(否则变 422),留给 toolcore 出结构化 `bad_arg`([src/pharos/service.py:43](../../src/pharos/service.py#L43) 注释 + [src/pharos/toolcore.py:221-224](../../src/pharos/toolcore.py#L221));
- `_safe_doc_call` 统一异常映射时,**无权与不存在同响应**(不泄存在性),sidecar 损坏映射成 `config_error` 而非掩盖成 `no_access`,其余异常一律通用降级、不向不可信 agent 泄内部消息/栈([src/pharos/toolcore.py:198-208](../../src/pharos/toolcore.py#L198))。

### 2.5 锁模型:检索资源锁内、LLM 锁外(D8,现为"锁下沉"版)

FastAPI 的 sync 端点跑在线程池里,会并发进 retriever。这里有一条完整的演进弧线:

1. **原设计**:一把 `LockedRetriever` 大锁串行全部检索调用——检索在锁内、LLM 网络调用在锁外;
2. **SCALE_OUT 阶段 B 审查发现**:remote 推理后端的 HTTP 重试退避 sleep 会**持大锁阻塞整副本**(预热/滚动重启期一次退避,所有查询排队);
3. **锁下沉到资源类**([src/pharos/engine.py:6-12](../../src/pharos/engine.py#L6)):Qdrant 单 client 段 → `Store._lock`([src/embedder/store.py:32](../../src/embedder/store.py#L32),嵌入式非线程安全);GPU 前向 → `Dense/Reranker._fwd_lock`(remote 模式 override 成 nullcontext);query LRU → `_cache_lock`。remote 的 HTTP + 退避从此落在所有锁外。

`/v1/ask` 的关键不变量照旧成立:检索的 Qdrant/GPU 段在资源锁内,随后数秒到数十秒的 DeepSeek 调用**不持任何锁**([src/pharos/service.py:339](../../src/pharos/service.py#L339) 注释),一次慢生成不会饿死其他会话的检索。

同一轮对抗评审还抓到一个并发串味:Generator/LLM 共享单例时,`llm.last_finish_reason` 在并发 ask 下会跨请求污染(A 拿到 B 的截断标志)。修法不是加锁——加锁会把 LLM 网络调用串行化,违反 D8——而是 **Generator per-thread 惰性构建**([src/pharos/service.py:122-123](../../src/pharos/service.py#L122)、[304-310](../../src/pharos/service.py#L304)):线程池有界所以实例数有界,同线程内 answer→读 finish_reason 无并发窗口。**用执行模型消灭共享,而不是给共享加锁。**

⚠ 注意:[../DESIGN.md](../DESIGN.md) D8 原文仍描述大锁版(文档尚未回填);面试与学习以代码为准,讲"不变量 + 锁下沉演进"。

### 2.6 smart-ask:闭管道的有界智能(D9)

用户用默认参数问"Netflix 2011-2015 每年净利润"只得到三年——五年表在库里,但排序把它排在 top-k 窗口外。旋钮(kind=table/rerank)都存在,但**用户不该需要懂旋钮**。Pharos 的答案是两层([src/pharos/service.py:331-353](../../src/pharos/service.py#L331)、[src/pharos/smart.py:25-36](../../src/pharos/smart.py#L25)):

- **第 1 层 hints**:最终答案命中拒答模式时,按本次请求参数生成 ≤3 条可操作建议,不重复建议已自动做过的动作;正常答案零打扰。
- **第 2 层 失败驱动表格补检**:第一轮完全纯净;仅当 `looks_numeric(query)` 且用户未显式给 kind 且第一轮拒答时,带 `kind=table, top_k=5, rerank_top_n=50` 的补检腿重问一轮(命中与主检索**并集**,硬上限 1 次)。**重试择优**:重试答案只有不再是拒答才替换,否则保留第一轮诚实拒答 + hints 并留痕 `table_leg_retry_discarded`。

三条红线:一切自动行为在响应 `auto` 字段留痕;`PHAROS_SMART_ASK=off` 一键纯净;数值判定/拒答判定/腿参数单一来源自 `generator.signals`——eval 考卷跑的就是生产行为。为什么是"失败驱动"而不是"前置腿",见 §3 的实测裁决。

### 2.7 可观测:三条纪律(D11)

两个 async 中间件:`_auth` 在内层,`_observe` 在外层且 **try/finally**——处理器崩溃也记录(rec 带 `crashed=true`),否则最严重的错误在观测里恰好隐形([src/pharos/service.py:167-197](../../src/pharos/service.py#L167))。三条纪律:

1. **200 也可能是失败**:errors 计入 http≥400 **或**结构化业务失败(status 不在 ok/empty)([src/pharos/service.py:182-184](../../src/pharos/service.py#L182))——只看 http 码会漏计一切返回 200 的领域错误;
2. **观测绝不拖垮服务**:JSONL 落盘走后台单写线程 + 有界队列,`write` 只做 `put_nowait`,队满立即丢弃并计数([src/pharos/obs.py:112-125](../../src/pharos/obs.py#L112))。这是阶段 F 审查的产物:原先在事件循环里同步 open/append,共享 bind-mount 卷 IO 一卡顿会冻结整个事件循环——**副本所有端点连同健康探针一起停摆,观测反而成了可用性单点**;
3. **不落敏感信息**:绝不落盘 key 本体(只记身份 name);query 默认截断 120 字、`PHAROS_LOG_QUERIES=off` 删除,且隐私边界收在入队前单点执行([src/pharos/obs.py:115-120](../../src/pharos/obs.py#L115))——"截断只在中间件做"的话任何新写入方都会漏。`/v1/stats` 在 keys 模式下 admin key 才可读(聚合查询模式也是信息,[src/pharos/service.py:258-268](../../src/pharos/service.py#L258))。

### 2.8 探针体系:liveness/readiness 分离 + 抗饥饿(阶段 F)

`/healthz` 是 liveness(进程活着,纯内存读);`/readyz` 是 readiness(探下游可达且 collection 存在),编排/healthcheck 据此决策——下游抖动只摘本副本流量,不触发 crashloop([src/pharos/service.py:211-256](../../src/pharos/service.py#L211))。

最好的 SRE 故事在"抗饥饿":上多副本前的全量审查发现,sync 探针会与 `/v1/ask`(LLM 数十秒)、inference hang 时最长 ~361s 的 `/v1/retrieve` **共享 anyio 默认 40 线程池**。高负载下探针在池里排队饥饿 → healthcheck 超时判 unhealthy → nginx 把**正在正常干活的健康副本全摘掉** = 全局 outage。负载越高,"我还健康"的信号越假——恰好在最不该摘副本的时刻摘副本。修法:

- 探针全 async 化(healthz 在事件循环直跑不进线程池);
- readyz 的 qdrant 阻塞调用 offload 到专用 `_PROBE_LIMITER`(8 线程,与业务池隔离,[src/pharos/service.py:38-40](../../src/pharos/service.py#L38)、[236-238](../../src/pharos/service.py#L236));
- inference 探活用 AsyncClient 且显式短超时 `httpx.Timeout(1.5, connect=1.0)`([src/pharos/service.py:250](../../src/pharos/service.py#L250))——评审又抓到默认 3s 每阶段最坏 ~9s 会超 healthcheck 5s,探针自己把自己误判。

安全同样收口:探针未鉴权,异常只记服务端日志、响应体不回 `str(e)`(否则泄内网 qdrant/inference host:port),collection 缺失只回 `collection_missing` 不回集合名。优雅停机预算自洽:uvicorn `timeout_graceful_shutdown=25` < compose `stop_grace_period` 30s([src/pharos/cli.py:34-37](../../src/pharos/cli.py#L34)),SIGKILL 前 drain 在途请求 + lifespan flush 日志队列(timeout=5.0,[src/pharos/service.py:109-115](../../src/pharos/service.py#L109))。

F-2 端到端实测(3 副本 + nginx):轮询分发 6/6/7 近均匀;持续 50 请求中 `docker kill` 一副本,**50/50 全 200 零失败**([../SCALE_OUT.md](../SCALE_OUT.md) F-2)。

### 2.9 两个"系统不报错但把人引向死路"的教训

服务层还有两类不体现在架构图上、却最能暴露工程成色的 bug,都被对抗评审抓到过:

**错误提示的死循环**(P1 评审 C2)。toolcore 的 `no_identity` 提示早年写的是"请设置 RAG_TENANT"(引擎期命名),而 Pharos 服务只读 `PHAROS_TENANT`。用户照 hint 做 → 设了 RAG_TENANT → 重启 → 仍然全空 → hint 还是让他设 RAG_TENANT——**错误提示本身制造了一个无法逃出的排障死循环,比没有提示更糟**。后来命名空间统一为 PHAROS_*,错误从源头消除([src/pharos/toolcore.py:45-46](../../src/pharos/toolcore.py#L45)),服务绑定层的 `_adapt` 退化为防线([src/pharos/service.py:139-145](../../src/pharos/service.py#L139))。提炼:fail-closed 系统的错误提示必须和配置面同步演进,"提示指向的动作做了没用"是一类值得专门审计的 bug。

**配置透传坑的复发**(阶段 F 审查)。D 阶段加 `qdrant_url` 时曾宣称"透传三出口"却漏改 engine,被自己的核实纪律逮住;F 阶段全量审查发现同款坑在另一个开关上复发:`mcp_stdio._config` 构造 EmbedConfig 时漏了 `inference_url`。agentic 出口明明配了 `PHAROS_INFERENCE_URL`,却在这里静默丢失;slim(无 torch)环境首查走 local 路径,import torch 直接崩,又被 toolcore 的宽兜底吞成 `backend_unavailable`——三层掩盖后,用户只看到"后端暂不可用"。修法是补透传并顺带对齐模型路径/gpu_name([src/pharos/mcp_stdio.py:56-66](../../src/pharos/mcp_stdio.py#L56)),配守护测试 `test_mcp_stdio_config_passes_inference_url`(删透传即红)。结构性教训:**每加一个生产配置开关,必须枚举全部消费出口并各配一条"删了透传就红"的守护测试**——"配置静默丢失"比配置错误更危险,因为系统看起来在正常工作。

### 2.10 MCP 薄适配器:同名同参同契约的第二种绑定

`pharos mcp` 只 import mcp + httpx + toolcore,六个工具原样转发 HTTP([src/pharos/mcp_adapter.py:1-12](../../src/pharos/mcp_adapter.py#L1))。`_call` 统一错误映射、永不向 agent 裸抛([src/pharos/mcp_adapter.py:42-64](../../src/pharos/mcp_adapter.py#L42)):网络错 → `backend_unavailable` 且 hint 给出恢复动作(`pharos serve`);401 → `unauthorized`;3xx → 结构化降级(httpx 默认不跟随重定向,否则误报"非 JSON");4xx → `contract_mismatch` 不可重试(§4 修复);5xx → 可重试。读超时 600s(守护进程首查 lazy load 8B 模型,适配器不能先超时);路径参数 doc_id 一律 `quote(safe='')`,空 doc_id 本地即拒([src/pharos/mcp_adapter.py:95-97](../../src/pharos/mcp_adapter.py#L95))。工具 docstring 与 stdio 直连逐字同文,由结构化回归测试(`test_transports_contract_no_drift`)钉住不漂移。

---

## 3. 为什么这么设计:被否决的备选

| 备选 | 否决理由 | 数据/证据 |
|---|---|---|
| 每消费方各开索引 | 嵌入式 Qdrant 单客户端锁直接堵死 | 第二进程打开同路径即报错(eval 为此专门 copytree 避锁) |
| stdio 直连作默认 | 每会话重付 1-2min 模型加载 | 保留作 `--direct` fallback;守护进程模式适配器毫秒级连上 |
| Qdrant server 模式(初期) | 单机小团队下运维复杂度不抵收益 | 后在 SCALE_OUT 阶段 D 兑现;且 DESIGN 顶部自我批注"换 url 即可"是过度简化——实际要 store 三分支 + 全出口透传 + 迁移 + ACL 越权重测 |
| HTTP 层复制 toolcore 逻辑 | 同一语义两处实现必然漂移 | 拆分时原 test_tools.py 一行未改 22 项全绿 = 纯移动无回归 |
| 枚举校验放 pydantic | 会变 422,破坏 agent 的 200+status 状态机 | `test_retrieve_bad_arg_structured_not_422` 钉住 |
| SSO/OIDC / 数据库存密钥 / key 热加载 | 当前规模不值得引 IdP;文件 + 重启秒级足够;热加载增加状态一致性面 | 轮换 = 改文件 + systemctl restart |
| 前置表格补检腿 | 88 题 A/B 实测:表格题 0.625→0.875 **但误伤 5 道散文题**(0.861→0.792) | 失败驱动版:表格 0.625→0.75+、散文持平、忠实度全卷 1.000,总体更优 |
| 重试无条件替换第一轮 | 部分回答夹带"未提供 X"的错误缺失声明(X 实际在 context 里),忠实度 1.0→0.93 | 改"重试择优":宁保留诚实拒答 |
| 默认全局 rerank / 默认加大 top_k / 全量查询翻译 / 闭管道内多轮循环 | 每问 +3~5s / 只加噪声 / +1 次 LLM 调用 / 那是 MCP 出口的职责(agent 编排实测净负 Δ−0.097,口径见 §8) | 88 题实测(DESIGN D9) |
| 保留大锁到多副本 v1 | 阶段 B 审查发现 remote 重试退避持锁 = 整副本停摆 | `tests/engine/test_concurrency.py` 锁死"remote encode 退避在 search_with_context 之外" |
| 跨副本共享去重(Redis/粘滞) | 去重是便利不是正确性,规模到了再做 | nginx 轮询下去重效果掉到 ~1/N,是声明过的可接受降级(SCALE_OUT F-3) |
| Prometheus/Grafana | 团队规模不值得运维一套监控栈 | JSONL 可被任何工具后加工;指标重启归零列 TODO |

⚠ 数据口径提醒:smart-ask 的 0.625/0.875/0.861 等数字全部来自 **88 题**考卷双版实测,不可与早期 72 题基线混用;引用时注明口径。

腿参数还有一个教训值得单独记:`rerank_top_n=30` 装不进粗排 31-50 名的五年表(重试形同虚设),50 一发命中——**精排纠正的是排序,前提是候选池里得有它**。

---

## 4. 实战复盘:本次对抗审查(2026-07-07)

写这套学习文档前,对服务层做了一轮"分析师深读 → 独立验证员先试图反驳"的对抗审查。全仓证据:修复前基线 `pytest tests -q` = 224 passed / 4 skipped,修复后 259 passed / 5 skipped(新增 36 用例)。服务层落地 4 项、留档 2 项。

### 已修的 4 项(症状 → 根因 → 修法 → 测试)

1. **stats 键基数无界(medium)**。症状:长期运行的守护进程内存缓慢增长,`/v1/stats` 快照可被撑爆。根因:指标键用原始 URL 路径,`/v1/documents/abc` 与 `/v1/documents/xyz` 是两个键,每个新键分配一个 maxlen=1000 的延迟 deque——枚举 doc_id(未鉴权 404 也计量)= 低速 DoS。修法:改用**路由模板**做键(`/v1/documents/{doc_id}`),无匹配路由归并 `/v1/_unmatched`,键集合 = 已注册路由数 + 1 天然有界;JSONL 日志的 ep 保留原始路径(调试价值在磁盘不在内存)([src/pharos/service.py:185-191](../../src/pharos/service.py#L185))。测试:`test_stats_keys_bounded_by_route_template`、`test_stats_unauthorized_requests_bounded`。
2. **healthz 信息收敛(low)**。症状:未鉴权的 `/healthz` 返回 collection 名与 llm_model。根因:readyz 因评审刻意不回集合名、不回 `str(e)`,healthz 却回了——同一个评审建立的信息边界被隔壁端点架空。修法:healthz 收敛为 `{status, service, version, tenant_bound, uptime_s}`,敏感字段挪进 admin-gated 的 `/v1/stats`([src/pharos/service.py:211-222](../../src/pharos/service.py#L211)、[264-267](../../src/pharos/service.py#L264))。教学点:**安全加固要按"信息边界"审计,不是按端点逐个看**。
3. **adapter 4xx 映射(low)**。症状:适配器把所有非 401 的 4xx(含版本漂移导致的 422)标 `retriable=true` 并建议"请稍后重试"。根因:契约错误是永久性的,重试永不修复,retriable 语义会驱动 agent 无效循环且误导排障方向。修法:4xx → `contract_mismatch` + `retriable=false` + hint 指向"核对适配器与服务版本/PHAROS_URL";仅 ≥500 保留 `backend_unavailable`([src/pharos/mcp_adapter.py:52-60](../../src/pharos/mcp_adapter.py#L52))。测试:`test_422_maps_contract_mismatch_not_retriable`、`test_404_maps_contract_mismatch_not_retriable`。
4. **flush 真 timeout(low)**。症状:`RequestLog.flush(timeout)` 的参数被静默忽略,实现无条件 `q.join()`。根因:stdlib `queue.join()` 不支持超时;写线程卡在 bind-mount IO 挂起时(正是该模块要防的场景),优雅停机会冻死到 stop_grace_period 被 SIGKILL——"drain 期不丢日志"的目标在最需要它的故障下反而失效。修法:用 `all_tasks_done` 条件变量带 deadline 轮询,超时 warning 后放弃返回;lifespan 传 `timeout=5.0`(25s drain + 5s flush ≤ 30s 宽限,预算自洽)([src/pharos/obs.py:92-110](../../src/pharos/obs.py#L92)、[src/pharos/service.py:113](../../src/pharos/service.py#L113))。测试:`test_reqlog_flush_timeout_returns_and_warns`(注入卡死的假写路径,断言按时返回)、`test_reqlog_flush_timeout_normal_drain`。

### 留档不改的 2 项——"确认了但不改"也是工程判断

- **service#0(/readyz 绕过 Store._lock 直读嵌入式 client):refuted**。这是本轮唯一被验证员**反驳成功**的服务层报告,教学价值恰恰在论证过程:表面看,readyz 直接调 `state.retriever.store.client.collection_exists(...)`([src/pharos/service.py:236-238](../../src/pharos/service.py#L236))绕过了所有业务方法都持有的 `Store._lock`,像一个并发 bug。反驳链:① serve 进程**零写路径**(建库在 indexer,文档明确要求先停 serve);② `collection_exists` 在 QdrantLocal 里是纯字典成员读;③ store.py 注释自声明的并发模型是"读读安全、读写不安全"([src/embedder/store.py:4-5](../../src/embedder/store.py#L4))——三条合起来,竞态触发条件不可达。结论:不改码,留一条"若未来 serve 引入在线写端点,此处需改走带锁方法"的档。**一个"看起来像并发 bug"的报告,要用进程的读写路径清单去证伪,而不是凭"没锁 = 危险"的直觉去修**——盲目加锁反而把探针耦合进业务锁,重新制造探针饥饿。
- **service#5(create_app 写进程级 env 兑现 toolcore 预算):两轮验证结论冲突,保守留档**。`create_app` 把 `cfg.max_context_tokens` 写进 `os.environ`([src/pharos/service.py:86](../../src/pharos/service.py#L86)),toolcore 每次检索现读 env([src/pharos/toolcore.py:55-61](../../src/pharos/toolcore.py#L55))——同进程先后建两个不同预算的 app 会互相覆盖。一轮验证判 confirmed(测试套件确实同进程多 app),另一轮判 refuted(生产入口一进程一 app,触发不可达)。**两轮结论冲突时的处置纪律:不武断采信任何一方,按保守处理留档**——生产路径不改(收益不抵改签名的折腾),但写明"若未来同进程多 app,应把预算参数化进 toolcore(有 returned_keys 依赖注入的先例)"。

---

## 5. 面试怎么讲

### 30 秒版

"我把一个多格式 RAG 做成了守护进程服务:两个硬约束——嵌入式向量库单客户端独占锁、8B 嵌入模型加载 1-2 分钟——逼出'守护进程独占资源、一切消费走 HTTP'的形态,三个入口(HTTP、MCP 薄适配器、stdio 直连降级)共用 toolcore 一套工具契约防漂移。服务面做了多身份 fail-closed 鉴权、opt-in 会话去重、200+status 的结构化错误契约、'检索锁内 LLM 锁外'的锁模型,以及探针与业务线程池隔离的健康体系。上 nginx 三副本实测 kill 一副本 50/50 请求零失败。"

### 3 分钟版(结构化展开)

1. **形态怎么来的**:不是"想做微服务",是资源约束推导——嵌入式 Qdrant 第二进程打开即报错、模型每进程重付 1-2 分钟。守护进程独占 + HTTP 共享是唯一让多个 agent 会话共享热后端的形态;stdio 直连保留作降级路径。
2. **防漂移**:六个工具的校验/去重/预算/错误映射收在纯 stdlib 的 toolcore,HTTP 与两条 MCP 绑定零语义复制;这层契约经五轮对抗评审打磨(预算计入表格 content_raw、去重键抗 anchor 漂移、登记推迟到预算后),复制必漂移。
3. **安全模型**:"部署即授权"——绑非回环没配多身份 keys 直接拒绝启动;keys 文件任何格式错拒绝启动;身份服务端权威决定,客户端不可篡改;无权与不存在同响应不泄存在性。
4. **并发**:锁模型有演进弧线——大锁(检索锁内/LLM 锁外)→ 审查发现 remote 重试退避持大锁整副本停摆 → 锁下沉到资源类;并发串味用 per-thread Generator 解决而不是加锁(加锁违反"LLM 在锁外"的不变量)。
5. **数据点**:闭管道忠实度 ≈1.0、正确性 0.847(**Tier2 双 Claude 裁判,72 题口径**);smart-ask 用失败驱动而非前置腿,是 **88 题** A/B 实测裁决——前置腿表格 0.625→0.875 但误伤 5 道散文题,失败驱动表格 0.625→0.75+、散文持平、忠实度全卷 1.000。(两组数字口径不同:88 题 DeepSeek Tier1 的闭管道正确性是 0.818,不可与 72 题的 0.847 相减。)
6. **可用性**:探针饥饿(sync 探针与业务共享 40 线程池,负载越高就绪信号越假)→ 探针 async 化 + 专用 8 线程 limiter;3 副本 kill 实测 50/50 零失败。

---

## 6. 追问预演

1. **"为什么不一开始就上 Qdrant server?"** 要点:决策是分阶段的——单机小团队下,server 模式引入常驻服务运维 + 数据迁移,收益不抵复杂度;规模需求出现后在 SCALE_OUT 阶段 D 兑现。加分点:主动承认当年"换 url 即可"的预估是过度简化,实际要 store 三分支 + 全出口配置透传 + 迁移脚本 + server 模式 ACL 越权重测(`:memory:` 版会假绿)。
2. **"错误为什么都返回 200?监控怎么办?"** 要点:消费方是 agent,领域状态机(no_access/bad_arg/retriable)必须结构化、与传输层语义分离;监控侧对应地**不能只看 http 码**——errors 判定是"http≥400 或 status 不在 ok/empty",否则 200 包裹的失败全部漏计。延伸(诚实):200 包裹也让 nginx 的被动摘除(`proxy_next_upstream http_5xx`)对病副本失明,已确认待修(deferred deploy#1:对 `inference_unavailable/backend_unavailable` 改用 503,body 不变)。
3. **"会话去重为什么不用 Redis?多副本下不就失效了?"** 要点:去重是 opt-in 的 token 节省便利,**不是正确性**——轮询下掉到 ~1/N 是声明过的可接受降级;引入 Redis = 引入新的共享状态与故障域,规模到了再做(粘滞或共享存储两条路都留了)。关键词:便利与正确性分级、声明式降级。
4. **"身份 name 为什么禁止 `|`?"** 要点:这是从数据结构反推的输入校验——去重登记键是 `name|sid` 字符串拼接,含分隔符会造成命名空间碰撞(`'a'+'b|c'` 与 `'a|b'+'c'` 同键);重名同理(两身份共享去重空间 = 串味)。能把"一条校验规则的可推导性"讲清楚,比背十条校验规则加分。
5. **"探针为什么会把健康副本搞下线?"** 要点:sync 探针与业务共享 anyio 默认 40 线程池;LLM 数十秒 + inference hang 时单请求最长 ~361s 重试,高负载下探针排队饥饿 → healthcheck 假 unhealthy → LB 摘副本 → 存活副本更挤 → 雪崩。修法关键词:探针可靠性等级必须高于业务、async 化、专用 CapacityLimiter、显式短超时(默认 3s×3 阶段 ~9s > healthcheck 5s 的自我误判)。
6. **"检索为什么要锁?锁粒度怎么定的?"** 要点:嵌入式 Qdrant 单 client 与 GPU 前向非线程安全,必须串行;但锁的范围要以"谁是真正的非线程安全资源"为界——大锁把 remote 重试退避也圈了进去,一次退避阻塞整副本,所以下沉到 Store/_fwd_lock/_cache_lock 三类资源锁;LLM 纯网络 IO 永远在锁外。server 模式下连 `Store._lock` 都可删(Qdrant server 并发安全)。
7. **"smart-ask 会不会变成隐形 agent?"** 要点:四条边界——失败驱动(答对的题永不触发,零误伤面)、硬上限 1 次重试、`auto` 字段留痕、一键关;且重试择优宁保留诚实拒答(部分回答夹带错误缺失声明,忠实度 1.0→0.93)。数据裁决:前置腿方案两版都跑了 88 题,被误伤数据否决。
8. **"日志里记什么、不记什么,怎么定的?"** 要点:绝不落 key 本体(只记身份 name);query 默认截断 120 字、可关;隐私边界单点收口在入队前(防新写入方漏截断);观测失败绝不拖垮服务(有界队列满即丢 + 计数暴露);/v1/stats 要 admin key(聚合查询模式也是信息)。

---

## 7. 动手实验

### Lab 1(CPU):起一个 fake 后端守护进程,亲手观察 D6/D7

整个服务层可注入 fake retriever,无 GPU/网络即可完整跑起来。前置:仓库开发环境(`pip install -e .[dev]`;仓库标准环境为 WSL `conda activate navikb`,Windows 侧装齐依赖亦可)。

```bash
cd <repo>/tests
python -c "
import _fakes, uvicorn
app = _fakes.make_app(retriever=_fakes.FakeRetriever(
    results_factory=lambda: [_fakes.make_res(_fakes.make_hit(), ctx_text='big', anchor=[1,5])]))
uvicorn.run(app, port=8788)
" &
# 1) 同会话打两次:第二次 hits[0].context_status=already_returned 且 text 为空(降级成指针)
curl -s -XPOST localhost:8788/v1/retrieve -H 'Content-Type: application/json' \
     -H 'X-Pharos-Session: A' -d '{"query":"q"}'
# 2) 不带会话头打两次:两次都是 full_section(去重 opt-in)
# 3) 枚举校验不走 422:HTTP 200 + status=bad_arg
curl -s -XPOST localhost:8788/v1/retrieve -H 'Content-Type: application/json' -d '{"query":"q","mode":"weird"}'
# 4) /v1/stats:上面那次 bad_arg 被计入 errors(200 也算失败)
curl -s localhost:8788/v1/stats
```

对照读三个测试加深理解:`test_session_dedup_and_isolation`(tests/test_service.py)、`test_keys_mode_session_isolated_across_users`(tests/test_team.py,伪造相同 session id 互不可见)、`test_observe_records_on_handler_crash`(tests/test_team.py,处理器崩溃仍计数落盘)。全量回归:`pytest tests/test_service.py tests/test_sessions.py tests/test_smart.py tests/test_team.py tests/test_adapter.py -q` 应全绿——这是"toolcore 依赖注入 + app 工厂全可注入"分层的直接红利。

### Lab 2(CPU):fail-closed 三连——把"校验规则可推导"亲眼看一遍

```bash
# 1) 坏 keys 文件:name 含 '|' → SystemExit,报错原文点名"会话隔离前缀分隔符"
python -c "
from pharos import identity as I
import json, tempfile
p = tempfile.mktemp()
open(p, 'w').write(json.dumps({'keys': [{'key': 'x'*20, 'name': 'a|b', 'tenant': 't'}]}))
I.load_keys(p)"
# 变体:两条重名 name 试一次;key 改成 'short' 试一次 —— 每种格式错都是指名道姓的 SystemExit

# 2) 非回环守卫:绑 0.0.0.0 且无 keys → 拒绝启动
cd <repo>/tests && python -c "
from _fakes import make_app, make_cfg
make_app(cfg=make_cfg(host='0.0.0.0'))"

# 3) keys new 往返:生成 pk_alice_<32hex> 且只打印一次;第二次重名拒绝
python -m pharos keys new alice --tenant demo --file /tmp/k.json
python -m pharos keys new alice --tenant demo --file /tmp/k.json
```

### Lab 3(可选,GPU + WSL + Docker):多副本 kill 无感 + 去重降级实测

前置:必须在 WSL 内跑 compose,Docker Desktop 对 Ubuntu 发行版开 WSL Integration(否则 bind mount 全崩,见 [../SCALE_OUT.md](../SCALE_OUT.md) 环境前置)。

```bash
docker compose --env-file .env.compose up -d --build --scale pharos=3
for i in $(seq 1 50); do curl -s -o /dev/null -w '%{http_code}\n' -XPOST localhost:8080/v1/retrieve \
  -H "X-API-Key: <key>" -H 'X-Pharos-Session: S' -d '{"query":"净利润"}'; done &
docker kill $(docker compose ps -q pharos | head -1)
```

预期:50/50 全 200(nginx `proxy_next_upstream` + `non_idempotent` 把在途 POST 转到健康副本);同一会话的 already_returned 比例掉到约 1/N——亲手验证 F-3 声明的"有意接受的降级"。

---

## 8. 诚实边界

面试中主动承认这些,比等着被问出来强得多:

1. **多副本下的进程内状态**:SessionRegistry 去重效果掉到 ~1/N、Stats 只反映单副本且重启归零。话术:"这是声明过的分级降级——去重是便利不是正确性;要精确需要 session 粘滞或 Redis,规模到了再引,当前不为便利引入新故障域。"
2. **200 包裹错误对 LB 失明**(已确认待修,deferred deploy#1):`inference_unavailable/backend_unavailable` 以 200 返回,nginx 被动摘除永不触发,病副本持续吃 1/N 流量。修法草图已定(此类状态改 503、body 结构不变),需 compose 环境实测后落地。
3. **readyz 深度不足**(deferred deploy#2):只查 collection 存在不查非空——迁移中断留下的空 collection 下 readyz 仍绿,查询全 empty 无告警,正是自己定义过的"静默数据损坏"类别。
4. **无请求取消传播**(deferred deploy#0):nginx 130s 超时后客户端拿 504,但 pharos 工作线程仍继续重试满 ~361s 烧线程;重试链无壁钟总 deadline。承认:"超时预算两端没有对齐成显式等式,这是下一轮要修的。"
5. **agentic Δ−0.097 的幅度存疑**(deferred eval#0):eval 的 agentic/decompose 路径绕过生产 Generator,缺表格 content_raw 补回与 section_path 面包屑——对 agentic 系统性不利——Δ 的方向可信,但幅度很可能被高估(这一点已确认),须复跑后修正。引用这个数字时要带上这个保留。
6. **文档滞后于代码**:DESIGN D8 仍写大锁版(代码已锁下沉);DESIGN 称"/healthz 是唯一免鉴权端点"但代码同时豁免 /readyz;API.md 漏记 /readyz 端点与 `inference_unavailable` 状态。话术:"审查发现了这批 doc-code 漂移并已列入回填清单——这也是为什么我坚持'以代码为准 + 文档标注已交付边界'的习惯。"
7. **create_app 的 env 通道**(service#5):同进程多 app 会互相覆盖预算;生产触发不可达所以保守留档,但承认这是拿一条全局通道来实现注入语义,终归是个妥协。

---

*上一篇:[07 评估方法论](07-evaluation.md) · 工程原文:[../DESIGN.md](../DESIGN.md) / [../SCALE_OUT.md](../SCALE_OUT.md) / [../API.md](../API.md)*
