# 组件异议与修复留痕(供一起 review)

> 约定:做 Pharos 时对既有组件(chunk-test-repo)的**异议、修复、刻意不修**都记在这里,
> 每条给动机 + 动作 + 验证。组件仓的改动同时有独立 commit 可查。

## 已修复

### N1:mcp_server 工具语义与 stdio transport 耦合 —— 拆出 toolcore(引擎 commit `7fbf709`)

- **异议**:六个工具的校验/结构化构建/去重/预算/错误映射(经 R1-R5 五轮评审打磨的核心资产)
  全在 server.py 里与 FastMCP 绑定。Pharos 的 HTTP 端点若复制这套逻辑,契约必然漂移
  (同一个 already_returned 语义两处实现,改一处忘一处)。
- **动作**:纯移动拆分 `mcp_server/toolcore.py`(transport 无关、纯 stdlib、retriever/user 依赖注入),
  server.py 保留 stdio 绑定 + 显式 re-export。**逻辑零改动**。
- **验证**:引擎原 test_tools.py **一行未改** 22 项全绿 + embedder test_store 7 项全绿。
- **顺手加固**:server.py 把自身目录插入 sys.path(原先只在"作为脚本跑"时可解析 toolcore,
  被外部 import 时会挂)。

### N2:`_RETURNED_KEYS` 进程级去重在多会话下会串 —— Pharos 侧兑现 per-session(设计预告的坑)

- **异议**:server.py B5.A 注释自己写了"换 HTTP/SSE 多会话 transport 前必须改成 per-(session,user)
  隔离"。这不算发现新 bug,但 Pharos 恰好就是那个"换 transport"的时刻,**必须兑现,否则是真泄漏**:
  A 会话取过的段,B 会话被误标 already_returned(正文清空,B 从没收到过)。
- **动作**:Pharos `sessions.py`(有界 LRU SessionRegistry)+ `X-Pharos-Session` 头 opt-in;
  toolcore 的 `returned_keys` 本来就是参数化的,引擎零改动。
- **验证**:CPU 测试(同会话去重/跨会话隔离/无头不去重)+ GPU 冒烟实测(smoke1 全
  already_returned、smoke2 隔离)。

## 刻意不修(记录在案,可再议)

### N3:`Generator.answer` 不支持检索过滤(doc_ids/doc_type/kind/strategy)

`/v1/ask` 目前只透传 top_k/rerank——想"只在财报里问"得走 agentic 或 /v1/retrieve 自己拼。
**不修的理由**:改 Generator.answer 签名会牵动 eval(run_eval/decompose 都调它),而 v0.1 的
闭管道定位是"整库问答";按 doc 过滤的需求先用 agentic 出口顶。列入 TODO(P2),真要做时
在 generator 仓加可选 kwargs 并补 eval 回归。

### N4:index_real.py 写死路径/ACL —— 不改脚本,产品化到 `pharos index`

引擎的 index_real.py 保留作历史脚本(它建了现在的 ~/rag_real);Pharos 的 indexer.py 把
corpus/dest/collection/ACL 全参数化。**不动原脚本的理由**:它是"当时怎么建库"的可复现记录。

### N5:引擎 stdio server 保留,不下线

守护进程没跑时,引擎仓 .mcp.json 的直连模式仍可用(代价:每会话重付模型加载)。两种模式
不能同时打开同一索引(单客户端锁)——已在两边 README 说明。**不下线的理由**:它是唯一
不依赖守护进程的消费方式,留作降级路径。

### N6:`OpenAICompatibleLLM` 无重试

DeepSeek 偶发 5xx/超时会直接把 /v1/ask 打成 ask_failed(retriable=true,客户端可重试)。
**不修的理由**:重试放客户端语义更清晰(答案不幂等,服务端自动重试会加倍延迟与费用);
若实测故障率高再加有界重试,列 TODO(P3)。

## 对抗评审 P1(2026-07-02)结果与处置

三视角(安全/正确性·并发/契约·文档)× 每条 2 反驳者。39 个 agent;部分验证子任务撞会话额度,
未验完的 13 条由主线逐条对照源码核实(全部属实)。**修复后全部钉了回归测试**(tests/test_review_fixes.py)。

**Confirmed(评审确认)**
- C1 indexer:`restricted`+空 `allow` 建库成功但任何身份检索不到、零告警 → 建库入口显式拒绝;
- C2 no_identity hint 指示设 `RAG_TENANT` 而 Pharos 只读 `PHAROS_TENANT`(照做仍全空,死循环误导)
  → service 绑定层把契约文本翻译成 PHAROS_* 配置名。

**自核实属实并修**(验证 agent 未跑完的):适配器 doc_id 无 URL 编码(#// 截断/改路由)→ quote;
空 doc_id 打到别的路由(307)→ 本地 bad_arg + _call 加 3xx 分支;/v1/ask 并发下共享 LLM 的
last_finish_reason 跨请求串味 → Generator 改 per-thread(threading.local,同线程内无并发窗口);
Generator 工厂只捕 ValueError(openai 缺包等裸抛 500)→ 兜底 ask_failed;.env 行内注释/引号不剥 +
int() 无保护崩启动 → _parse_env_value/_int_env;PHAROS_QDRANT_PATH/SIDECAR_DIR 覆盖不 expanduser
(字面 "./~" 开空库)→ expanduser;CLI 带 --url 跳过 .env 加载(API key 丢失恒 401)→ _client 恒 load_env;
适配器与引擎 docstring 3 处不同文 → 逐字对齐 + 同文回归测试;引擎 server.py list_documents docstring
漏 coverage → 补;引擎 mcp_server/README 未提 toolcore 分层与 Pharos/锁互斥 → 补(即 N5 声称落实);
API.md meta 漏 requested_k/rerank、documents 漏 retriable → 补;.env.example 缺 4 个实读变量 → 补。

**Refuted(证伪不修,留档)**:API key 非恒时比较(LAN 明文 HTTP 下时序测不出,能测者已在信任边界内);
X-Pharos-Session 碰撞 griefing(单身份模型下同边界,有更强手段,无边际能力);/v1/ask citations 缺
untrusted 标记(唯一消费者 CLI 只打印元数据,无下游 LLM 路径;引用原文默认不返回)。

## 观察(无动作)

- FastAPI `on_event` 已废弃 → Pharos 直接用 lifespan(自家代码,不涉组件)。
- toolcore 的交付预算仍走 `RAG_MAX_CONTEXT_TOKENS` 环境变量(引擎契约);Pharos 用
  `PHAROS_MAX_CONTEXT_TOKENS` 在 create_app 时兑现成该 env。参数化成函数入参更干净,
  但会改 toolcore 签名,收益小于折腾,不动。
