# 引擎组件的已知决策与历史(单仓后,这些曾是跨仓评审记录)

> **历史说明**:下面 N1-N8 与 P1 评审最初都是"对独立引擎仓的异议 / 修复 / 刻意不修"的跨仓留痕。
> 引擎已折叠进 pharos 单仓(`src/{chunker,embedder,generator,pharos}`),那条两仓边界不复存在。
> 这些条目保留作**引擎内部演进的历史与决策依据**;涉及"另一个仓 / 契约漂移"的措辞已按单仓现状订正。

> 约定:对引擎组件的**异议、修复、刻意不修**记在这里,每条给动机 + 动作 + 验证。
> 相关改动在本仓 git 历史中有对应 commit 可查。

## 已修复

### N1:mcp_server 工具语义与 stdio transport 耦合 —— 拆出 toolcore

- **异议**:六个工具的校验/结构化构建/去重/预算/错误映射(经 R1-R5 五轮评审打磨的核心资产)
  全在 stdio server 里与 FastMCP 绑定。Pharos 的 HTTP 端点若复制这套逻辑,契约必然漂移
  (同一个 already_returned 语义两处实现,改一处忘一处)。
- **动作**:纯移动拆分出 `toolcore.py`(transport 无关、纯 stdlib、retriever/user 依赖注入),
  stdio 侧(现 `src/pharos/mcp_stdio.py`)保留 stdio 绑定 + 显式 re-export。**逻辑零改动**。
- **验证**:原 test_tools.py **一行未改** 22 项全绿 + embedder test_store 7 项全绿(现均在 `tests/engine/`)。
- **单仓后失效项**:原先 stdio server 把自身目录插入 sys.path(为让被外部 import 时能解析 toolcore)。
  单仓 src-layout(`pip install -e .`,import 名不变)后无需自插路径,该加固已随之移除,N/A。

### N2:`_RETURNED_KEYS` 进程级去重在多会话下会串 —— Pharos 侧兑现 per-session(设计预告的坑)

- **异议**:stdio server B5.A 注释自己写了"换 HTTP/SSE 多会话 transport 前必须改成 per-(session,user)
  隔离"。这不算发现新 bug,但 Pharos 恰好就是那个"换 transport"的时刻,**必须兑现,否则是真泄漏**:
  A 会话取过的段,B 会话被误标 already_returned(正文清空,B 从没收到过)。
- **动作**:Pharos `sessions.py`(有界 LRU SessionRegistry)+ `X-Pharos-Session` 头 opt-in;
  toolcore 的 `returned_keys` 本来就是参数化的,toolcore 本身零改动。
- **验证**:CPU 测试(同会话去重/跨会话隔离/无头不去重)+ GPU 冒烟实测(smoke1 全
  already_returned、smoke2 隔离)。

### N7:数值范围错答(分部被引申成总体)—— 引擎 generator 修复(2026-07-03)

- **异议**:实测唯一一次"自信错答"(分部营收当总营收,带真实引用极难被发现)。诊断出**双根因**:
  ① prompt 无数值范围约束;② 范围证据(小节路径)根本不进 prompt——表格正文无一字分部字样,
  模型无从判断。只修①实测无效,①+②齐了才生效。
- **动作**:SYSTEM 加窄靶数值范围约束(非 R3 被 revert 的全称收紧);context source 行并入
  section_path 面包屑(顺带提升全部问题的溯源质量)。
- **验证**:三关全过——错答 case 转为"标注范围+拒引申";正确 case 无误伤;同 DeepSeek 裁判
  72 题前后对比忠实度 0.972→1.000、正确性持平。详见 pharos TESTING §3。

### N8:表格块检索文本增强 —— 引擎 chunker 修复(2026-07-03,`2bd97a5`)

- **异议**:表格块可检索信号只有 caption 一句,表的语义(列头/行标签)全锁在 content_raw
  不可检索——数字类问题表格被散文挤出 top-k(N3/N7 案例的检索侧根因)。
- **动作**:`_table_signal`(表头+行标签,数据单元格不进)+ 面包屑拼入检索文本;
  **存在性门控**防幽灵块(无门控会 +23 块并平移 chunk id——重建时发现并当场修正,细节见
  TESTING §3)。两索引全量重建。
- **验证**:44 chunker 测试;72 题回归正确性持平、3 题冗余 gold 位移(全部仍答对)、
  动机案例中文问法直接答对。裁决:保留(破坏度视角,解锁错答类 > 冗余召回位移)。
- **连带发现**:gold 集无表格题 = eval 测量盲区(列 TODO:gen_gold 定向补采)。

## 刻意不修(记录在案,可再议)

### N3:`Generator.answer` 不支持检索过滤 —— **已于 2026-07-02 落地**(状态流转:不修→修)

原判"先用 agentic 出口顶,列 TODO"。用户实测暴露真实痛点:"Netflix 2015 营收"类
**数字埋在表里**的题,通用问法下表格块被 MD&A 散文挤出 top-k,top_k/rerank 都救不动,
而 `kind=table` 过滤一击命中(数字在 p.16 Selected Financial Data 表,库内验证)。
**动作**:引擎 Generator.answer 加可选 doc_ids/doc_type/kind/strategy——**按需传**
(不设的参数不出现在调用里),老窄签名 retriever(单测 mock/smoke)与既有调用零影响;
eval 只用 top_k/rerank 关键字(run_eval.py:80 核实),不受牵动,故未重跑 72 题评估。
`/v1/ask` 与 `pharos ask --kind/--doc-type/--doc-id/--strategy` 透传。
**验证**(当时口径,单仓统一 pytest 前的分仓计数):generator 侧 17 测(+1 透传/窄签名兼容)、
产品侧 38 测(+2)。真库实证(诚实版):
英文总营收措辞 + `--kind table --rerank` → **正确答出 $6,779,511 千美元并引用 p.16 汇总表**;
纯中文问法 + kind=table + rerank → 仍未召回 p.16 表但**诚实拒答**(明说只见到分部数据);
⚠ 中文 + kind=table + top_k 15 **无 rerank** 曾把分部营收(4,180,339)错答成公司营收——
跨语言数字题的残留缺口与使用指引见 TODO(P2)与 TESTING §3。**数字题推荐姿势:
`--kind table --rerank` + 用文档语言的关键词(英文财报用 "total revenues")**。

### N4:index_real.py 写死路径/ACL —— 不改脚本,产品化到 `pharos index`

引擎的 index_real.py 保留作历史脚本(它建了现在的 ~/rag_real);Pharos 的 indexer.py 把
corpus/dest/collection/ACL 全参数化。**不动原脚本的理由**:它是"当时怎么建库"的可复现记录。

### N5:stdio-direct 直连模式保留,不下线

守护进程没跑时,`pharos mcp --direct`(stdio 直连、无守护进程)仍可用(代价:每会话重付模型加载)。
它与 `pharos serve` 守护进程不能同时打开同一索引(单客户端锁)——已在文档说明。**不下线的理由**:
它是唯一不依赖守护进程的消费方式,留作降级路径。(单仓后三入口统一走同一 toolcore:
`pharos serve` HTTP 守护 / `pharos mcp` stdio→HTTP 适配器 / `pharos mcp --direct` stdio 直连。)

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
  → service 绑定层把契约文本翻译成 PHAROS_* 配置名。(单仓后命名空间已统一为 `PHAROS_*`,
  `RAG_*` 仅作单版本 DEPRECATED 别名保留,此错源头消除。)

**自核实属实并修**(验证 agent 未跑完的):适配器 doc_id 无 URL 编码(#// 截断/改路由)→ quote;
空 doc_id 打到别的路由(307)→ 本地 bad_arg + _call 加 3xx 分支;/v1/ask 并发下共享 LLM 的
last_finish_reason 跨请求串味 → Generator 改 per-thread(threading.local,同线程内无并发窗口);
Generator 工厂只捕 ValueError(openai 缺包等裸抛 500)→ 兜底 ask_failed;.env 行内注释/引号不剥 +
int() 无保护崩启动 → _parse_env_value/_int_env;PHAROS_QDRANT_PATH/SIDECAR_DIR 覆盖不 expanduser
(字面 "./~" 开空库)→ expanduser;CLI 带 --url 跳过 .env 加载(API key 丢失恒 401)→ _client 恒 load_env;
适配器与 stdio docstring 3 处不同文 → 逐字对齐 + 同文回归测试(单仓后该契约由**结构化测试**兜:
适配器 vs mcp_stdio docstring 相等 + `_INSTRUCTIONS` 单一来源于 toolcore);stdio server list_documents docstring
漏 coverage → 补;mcp-server 文档未提 toolcore 分层与守护进程/锁互斥 → 补(即 N5 声称落实);
API.md meta 漏 requested_k/rerank、documents 漏 retriable → 补;.env.example 缺 4 个实读变量 → 补。

**Refuted(证伪不修,留档)**:API key 非恒时比较(LAN 明文 HTTP 下时序测不出,能测者已在信任边界内);
X-Pharos-Session 碰撞 griefing(单身份模型下同边界,有更强手段,无边际能力);/v1/ask citations 缺
untrusted 标记(唯一消费者 CLI 只打印元数据,无下游 LLM 路径;引用原文默认不返回)。

## 观察(无动作)

- FastAPI `on_event` 已废弃 → Pharos 直接用 lifespan(自家代码,不涉组件)。
- toolcore 的交付预算历史上走 `RAG_MAX_CONTEXT_TOKENS` 环境变量;单仓统一命名空间后读
  `PHAROS_MAX_CONTEXT_TOKENS`(`RAG_*` 作 DEPRECATED 别名一版),create_app 时兑现成该 env。
  参数化成函数入参更干净,但会改 toolcore 签名,收益小于折腾,不动。
