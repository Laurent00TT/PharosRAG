# Pharos 设计文档

> 状态:已实现、实测、经对抗评审(证据见 [TESTING.md](TESTING.md))。面向小团队内部知识库部署。

## 1. 目标与非目标

**目标**:把本仓内的四个组件(chunker / embedder / generator / pharos)组装成一个**面向小团队内部知识库**的完整 RAG 服务,
提供两种消费方式且共用同一检索栈、同一套契约:

1. **闭管道问答**(HTTP `/v1/ask`):一问一答,检索→grounding prompt→DeepSeek→带引用答案。
   系统评估(异厂 Claude 裁判)已定论这是**默认最优消费方式**:忠实度 ≈1.0、正确性 0.847、最便宜。
2. **agentic RAG**(MCP):agent 自己驱动检索工具(何时搜/怎么改写/要不要多跳)。
   评估显示 agentic 在简单题上净负(Δ−0.097),但对交互式深挖/跨文档浏览是闭管道给不了的形态——两者是**互补**,不是竞争。

服务面覆盖:多身份鉴权(keys 模式,§D10)、请求日志与指标(§D11)、systemd 托管、备份恢复。

**非目标**(明确不做,理由见 [ROADMAP.md](ROADMAP.md)):HTTPS/公网终结(内网信任边界 + key,要远程走隧道)、
SSO/OIDC、解析编排(MinerU 调用在本仓 `scripts/`)、前端 UI。
（原非目标"水平扩展/多副本"已在阶段 A–F 交付:拆 GPU 推理层 + 应用脱 torch + Qdrant server + nginx 多副本,
见 [SCALE_OUT.md](SCALE_OUT.md)。⚠ 下方 D1 "Qdrant server 换 url 即可"是**仅配置面**的简化——实际还需 store 三分支 + 全出口透传 + 数据迁移 + server-mode ACL 越权重测。）

## 2. 核心架构决策

### D1:守护进程独占资源,一切消费走 HTTP(本项目最重要的决策)

**问题**:stdio MCP 直连模式(`pharos mcp --direct`,`src/pharos/mcp_stdio.py`)有两个实测痛点:
- 嵌入式 Qdrant **单客户端独占锁**——第二个进程打开同一索引直接报错(eval 为此专门 copytree 避锁);
- dense 模型(Qwen3-VL 8B)**加载 1-2 分钟**——stdio 每会话一进程,每开一个 Claude Code 会话重付一次。

**决策**:`pharos serve`(FastAPI)是系统里**唯一**碰 Qdrant 与 GPU 的进程;
闭管道问答与 MCP 都从 HTTP 走。冒烟实测:适配器毫秒级连上,首查 19s(模型热缓存),后续查询秒级。

**否决的备选**:
- *每消费方各开索引*:被单客户端锁直接堵死;
- *MCP 进程内直连检索栈(`pharos mcp --direct`)*:本仓保留作 fallback(不跑守护进程时,stdio 直连、无 daemon),但每会话重付加载,不作为产品默认;
- *Qdrant server 模式(docker)*:解锁多客户端,但引入常驻服务运维 + 数据迁移,当前规模(单实例/小团队)收益不抵复杂度。规模上去后这是 v2 的自然升级路径(EmbedConfig 换 url 即可)。

### D2:MCP = 零 GPU 依赖的薄适配器

`pharos mcp` 只 import mcp + httpx + toolcore(纯 stdlib),六个工具原样转发 HTTP。
守护进程未启动时返回结构化 `backend_unavailable`(hint 指向 `pharos serve`),不裸抛。
工具名/入参/返回契约与本仓 stdio 直连(`pharos mcp --direct`,`src/pharos/mcp_stdio.py`)**完全一致**——agent 侧无感切换。

### D3:工具语义单一来源(toolcore)

HTTP 端点与 MCP 适配器的校验/结构化结果/去重/预算/错误映射/_INSTRUCTIONS 全部来自
`src/pharos/toolcore.py`(留痕见 [COMPONENT_NOTES.md](COMPONENT_NOTES.md))。**动机**:这层契约经 R1-R5 五轮对抗评审打磨
(already_returned/omitted_budget/预算含资产/无权不泄存在性…),复制一份必然漂移。
折入本仓后 toolcore 是普通包内模块(`import`,不再按文件路径 importlib 加载)。

### D4:单仓自包含(path-dep 已作废)

四个组件(chunker / embedder / generator / pharos)现已折入本仓 `src/`,src-layout 可编辑安装
(`pip install -e '.[dev]'`),导入名不变。**历史**:早期 pharos 是薄产品壳、经 path-dep(sys.path 插引擎
src)消费一个独立引擎仓——该跨仓接缝(含 `PHAROS_ENGINE` 定位、`bootstrap()`/版本守卫)**已随合仓拆除**。
`engine.py` 现只是 `LockedRetriever` + `build_*`,走本仓普通包导入,不再有 sys.path 注入与跨仓漂移面。

### D5:部署即授权 —— 身份是安全边界的入口

**核心不变量**:能连上端口的人 = 该身份的全部可见内容,所以身份必须由服务端权威决定、
客户端不可经参数篡改;tenant 未设 → toolcore 层 fail-closed(`no_identity`,空结果)。
身份的三种绑定方式(keys / legacy / open)与多身份产品化见 **§D10**;此处只立不变量:
默认绑 127.0.0.1、非回环强制鉴权、`/healthz` 是唯一免鉴权端点。
身份只回答"谁在问",可见性("能看什么")由 embedder 层 ACL 硬过滤兑现(与 stdio 直连路径同一 fail-closed 模型)。

### D6:去重 opt-in,per-session 隔离

stdio 直连路径的 `_RETURNED_KEYS` 是进程级的(stdio 下进程=会话,成立;其注释明确预告
"换 HTTP 多会话前必须 per-session 隔离")。Pharos 兑现它:`X-Pharos-Session` 头 → SessionRegistry
(有界 LRU 64 会话)取该会话的 returned_keys;**不带头 = 不去重**(curl 一次性调用不该有
跨调用状态)。MCP 适配器每进程一个 uuid,自动获得会话语义。冒烟实测:同会话第二次全
already_returned,新会话完全隔离。

### D7:领域结果一律 HTTP 200 + status 字段

`no_identity`/`empty_query`/`bad_arg`/`no_access`/`backend_unavailable`… 都是**领域结果**
(agent/客户端要程序化决策的状态机),统一 200 + JSON status,与 toolcore 契约一致;
HTTP 状态码只留给传输层语义(401 auth、422 请求体不是合法 JSON、5xx 崩溃)。
mode/strategy 等枚举校验故意**不在 pydantic 层做**(否则变 422),留给 toolcore 出结构化 bad_arg。

### D8:/v1/ask 的锁模型 —— 检索在锁内,LLM 在锁外

嵌入式 Qdrant 与 GPU 前向非线程安全,FastAPI 线程池会并发跑 sync 端点 → 所有 retriever 调用
经 `LockedRetriever` 串行化。Generator 依赖注入的正是这个加锁代理,所以 `/v1/ask` 的检索段
持锁、随后数秒-数十秒的 DeepSeek 网络调用**不持锁**——不阻塞其他会话的检索。

### D9:smart-ask —— 闭管道的有界智能(2026-07-03,用户体验驱动)

**问题**:用户用默认参数问"Netflix 2011-2015 每年净利润",只得到三年——正确答案在五年表里,
但排序把它排在窗口外。旋钮(kind/rerank/措辞)都存在,但**用户不该需要懂旋钮**。

**设计红线**:闭管道不变成隐形 agent(agent 编排实测净负 −0.097);一切自动行为响应留痕
(`auto` 字段);`PHAROS_SMART_ASK=off` 一键纯净模式;默认行为变化必须过 88 题考卷。

**第 1 层 hints**:答案命中拒答/部分拒答模式时,按本次请求参数生成 ≤3 条可操作建议
(kind=table / 文档语言关键词 / rerank / 用 retrieve 自查排名),不重复建议已自动做过的动作。
正常答案零打扰。

**第 2 层 失败驱动表格补检**:第一轮**纯净**(与无 smart 完全同路径);数值型问题
(`generator.signals.looks_numeric`,零 LLM)**且第一轮拒答/部分拒答**时,带一条
`kind=table, top_k=5, rerank(top_n=50)` 的检索腿重问一轮(硬上限 1 次;命中与主检索**并集**
——decompose 教训:替换会窄化)。用户显式给 kind 时尊重用户不叠加。
**重试择优采用**:重试答案只有**不再是拒答**(完整答出)才替换;部分回答会夹带"未提供 X"的
错误缺失声明(X 实际在 context 里,88 题实测忠实度 1.0→0.93)——那种情况保留第一轮的
诚实拒答 + hints。忠实度是本系统头牌,排序在"多答一点"之前。

**为什么是失败驱动而不是前置腿(88 题实测裁决,两版都跑了)**:前置腿把表格题 0.625→0.875,
但**误伤 5 道原本答对的散文题**(腿带进的相近数值把模型带偏/吓保守),散文 0.861→0.792;
失败驱动下"答对的题永不触发",零误伤面:表格 0.625→0.75+、散文持平、忠实度全卷 1.000,
总体更优。教训:**默认行为的智能必须只作用于失败路径**——成功路径上的任何"帮忙"都是风险。

**腿参数教训**:rerank_top_n 必须 ≥ "对的块在粗排的最差名次"(实测 30 装不进粗排 31-50 名的
五年表,重试形同虚设;50 一发命中)。精排纠正的是排序,前提是候选池里得有它。

**单一来源**:数值判定/拒答判定/腿参数在 `src/generator/signals.py`,pharos 与
`run_eval --smart-tables` 共用——考卷跑的就是生产行为。

**否决的备选**:前置补检腿(上述实测);默认全局 rerank(每问 +3~5s,散文题收益甚微);
默认加大 top_k(实测只加噪声);全量查询翻译(每问 +1 次 LLM 调用);闭管道内多轮循环(MCP 出口的职责)。

## 2b. 服务面:多身份 / 可观测 / 分层(D10-D12)

**分层原则(D12):身份在服务层,ACL 在检索层。** 合仓后这是**本仓内部的模块分层**(不再是跨仓不变量):
身份回答"谁在问"(pharos 服务层的 identity 模块关注点),可见性回答"能看什么"(embedder 层关注点,
多租户 ACL 机制经 acl_regression 五用户矩阵验证)。两者正交:服务层做多身份鉴权时,可见性硬过滤的兑现
**收敛在 embedder ACL 一处**——这是分层职责的检验,身份不越界替 ACL 决策、ACL 不感知具体身份来源。

### D10:多身份模型 —— API key → 身份,三种模式,fail-closed

- **keys 模式**(团队部署,默认):`PHAROS_KEYS_FILE` 指向 JSON(`{"keys":[{"key","name","tenant",
  "principals",["admin"]}]}`,gitignored、建议 chmod 600)。每个 /v1/* 请求经 X-API-Key
  解析成身份(name + User),**未知/缺失 key 一律 401**;检索时把该身份的 User 传给
  检索栈(embedder ACL 硬过滤兑现"能看什么")。
- **legacy / open 模式**(单人 / 本地开发):只设 PHAROS_API_KEY = 单密钥门槛;都不设 = 仅回环无鉴权。
  ⚠ 两者都是**单身份**:所有客户端共享启动绑定的那一个身份,**不是多租户**——多用户务必用 keys 模式。
- **name 约束**(fail-closed 校验):身份 name 必须**唯一**且**不含 `|`**(它是会话去重登记键
  `身份名|会话id` 的前缀,重名/含分隔符会让命名空间碰撞);违反则拒绝启动。
- **fail-closed 守卫**:`PHAROS_HOST` 绑定非回环地址而未配 keys 模式 → **拒绝启动**
  (部署即授权,不允许把整库裸奔到局域网);keys 文件格式错 → 拒绝启动,不静默降级。
- **会话隔离**:returned_keys 登记键 = `身份名|会话id`——不同用户即使伪造相同
  X-Pharos-Session 也互不可见(单身份下会话碰撞无害,多用户下该前提不再成立)。
- **轮换** = 编辑 keys 文件 + `systemctl restart pharos`(秒级);不做热加载(简单>优雅,
  团队规模下重启无感)。密钥生成器:`pharos keys new <name> --tenant --principals`。
- **否决的备选**:SSO/OIDC(当前规模不值得引入 IdP 依赖);数据库存密钥(文件+重启足够);
  热加载(增加状态一致性面,收益小)。

### D11:可观测性 —— 请求日志 + 内存指标,够用为度

- **请求日志**:JSONL 追加(`PHAROS_LOG_DIR`,默认 ~/pharos_logs),每行 {ts, ep, user, http, ms,
  status, n/n_citations/auto/refusal, query(截断,`PHAROS_LOG_QUERIES=off` 可关)}。`user` = keys 模式下
  的身份名,legacy/open 单身份模式恒为占位 `default`/`local`(**绝不落 key 本体**)。
  **绝不落盘 key 本体**;query 记录默认开(内网调试价值 > 隐私风险,可关并已文档化)。
- **指标**:进程内存(计数器 + 每端点延迟环形队列)→ `/v1/stats`(keys 模式下 admin key
  才可读——聚合查询模式也是信息);重启归零(当前可接受,持久化指标见 TODO,需要时再做)。
- **否决的备选**:Prometheus/Grafana(团队规模不值得运维一套监控栈;JSONL 可被任何工具后加工)。

## 3. 命名

**Pharos(法罗斯)**:亚历山大灯塔,古代世界七大奇迹,守在亚历山大图书馆旁。
"图书馆 + 导航"正是本系统:私人藏书(77 篇多格式文档起步)+ 检索导航。CLI 顺口:
`pharos serve / ask / mcp / index`。

## 4. 风险与已知限制

| 风险 | 现状 | 缓解 |
|---|---|---|
| 守护进程单点(无多副本) | 当前规模可接受 | systemd 自愈 + 适配器结构化降级 + hint 指向恢复动作 |
| 适配器与 stdio 直连契约漂移 | 合仓后为单仓结构化契约测试(适配器 vs `mcp_stdio` docstring 相等 + `_INSTRUCTIONS` 单一来源自 toolcore) | 一条 pytest 套件(179 passed)全绿才算过;COMPONENT_NOTES 记录接缝 |
| index 与 serve 抢锁 | 单客户端锁 | indexer 捕获锁错误给明确提示;文档要求先停 serve |
| 端口暴露即数据暴露 | 默认 127.0.0.1 | PHAROS_API_KEY;公网/HTTPS 明确非目标 |
| LLM 上游故障 | /v1/ask 返回 ask_failed(retriable) | 细节只进服务端日志,不外泄 |
