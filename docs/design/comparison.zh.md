<div align="right">

**中文** | [English](comparison.md)

</div>

# 对比

> 如果你在搜 "yet another RAG library",这页给出 NaviKB 是否适合你的简短答案。

## TL;DR

| 用例 | 最适配 |
|---|---|
| "我想要一个 Python 库去搭 RAG pipeline" | **LlamaIndex** 或 **LangChain** |
| "我想 5 分钟内通过 UI 跟我的文档聊天" | **Verba**、**Open WebUI + RAG plugin**,或托管(NotebookLM、Perplexity Spaces) |
| "我想要 LLM 抽取的实体 + 关系索引成图" | **GraphRAG** (Microsoft) 或 **LightRAG** |
| "我在研究 navigable RAG 算法,要写 paper" | **NaviRAG**、**CORPUS2SKILL**、**Don't Retrieve, Navigate** |
| "我想运营一个小团队 KB,带 auth、audit、软删除、备份、MCP,全本地" | **NaviKB** ← 你在这 |

定位决策: NaviKB 是**你运营的系统**,不是**你组合的库**。如果你需要的是用于自由组合的基础库,下面列出的库替代品能给你更多灵活性和更快的起步。如果你想要一个能直接**跑起来**、让小团队可以放心依赖其运维默认值的系统,NaviKB 正是为这个缺口而生。

## 详细对比

### vs LlamaIndex

[LlamaIndex](https://www.llamaindex.ai/) 是最成熟的开源 RAG / agent 框架。
它是一个**库** —— 你在 Python 代码里组合 node、retriever、query engine、
agent。

| 维度 | LlamaIndex | NaviKB |
|---|---|---|
| 形状 | Python 库 | 端到端系统(server + worker + 脚本) |
| 层次检索 | `SummaryIndex`、`DocumentSummaryIndex`、`TreeIndex` 可用但可选 | NavIndex 是主访问路径 |
| Pointer-only 导航 | 不是一等;retriever 默认返回文本 | 一等: `kb_search_nav` 返回指针,不带文本 |
| MCP-native | 通过第三方包有 adapter | 内置 `/mcp` sub-app,一等 tool surface |
| 引用纪律 | 调用者负责 | API 表面强制(evidence_fields vs hint_fields) |
| Auth / audit / backup / GC | "自己搭" | 端到端设计在内 |
| 成熟度 | 成熟,大社区,很多 integration | Pre-1.0(研究 / 示例, 早期 alpha) |

**选 LlamaIndex** 如果你需要最大灵活性、正在搭建不适合任何固定系统模板的自定义 RAG pipeline、需要与大量 vector DB / LLM provider 对接、或者处于 prototyping 阶段需要频繁替换组件。

**选 NaviKB** 如果你想要一个能直接搭起来、配置好就能用、在运维压力下(delete、restore、backup、多进程 maintenance)也能放心信赖的系统,而不想自己从头搭那一层。

### vs LangChain

[LangChain](https://www.langchain.com/) 是另一个主导框架。同样的"库 vs 系统"
区别适用于 NaviKB。

LangChain 具体优化的是 **chain 组合** —— 在 DAG 里编排 tool、retriever、LLM 的调用顺序。NaviKB 选择了相反的立场:访问模式(navigation-first + 4 通道检索 + RRF 融合 + cross-encoder rerank + evidence/hint 分区)**固定内置**。operator 直接使用这套模式,无法也无需自行重组。

这在你需要有主见的固定模式时是优点。当你的问题不适合这套模式时,就是局限(例如你在搭需要自定义 plan-execute-observe loop 的多模态 agent)。LangChain 更灵活,NaviKB 更有主见。

### vs Haystack

[Haystack](https://haystack.deepset.ai/) 介于 LlamaIndex 和完整系统项目之间。
它有 pipeline 抽象和合理的默认值,但部署运维仍需自行负责。

跟 NaviKB 最大的区别是**默认访问模式**: Haystack 围绕检索 pipeline 中心化
(Retriever → Reader → Generator)。NaviKB 围绕导航图;检索是几个组件
之一,不是脊柱。

对于已经在用 Haystack 且 retriever 运转良好的项目,继续沿用很可能才是正确选择。NaviKB 不是 drop-in 替代品。

### vs GraphRAG

[Microsoft GraphRAG](https://github.com/microsoft/graphrag) 走 "知识图谱
来自 LLM 抽取" 路线: 把每份文档喂给 LLM,抽实体和关系,构图索引,查询时
遍历图。

这是一个强大且跟 NaviKB 非常不同的设计哲学:

| 维度 | GraphRAG | NaviKB |
|---|---|---|
| 图来源 | LLM 抽取的实体 + 关系 | 文档自身的结构索引(标题、表格、图) |
| 对图的信任 | operator 必须信任 LLM 的抽取质量 | 图是 parser 确定的;alias 是明确的人工策展 |
| Ingest 成本 | 高(每 chunk 一次或多次 LLM 调用) | 低(ingest 的结构 pass 里没 LLM 调用;description 通道可选) |
| 最适合 | 跨文档围绕发现的概念合成 | 帮读者导航他们手头真有的文档 |
| 部署 | 为企业栈设计(Neo4j 级基础设施) | 为单机 + SQLite 设计 |

GraphRAG 和 NaviKB 其实**并不竞争** —— 它们回答的是不同问题。如果你的问题是"我的语料在话题 X 上跨文档综合起来说了什么",GraphRAG 更合适。如果你的问题是"这份手册里 Y 流程在哪里",NaviKB 才是合适的选择。

### vs NaviRAG 和 "Don't Retrieve, Navigate"

这些是 2026 年的研究论文,探索的方向与 NaviKB 相同(以导航作为主访问模式):

- [NaviRAG (arXiv:2604.12766)](https://arxiv.org/abs/2604.12766):
  "locate first, then forage"。LLM agent 导航一个 distilled 层次表示,
  迭代识别信息缺口。
- ["Don't Retrieve, Navigate" (arXiv:2604.14572)](https://arxiv.org/abs/2604.14572):
  把企业知识 distill 成 "navigable agent skills"。

| 维度 | 研究论文 | NaviKB |
|---|---|---|
| 目标 | 算法贡献 | 生产系统 |
| 层次来源 | LLM 从语料 distill | parser 从文档结构抽取 |
| 实现状态 | 参考代码(NaviRAG: 5 stars, "active organization 中";DRN: 只论文) | 800+ 测试,pre-release |
| 运维表面 | 无(研究 demo) | Auth、audit、backup、GC、MCP server |
| 引用纪律 | 没显式处理 | 强制 |

NaviKB **不是** NaviRAG 的实现。它早于那篇论文存在,采用不同机制(parser 解析标题,而非 LLM distill)。命名上的重叠是有意为之 —— NaviKB 将自己定位在 "navigable KB" 这一研究方向上 —— 但两者在不同层面解决不同问题。

如果你是研究者在比较检索算法,读论文就够了,NaviKB 与此无关。如果你是 operator,想在这个季度针对真实文档跑起一个 navigable KB,NaviKB 是这类设计理念在生产栈层面的答案。

### vs Verba

[Verba](https://github.com/weaviate/verba) 是 Weaviate 的开源 "开箱即用 RAG"
项目 —— Weaviate 之上的 web UI,带 ingest、聊天、基础配置。

| 维度 | Verba | NaviKB |
|---|---|---|
| 接口 | Web 聊天 UI(主) | Agent 用 MCP + 脚本用 REST(没自带 UI) |
| 向量库 | Weaviate | Qdrant(默认本地文件模式) |
| 目标用户 | 想跟文档聊天的终端用户 | 想把 KB 暴露给多个 Agent 的 operator |
| 引用纪律 | 标准 RAG(chunk + cite) | Evidence vs hint 分区 |
| Auth | 单用户(默认部署) | 第一天就多用户(manage_users.py) |

终端用户想要浏览器里的聊天界面,选 Verba。如果目标消费者是 Agent(Claude Code、Continue、Cline 等),且你对"谁能做什么"的管控需求超出单用户范围,选 NaviKB。

### vs 托管服务(NotebookLM、Perplexity Spaces 等)

托管服务是不同类别。它们更容易上手,更难控制。它们通常:

- 不暴露 Agent 工具表面(封闭 UI)
- 不让你在敏感语料上自托管
- 不给你想要的 regulated 文档审计
- 在聊天体验上很好 —— NaviKB 明确不提供这个

如果你的语料可以放心发给 Google 或 Perplexity 处理,托管服务是阻力最小的路径。如果数据必须留在本机或局域网内,NaviKB 正好填补托管服务覆盖不到的位置。

## 诚实的弱点

这一节列出上面对比中,若有人在评估 NaviKB、会对其不利的地方。

### NaviKB 比替代品弱的地方

- **更年轻、经受的实战更少。** LlamaIndex、LangChain、Haystack、Verba、GraphRAG 拥有庞大用户群和多年打磨。NaviKB 是早期 alpha 参考实现 —— 代码已公开,但全新机器上的安装体验与广泛真实使用均尚未得到验证(见 [docs/status.zh.md](../status.zh.md))。
- **生态更小。** 没有支持 50 种文档格式的 connector、没有对接 100 个向量库的 integration、也没有针对每个 LLM provider 的示例 notebook。NaviKB 挑选了一小套组件(parse 用 MinerU、文本+视觉 embed 用 Qwen3-VL、稀疏用 MILCO、向量用 Qdrant)并将其良好支撑;添加新的组合需要写代码,而不是改 config。
- **没有 UI。** 这是刻意的 —— UI 就是 Agent —— 但如果你想给非 Agent 用户提供"跟文档聊天"的体验,NaviKB 是错误的工具。
- **没有托管服务选项。** NaviKB 是本地优先的项目,并打算一直如此。需要 SaaS 经济模式的 operator 应该考虑 LlamaCloud、Weaviate Cloud 等。
- **真实文档的 evaluation 尚未公开。** 800+ 条测试衡量的是合成/单元范围输入上的正确性。真实文档的 Recall/NDCG/MRR benchmark 是公开发布阻塞清单的一部分。在它落地之前,任何在检索质量上做横向对比的人比较的都是设计声明,而非实测数据。

### 替代品比 NaviKB 弱的地方

- **运维深度。** 多数 RAG 框架将 auth、audit、软删除、maintenance flag、备份、GC 留给用户"自行搭建"。NaviKB 将这些全部纳入设计。
- **引用纪律。** 多数栈把 retriever 返回的全部内容直接交给 LLM,靠它自觉引用。NaviKB 在 API 表面明确分区 evidence vs hint。
- **跨进程正确性。** 多数项目不将 ingestion worker 与 serving 进程分离;分离的方案(例如 Celery 搭配 web app)依赖 message broker。NaviKB 用 SQLite 作为跨进程协调器,以局限于单机的代价换掉了 broker 的复杂度。
- **Navigation-first 作为设计承诺。** 许多框架将层次检索作为可选项提供。NaviKB 将它设为默认,并在 API 中显式化 pointer-only 契约。

## 怎么保持这个文档诚实

这份文档难免会在具体细节上有所偏差。对比文档都有这个问题 —— 项目在变、我们对它们的理解在变、我们作为 NaviKB 作者也有自己的立场。

纠错机制: 如果你维护或使用上面任何项目,且认为本页对其描述有误,欢迎[开 issue](../../../issues) 并附上 `comparison-correction` 标签。我们会逐一阅读。未必每条都同意,但会就技术 merit 认真讨论。

## 接下来读什么

- [`architecture.zh.md`](architecture.zh.md) —— NaviKB 详细长什么样
- [`navigation-first.zh.md`](navigation-first.zh.md) —— 驱动上面多数差异化
  的设计承诺
- [`security-model.zh.md`](security-model.zh.md) —— 对比中最强烈倾向 NaviKB
  的运维层
- [`../status.zh.md`](../status.zh.md) —— 诚实的当前状态,包括公开发布
  阻塞项
