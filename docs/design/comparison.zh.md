<div align="right">

**中文** | [English](comparison.md)

</div>

# 对比

> 如果你在搜 "yet another RAG library",这页是 NaviKB 是不是你想要的东西
> 的简短答案。

## TL;DR

| 用例 | 最适配 |
|---|---|
| "我想要一个 Python 库去搭 RAG pipeline" | **LlamaIndex** 或 **LangChain** |
| "我想 5 分钟内通过 UI 跟我的文档聊天" | **Verba**、**Open WebUI + RAG plugin**,或托管(NotebookLM、Perplexity Spaces) |
| "我想要 LLM 抽取的实体 + 关系索引成图" | **GraphRAG** (Microsoft) 或 **LightRAG** |
| "我在研究 navigable RAG 算法,要写 paper" | **NaviRAG**、**CORPUS2SKILL**、**Don't Retrieve, Navigate** |
| "我想运营一个小团队 KB,带 auth、audit、软删除、备份、MCP,全本地" | **NaviKB** ← 你在这 |

定位决策: NaviKB 是**你运营的系统**,不是**你组合的库**。如果你想要搭积木
的原料,下面列的库替代品会给你更多灵活性和更快的起点。如果你想要一个能
**跑**的东西、带小团队能信任的运维默认值,NaviKB 瞄准那个缺口。

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
| 成熟度 | 成熟,大社区,很多 integration | Pre-1.0(Stage 0 设计预览) |

**选 LlamaIndex 当** 你要最大灵活性、你在搭一个不合任何系统模子的自定义
RAG pipeline、你需要跟很多具体 vector DB / LLM provider 的 integration、
或者你在 prototyping 想换组件。

**选 NaviKB 当** 你想要一个能立起来、配置好、信任它在运维压力下表现
(delete、restore、backup、多进程 maintenance)的系统,不需要自己写那一层。

### vs LangChain

[LangChain](https://www.langchain.com/) 是另一个主导框架。同样的"库 vs 系统"
区别适用于 NaviKB。

LangChain 具体优化的是 **chain 组合** —— 在 DAG 里排 tool、retriever、LLM
的顺序。NaviKB 选反过来的立场: 访问模式(navigation-first + 4 通道检索 +
RRF 融合 + cross-encoder rerank + evidence/hint 分区)**焊死**。operator
拿到这个模式;他们不能重建它。

这在你想要 opinionated 模式时是好的。在你的问题不合这个模式时是坏的(例如
你在搭需要自定义 plan-execute-observe loop 的多模态 agent)。LangChain 更
灵活。NaviKB 更 opinionated。

### vs Haystack

[Haystack](https://haystack.deepset.ai/) 介于 LlamaIndex 和完整系统项目之间。
它有 pipeline 抽象和合理的默认值,但希望你自己部署运维。

跟 NaviKB 最大的区别是**默认访问模式**: Haystack 围绕检索 pipeline 中心化
(Retriever → Reader → Generator)。NaviKB 围绕导航图;检索是几个组件
之一,不是脊柱。

对已有 Haystack 和能 work 的 retriever 的项目,正确的做法可能是继续用它。
NaviKB 不是 drop-in 替代品。

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

GraphRAG 和 NaviKB 其实**不在竞争** —— 它们在回答不同问题。如果你的问题是
"我的语料对话题 X 跨文档合成怎么说",GraphRAG 是更好的形状。如果你的问题
是 "这份手册的 Y 流程在哪",NaviKB 是。

### vs NaviRAG 和 "Don't Retrieve, Navigate"

这些是 2026 研究论文,探索 NaviKB 选的同一方向(导航作为主访问模式):

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

NaviKB **不是** NaviRAG 的实现。它在那篇论文之前就存在,用不同机制
(parser 标题,不是 LLM distill)。命名重叠是有意的 —— NaviKB 在 "navigable
KB" 研究方向给自己定位 —— 但项目在不同层解决不同问题。

如果你是研究者比较检索算法,读论文,忽略 NaviKB。如果你是 operator 想在这
季度对真实文档跑一个 navigable KB,NaviKB 是那一类设计底下的生产栈答案。

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

终端用户想要浏览器里的聊天界面,选 Verba。目标消费者是 Agent(Claude Code、
Continue、Cline 等)且你在意"谁能做什么"超出单用户,选 NaviKB。

### vs 托管服务(NotebookLM、Perplexity Spaces 等)

托管服务是不同类别。它们更容易上手,更难控制。它们通常:

- 不暴露 Agent 工具表面(封闭 UI)
- 不让你在敏感语料上自托管
- 不给你想要的 regulated 文档审计
- 在聊天体验上很好 —— NaviKB 明确不提供这个

如果你的语料是你可以发给 Google 或 Perplexity 的东西,托管服务是阻力最小
路径。如果它需要留在你的机器或 LAN 上,NaviKB 坐在它们去不到的地方。

## 诚实的弱点

这一节列上面对比里如果有人在评估我们、会向 NaviKB 倾斜不利的地方。

### NaviKB 比替代品弱的地方

- **代码还没公开。** LlamaIndex、LangChain、Haystack、Verba、GraphRAG 都有
  能装的包。NaviKB 有文档。差距是真的,只有我们达到 Stage 2 才会关闭
  (见 [docs/status.zh.md](../status.zh.md))。
- **生态更小。** 没有 50 种文档格式的 connector、没有 100 个向量库的
  integration、没有给每个 LLM provider 的示例 notebook。NaviKB 挑了一小套
  (parse 用 MinerU、文本 embed 用 BGE-M3、视觉用 ColQwen2、向量用 Qdrant)
  把它们支撑好;加新组合要写代码,不是改 config。
- **没 UI。** 这是刻意的 —— UI 就是 Agent —— 但如果你想给非 Agent 用户
  一个"跟文档聊天"体验,NaviKB 是错的工具。
- **没托管服务选项。** NaviKB 是本地优先,打算一直这样。要 SaaS 经济模式
  的 operator 应该看 LlamaCloud、Weaviate Cloud 等。
- **真实文档的 evaluation 还没公开。** 800+ 测试数衡量合成 / 单元 scope
  输入上的正确性。真实文档 Recall/NDCG/MRR benchmark 是公开发布阻塞清单的
  一部分。它落地之前,任何在检索质量上做对比的人都是在对比设计声明,不是
  测量数。

### 替代品比 NaviKB 弱的地方

- **运维深度。** 多数 RAG 框架把 auth、audit、软删除、maintenance flag、
  备份、GC 当作"自己搭"。NaviKB 把它们设计在内。
- **引用纪律。** 多数栈把 retriever 返回的所有东西给 LLM,信任它负责任地
  引用。NaviKB 在 API 表面分区 evidence vs hint。
- **跨进程正确性。** 多数项目不把 ingestion worker 跟 serving 进程分开;
  分开的(例如 Celery 配上一个 web app)依赖 message broker。NaviKB 用
  SQLite 作跨进程协调器,付小部署的代价(一台机器)跳过 broker 复杂度。
- **Navigation-first 作为设计承诺。** 很多框架有层次检索作为选项。NaviKB
  把它当作默认,并把 pointer-only 契约在 API 里显式化。

## 怎么保持这个文档诚实

这份文档会把具体的事写错。对比文档总是这样 —— 项目在变、我们对它们的理解
在变、我们作为 NaviKB 作者有自己的偏见。

修法机制: 如果你维护或使用上面任何项目并认为本页对它描述错了,
[开 issue](../../../issues) 带 `comparison-correction` 标签。我们会读每一个。
我们可能不同意每一个,但我们会在技术 merit 上对话。

## 接下来读什么

- [`architecture.zh.md`](architecture.zh.md) —— NaviKB 详细长什么样
- [`navigation-first.zh.md`](navigation-first.zh.md) —— 驱动上面多数差异化
  的设计承诺
- [`security-model.zh.md`](security-model.zh.md) —— 对比中最强烈倾向 NaviKB
  的运维层
- [`../status.zh.md`](../status.zh.md) —— 诚实的当前状态,包括公开发布
  阻塞项
