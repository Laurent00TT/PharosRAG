<div align="right">

**中文** | [English](navigation-first.md)

</div>

# Navigation-First

> NaviKB 最重要的设计决策。架构里其它所有东西都是这一条的下游。

## 一段话立场

文档不是一串扁平的 chunk,它是一个**可导航的结构**。读者(无论人类还是
Agent)首先关心的是**到达正确的位置**,其次才是**从那个位置取得正确的文本**。
多数 RAG 栈把这个顺序颠倒了 —— 它们先检索文本,结构还原即便有也是事后拼补的。
NaviKB 从结构出发:把结构索引成指针、作为一等 tool surface 暴露给 Agent,
把检索定位为**导航收窄候选空间之后的轻量补充**。

本文说明我们为何做出这个选择、数据结构是什么形态、设计在哪里发挥优势、哪里存在
已知的局限。

## chunk-first 检索的问题

设想一份 200 页的运维手册。问"Sev-1 事件的升级流程是什么?"。在典型的
RAG pipeline 里:

1. query 用 sentence-transformer 嵌入
2. 在 ~2000 个 chunk(每个 ~300 tokens)的向量库上做 cosine 相似度,取 top-5
3. top-5 chunks 拼起来,加个"用这些作上下文"的指令前缀,发给 LLM

这套方法"够用"到足以成为行业默认。但它隐藏的失败模式是真实的问题:

- **丧失局部性。** Chunk N 和 Chunk N+1 在手册里可能物理相邻,但对 query
  的打分却大相径庭。读者看到的上下文是第 7 章与第 12 章内容拼凑而成的怪物。
- **丧失章节级意图。** "5.3 升级" 这个标题本身就是强信号 —— 比它下面任何
  单句都更强。Heading 文字在 chunking 时往往被丢弃,或仅作为前缀而被稀释。
- **丧失关系结构。** 读者的问题也许更适合由**下一节** —— "5.4 事后
  评审" —— 来回答,而扁平 retriever 根本无从知晓这一点。
- **Agent 无法追问"这章里还有什么?"** 因为索引根本没有把"章"作为概念暴露出来。

这些不是任何具体实现的 bug,而是数据模型本身造成的必然结果。

## 位置先的替代方案

NaviKB 将每份文档的结构与内容分开索引。结构索引称为 `NavIndex`,每行存储一个
**NavEntry**,NavEntry 是"读者可以落脚的、在结构上有意义的单元":一节、
一子节、一张表、一个图、一段页码范围。

```python
@dataclass
class NavEntry:
    entry_id: str                    # 稳定 id
    label: str                       # "5.3 升级流程"
    normalized_label: str            # "5.3 升级流程"
    entry_type: str                  # "section" | "table" | "figure" | "page_range"
    source: str                      # "parser_heading" | "manual_alias" | ...
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    order_index: int                 # 在文档里的序列化顺序
    parent_entry_id: str | None      # 章节树
    resource_uris: list[str]         # ["kb://documents/D-42/pages/87", ...]
    source_ref_ids: list[str]        # 可追溯到 parser 输出
    aliases: list[str]               # 人工策展的替代 label
```

**关键观察:** 没有 `text` 字段,没有 `embedding`,没有 `description`。
NavEntry 是**纯指针**。它告诉你**去哪里**,而不是那里有什么。

这是架构层面的承诺。一旦接受 NavIndex 只存储指针,以下几点便顺理成章:

- 索引体积小(典型文档几百 KB),可以整个载入内存
- Agent 可以遍历索引而不会耗尽 token budget
- 文档变更后的索引更新很快(只需重建受影响的 entry,无需重新 embed)
- 搜索延迟由 entry 数量决定,与语料总量无关

## Agent 看到什么

NaviKB 的 MCP 界面将"导航"作为一等工具暴露,与"检索"明确区分:

| 工具 | 返回 | 什么时候用 |
|---|---|---|
| `kb_search_nav(query)` | NavEntry 列表(仅指针) | 问题映射到节 / 章 / 表时 —— "X 在哪讲?" |
| `kb_expand_nav_entry(entry_id)` | NavEntry + 父 + 子 + 兄弟 | 已经定位到正确 entry,想看周围结构 —— "这章里还有什么?" |
| `kb_get_toc(doc_id)` | 整份文档的完整 NavEntry 树 | 决定 fetch 什么之前,想浏览文档结构 |
| `kb_search(query)` | Evidence 命中 + 文本 + cite ID | 直接 evidence 检索 —— 当你已经大致知道想要什么文本时 |
| `kb_hybrid_search(query)` | Evidence 命中 + 建议的 NavEntry fetch | 常见路径 —— 同一次调用既拿 evidence 又拿导航 hint |

MCP 的 `kb_answer_from_originals` prompt 明确引导 Agent:先调用
`kb_hybrid_search`,再**通过 URI 取回原始资源**后引用。指针优先的
设计使这一工作流顺畅自然:Agent 始终同时掌握"位置"和"内容"。

## 检索仍然重要 —— 但它是第二步

Navigation-first 不等于"不要检索"。四通道检索栈(text dense、text sparse、
vision、description)仍是骨干,覆盖单靠导航结构无法触达的场景:

- 用正文同义改写提问、没有对应 heading 可以匹配的情况
- 跨文档查询:答案同时分布在文档 A 的 5.3 节**和**文档 B 第 87 页的图中
- 发现用户不知道存在的相关内容

这是关于**优先级**与**首次响应**的选择。当两条通道都有输出时,NaviKB 返回:

```text
evidence (可引用):                N 个 text/vision chunk,带 cross-encoder rerank
suggested_fetches (导航):         M 个 NavEntry 指针,覆盖周围结构
```

由 Agent 自行决定是直接引用,还是先进行导航。

## 为什么"做对"很难

navigation-first 若要真正胜过 chunk-first,三个条件必须同时满足:

### 1. Parser 得产出真结构

如果 parser 只输出"第 N 页文本",没有 heading、没有表格边界、没有图引用,
NavIndex 就会退化为"一页一个 entry" —— 比 chunking 提供的信号还少。NaviKB 依赖
MinerU(cloud API 或本地部署)的解析质量;实现上支持降级回退(以页作为 entry),
但设计前提是有真正的结构化 parser。

### 2. Label 得有意义到能 match query

"5.3 升级流程" 可以自然匹配 query "升级","Section 5.3" 则不行。真实文档
情况参差不齐;NaviKB 允许通过 **alias**(经 `kb_add_nav_alias` 添加的人工
标签)来修补 parser label 质量差的高频 entry。Alias 是 NavEntry 的一等字段,
持久化存储、可审计、可被 `kb_search_nav` 检索。

### 3. Agent 得真的用导航

一个不加引导的 Agent 若忽略 `kb_search_nav`、只调 `kb_search`,其行为本质上
仍是 chunk-first,只是多了几步而已。NaviKB 的 MCP prompt(`kb_answer_from_originals`)
和工具描述都经过调优,引导 Agent 先进行导航。我们将 prompt engineering 视为系统
的组成部分,而非留给下游调用者自行解决的问题。

## NavIndex 故意不做什么

为了让设计保持诚实,NavIndex **不**打算做以下这些事:

- **不是知识图谱。** 不做实体抽取,不做关系推断。"边"仅限于文档自身结构中的
  parent / child / next / prev,以及人工 alias。我们认为 LLM 抽取的边在
  小团队规模下还不足够可靠,无法替代这些。
- **不是搜索引擎。** 只做词法子串匹配和 alias 匹配 `normalized_label`,仅此而已。
  若需要语义相似,则已跨入检索层 —— 这完全合理,同一次 query 调用可以兼顾,
  但两者的契约不同。
- **不与内容共存。** NavIndex 是独立的 SQLite 库(`nav_index.db`)。
  这是刻意的设计:一次大规模 ingest 可以重建内容通道而不触动 NavIndex,
  一次快速的人工编辑(alias、hide)可以更新 NavIndex 而不触动 Qdrant。
- **不由 LLM 生成。** Alias 明确由人工策展(或 parser 抽取)产生。我们不允许 LLM
  以"方便用户"为由凭空构造层次结构。

## 我们接受的 trade-off

| 放弃什么 | 换到什么 |
|---|---|
| 在无真实结构的文档(一大块扁平文本)上表现较差 | 在结构良好的文档上表现出色 —— 涵盖多数运维手册、技术规范、法律文件、论文、书籍 |
| 无法推断文档本身不存在的关系(没有跨文档"主题"聚类) | 不存在 LLM 幻觉关系污染导航层的风险 |
| Ingest 时单文档成本较高(必须运行结构 parser) | Serving 时单次 query 成本较低(索引小,导航路径上无 LLM 参与) |
| Operator 偶尔需要为高频 query 策展 alias | 策展**本身就是**长期质量杠杆 —— eval 结果随时间持续改善,无需重新训练 |

## 跟邻近思路的对比

此处仅作概览,完整对比见 [`comparison.zh.md`](comparison.zh.md)。

- **vs LlamaIndex / Haystack 的层次 retriever:** 它们沿层次检索,但仍对每个
  chunk 进行 embed。NaviKB 让层次保持纯指针形态,将检索作为可独立组合的另一层。
- **vs GraphRAG:** GraphRAG 构建的是 LLM 抽取实体和关系所形成的图。NaviKB 使用的
  是文档自身的结构图。问题不同,信任模型也不同。
- **vs NaviRAG / "Don't Retrieve, Navigate":** 这些是 2026 年的研究论文,通过不同
  机制(LLM-distilled 层次、agent skill)探索同一方向。NaviKB 的贡献在于
  **运营层面**:一个今天就能自行部署的 navigation-first 系统,具备完整的
  auth/audit/backup 支撑。

## 这个设计在哪里可能是错的

以下两个场景可以证伪 navigation-first 的论点:

1. **语料本身没有可导航的结构。** 如果索引的是非结构化邮件归档、原始文本抓取、
   缺乏说话人或时间戳的对话记录,NavIndex 会退化,结果会输给一个调优良好的
   chunk-first 系统。
2. **Agent 无法基于导航进行推理。** 如果下游 LLM 的能力弱到或不稳定到这种程度——
   给它"这是一组指针"比"这是一大段文本"反而产生更差的结果——那么 navigation-first
   非但无益,反而有害。我们尚未严格测量过模型能力的下限 —— 这是 [docs/status.zh.md](../status.zh.md)
   中阻塞公开发布的事项之一。

如果以上任一条符合你的实际情况,NaviKB 不是合适的工具。我们会在文档中坦诚说明,
而不是设法将设计硬塞进这些场景。

## 接下来读什么

- [`architecture.zh.md`](architecture.zh.md) —— navigation-first 决策如何塑造
  栈的其余部分
- [`security-model.zh.md`](security-model.zh.md) —— alias 策展、隐藏、重建
  如何跟 auth + audit 交互
- [`comparison.zh.md`](comparison.zh.md) —— 与具体替代方案的明确对位
