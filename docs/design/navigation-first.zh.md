<div align="right">

**中文** | [English](navigation-first.md)

</div>

# Navigation-First

> NaviKB 最重要的设计决策。架构里其它所有东西都是这一条的下游。

## 一段话立场

文档不是一串扁平的 chunk,它是一个**可导航的结构**。读者(无论人类还是
Agent)关心**先到达正确的位置**,然后才关心**从那个位置拿到正确的文本**。
多数 RAG 栈把这个顺序反了 —— 它们先检索文本,如果还原结构那也是后补的。
NaviKB 从结构开始: 把结构索引成指针、作为一等 tool surface 暴露给 Agent,
把检索当作**导航缩小候选空间之后的廉价补充**。

本文说明我们为什么这么选、数据结构长什么样、设计在哪里 pay off、哪里有
已知的局限。

## chunk-first 检索的问题

设想一份 200 页的运维手册。问"Sev-1 事件的升级流程是什么?"。在典型的
RAG pipeline 里:

1. query 用 sentence-transformer 嵌入
2. 在 ~2000 个 chunk(每个 ~300 tokens)的向量库上做 cosine 相似度,取 top-5
3. top-5 chunks 拼起来,加个"用这些作上下文"的指令前缀,发给 LLM

这个方法 work 得"足够多"以至于变成了默认。但它隐藏的失败模式是真问题:

- **失去局部性。** Chunk N 和 Chunk N+1 可能在手册里物理相邻,但对 query
  打分非常不同。读者看到的上下文是混了第 7 章和第 12 章的 Frankenstein。
- **失去章节级意图。** "5.3 升级" 这个标题是个强信号 —— 比它下面任何
  单句都强。Heading 的文字往往在 chunking 时丢了,或者只作为 prefix 被
  稀释。
- **失去关系结构。** 读者的问题可能其实更适合用**下一节** —— "5.4 事后
  评审" —— 来回答,而扁平 retriever 完全没办法知道这点。
- **Agent 无法问"这章里还有什么?"** 因为索引根本没把"章"当概念暴露。

这些不是任何具体实现的 bug。它们是数据模型的后果。

## 位置先的替代方案

NaviKB 把每份文档的结构和内容分开索引。结构索引叫 `NavIndex`,每行存一个
**NavEntry**,NavEntry 是 "读者可以落脚的、结构上有意义的单元": 一节、
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

**关键观察:** 没有 `text` 字段。没有 `embedding`。没有 `description`。
一个 NavEntry 是**纯指针**。它告诉你**去哪里**;它不告诉你那里有什么。

这是架构层的承诺。一旦你接受 NavIndex 只装指针,几件事就顺理成章:

- 索引很小(典型文档几百 KB),可以整个 load 进内存
- Agent 可以 walk 它而不会耗光 token budget
- 文档变更后的索引更新很快(只重建受影响的 entry,不需要重新 embed)
- 搜索延迟由 entry 数量决定,跟语料大小无关

## Agent 看到什么

NaviKB 的 MCP 表面把"导航"作为一等工具暴露,跟"检索"明确区分:

| 工具 | 返回 | 什么时候用 |
|---|---|---|
| `kb_search_nav(query)` | NavEntry 列表(仅指针) | 问题映射到节 / 章 / 表时 —— "X 在哪讲?" |
| `kb_expand_nav_entry(entry_id)` | NavEntry + 父 + 子 + 兄弟 | 已经定位到正确 entry,想看周围结构 —— "这章里还有什么?" |
| `kb_get_toc(doc_id)` | 整份文档的完整 NavEntry 树 | 决定 fetch 什么之前,想浏览文档结构 |
| `kb_search(query)` | Evidence 命中 + 文本 + cite ID | 直接 evidence 检索 —— 当你已经大致知道想要什么文本时 |
| `kb_hybrid_search(query)` | Evidence 命中 + 建议的 NavEntry fetch | 常见路径 —— 同一次调用既拿 evidence 又拿导航 hint |

MCP 的 `kb_answer_from_originals` prompt 明确训练 Agent: 先用
`kb_hybrid_search`,然后**通过 URI fetch 原始资源**再引用。指针-first
的设计让这个 workflow 自然: Agent 始终同时有"在哪"和"是什么"。

## 检索仍然重要 —— 但它是第二步

Navigation-first 不等于"不要检索"。4 通道检索栈(text dense、text sparse、
vision、description)仍然是骨架,覆盖导航靠结构够不到的场景:

- 用 body text 的同义改写问的问题,没有 heading 匹配
- 跨文档查询: 答案在文档 A 的 5.3 节**且**在文档 B 第 87 页的图里
- 露出用户都不知道存在的相关内容

选择是关于**优先级**和**首次响应**的。当两边都有话要说时,NaviKB 返回:

```text
evidence (可引用):                N 个 text/vision chunk,带 cross-encoder rerank
suggested_fetches (导航):         M 个 NavEntry 指针,覆盖周围结构
```

由 Agent 决定是直接引用还是先导航。

## 为什么"做对"很难

navigation-first 要真的赢过 chunk-first,三件事必须同时成立:

### 1. Parser 得产出真结构

如果 parser 只 emit "第 N 页文本",没有 heading、没有表格边界、没有图引用,
NavIndex 退化成 "一页一个 entry" —— 比 chunking 的信号还少。NaviKB 依赖
MinerU(cloud API 或本地)的解析质量;实现支持降级 fallback(页即 entry),
但设计假设是有真 parser。

### 2. Label 得有意义到能 match query

"5.3 升级流程" trivially match query "升级"。"Section 5.3" 不行。真实文档
有混杂的情况;NaviKB 允许 **alias**(通过 `kb_add_nav_alias` 加的人工
label)修补 parser label 不好的高流量 entry。Alias 是 NavEntry 的一等字段,
持久化、审计、可被 `kb_search_nav` 搜到。

### 3. Agent 得真的用导航

一个 naive Agent 忽略 `kb_search_nav` 只调 `kb_search`,就是 chunk-first
行为加额外步骤。NaviKB 的 MCP prompt(`kb_answer_from_originals`)和工具
描述都调过,推 Agent 先导航。我们把 prompt engineering 当作系统的一部分,
不是下游调用者应该自己再造的轮子。

## NavIndex 故意不做什么

为了让设计诚实,这些事 NavIndex **不**打算做:

- **不是知识图谱。** 不做实体抽取。不做关系推断。"边"就是文档自身结构里的
  parent / child / next / prev,加人工 alias。我们不相信 LLM 抽取的边在
  小团队规模下足够可靠,可以替代这个。
- **不是搜索引擎。** Lexical 子串 + alias match `normalized_label`。就这些。
  想要语义相似,你已经离开导航进入检索层了 —— 这没问题,是同一个 query
  调用,但契约不一样。
- **不与内容存在一起。** NavIndex 是独立的 SQLite 库 (`nav_index.db`)。
  这是刻意的: 一次 heavy ingest 可以重建内容通道而不动 NavIndex,一个
  快速人工编辑(alias、hide)可以更新 NavIndex 而不动 Qdrant。
- **不由 LLM 生成。** Alias 明确是人工策展(或 parser 抽出)。我们不让 LLM
  "为用户方便"幻觉一个层次结构。

## 我们接受的 trade-off

| 放弃什么 | 换到什么 |
|---|---|
| 没真结构的文档(一坨巨大文本)上表现差 | 在结构良好的文档上表现优秀 —— 多数运维手册、技术规范、法律文件、论文、书 |
| 不能推断文档没有的关系(没有跨文档"主题"聚类) | 没有 LLM-幻觉关系污染导航表面的风险 |
| Ingest 时每文档成本更高(必须跑结构 parser) | Serving 时每 query 成本更低(索引小,导航路径上无 LLM) |
| Operator 偶尔要为高流量 query 策展 alias | 策展**就是**长期质量杠杆 —— eval 结果随时间改善,无需重训 |

## 跟邻近思路的对比

这里只是概览,完整对比在 [`comparison.zh.md`](comparison.zh.md)。

- **vs LlamaIndex / Haystack 的层次 retriever:** 它们沿层次检索,但仍 embed
  每个 chunk。NaviKB 让层次保持指针-only,让检索作为可组合的另一件事。
- **vs GraphRAG:** GraphRAG 构造 LLM-抽取实体和关系的图。NaviKB 用文档自身
  的结构图。不同问题、不同信任模型。
- **vs NaviRAG / "Don't Retrieve, Navigate":** 这些是 2026 研究论文,用不同
  机制(LLM-distilled 层次、agent skill)探索同一方向。NaviKB 的贡献在
  **运营侧**: 一个你今天就能自己跑的 navigation-first 系统,带真的
  auth/audit/backup 故事。

## 这个设计在哪里可能是错的

两个会证伪 navigation-first 论点的场景:

1. **语料没结构可导航。** 如果你索引的是非结构化邮件 dump、原始文本爬取、
   没说话人或时间戳的对话 transcript,NavIndex 退化,你会输给一个调好的
   chunk-first 系统。
2. **Agent 不能基于导航推理。** 如果你下游的 LLM 小到 / 不可靠到给它
   "这是一组指针"产生比"这是一面墙的文本"更差的结果,navigation-first 反而
   伤了你。我们还没严格测量过模型能力的下限 —— 这是 [docs/status.zh.md](../status.zh.md)
   里阻塞公开发布的项目之一。

如果以上任一条是你的情况,NaviKB 不是合适的工具。我们会在文档里诚实地说,
不会想方设法 retrofit 设计去覆盖。

## 接下来读什么

- [`architecture.zh.md`](architecture.zh.md) —— navigation-first 决策如何塑造
  栈的其余部分
- [`security-model.zh.md`](security-model.zh.md) —— alias 策展、隐藏、重建
  如何跟 auth + audit 交互
- [`comparison.zh.md`](comparison.zh.md) —— 与具体替代方案的明确对位
