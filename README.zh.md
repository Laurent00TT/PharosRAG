<div align="right">

**中文** | [English](README.md)

</div>

# NaviKB

**Navigable Knowledge Base** —— 一个 navigation-first、本地优先的知识库,为个人和小团队设计。

> ⚠ **状态: 设计阶段,公开预览。** 当前仓库只包含设计文档,代码在私有开发中。
> 完整发布计划和验收标准见 [docs/status.zh.md](docs/status.zh.md)。

---

## 它是什么

NaviKB 围绕两个主流 RAG 做得不太对的判断展开:

1. **导航,而不是检索,才是知识访问的主路径。** 多数 RAG 把文档视作扁平的
   chunk 集合,靠 dense similarity 找正确文本。NaviKB 把每份文档的**结构**
   (标题、表格、图、页码范围)索引到一个 pointer-only 的 **NavIndex**,让
   Agent 像人类读者一样沿着结构走。
2. **生成内容不是 evidence。** LLM 对一页的描述是 recall hint。把它当作
   原始证据引用 = 幻觉放大器。NaviKB 在每个响应里都把字段分成
   `evidence_fields`(已解析的文本、图片 URL、figure caption)和
   `hint_fields`(LLM 生成的描述),并明确告诉 Agent 哪些可引用。

NaviKB 是为个人 + 小团队的真实工作形态设计的:几百到几千个 PDF、少数几个
operator、强偏好本地运行、不依赖 SaaS。

## 它不是什么

- **不是托管服务。** 所有东西跑在你自己的机器或局域网。
- **不是 vector-search 套壳。** 检索是 4 通道之一(text dense、text sparse、
  vision、description),用 RRF 融合 + rerank —— 但始终锚定在可导航的结构上。
- **不是知识图谱。** 不做实体抽取、不做本体归纳、不让 LLM 凭空建关系边。
  唯一的"图"就是文档自带的目录结构。
- **不是企业平台。** 没有多租户编排、没有 Kubernetes operator、没有
  Celery / Redis / Kafka。SQLite + Qdrant + worker 进程就是全栈。

## 为什么是这个形状

设计是对 2026 年 RAG 两个趋势的反方向选择:

- **越做越重的基础设施** —— 托管 vector DB、graph DB、agent 平台、observability
  栈。对有专职 ops 团队的企业很好;对个人和 2-5 人小团队很不友好。
- **越来越脱离文档结构** —— chunk 一切、embed 一切、靠相似度祈祷找到。
  失去了让真实文档 navigable 的所有 affordance(章节、figure、表格、
  "见第三章"这种引用)。

NaviKB 在两件事上都选了相反的立场:**小体积、深度尊重结构、MCP-native**。

形状的细节见 [docs/design/architecture.zh.md](docs/design/architecture.zh.md),
设计 rationale 见 `docs/design/navigation-first.zh.md`(下一份)。

## 差异化对比

| 维度 | NaviKB | 主流 RAG (LangChain / LlamaIndex) | GraphRAG | NaviRAG (research) |
|---|---|---|---|---|
| 主访问模式 | 指针导航 + 4 通道检索 | 扁平 chunk 上的 dense 检索 | LLM 抽取实体后的图遍历 | LLM 驱动的层次探索 |
| 本地优先 | ✅ 默认 | ⚠ 可以但少 | ❌ 通常需要 Neo4j 级基础设施 | ❌ 研究阶段 |
| MCP-native | ✅ 一等公民 tool surface | ⚠ 靠 adapter | ❌ | ❌ |
| Citation 纪律 | ✅ 强制 evidence/hint 分区 | ❌ 调用方自己定 | ❌ | ❌ |
| 运维表面(auth/audit/backup/GC) | ✅ 端到端 | ❌ "自己搭" | ⚠ 仅企业栈 | ❌ |
| 状态 | 设计公开,代码私有 | 成熟 | 成熟 | 研究原型 |

更完整的对比 + 论证将放在 `docs/design/comparison.zh.md`。

## 文档地图

| 文档 | 状态 | 用途 |
|---|---|---|
| [README](README.zh.md)(当前) | ✅ 公开 | 30 秒理解 |
| [docs/status.zh.md](docs/status.zh.md) | ✅ 公开 | 代码状态、发布时间、release 阻塞项 |
| [docs/design/architecture.zh.md](docs/design/architecture.zh.md) | ✅ 公开 | 四层架构: ingest → storage → retrieve → serve |
| `docs/design/navigation-first.zh.md` | 🔄 进行中 | 为什么 NavIndex;指针-only 设计;vs chunking 的取舍 |
| `docs/design/security-model.zh.md` | 🔄 进行中 | hard-cutover auth、two-phase audit、软删除、maintenance flag、GC retention |
| `docs/design/comparison.zh.md` | 🔄 进行中 | 与 LangChain / LlamaIndex / GraphRAG / NaviRAG 的深度对比 |
| [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) | ✅ 公开 | 当前阶段欢迎和不欢迎的贡献类型 |

## 快速链接

- 📋 [状态 & 发布计划](docs/status.zh.md)
- 🏛 [架构](docs/design/architecture.zh.md)
- 💬 [讨论: 开 issue](../../issues) —— 欢迎设计反馈
- 🤝 [贡献指南](CONTRIBUTING.zh.md)
- 📜 [License: Apache-2.0](LICENSE)

## 命名由来

**NaviKB** = **Navi**gable + **K**nowledge **B**ase。

读作 "nah-vee-K-B"(三音节)或 "navy-K-B"(偷懒的两音节版本)。把项目定位到
2026 年正在成型的 *navigable knowledge base* 研究方向
(见 [NaviRAG](https://arxiv.org/abs/2604.12766)、
["Don't Retrieve, Navigate"](https://arxiv.org/abs/2604.14572)),但聚焦在
**不同的层**:那些是检索算法的研究原型,NaviKB 是支撑这类设计的生产运维栈。
