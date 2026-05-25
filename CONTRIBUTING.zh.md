<div align="right">

**中文** | [English](CONTRIBUTING.md)

</div>

# 为 NaviKB 做贡献

NaviKB 在 [Stage 0: 设计预览](docs/status.zh.md) 阶段。这意味着当前
贡献的形态跟典型开源项目不一样。本文说明欢迎什么、不欢迎什么、为什么。

## 当前欢迎的

### ✅ 设计反馈

`docs/design/` 里的设计文档是这个阶段的真实交付物。仔细读完并对其中
假设提出反对,是你能做的最有价值贡献。

**怎么做:** 开 issue 带 `design-feedback` 标签。引用具体文档和章节
(`docs/design/architecture.zh.md §4 Serve`),说明你质疑的假设是什么,
你认为替代方案是什么样子。

**好反馈示例:**

- "四通道检索说明里说你们用 RRF 在四个 rank 上融合。考虑过 Sparse-Dense
  Fusion ([参考论文]) 吗?它可以让你训一个 late-fusion 权重 per channel。"
- "跨进程 cache-epoch 计数器假设两个进程都能写同一个 SQLite 文件。在
  Docker Compose 部署 + worker 在独立 container + 共享 volume 时是 work
  的。在 Kubernetes 持久卷部署里,锁语义没测过。值得在 architecture 文档
  里加个 note。"
- "Status 文档说 'navigation-first' 是差异化点。但 haystack 和 llamaindex
  都有 document-tree retriever。值得在 comparison.zh.md 里直接对比。"

### ✅ 对比 + 定位输入

如果你维护或使用相关项目(LlamaIndex、LangChain RAG、GraphRAG、NaviRAG、
Verba 等),并且认为我们的 `comparison.zh.md` 描述错了,开 issue。我们想要
准确的对比,不是奉承 NaviKB 的对比。

### ✅ 文档错别字 / 清晰度问题

直接开 PR。这些会被 merge。如果你能同时改中英两个版本最好;如果只改一份,
在 PR 里说明,我们会同步另一份。

## 当前不接受的

### ❌ 对(不存在的)实现的代码 PR

代码不在这个仓库里(原因见 [docs/status.zh.md](docs/status.zh.md))。
添加 `src/`、`tests/`、`scripts/` 等的 PR 会被关闭并指向本文档。这不是
对 PR 的评价;而是 merge 代码会让我们宣传一些没做过第二人验证的事,这
正好越过了我们设的门槛。

### ❌ 暗示改变项目 scope 的功能请求

我们明确标了 out-of-scope 项(多机部署、LLM 驱动的本体归纳、托管 SaaS
—— 见 [architecture.zh.md § 故意不做的事](docs/design/architecture.zh.md#故意不做的事))。
要这些的 issue 会被礼貌地关闭并指向那张清单。

如果你认为某个 out-of-scope 项应该 in-scope,那是个设计讨论 —— 开
`design-feedback` issue,不是 feature request。

### ❌ "这玩意什么时候 ship?"

发布计划在 [docs/status.zh.md](docs/status.zh.md)。它是 trigger-based 的,
不是 date-based 的。要日历日期不会得到 —— 因为我们内部承诺的每一个日历
日期都跳票了。

## 怎么开一个好的 design-feedback issue

```
Title: [design-feedback] §<章节>: <一句话总结>

Document: docs/design/architecture.zh.md (或别的)
Section: §3 Retrieve
具体段落: "RRF over 分数融合..."

我质疑的假设:
  <你对文档主张的理解>

我为什么觉得值得 revisit:
  <证据、参考、替代方法>

什么能让我接受任一方:
  <什么数据 / 论证能解决这个>
```

我们承诺读每个 issue。我们不承诺只要给反馈就改设计 —— 设计必须真的
变好,不是因为有反馈就妥协。

## 双语维护

每份文档都有 English (`<name>.md`) 和中文 (`<name>.zh.md`) 两个版本。
它们期望**内容等价**,不是逐字翻译。

当你改一份时,在同一个 PR 里改另一份。如果你只会一种语言,就只写英文版,
在 PR 描述里说中文 mirror 待更新 —— maintainer 会在 merge 之前同步中文。

只改一种语言又不承认另一种的 PR 会被要求要么补完 mirror,要么降级 PR
的 scope。

## Code of Conduct

参与任何 NaviKB 空间(issue、discussion、未来的 PR)受 [Contributor
Covenant](CODE_OF_CONDUCT.md) 约束。短版本: 假定善意、给技术 specific
而非人身定性、不要把别的论坛的纠纷拖到这里。

## Maintainer 响应时间

这是当前只有一个 maintainer 的项目。Issue 响应是 best-effort,通常
一周内。如果真的紧急(对已记录设计的安全担忧 —— 是的,即便代码还没出
来)email 比 issue 快。Email 地址会在 maintainer 的 GitHub profile 上。
