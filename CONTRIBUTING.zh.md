<div align="right">

**中文** | [English](CONTRIBUTING.md)

</div>

# 为 NaviKB 做贡献

NaviKB 是一个早期 alpha 的**研究 / 示例**项目（见 [status](docs/status.zh.md)）。
代码已公开且能跑，但它不是开箱即用的产品 —— 所以最有用的贡献也反映这个阶段。

## 当前最有用的

### 🐛 复现报告与 bug 修复

你能做的最有用的事，是**在干净机器上装起来，然后告诉我们哪里坏了。**
干净机器安装尚未被广泛复现，所以一份安装报告（什么 work、什么不 work、你的
OS / Python 版本）—— 或一个修你撞到的 break 的 PR —— 极有价值。开 issue，或发一个
聚焦的 PR 并说明你如何复现了问题。

### 🧪 测试与文档

增加测试覆盖、修一个 flaky / 依赖环境的测试、或让文档更清楚的 PR 都欢迎。
文档方面，如果可以，请同时改中英两版（见下面「双语维护」）。

### 💬 设计、对比、定位反馈

设计是这里贡献的真实一部分。如果你认为某个设计决策错了，或 `comparison.zh.md`
描述错了某个相关项目（LlamaIndex、LangChain、GraphRAG、NaviRAG、Verba……），
开 issue 带 `design-feedback` 标签。我们想要准确的对比，不是奉承 NaviKB 的。

**一个好的 design-feedback issue：**

```text
Title: [design-feedback] <章节>: <一句话总结>
Document: docs/design/architecture.zh.md（或别的）+ 章节
我质疑的假设: <文档主张了什么>
为什么值得 revisit: <证据、参考、替代方法>
什么能让我接受任一方: <什么数据 / 论证能解决>
```

## 较大的代码改动前，先开 issue

任何超出聚焦修复的改动 —— 新功能、重构、依赖变更 —— 请先开 issue 讨论。两个原因：

- 这是早期 alpha；内部还在动，对着移动目标发大 PR 对谁都难受。
- 有些东西是**故意 out-of-scope**（多机部署、LLM 驱动的本体归纳、托管 SaaS
  —— 见 [architecture 文档](docs/design/architecture.zh.md)）。要这些的 issue 或 PR
  会被礼貌地关闭并指向那张清单。如果你认为某个 out-of-scope 项应该 in-scope，
  那是 `design-feedback` 讨论，不是 feature request。

## "什么时候稳定？"

诚实的成熟度和路线图在 [docs/status.zh.md](docs/status.zh.md)。它是 trigger-based 的，
不是 date-based 的 —— 我们内部承诺的每个日期都跳票了。

## 双语维护

每份文档都有 English（`<name>.md`）和中文（`<name>.zh.md`）版本，期望**内容等价**，
不是逐字翻译。改一份时在同一个 PR 里改另一份 —— 或注明 mirror 待更新，maintainer
会在 merge 前补。

## Code of Conduct

参与任何 NaviKB 空间（issue、discussion、PR）受 [Contributor Covenant](CODE_OF_CONDUCT.md)
约束。短版本：假定善意、给技术 specific 而非人身定性、不要把别的论坛的纠纷拖进来。

## Maintainer 响应时间

当前只有一个 maintainer。Issue 响应 best-effort，通常一周内。对于你不想公开的
安全担忧，email（地址在 maintainer 的 GitHub profile 上）比 issue 快 —— 见
[SECURITY.md](SECURITY.md)。
