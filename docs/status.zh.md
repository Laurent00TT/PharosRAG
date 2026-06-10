<div align="right">

**中文** | [English](status.md)

</div>

# 状态 & 诚实成熟度

> **诚实摘要：** 代码已公开且能跑。这是一个**早期 alpha** 的研究 / 示例参考实现 ——
> 不是开箱即用的产品，也不是 semver 意义上的"已发布版本"。本页如实说明"能用什么、不能用
> 什么、真实风险在哪"。

---

## 今天能用什么

- 设计经过六个内部 review 阶段迭代，每个 PR 都有设计 spec 和代码 review。
- 实现通过自己的测试套件（800+ 测试），覆盖检索、auth、audit、软删除、maintenance flag、
  备份、GC、MCP tool 面。
- 核心不变量 —— 跨进程 maintenance gate、两阶段 audit、evidence/hint 分区、navigation-first
  检索 —— 都已就位且有测试。
- 在维护者机器上跑通了参考模型栈的端到端流程（Qwen3-VL 嵌入+重排、MILCO 学习式稀疏、Qwen3.6-VL
  页面描述、MinerU 版面解析）。

## 还不能用 / 尚未验证

- **复现性只做了轻度验证。** 从源码安装（`pip install -e ".[server,dev]"`）加 `import kb`
  在干净的 Python 3.12 虚拟环境里成功。但**尚未**被第二个人、或在第二种 OS 上复现 ——
  这一点、加上更重的可选 `[mineru]` extra 和 live 模型服务，是尚未填补的空白。
- **评测体系仍在完善。** golden set 大部分是合成的；广泛的真实文档评估是当前最大的质量不确定性。
- **没有发布的包。** 暂无 PyPI 发布或容器镜像；安装从源码进行。
- **模型 / 部署矩阵是参考，不是测过的网格。** 只有上面那套参考栈端到端跑过。其他模型端点
  *应该*能用 —— 一切都在 `*_SERVER_URL` / `*_MODEL_ID` 之后 —— 但未经验证。

## 为什么现在公开，作为"研究 / 示例"

两个原因。

第一，**当前阶段，设计本身就是贡献。** NaviKB 的几个决策 —— navigation-first 作为主访问模式、
evidence/hint 分区、小团队栈上的两阶段 audit、跨进程 cache epoch —— 单独拎出来就值得阅读和
探讨，跟打包打磨到什么程度无关。公开代码让这个讨论从假设变成具体。

第二，**画饼式的 vaporware 既廉价又无用；诚实的 alpha 才有用。** 与其捂着代码等一切完美，不如交出一个
能跑的参考实现并对差距诚实。如果你装起来某处坏了，那正是关闭复现性差距最需要的反馈。

## 你能怎么帮

- **🐛 开 issue** —— 安装坏了、文档不清、或某个对比错了。第二人安装报告是现在最有用的东西。
- **📋 读设计文档** —— [architecture](design/architecture.zh.md)、
  [navigation-first](design/navigation-first.zh.md)、
  [security-model](design/security-model.zh.md)、
  [comparison](design/comparison.zh.md)。
- **⭐ Star / watch** —— 跟进粗糙之处被逐步磨平的过程。

## 路线图（诚实，无日期）

大致按优先级，最可能变化的：

1. 可复现的干净机器安装 —— 一次依赖体检 + 一份第二人安装报告。
2. 一份真实文档评测报告（Recall@k、NDCG、MRR），基于至少一个非合成语料，golden set 公开。
3. 一份记录在案的威胁模型，加一次作者以外的人做的 auth/audit review。
4. 一个验证过的"自带模型"矩阵，超出单一参考栈。

我们不给日历日期 —— 内部设过的每个日期都没能兑现。上面是真实的待办优先级，不是发布时间表。

---

*研究 / 示例 · 早期 alpha。最后更新：2026-06-05。*
