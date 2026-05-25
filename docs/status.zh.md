<div align="right">

**中文** | [English](status.md)

</div>

# 状态 & 发布计划

> **诚实摘要:** 设计已完成,实现在维护者机器上工作正常,**但尚未发布**。
> 本页告诉你今天的真实状态、阻塞公开发布的事项、以及时间线。

---

## 今天真实的状态

- 设计经过六个阶段的内部迭代(small-team 阶段 T1a → T5),每个 PR 都有
  明确的设计 spec 和代码 review
- 实现在私有开发中通过了 828 个自己的测试,覆盖检索、auth、audit、
  软删除、maintenance flag、备份、GC、MCP tool 表面
- 核心不变量(跨进程 maintenance gate、两阶段 audit、evidence/hint 分区、
  navigation-first 检索)都已就位且有测试

## 今天**还不**真实的事

- 代码**还没公开**。原因见下方 [为什么代码还没公开](#为什么代码还没公开)
- PyPI / 任何 container registry 上**没有发布的包**
- 可复现性**未经验证**: 在维护者机器上能跑,但还没被第二个人独立装起来
- Eval golden set 大部分还是合成的;基于真实文档的评估是当前最大的开放
  质量风险

## 为什么代码还没公开

公开发布的门槛不是"测试都过了",而是:

1. **第二个人能在 30 分钟内在干净机器上完成安装 + 第一份文档 ingest**,
   只看公开文档
2. **Eval golden set 里至少有一个真实文档的 benchmark** —— 不只是合成
   query —— 覆盖我们真实预期的四类 query (fact lookup、流程 / how-to、
   表格 lookup、视觉 layout)
3. **威胁模型已记录**,auth 表面被作者以外的人 review 过
4. **模型升级路径是具体的** —— 当前代码栈用 BGE-M3 + ColQwen2 +
   bge-reranker-v2-m3。下一代栈(Qwen3-VL + MILCO + BGE-M4 等)已设计
   但未验证,发布时同时支持两个 embedding 版本是发布计划的一部分

这四件事都成真之前,管这项目叫"已发布"会错误描述它的成熟度。我们宁愿
under-promise。

## 发布计划

我们用三阶段公开发布:

### Stage 0 —— 设计预览(当前)

- **公开的:** 本仓库的文档
- **你可以:** 读设计、开 issue 提设计反馈、watch 仓库等 Stage 1 触发
- **你做不了:** 安装、ingest、搜索 —— 这里没有代码
- **持续时间:** 直到 Stage 1 验收标准达成

### Stage 1 —— Closed alpha

- **触发:** 一个小群体(5-10 个信任用户)独立装起 NaviKB,确认文档覆盖了
  安装路径
- **变化:** alpha 用户拿到私有 repo invite + 一份针对他们反馈循环优化的
  setup 文档
- **仍然私有:** 主代码 repo
- **预期持续:** 4-8 周迭代

### Stage 2 —— 公开开源

- **触发:** 上面的验收标准全部达成
- **变化:** 代码变成公开仓库(在本 GitHub 组织下);安装说明落到本仓库的
  docs
- **版本:** 从 v0.1.0 开始 semantic versioning(v0.x 阶段仍预期有破坏性变更)

我们不给任何阶段的日历日期,因为我们内部给过的每一个日历日期都跳票了。
上面的触发条件才是真正的 gate。

## 跟进进度

跟进真实进度的最快方式:

- **⭐ Star 或 watch 本仓库** —— Stage 1 邀请和 Stage 2 发布都会在这里宣布
- **🐛 开 issue** 带 `design-feedback` 标签 —— 每个 issue 都会被读,
  即便 Stage 2 之前不接受 code PR
- **📋 读设计文档** —— [architecture](design/architecture.zh.md)、
  [navigation-first](design/navigation-first.zh.md)、
  [security-model](design/security-model.zh.md)、
  [comparison](design/comparison.zh.md) 都已上线

## Stage 2 之前我们会公开的物料

为了让发布有说服力,这些 artifact 计划**在代码 drop 之前**就公开:

- 一份可复现的 evaluation 报告(Recall@k、NDCG、MRR),基于至少一个真实
  文档语料,golden set 公开
- Stage 1 用户的第二人安装报告(如果他们偏好,匿名处理)
- 一份记录在案的威胁模型,加 auth/audit 表面 review 笔记
- 模型升级 design spec,在新 embedding 栈上端到端验证过

任何一项没落地,发布日期就不落地。我们不会带着 README 里的
"TODO: evaluation" 发 public v0.1.0。

## 为什么要这么透明

两个原因。

第一,**vaporware 又便宜又没用**。任何人都能注册一个 repo 名 + 贴个 logo。
我们宁愿展示足够的设计让人相信,加上足够的诚实关于差距让人信任。

第二,**当前阶段设计本身就是贡献。** NaviKB 的几个决策(navigation-first
作为主访问模式、evidence/hint 分区、小团队栈上的两阶段 audit、跨进程
cache epoch)单独拎出来讨论就有价值,跟代码什么时候 ship 无关。那个讨论
就是 design preview 的目的。

---

*最后更新: 2026-05-25。*
