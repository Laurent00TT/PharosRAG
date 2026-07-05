# eval 基线锚(口径 anchor)

> 只记指标数字 + 来源,不含逐题 gold/答案(那些含研报节选,gitignored)。可复现见 [README.md](README.md) / [../docs/RUNBOOK.md](../docs/RUNBOOK.md)。

## Tier1(DeepSeek 自判,仓内可复现)

- **配置**:`--mode single --judge deepseek --smart-tables`,gold 88 题(散文 72 + 表格 16),eval 库 `~/rag_eval_big`(collection `evalbig`,15 篇 / 1409 chunk)。
- **复现于**:引擎折入 pharos 单仓后(`master`,合并提交 `b6fe69d`),2026-07-05,WSL navikb / RTX 4090。

| 指标 | 本次(单仓 master) | 迁移前权威基线 | Δ |
|---|---|---|---|
| 检索召回@k | **0.830** | 0.818 | +0.012 |
| 全召回 | **0.761** | —(0.792 为 72 题口径) | — |
| MRR | **0.629** | 0.627 | +0.002 |
| 引用召回 | **0.778** | 0.767 | +0.011 |
| 忠实度 | **0.977** | 0.977 | 0(精确) |
| 正确性 | **0.830** | 0.818 | +0.012 |
| 平均轮数 | 1.00 | 1.00 | 0 |

**结论**:五指标全部落在单轮 DeepSeek 裁判 **±2 题噪声**内(0.012×88 ≈ 1 题),忠实度逐字一致。
**迁移端到端保真** —— 同代码(引擎字节一致)+ 同 gold + 同 eval 库,单仓复现出与迁移前一致的数字。

## Tier2(双-Claude 异厂裁判,权威去偏)

Tier1 是同厂 DeepSeek 自判(循环偏差,看趋势)。**权威去偏**走异厂 Claude 裁判:2 遍独立 Claude pass、看**全 context**、双-pass AND(两遍都判 true 才算)。本次用 Claude Code 多-agent 编排跑通(单仓 master,2026-07-05,88 题 single),指纹校验通过:

| 指标(88 题,single) | Tier2 双-Claude(权威) | Tier1 DeepSeek 自判 |
|---|---|---|
| **忠实度** | **0.989** | 0.977 |
| **正确性** | **0.818** | 0.830 |
| 检索召回 / MRR / 引用召回 | 0.830 / 0.629 / 0.778(程序化,判官无关) | 同 |

**按 hop 正确性(AND)**:single-hop **0.889**(n=54)/ multi_intra **0.828**(n=29)/ multi_cross **0.000**(n=5)。
- 忠实度 0.989 ≈ 滴水不漏——grounding 契约稳(复现 R5 "≈1.0" 结论,系统宁可拒答不编);两 pass 仅 1 题忠实、1 题正确分歧(裁判一致性高)。
- 异厂 Claude 判比同厂 DeepSeek 自判**略严**(正确 0.818 vs 0.830,差 ~1 题),方向符合预期(去偏)。
- 硬骨头仍是 **multi_cross(跨文档综合)0.000**(n=5 偏小):卡在"拿到两篇块也合不出对比",是综合不是检索。

**为何"不在仓内可复现"**:这条裁判链路(2 遍独立 Claude 判 + 看全 context)靠 Claude Code 多-agent 编排;`verdicts.json`/`_judge/` 不入库(gitignored、可再生),换个没有 Claude 编排的环境复现不出这组权威数字。历史 72 题口径(补表格题前)另有一组(忠实 ≈1.0 / 正确 0.847),与 88 题**不可直接比**。

## 口径断代提醒

88 题(2026-07-03 补 16 表格题起)与历史 72 题聚合数字**不可直接对比**。gold/results/verdicts/baseline_*.json 均 gitignored(含私数据、可重生)。
