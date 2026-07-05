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

## Tier2(双-Claude 异厂裁判,权威、**不在仓内可复现**)

Tier1 是同厂 DeepSeek 自判,有循环偏差、只看趋势。**权威去偏数字**走双-Claude 裁判(R5 订正后,72 题口径:忠实度 ≈ 1.000、正确性 0.847),需 Claude Code 多 agent 编排产 `verdicts.json`(步骤见 [README.md](README.md) 去偏全流程),换环境需自备异厂裁判。

## 口径断代提醒

88 题(2026-07-03 补 16 表格题起)与历史 72 题聚合数字**不可直接对比**。gold/results/verdicts/baseline_*.json 均 gitignored(含私数据、可重生)。
