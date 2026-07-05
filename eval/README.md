# eval —— 端到端 / agentic RAG 评估闭环(Batch 7)

回答一个问题:**这套 RAG 到底答得对不对、有没有幻觉、检索有没有捞到该捞的、ACL 隔离扛不扛得住。**
区别于 [`eval/component_retrieval`](component_retrieval)(只测**检索组件**的 MRR/Recall),这里测**整条链**
(检索 → 生成 → 引用 → ACL),并且**全自动**:合成 gold、跑系统、LLM 当裁判,零人工标注。

## 迁移说明(引擎折入 pharos 单仓后)

- **三条评测轴,分目录不混淆**:本目录 `eval/` = 端到端五指标(皇冠);`eval/component_retrieval/` = 检索组件
  (BM25/BGE/RRF,MRR/Recall);`eval/component_chunking/` = 分块证据保全(vs MMDocIR,需自备外部标注 `MMDocIR_annotations.jsonl`)。
- **环境变量统一 `PHAROS_EVAL_*`**(`PHAROS_EVAL_SRC/COLLECTION/GEN_MODEL/JUDGE_MODEL`);`RAG_EVAL_*` 留一版弃用别名。
  `.env` 自动读 pharos 仓根的 `.env`(与守护进程同一份)。⚠ `PHAROS_EVAL_GEN_MODEL` **须与生产 `PHAROS_LLM_MODEL` 一致**。
- **跑 GPU eval 前先停 live 守护进程**:`sudo systemctl stop pharos`(避免与 daemon 抢 GPU rerank 致 OOM);
  嵌入式 Qdrant 单客户端锁则由 `_common.copy_demo` 的 copytree-到临时库规避(不抢锁、不污染原库)。
- **可复现性两档**:**Tier1**(仓内可复现)= `run_eval --judge deepseek` 自判(同厂,看趋势);
  **Tier2**(权威,**不在仓内可复现**)= 双-Claude 异厂裁判,需 Claude Code 多 agent 编排产 `verdicts.json`(步骤 3/6,不入库)。

## 脚本

| 脚本 | 干什么 | 依赖 |
|---|---|---|
| `gen_gold.py` | **7.0(快速版)** 从 demo 索引采样 chunk,DeepSeek 据每段生成 (问题, golden 答案) → `gold.jsonl` | CPU + DeepSeek |
| `run_eval.py` | **7.A/7.B** 跑系统出四指标 + 单跳 vs agentic 双层归因(`--judge none` 出 rows 交 Claude 裁判) | GPU + DeepSeek |
| `acl_regression.py` | **7.C** 自建 2 租户合成库,断言跨租户/无权/unset 身份对受限内容 **0 召回** | GPU |
| `index_eval_corpus.py` | 扩语料:15 篇 parsed/ → 更大评估库 `~/rag_eval_big` | GPU |
| `dump_chunks.py` | 扫库采样 → `_units/`(给 Claude 子 agent 造 gold 的输入) | CPU |
| `dump_judge_units.py` | `results_*.json` → `_judge/`(给 Claude 裁判 agent 的输入) | CPU |
| `aggregate.py` | 合并程序化指标 + Claude verdicts,按 hop 拆 + 双层归因 | CPU |

> 去偏全流程见下方「去偏全流程」节(gold 与裁判都用 Claude 子 agent,DeepSeek 只当被测系统)。

## 四个指标(run_eval)

- **检索召回@k / MRR**(组件层):golden chunk 有没有被召回、排第几 —— 只看检索,不看生成。
- **引用召回**:答案是否**真的引用了** golden chunk(`[cite:n]` → chunk_id),grounded 取证。
- **忠实度**(LLM-judge):答案每个论断是否都有上下文支持 —— 防幻觉。合理拒答("信息不足")算忠实。
- **正确性**(LLM-judge):答案是否与 golden 答案事实一致。

## 双层归因(7.B,`--mode both`)

把"检索组件的功劳"和"agent 多跳擦屁股的功劳"分开:

- **single**:闭管道单跳(retrieve 一次 → 生成),= 组件层基线。
- **agentic**:DeepSeek 判断上下文够不够,不够就**改写 query 再搜**,累积上下文再生成,= 模拟 agent。
- **Δ = agentic − single**:正的 `correctness Δ` = agent 的多跳真补上了单跳检索的缺口;`avg_rounds` 上升 = 多花的检索代价。

> 这一栏是 agentic RAG 的核心论据:如果 Δ≈0,说明单跳检索已够好、agent 多跳是浪费;Δ 明显为正才证明 agent 驱动有价值。

## 怎么跑

```bash
conda activate navikb
python eval/gen_gold.py --per-doc 6        # 生成 gold.jsonl(~24 条)
python eval/run_eval.py --mode both        # 四指标 + 双层归因;结果落 results_single.json / results_agentic.json
python eval/acl_regression.py              # ACL 隔离回归,退出码 0=无泄漏
```

可选:`--top-k 6`、`--rerank`(开 cross-encoder 精排,+16G 显存)、`--rounds 2`(agentic 最大轮数)、`--limit N`(冒烟)。

## 去偏全流程(异厂裁判 + 扩语料 + 多跳)—— 推荐

`run_eval --judge deepseek` 是 DeepSeek 自判,有同厂循环偏差。**坐实数字**走这条:gold 与裁判都用 **Claude 子 agent**
(异厂、前沿、数据不出 Claude Code 信任域),DeepSeek **只当被测系统**。

```bash
python eval/index_eval_corpus.py                                      # 1. 扩语料:15 篇 parsed/ -> ~/rag_eval_big
PHAROS_EVAL_SRC=~/rag_eval_big PHAROS_EVAL_COLLECTION=evalbig python eval/dump_chunks.py   # 2. 采样 -> _units/(给 gold agent)
#  3. 编排者(Claude)起 Workflow:子 agent 读 _units/ 造 gold(单跳+单篇多跳+跨文档)-> eval/gold.jsonl
PHAROS_EVAL_SRC=~/rag_eval_big PHAROS_EVAL_COLLECTION=evalbig python eval/run_eval.py --mode both --judge none   # 4. 跑系统出 rows
python eval/dump_judge_units.py                                       # 5. rows -> _judge/(给裁判 agent)
#  6. 编排者起 Workflow:2 个独立 Claude 裁判 pass 判 faithfulness/correctness -> eval/verdicts.json
python eval/aggregate.py                                              # 7. 合并 + 按 hop 拆 + 双层归因
```

> 步骤 3/6 的 Workflow 脚本不在 repo(由编排者按需生成);其余脚本入库。这是 Claude Code 内的多 agent 编排,换环境需自备异厂裁判。

## 权威运行(2026-06-30,**R5 方法论订正后** / 15 篇 / gold 72[单跳38/单篇多跳29/跨文档5] / 全 context Claude 裁判)

> ⚠ **R5 自审订正(重要)**:早先报的"忠实度 0.83 / 17% 无据论断"是 **eval bug**——`dump_judge_units` 把喂裁判的
> context 截到 5000 字(实测中位 16k),裁判只看到约 40% 段落 → 依据落在后段的论断被误判 unfaithful。去掉截断、让裁判
> 看**全 context** 重判后,**真实忠实度 ≈ 1.0**。下表为订正值,并由 `aggregate.py`(含指纹校验、paired 归因)复现。

| 指标 | single | agentic | decompose |
|---|---|---|---|
| **忠实度** | **1.000** | **1.000** | 0.972 |
| 正确性 | 0.847 | 0.750 | 0.831 |
| 检索召回 | 0.854 | 0.840 | 0.852 |
| 全召回 | 0.792 | 0.792 | 0.792 |
| 引用召回 | 0.750 | 0.729 | 0.771 |
| 平均轮数 | 1.00 | 1.22 | 1.78 |

按 hop 正确性(AND):single-hop 0.97/0.87/0.92;单篇多跳 0.83/0.72/0.82;跨文档多跳 0.00/0.00/0.20(n=5)。
双层归因(paired,公共判过题集):single→agentic Δ**−0.097**(n=72);single→decompose Δ**−0.014**(n=71)。
> 全 context **双裁判 AND**(pass1+pass2 全 216;decompose 缺 1 判,judged 71/72)。`aggregate.py` 复现,含指纹校验。

**订正后三条发现:**

1. **忠实度 ≈ 1.0,grounding 契约几乎滴水不漏。** 早先"17% 幻觉"是 CTX_CAP 截断伪影、非真幻觉 —— **这推翻了 B1"17%
   无据论断是开放项"的结论**。系统宁可拒答也不编,该安全性质比之前认为的强得多。**教训:评估管道自身的 bug 能凭空造出一个"结论"。**
2. **正确性瓶颈 = 表格数字(已大幅缓解)+ 跨文档综合(仍硬)。** 表格数字由 ③ 修复到 single-hop 0.97;剩下的硬骨头是
   multi_cross(0.00,卡在综合不是检索,n=5 偏小)。
3. **agentic/decompose 净负(现配对、公共题集,更干净):** single 每个 hop 都 ≥ agentic(Δ−0.083);decompose 只在跨文档
   微弱占优(正确 0.20 vs 0、检索 0.37 vs 0.20),整体 Δ−0.014。**closed pipeline 应为默认。**

## ③ 表格/数值 grounding 修复(2026-06-30,commit 见 git log)

**诊断**(逐题实跟,推翻了"content_raw 没喂"的初判):21 个正确性失败里 14 个"召回到却答错",真根因是
**(a)** `search_with_context` 按 section 去重把图表/表格**资产块折叠进散文兄弟**;**(b)** `assemble_big` 的 big-block
用 `_gather` 只取 `el.text/caption`、**不含资产的 `asset_content`**。图表/表里的数(在 `chunk.content_raw`)既被去重丢、
又被组装排除 —— "召回到等于没召回"。

**两处修复**:`embedder/retrieve.py` 资产块(chart/table)不参与 section 去重(按 chunk_id 独立成键);
`generator/generate.py` 资产命中把 `content_raw` 补回喂给 LLM(big-block 缺它)。

**效果**(single,Claude 单裁判 pass1,apples-to-apples vs 修复前 pass1):

| | 修复前 | 修复后 |
|---|---|---|
| 正确性 | 0.704 | **0.819 (+0.115)** |
| └ single-hop | — | 0.97 |
| └ multi_intra | — | 0.76 |
| 检索召回(agentic) | 0.764 | 0.840 |

针对性 before/after:4 道已知表格题(光刻胶份额 / NAND 现货价 / FLAN-T5 对比分 / Token 消耗)全部从"信息不足"→答出正确数。

**忠实度 prompt 尝试 = 负结果,且问题本不存在**:曾收紧 grounding prompt 想收"17% 无据论断",实测反噬(−0.12 忠实/
−0.04 正确)已回退。**R5 自审后发现:那 17% 本身就是 CTX_CAP 截断伪影**(裁判没看全 context)——真实忠实度 ≈ 1.0,**根本没有
缺口要修**。双重教训:①先确认问题真实存在再动手(prompt 尝试白折腾);②评估管道自身的 bug 会凭空造出一个假"结论"。

## decompose 多跳实验(B2/B3,2026-06-30)—— 负结果,留档

针对 cross-doc 检索 0.20、multi_intra 偏弱,加了 `--mode decompose`:**拆子问题→各检索→并集→合成**(区别于 agentic 的"改写替换 query")。
3 路对比(R5 订正后、全 context 单裁判、72 题,`aggregate.py` 可复现):

| 正确性 | single | agentic | decompose |
|---|---|---|---|
| single-hop (38) | **0.97** | 0.87 | 0.92 |
| multi_intra (29) | **0.83** | 0.72 | 0.82 |
| multi_cross (5) | 0.00 | 0.00 | **0.20** |
| **总体** | **0.847** | 0.750 | 0.831 |
| 平均检索轮 | 1.0 | 1.22 | 1.78 |

**结论:单跳闭管道整体最好且最省**(Δ vs agentic −0.097、vs decompose −0.014,paired)。agentic 每个 hop 都 ≤ single(多检索=干扰块稀释);
decompose 仅在 cross-doc 微弱占优(正确 0.20 vs 0、检索 0.37 vs 0.20),整体仍不及 single。cross-doc 卡在**综合**(拿到两篇块也合不出对比),不是检索。
**这个 workload(单跳为主、表格密集)上 agent 编排净负;closed pipeline 应为默认。**(cross-doc n=5 偏小,绝对值仅参考。)

## gold 扩充:补表格题(2026-07-03,gold 72 → 88)

**动机**:原 72 题全部采样自散文块(gen_gold.py 当年显式 SKIP table——那时表格块 text 只有 caption,
问不出题)。实测后果:表格检索文本增强(chunker `2bd97a5`)把"数字埋在表里"的真实题类从错答修到答对,
但在纯散文 gold 上只显示成本(3 题冗余 gold 位移)不显示收益——**考卷偏科,表格向改动没有对称的验收标尺**。

**做法**(`gen_gold_tables.py`):对 kind=table 且 content_raw≥120 字的块定向采样(15 篇全覆盖,每篇≤2),
DeepSeek 据表体 HTML 出"必须由表内数值回答"的题;**程序化 QC:答案里的数值必须逐字在表体中出现**,
不满足直接丢弃(实跑拦下 2 条出题幻觉)。产出 16 题(中英混合)追加进 gold.jsonl,原 72 题备份
`gold_backup_prose72.jsonl`,表格题单独留档 `gold_tables.jsonl`(带 `asset:"table"` 标记可拆分统计)。

⚠ **口径断代**:88 题起的聚合数字与历史 72 题数字**不可直接对比**;新基线见 docs/TESTING.md §3。

## 诚实警告(读数前必看)

1. **裁判选择 + 全 context**:上表用 **Claude 子 agent 裁判、喂全量 context**(异厂去偏)。**教训(R5)**:早先给裁判截 5000 字
   → 忠实度被误压到 0.83,以为有"17% 幻觉";喂全 context 后 ≈ 1.0。忠实度裁判**必须看到生成器实际用的全部段落**。
   `run_eval --judge deepseek` 是 DeepSeek 自判的快速版(同厂循环偏差),只看相对趋势。
2. **gold 由 Claude 生成**:单跳"问题源自单 chunk"=golden 无歧义可程序化判引用;多跳由 Claude 跨 chunk/跨 doc 构造。
   跨文档多跳样本少(n=5),其绝对值仅供参考。
3. **语料 15 篇**(5 英文论文 + 4 英文财报 + 6 中文研报,表格密集以压数值瓶颈);结论是"这组语料上的表现"。
4. **gold/results/中间产物不入库**:含文档摘录(可能含研报私数据)且可重生成,已 gitignore;脚本入库,跑一遍即重建。
