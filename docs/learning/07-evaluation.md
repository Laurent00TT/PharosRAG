# 07 评估方法论 —— 本项目最大的差异化

> **本篇导读**
> 这篇讲 pharos 的评估闭环:合成 gold → 跑真系统 → LLM 裁判 → 归因,以及支撑它的去偏设计、五指标分层、可复现性分档与"评估管道自己也会出 bug"的完整实战故事。
> **面试权重:全套最高。**"你怎么评估 RAG"是必问题,绝大多数候选人只有一个词的答案(ragas),这篇给你一套能连续应对 20 分钟追问的体系。
> 前置阅读:[01 RAG 总览](01-rag-overview.md)、[05 生成与 grounding](05-generation-grounding.md);工程细节见 [eval/README.md](../../eval/README.md) 与 [docs/TESTING.md §3](../TESTING.md)。

---

## 1. 概念底座:为什么"评估 RAG"比"搭 RAG"更难

任何 RAG 系统跑起来只要一周,但回答"它到底行不行"可以卡住一个团队一个季度。难点有三层:

**第一层:没有考卷。** 传统 ML 有带标注的测试集;你自己的文档库没有。公共 benchmark(HotpotQA、MS MARCO、RAGBench)测的是别人的语料分布——你的用户问的是"2015 年 Netflix 总营收多少",不是维基百科琐事。所以第一件事是造考卷,而造考卷的三条路各有致命伤:

| 方案 | 优点 | 致命伤 |
|---|---|---|
| 人工标注 | 质量高 | 贵、慢,个人/小团队做不起 |
| 真实查询日志采样 | 分布真实 | 冷启动系统没有日志;答案仍要人标 |
| LLM 合成 gold | 全自动、可批量 | **考卷本身可能是错的**(出题幻觉),且有同源偏置 |

**第二层:端到端分数无法定位问题。** RAG 是多级流水线(解析 → 切块 → 检索 → 组装 → 生成 → 引用),"正确率 0.8"不告诉你剩下 0.2 坏在哪一级。工业界的答案是分层指标:检索层用 Recall/MRR 这类程序化指标,生成层用忠实度(faithfulness,答案是否被上下文支撑)和正确性(correctness,答案是否符合事实)——后两者没有程序化算法,只能请 LLM 当裁判(LLM-as-judge),ragas、TruLens 都是这个思路的封装。

**第三层(最容易被忽视):测量仪器自己会坏。** LLM 裁判引入三个新失效面:

1. **同厂循环偏差(self-preference bias)**:用 GPT 生成答案再用 GPT 打分,模型系统性偏爱自家风格的输出。学术界有大量实证,但工程实践里"gold、被测、裁判三个角色是不是同一家模型"极少有人检查。
2. **裁判输入偏差**:裁判看到的上下文若与生成器实际看到的不一致(截断、摘要、重构),忠实度分数测的就不是系统,而是这个偏差。
3. **考卷偏科**:考卷的题型分布若与改动的受益面不对称,eval 会系统性反对正确的改动(收益不可见,成本全暴露)。

pharos 的评估子系统就是围绕"考卷怎么来、裁判怎么去偏、仪器坏了怎么发现"三个问题构建的。它最有价值的产出不是那几个分数,而是一个真实案例:**评估管道自身的一个截断 bug,凭空制造了"系统有 17% 幻觉"的假结论,团队还基于假结论白做了一轮修复**——这个故事在第 4 节。

---

## 2. Pharos 怎么做

### 2.0 数据流总览

```
                    ┌── Tier1(仓内可复现,快) ──────────────────────────┐
                    │  run_eval --judge deepseek:DeepSeek 自判,同厂     │
                    │  循环偏差,只看相对趋势                              │
                    └───────────────────────────────────────────────────┘
 gen_gold.py ──┐
 gen_gold_tables.py ─┤→ gold.jsonl(88 题)
 dump_chunks.py → _units/ → Claude 子agent 造多跳 gold ──┘
                    ┌── Tier2(权威,双 Claude 异厂裁判)────────────────┐
                    │  run_eval --judge none → rows(answer+ctx_text)   │
                    │  dump_judge_units.py → _judge/judge_NN.json       │
                    │  2 个独立 Claude 裁判 pass → verdicts.json         │
                    │  aggregate.py:指纹校验 + 双裁判 AND + paired 归因  │
                    └───────────────────────────────────────────────────┘
 acl_regression.py:独立的安全回归轴(65 项断言,退出码 0 = 投产门槛)
```

### 2.1 去偏三角:gold 与裁判用 Claude,DeepSeek 只当被测系统

评估里有三个 LLM 角色:出题人、被测系统、裁判。三者同厂就是"自出题、自作答、自打分"。pharos 的拆法:

- **被测系统** = 生产同款 DeepSeek。关键细节:GEN_MODEL 默认回退到生产的 `PHAROS_LLM_MODEL`([eval/_common.py:44](../../eval/_common.py#L44))——"测的就是部署配置"不是口号,是代码默认值。
- **出题人与裁判** = Claude(异厂前沿模型)。`run_eval --judge none` 只算程序化指标,把 rows(answer + 喂进 LLM 的完整 ctx_text + golden)落盘([eval/run_eval.py:225-226](../../eval/run_eval.py#L225));[eval/dump_judge_units.py](../../eval/dump_judge_units.py) 把 rows 切成每批 8 条的判分单元文件,由 Claude Code 编排 2 个独立 Claude 裁判 pass 产出 verdicts.json,[eval/aggregate.py](../../eval/aggregate.py) 取**双裁判 AND**(两个 pass 都判 true 才算 true)。
- gold 侧同理:[eval/dump_chunks.py](../../eval/dump_chunks.py) 产出 `_units/` 单元文件给 Claude 子 agent 造 gold(单跳 / 单篇多跳 / 跨文档三类),chunk 原文只进单元文件、不进编排者上下文——研报私数据不出 Claude Code 信任域,不外发第三方 API。

**可复现性诚实分两档**([eval/README.md](../../eval/README.md) 迁移节):Tier1 = `--judge deepseek` 仓内一条命令可复现,但同厂自判只看趋势;Tier2 = 双 Claude 异厂 AND,是权威数字,但编排 Workflow 刻意不入库(README 明说步骤 3/6 由编排者按需生成)。不假装"权威数字人人可复现"——这本身就是方法论的一部分。

### 2.2 五指标分层:程序化三项零裁判 + 裁判两项互相制衡

程序化指标由 [eval/run_eval.py:179-181](../../eval/run_eval.py#L179) 与 [:200-211](../../eval/run_eval.py#L200) 直接算,免裁判、可高频回归。`agg()` 落 5 个字段,rollup 口径把 retrieval_recall / retrieval_full 并为『检索召回』一项、avg_rounds 作代价单列——对外 headline 即**程序化三项(检索召回 / MRR / 引用召回)+ 裁判两项 = 五指标**,与 README/OVERVIEW 同口径。下表是这 5 个原始字段:

| 指标 | 度量什么 | 怎么算 |
|---|---|---|
| retrieval_recall | 检索层:该捞的捞到没有 | golden chunks 被召回的比例(多 gold 按命中比例) |
| retrieval_full | 检索层:是否全召回 | golden 集 ⊆ 召回集 |
| MRR | 检索层:排得靠前吗 | 最佳命中名次的倒数 |
| citation_recall | **grounding 是否真实** | 答案 `[cite:n]` 经 [Generator._parse_citations](../../src/generator/generate.py#L117) 映射回 chunk_id 后命中 golden 的比例 |
| avg_rounds | 代价 | 平均检索轮数,与 Δ 正确性配对读 |

citation_recall 值得单独讲:它把"答案是否真的建立在证据上"变成**可数的结构化事实**——不是问 LLM"你觉得有依据吗",而是数引用标记。引用标记用 `[cite:n]` 而非裸 `[n]`([src/generator/prompt.py:15](../../src/generator/prompt.py#L15) 的唯一 CITE_RE),因为检索正文常含脚注/参考文献的裸 `[n]`,共用标记会被 LLM 照抄、解析误映射成假引用。

裁判指标两项,方向**刻意相反**([eval/run_eval.py:29-37](../../eval/run_eval.py#L29)):

- FAITH_SYS 规定:合理拒答("信息不足")= faithful=**true**;
- CORRECT_SYS 规定:拒答但参考答案有实质内容 = correct=**false**。

系统靠全拒答刷不了忠实度榜(正确性会崩),靠瞎编刷不了正确性榜(忠实度会崩)——两个指标像一把钳子,夹出"宁可拒答不编造"这个真实的安全性质。88 题基线里表格 16 题忠实度满分 1.000 的构成正是:4 题检索 miss **全部诚实拒答、零瞎编**(口径:88 题,DeepSeek 裁判,[docs/TESTING.md §3](../TESTING.md))。

### 2.3 合成 gold:单 chunk 出题 = golden 无歧义

[eval/gen_gold.py](../../eval/gen_gold.py) 的核心取舍:从索引 scroll 全部 payload(纯 CPU,不编码),按 doc 分组后 `pick_evenly` 等距采样(确定性、无随机种子,覆盖文档首中尾,[gen_gold.py:33-39](../../eval/gen_gold.py#L33)),每个 chunk 喂给 DeepSeek JSON 模式生成(问题, 答案)。**问题源自单 chunk ⇒ golden_chunk_id 天然无歧义 ⇒ retrieval/citation recall 可程序化判定,零人工标注。**

代价与缓解都写在明面上:

- 代价①:问题偏单跳事实型 → 多跳 gold 由 Claude 子 agent 跨 chunk/跨 doc 构造补盲区(`_units/` 的 doc_NN / type_N 单元)。
- 代价②:同源偏置(问题措辞与 golden chunk 词汇重叠,检索偏容易)→ SYS prompt 强制"不得出现'根据该段落'类指代、不得宽泛到该段答不全、像真实用户问法"([gen_gold.py:25-30](../../eval/gen_gold.py#L25));README"诚实警告"节直接告诉读数人 gold 是生成的。
- 过滤:SKIP_KINDS 跳过 table/image/figure 等资产块、MIN_CHARS=150 过滤碎块([gen_gold.py:21-22](../../eval/gen_gold.py#L21))。

**表格 gold 的程序化 QC 闸门**是这条线上最有讲头的机制。[eval/gen_gold_tables.py](../../eval/gen_gold_tables.py) 对 kind=table 且 content_raw≥120 字的块出"必须由表内数值回答"的题,`answer_grounded` 做程序化验收([gen_gold_tables.py:48-53](../../eval/gen_gold_tables.py#L48)):答案里至少一个 ≥2 位数值(去逗号/空格规范化后)必须逐字出现在表体里,否则整条丢弃([:99-101](../../eval/gen_gold_tables.py#L99))——**"出题人自己编的数,不配进考卷"**。实跑拦下 2 条出题幻觉。LLM 出题人读大表会看错行列甚至编数,合成 gold 的最大风险是"考卷本身错",数值逐字匹配是最便宜的机械闸门(它的盲区见第 4 节延期项 eval#2)。

### 2.4 双层归因:single vs agentic vs decompose 的 paired 设计

这是"agentic RAG 到底值不值"的实验装置。三种模式([eval/run_eval.py:78-161](../../eval/run_eval.py#L78)):

- **single**:闭管道单跳,走生产 Generator——组件层基线;
- **agentic**:DeepSeek 判断上下文够不够(SUFFICIENCY_SYS),不够就**改写 query 替换重搜**,累积上下文再生成;
- **decompose**:先拆 1-4 个子问题(DECOMPOSE_SYS,跨文档对比必须每对象一个子问题),各自检索取**并集**(cap 14 防撑爆)再合成——区别于 agentic 的"改写替换会窄化丢另一跳"。

归因在 [eval/aggregate.py:114-128](../../eval/aggregate.py#L114) 做 **paired**:只在"两模式都被双裁判判过的公共题集"上相减——分母相同才可减,这是被踩过坑之后的纪律(见第 4 节)。按 hop(single / multi_intra / multi_cross)拆表。还有一个容易漏的口径细节:agentic/decompose 的 union_ids 逐轮 append 含重复,必须**保序去重**([run_eval.py:54-60](../../eval/run_eval.py#L54) 的 `_dedup`)后,MRR/best_rank 才与 single 的有序 hits 同口径。

结果(口径:72 题,Tier2 双 Claude 裁判,paired):single→agentic Δ**−0.097**(n=72),single→decompose Δ**−0.014**(n=71);agentic 每个 hop 都 ≤ single(多检索 = 干扰块稀释);decompose 仅跨文档微弱占优(正确 0.20 vs 0、检索 0.37 vs 0.20,n=5 偏小)。结论:**该 workload 上 closed pipeline 应为默认——agent 编排净负,负结果照发。**(这个 Δ 的幅度有一个已确认的测量瑕疵,见第 4 节 eval#0——诚实标注也是结论的一部分。)

### 2.5 指纹闸门 + 缺判 loud:结果与判分的对齐纪律

Tier2 流程里 results 与 verdicts 是分两步产出的文件,对齐靠行序——重跑 results 而没重判,第 i 行就可能已是另一道题,分数照出、全部张冠李戴,**而且没有任何报错**。pharos 的答案是把对齐做成机械强制:

- [dump_judge_units.py:44](../../eval/dump_judge_units.py#L44) 写 `_judge/fingerprint.json`,内容是 `{id: sha1(query)[:12]}`;
- [aggregate.py:67-71](../../eval/aggregate.py#L67) 消费 verdicts 前逐行比对 results 的 query 指纹,不一致直接 `SystemExit` **拒绝出数**。

缺判处理同样 loud 而非静默:[aggregate.py:45-50](../../eval/aggregate.py#L45) 的 `andflag` 规定任一 pass 未判/字段缺 → None(排除出分母),绝不做 `bool(None)=False` 的静默判假;每模式/每 hop 打印 n_judged,判空标 n/a 不混 nan。所有结论都建立在 eval 正确之上,**对齐必须是机械强制,不能靠人肉自觉**。

### 2.6 忠实度裁判的全 context 契约:裁判输入 = 生成器输入的原文

run_eval 把喂给 LLM 的 user message **原文**(PromptBuilder 组装的编号 passage 块 + 问题)直接存进 rows.ctx_text([run_eval.py:103](../../eval/run_eval.py#L103):`next(m.content for m in ans.raw_messages if m.role == "user")`),dump_judge_units 原样交给裁判——裁判看到的与生成器看到的**逐字节相同**,零重构偏差。[dump_judge_units.py:21](../../eval/dump_judge_units.py#L21) 的 `CTX_CAP = 200000` 实为"不截断"(ctx_text 中位 ~16k),仅防病态超长。

这一行为什么是整个子系统最重要的方法论,第 4 节的 0.83 假结论故事会给出血的证明。

### 2.7 ACL 回归的三层对抗设计:65 项断言

[eval/acl_regression.py](../../eval/acl_regression.py) 是独立的安全评估轴。demo 库全 public 测不出隔离,所以脚本用真 Chunker+Embedder 现建 2 租户 4 文档合成库,**每篇带独有 sentinel 原文**([acl_regression.py:29-42](../../eval/acl_regression.py#L29)),65 项断言分五节,层层反"假绿":

1. **召回隔离矩阵**(5 身份 × 4 sentinel 精确查询,[:82-90](../../eval/acl_regression.py#L82)):拿原文当 query 意味着哪怕字面精确匹配也必须 0 召回——证明 ACL 硬过滤**先于相关性**,而非"搜不到"侥幸。
2. **授权正向**([:93-98](../../eval/acl_regression.py#L93)):有权身份必须搜得到,防"全拒假过"——一个把所有请求都拒掉的系统能通过任何隔离测试。
3. **get_document 直读 fail-closed**:无权 → PermissionError。
4. **expand 跨 ACL**:拿 docA 真 chunk_id 以无权身份 expand → None。
5. **禁出口闸**([:123-138](../../eval/acl_regression.py#L123)):monkeypatch [store.acl_admits](../../src/embedder/acl.py#L29) 恒 True,**废掉出口复核后重跑全矩阵仍须 0 泄漏**——证明 RRF fusion prefetch 级下推本身就挡住越权,而非靠出口闸兜底掩盖回归。"嵌入式 QdrantLocal 的 fusion 丢顶层 should 过滤"是真踩过的坑;防御纵深会掩盖内层回归,回归测试要能**关掉外层证内层**。

退出码 0 = 无泄漏,可直接当投产前门槛。(历史文档 REVIEW_PLAN.md 记"44 项",那是禁出口闸新增前的计数,现行代码为 65 项。)ACL 设计本身见 [04 ACL 与安全](04-acl-security.md)。

### 2.8 两个工程细节:锁规避与 eval/产品同源

**copy_demo**:嵌入式 Qdrant 是单 client 独占锁,live 守护进程/MCP server 正持有 ~/rag_demo 锁时,评估脚本直开同目录会崩 "already accessed by another instance"。[_common.copy_demo](../../eval/_common.py#L49) 统一先 copytree 到临时目录再开——不抢锁、不污染原库,评估随时可跑,不要求用户先拆生产环境。

**--smart-tables 与产品同源**:eval 的失败驱动表格补检([run_eval.py:78-104](../../eval/run_eval.py#L78))与生产 smart-ask([src/pharos/service.py:336-353](../../src/pharos/service.py#L336))复用同一套信号函数——[generator/signals.py](../../src/generator/signals.py) 的 `looks_numeric` / `is_refusal` / `DEFAULT_TABLE_LEG` 单一来源,防两处词表漂移。**eval 与产品必须测同一策略同一参数,否则 eval 数字对产品无预测力**;代码注释里钉死"前置腿版本已被 88 题实测否决,勿改回"。

---

## 3. 为什么这么设计:被否决的备选

**备选 A:直接用 ragas / 现成评估框架。** 否决理由有三:① ragas 默认单模型裁判,同厂循环偏差没有解;② 它测不了本项目特有的资产(citation_recall 依赖自家 `[cite:n]` 结构化引用协议、ACL 隔离依赖自家权限模型);③ 私有研报数据要外发第三方 API。保留其思想(faithfulness/correctness 双裁判指标),自建管道。

**备选 B:同厂自判(DeepSeek 全包)。** 没有完全否决,而是降级保留为 **Tier1**([run_eval.py:225-226](../../eval/run_eval.py#L225) 的 `--judge deepseek`):快速回归看相对趋势可以,坐实绝对数字不行。这个分档本身是设计——比"假装没有循环偏差"和"每次都跑昂贵的异厂双裁判"都诚实。

**备选 C:人工标注 / 真实查询日志。** 太贵 / 个人系统无日志积累,都被"全自动 + 可程序化判定"压倒。代价(单跳偏置、同源偏置)靠缓解手段加明文警告来管,而不是假装它不存在。

**备选 D:表格补检做成前置腿(每题都带)。** 四轮 88 题实验否决(口径:88 题,DeepSeek 裁判,[TESTING.md §3](../TESTING.md) smart-ask 实录):

| 版本 | 表格16 | 散文72 | 忠实度 | 裁决 |
|---|---|---|---|---|
| 基线 | 0.625 | 0.861 | 0.977 | — |
| ① 前置腿 | 0.875 | **0.792** | 0.966 | 否决:相近数值误伤 5 道原本答对的散文题 |
| ② 失败驱动,rerank_top_n=30 | 0.750 | 0.847 | 1.000 | 假象:对的块在粗排 31-50 名,精排池装不进,腿形同虚设 |
| ③ top_n=50,无条件采用重试 | 0.625 | 0.833 | **0.932** | 否决:部分回答夹带"未提供 X"的错误缺失声明(X 就在 context 里) |
| ④ 失败驱动 + 择优采用 | 0.688 | 0.833 | 0.977 | **采纳**:弃用路径 4 题零损失,采用路径 1 题 ✗→✓ |

沉淀的四条方法论(每条都来自一轮否决):智能只作用于失败路径;精排池深度 ≥ 对的块的粗排最差名次;"全拒答变部分回答"会引入新失败面;±2 题噪声底噪下横向比较必须配对归因(rows 的 retried/retry_kept 标记就是为此内建的)。

**备选 E:新旧 gold 混着比。** gold 72→88 后强制**口径断代**:原 72 题备份 `gold_backup_prose72.jsonl`([gen_gold_tables.py:110-111](../../eval/gen_gold_tables.py#L110) 只备份一次),表格题带 `asset:"table"` 标记单独留档可拆分统计,TESTING 立新基线。考卷变了就重新画基线,88 题的 0.818 与 72 题的 0.847 **不可相减**——它们连裁判都不同(前者 DeepSeek Tier1,后者 Claude Tier2)。这个纪律看似琐碎,却是"数字可信"的地基:任何跨口径的减法都在制造假结论。

为什么要补表格题?因为踩过"考卷偏科制造测量盲区"的坑:表格块检索文本增强(chunker `2bd97a5`)把"数字埋在表里"的真实题类从错答修到答对,但 72 题全散文 gold 上只显示成本(检索召回 −2.1pp,3 题冗余 gold 位移)不显示收益(0 题上升)——逐题诊断确认 3 题位移全是双 gold 题的冗余 gold 被挤掉、答案全部仍对,才裁决保留增强,随后补 16 道表格题让考卷与改动对称。**改动的验收标尺必须覆盖改动的受益题型,否则 eval 会系统性反对正确的改动。**

---

## 4. 实战复盘

### 4.1 忠实度 0.83 假结论:评估管道自己的 bug 凭空造出"17% 幻觉"(R5.H1)

这是全项目最值得复述的故事,完整链条:

1. **假象**:权威运行报出忠实度 0.83,即"17% 论断无据",被列为开放项 B1。问题看起来真实且顽固。
2. **基于假结论的白折腾**:团队收紧 grounding prompt(全称句级约束"EVERY sentence MUST be supported"),实测**反噬**——忠实度 −0.12、正确性 −0.04,回退([src/generator/prompt.py:40-42](../../src/generator/prompt.py#L40) 注释留痕)。
3. **对仪器的对抗审查**:R5 评审把 eval 方法论自身当靶子。发现 dump_judge_units 有 `CTX_CAP=5000` 的**头截断**,而 ctx_text 实测中位 16k——裁判只看到约 40% 的段落。
4. **坐实**:逐条复核 16 个 unfaithful 判例,12 个(75%)的被引证据恰好落在被截掉的后段——"依据在后段"的论断被系统性误判 unfaithful。
5. **修复与重判**:去掉截断(CTX_CAP 设 200000 仅防病态超长),双裁判看全 context 重判全部 216 项(72 题 × 3 模式):真实忠实度 **≈1.0**(single/agentic 1.000、decompose 0.972)。B1 结论被推翻——系统宁可拒答也不编,安全性质比以为的强得多;之前的 prompt 收紧是在修一个**不存在的问题**。

沉淀两条铁律:①先确认问题真实存在再动手;②忠实度裁判必须看到生成器实际用的全部段落——这就是 2.6 节"ctx_text 直接存 user message 原文"契约的由来。面试一句话:**修系统之前,先怀疑测量仪器。**

### 4.2 归因数字复现不出来:aggregate 的 paired 重构(R5.H2/M1)

README 里的跨文档/decompose 数字用仓里提交的 aggregate.py 复现不出:旧版硬编码 single/agentic 两模式、读盘上残留的旧 verdicts(agentic 缺 8 条跨文档判分)、双层归因用**不同分母**直接相减——cross-doc 打出 nan,结论悬空。重构后:参数化 MODES、paired 公共题集、缺判 loud、sha1 指纹闸门(2.5 节),Δ−0.097/−0.014 可由 aggregate.py 一键复现;后补 pass2 双裁判 AND 复核全 216 项,数字与单裁判一致(agentic 0.764→0.750 略严)。教训:**发布的每个数字都要能从提交的代码 + 数据重放出来。**

### 4.3 .env 时序 bug:配置静默失效比崩溃更危险

eval 各脚本依赖 _common 的模块级常量(EVAL_SRC/GEN_MODEL/JUDGE_MODEL)读 `PHAROS_EVAL_*`;对抗总审发现 `.env` 的加载调用写在常量定义**之后**——.env 里配的评估库/模型全部静默落回默认值,脚本拿错误的库和模型跑出"看起来正常"的数字。修复:load_env() 提到常量之前([_common.py:35-36](../../eval/_common.py#L35) 有"评审修"注释留痕),setdefault 语义保证显式环境变量仍优先。这类 bug 能跑通,却测错了对象;评估基础设施对它必须零容忍。

### 4.4 本轮对抗审查:1 项已修,3 项确认但延期

写这套文档前,对 eval 子系统又做了一轮对抗审查(每条发现先派独立验证员试图反驳,反驳失败才算 confirmed)。结果 4 条 confirmed,处置分流本身是教学点——**"确认了但不能马上改"是工程判断,不是拖延**:

**已修 eval#1:dump_chunks 锁规避对齐**(fixes_applied.md #17)。症状:dump_chunks.py 直开原库不拷贝,与其余 eval 脚本的 copy_demo 策略不一致——_common.py 自述"统一 copytree"却有例外,目标库被 daemon 持锁时该脚本会崩,运行期间还反向独占锁阻塞 daemon。根因:纯遗漏。修法:改走 copy_demo 临时副本,与 gen_gold 同款([dump_chunks.py:42-44](../../eval/dump_chunks.py#L42) 现状)。为什么能马上修:CPU-only scroll、不改变任何评估输出内容、拷贝成本已被 gen_gold_tables 证明可接受。

**延期 eval#0(medium):三路对比对 agentic/decompose 系统性不利。** agentic/decompose 的 context 组装绕过生产 Generator,只取 `ctx.text or hit.text`([run_eval.py:115/148](../../eval/run_eval.py#L115)),缺两个已上线的修复:③ 资产 content_raw 补回([generate.py:75-80](../../src/generator/generate.py#L75))与 § section_path 面包屑([generate.py:85-89](../../src/generator/generate.py#L85))。后果:88 题口径下(16 道表格题)表格块 hit.text 只有 caption+检索信号、数据在 content_raw,agentic 路径"召回到也答不出"——**Δ(agentic−single) 被高估为 agent 编排的锅,部分其实是评测实现少装了两个修复**;验证员甚至确认 asset_no_prose 的表格块会因 text 为空被整体跳过,比声称的更严重。为什么延期:修法(抽公共 `build_context_entry` 三处复用)本身是 CPU 可测的小改,但**修完必须 GPU 复跑 88 题三路对比、更新已发布的 Δ 结论**——改数字的事不能和写文档混在一轮做。文档处置:凡引用 Δ−0.097 处诚实标注该瑕疵(本篇亦然)。

**延期 eval#2(medium):表格 gold 的 QC 闸门防"编造数"不防"错配数"。** [answer_grounded](../../eval/gen_gold_tables.py#L48) 只要求答案里任一 ≥2 位数值在表体出现;出题人读错行列(把 A 分部 2023 的数答给 B 分部 2024)产生的错数**必然也在表内**,照样过闸。16 题小样本下 1-2 条错 gold 就是 6-12 个百分点的正确性系统性低估。为什么延期:只影响未来的 gold 重生成,而重生成 = 又一次口径断代;正确顺序是先用"答案回读"(独立再调一次 LLM 据表作答、比对数值)**回溯审计现有 16 题**,把假设性污染变成实测结论,再决定是否重生成。

**延期 eval#3(low):gen_gold_tables 的 cap 按 doc_id 字典序吃满。** [gen_gold_tables.py:79-85](../../eval/gen_gold_tables.py#L79) 循环内 `len(rows)>=cap` 即 break,30 候选竞争 16 名额时字典序靠前的英文文档优先占满——验证员实测:6 篇中文研报只有 1 篇进了考卷,README"15 篇全覆盖"的陈述实际为假(那次实跑的偶然结果)。跨语言表格弱点(已知残留缺口)可能恰好不在考卷上。修法(轮转采样:每篇先取 1 道再取第 2 道)同样绑定 gold 重生成,与 eval#2 一并落地。

注意 eval#2/#3 的共性:**它们是"考卷生成器"的 bug,不是"系统"的 bug**——这一类问题恰恰是 4.1 节教训的延伸:合成评估里,出题、判分、聚合的每一环都是测量仪器,都要被对抗审查。

---

## 5. 面试怎么讲

### 30 秒版(电梯稿)

"我没有用 ragas,而是自建了一套去偏评估闭环:gold 生成和裁判用 Claude,被测系统用生产同款 DeepSeek——三个角色异厂,消掉 LLM 自评的循环偏差。指标分两层:检索召回/MRR/引用召回是程序化的,零裁判成本;忠实度和正确性用双裁判取 AND,且两个指标方向相反——合理拒答算忠实但算错,系统刷不了任何单边的分。这套管道抓到过一个很有代表性的 bug:裁判的 context 被截断到 40%,凭空造出'17% 幻觉'的假结论,我们还基于它白修了一轮 prompt——所以我的核心观点是,评估管道本身也是要被评估的系统。"

### 3 分钟版(结构化展开)

1. **考卷怎么来**(40s):单 chunk 出题让 golden_chunk_id 无歧义,检索/引用召回可程序化判定,零人工标注;代价是单跳偏置和同源偏置,用 Claude 子 agent 造多跳题、prompt 禁指代来缓解。表格题有程序化 QC:答案数值必须逐字在表体出现,实跑拦下 2 条出题幻觉——"出题人自己编的数,不配进考卷"。
2. **裁判怎么去偏**(40s):三角色异厂 + 双裁判 AND + 一条最重要的契约——裁判看到的 context 与生成器喂给 LLM 的 user message 逐字节相同,不重构不摘要。可复现性诚实分档:Tier1 同厂自判看趋势,Tier2 异厂双裁判坐实数字。
3. **数据点一:假结论故事**(50s):CTX_CAP=5000 头截断 → 裁判只见中位 40% 段落 → 忠实度假报 0.83 → 基于假结论收紧 prompt 反噬 −0.12 → 对抗审查发现截断、逐条复核 16 个 unfaithful 判例 75% 命中"被引段落被截掉" → 全 context 重判 216 项,真实忠实度 ≈1.0。修系统之前先怀疑测量仪器。
4. **数据点二:敢发布负结果**(30s):paired 双层归因测 agent 编排价值,single→agentic Δ−0.097、single→decompose Δ−0.014(72 题公共判过题集)——"这个 workload 上 closed pipeline 应为默认"是拿数据说的,不是立场。同时主动承认这个 Δ 有已确认的测量瑕疵(agentic 路径少装两个生产侧修复,Δ 幅度偏大),已列入待复跑。
5. **收尾**(20s):还有一条独立的安全评估轴——ACL 回归 65 项断言,包括 monkeypatch 废掉出口复核后仍须 0 泄漏,证明下推过滤本身挡住越权而非兜底假绿。评估、安全回归、产品决策(smart-ask 四轮实验)共用这一套基础设施。

---

## 6. 追问预演

**Q1:为什么不用 ragas?**
要点:不是没听过,是三个具体理由——单模型裁判无解同厂偏差;citation_recall/ACL 这类自有资产它测不了;私数据不外发。补一句"faithfulness/correctness 的指标思想我保留了,替换的是实现"。关键词:self-preference bias、结构化引用协议、数据边界。

**Q2:gold 是 LLM 生成的,考卷本身错了怎么办?**
要点:分层承认+分层设防。散文题:单 chunk 出题 + prompt 约束 + README 明文警告;表格题:answer_grounded 数值逐字闸门,实跑拦下 2 条。然后**主动**说闸门盲区:防编造不防错配(读错行列的数也在表内),已确认、修法是答案回读,绑定下次 gold 重生成——能主动讲出自己 QC 的盲区,比"我们有 QC"高一档。

**Q3:忠实度 ≈1.0,是不是裁判太松?**
要点:三个反证。①双裁判 AND 只会更严不会更松;②同一套裁判在 CTX_CAP bug 期间判出过 0.83,说明它判得动 unfaithful;③正确性同期只有 0.85 上下,两指标方向相反,若裁判整体放水两个都该虚高。再补机制解释:闭管道零召回时确定性返回"信息不足"不经 LLM,拒答算忠实是指标定义使然。

**Q4:Δ−0.097 说 agent 没用,会不会是你的 agentic 实现太弱?**
要点:先认——部分成立且已量化确认(eval#0:agentic 路径缺 content_raw 补回与面包屑,对表格题系统性不利,Δ 幅度被高估)。再守——方向大概率不翻:72 题散文 gold 时代(资产影响小)agentic 就每个 hop 都 ≤ single,机制解释(多轮检索 = 干扰块稀释、改写替换丢跳)独立于该瑕疵;decompose 在跨文档确实占优,说明实验装置分得出细粒度差异。关键词:paired、公共题集、待 GPU 复跑更新。

**Q5:88 题基线和 72 题权威运行,数字为什么不一致?**
要点:口径断代——考卷不同(72 全散文 vs 88 含 16 表格题)且裁判不同(Tier2 Claude 双裁判 vs Tier1 DeepSeek 自判),任何跨口径减法都是制造假结论。展示你对"基线管理"的意识:原卷备份、asset 标记拆分统计、TESTING 立新基线。

**Q6:双裁判 AND 会不会把分数压得虚低?**
要点:实测差异很小(agentic 单裁判 0.764 → AND 0.750),AND 买到的是"单裁判抽风不污染结论";缺判按 None 排除分母而非判假,配 n_judged loud 打印。可反问式延伸:更贵的做法是三裁判多数决,在这个体量上性价比不如 AND。

**Q7:这套东西换一家公司还能用吗?**
要点:分可迁移的方法论(三角色异厂、裁判全 context 契约、paired 归因、口径断代、指纹闸门、"考卷与改动对称")和绑定本项目的实现(Claude Code 编排、[cite:n] 协议)。方法论清单本身就是答案——它们全部不依赖任何具体模型或框架。

**Q8:单轮 eval 的噪声多大?你怎么处理?**
要点:实测底噪 ±2 题(88 题上 ≈2pp)——五轮实验里有 2 道题反复横跳。所以这个粒度的横向比较一律配对归因:rows 内建 retried/retry_kept 标记,smart-ask 每轮都做未触发面配对检查(81 题与基线仅 2 题翻转,恰为已知不稳定题)。关键词:配对消噪、失败面切片,而非只报均值。

---

## 7. 动手实验

两个实验都是**纯标准库、CPU 即可**,在仓根跑(Windows Git Bash / WSL 均可;产物均已 gitignore,做完删除)。

### 实验一:CTX_CAP 截断伪影重现——0.83 假结论是怎么造出来的

先构造一条"证据在 context 尾部"的 results 行,再看裁判单元文件里证据是否幸存:

```bash
cd eval
python - <<'PY'
import json
ctx = "PAD。" * 2000 + " 依据:2015 总营收 $6,779,511 千美元"
rows = [{"query": "总营收?", "hop": "single", "golden_answer": "$6,779,511",
         "answer": "$6,779,511 [cite:1]", "ctx_text": ctx,
         "retrieval_hit_frac": 1, "retrieval_full": True, "rank": 1,
         "citation_recall": 1, "n_citations": 1, "n_rounds": 1}]
json.dump({"agg": {}, "rows": rows}, open("results_single.json", "w", encoding="utf-8"))
PY
python dump_judge_units.py single
python - <<'PY'
import json
i = json.load(open("_judge/judge_00.json", encoding="utf-8"))["items"][0]
print(len(i["contexts"]), "依据" in i["contexts"])
PY
```

预期输出 `8027 True`——裁判能看到依据。然后把 [dump_judge_units.py:21](../../eval/dump_judge_units.py#L21) 的 `CTX_CAP` 临时改成 5000,重跑上面后两步:输出变为 `5000 False`——支撑答案的证据被头截断吃掉,裁判**必然**判 unfaithful。这正是 R5.H1 假报"17% 幻觉"的机制:证据位置越靠后越冤。做完把 CTX_CAP 改回 200000。

### 实验二:指纹闸门演练——亲手触发"重跑未重判,拒绝出数"

```bash
cd eval
python - <<'PY'
import json, os
os.makedirs("_judge", exist_ok=True)
rows = [{"query": "Q1", "hop": "single", "golden_answer": "A", "answer": "B [cite:1]",
         "ctx_text": "C", "retrieval_hit_frac": 1.0, "retrieval_full": True, "rank": 1,
         "citation_recall": 1.0, "n_citations": 1, "n_rounds": 1}]
json.dump({"agg": {}, "rows": rows}, open("results_single.json", "w", encoding="utf-8"))
json.dump({"pass1": [{"id": "single#000", "faithful": True, "correct": True}],
           "pass2": [{"id": "single#000", "faithful": True, "correct": True}]},
          open("verdicts.json", "w", encoding="utf-8"))
json.dump({"single#000": "deadbeef0000"}, open("_judge/fingerprint.json", "w", encoding="utf-8"))
PY
python aggregate.py        # 应 SystemExit:"对齐失败……拒绝出数"
python - <<'PY'
import json, hashlib
json.dump({"single#000": hashlib.sha1("Q1".encode()).hexdigest()[:12]},
          open("_judge/fingerprint.json", "w", encoding="utf-8"))
PY
python aggregate.py        # 指纹修正后:打印 single 报表(n=1, judged=1, 正确性 1.000)
```

第一次跑触发 [aggregate.py:71](../../eval/aggregate.py#L71) 的拒绝出数——体感一下"防张冠李戴是机械强制的,不靠人自觉"。做完删掉 `eval/` 下的 `results_single.json`、`verdicts.json`、`_judge/`。

### GPU/WSL 附加(前置:WSL + conda pharos + 4090,先 `sudo systemctl stop pharos`)

- 全量 ACL 回归:`python eval/acl_regression.py; echo exit=$?` ——五节 65 项全 PASS、退出码 0,第五节现场演示"废掉出口闸仍 0 泄漏"。
- Tier1 冒烟:`python eval/run_eval.py --mode both --judge deepseek --limit 5`(需 .env 有 DEEPSEEK_API_KEY;无 gold 先跑 `python eval/gen_gold.py --per-doc 6`)——逐题打印五指标,结束输出双层归因 Δ,rows 里能看到 ctx_text 与 retried/retry_kept 字段。

---

## 8. 诚实边界

面试中被问到"这套评估的弱点"时,主动交出这份清单比被动挨问强:

1. **合成 gold 的同源偏置未量化。** 问题措辞与 golden chunk 词汇重叠,检索侧分数偏乐观;缓解手段(prompt 禁指代、多跳补题)有,但"偏乐观多少"没有对照实验。话术:"我知道方向,不知道幅度——量化它需要一批人工改写的 paraphrase 对照卷,在 backlog 里。"
2. **表格 QC 闸门防编造不防错配**(eval#2,已确认待修)。16 题小样本下 1-2 条错 gold 即 6-12pp 波动;正确的下一步是答案回读式回溯审计,而不是急着重生成考卷再断代一次。
3. **表格题覆盖面结构性偏斜**(eval#3,已确认待修)。字典序吃满导致中文研报 6 篇只进 1 题,跨语言表格弱点可能恰好不在考卷上——README 的"15 篇全覆盖"是那次实跑的偶然结果。
4. **三路对比对 agentic 系统性不利**(eval#0,已确认待复跑)。Δ−0.097 的方向有多重独立证据,但幅度被高估;引用这个数字时必须带此标注。
5. **Tier2 不在仓内可复现。** 双 Claude 裁判依赖 Claude Code 多 agent 编排,Workflow 不入库;换环境需自备异厂裁判。这是"数据不出信任域"换来的代价,分档诚实标注,不掩饰。
6. **样本量与语料边界。** 跨文档多跳 n=5,其绝对值仅供参考;整套结论是"这 15 篇语料(5 英文论文 + 4 英文财报 + 6 中文研报)上的表现",不外推。单轮噪声底噪 ±2 题。
7. **裁判指标未做人工抽检对齐。** 双裁判 AND 防单裁判抽风,但"Claude 裁判 vs 人类判断"的一致率没测过——严格说,忠实度 ≈1.0 是"两个独立 Claude pass 都认为无幻觉",不是人工核验结论(CTX_CAP 复盘时逐条人工复核过 16 个 unfaithful 判例,是最接近的一次)。
8. **文档与代码有几处已知漂移。** eval/README 一处节标题写"四个指标"(实际程序化 5 字段 + 裁判 2 字段)、历史评审文档记 ACL 断言 44 项(现 65)——以代码为准,本篇锚点均按当前代码核验。

---

*系列导航:[06 Agentic 与 MCP](06-agentic-mcp.md) ← 本篇 → [08 服务架构](08-service-architecture.md);方法论故事集见 [10 方法论故事](10-methodology-stories.md),面试速查见 [11 面试 QA](11-interview-qa.md)。*
