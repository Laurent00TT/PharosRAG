# embedder 检索质量评估

> 2026-06-28。本文记录 embedder 组件"质量封板"的评估:过程、方法论、结果与决策。
> 可复现脚本见 [`eval/component_retrieval/`](../../../eval/component_retrieval/);设计背景见 [DESIGN.md](DESIGN.md)。

## 1. 目的

chunker 有数据驱动的切块评估(`eval_chunks.py` + MMDocIR ground-truth F1)。embedder 此前只完成**结构/安全封板**(对抗 review 抓 13 个 bug),缺**检索质量的数据验证**。本评估补这块欠账,用真实语料构造检索基准,把 3 个待验证项用数据钉死:

1. **sparse 选型**:BM25 vs BGE-M3-sparse,哪个该进 hybrid。
2. **RRF 权重**:dense 与 sparse 融合的最优比例 + hybrid 是否真的优于单路。
3. **est_tokens 重标**:chunker 的 char/token 启发式(中 1.7、英 4.0)是否该用 Qwen3-VL 真 tokenizer 重标。

## 2. 方法论

### 2.1 评估基准

- **语料**:从 `parsed/` 选 14 文档 → chunker 切出 **1067 chunk**。覆盖 academic_paper / law / financial_report_en / **financial_research_zh** / government,中英混(中文 109 chunk,补 chunker 评估的中文盲区)。
- **两类查询**(核心设计):

  | 类型 | 数量 | 构造 | golden | 测什么 |
  |---|---|---|---|---|
  | **精确词** | 25 | 程序化挖语料里 `df==1` 的稀有精确串(`Section 6.1`/`CLEF-2021`/`$1196`/长编号) | 含该串的唯一 chunk | sparse 一字不差命中 |
  | **语义** | 31 | agent 读 chunk 生成自然问题,**刻意避开原词**(同义/概括表达) | 被改写的 chunk | 语义召回(换种说法也能召回) |

- **为什么必须两类**(防评估偏置):只测精确词,BM25 几乎注定赢(它就是 term match);只测语义,dense 注定赢。两类分别对应 sparse / dense 的设计目的,合起来才公平。语义查询若照搬原词就退化成精确匹配,故 agent prompt 强制"避开专有名词/数字/关键短语"。

### 2.2 指标

- **MRR**(Mean Reciprocal Rank):golden 排名倒数的均值,排得越靠前越高。
- **Recall@10**:golden 落在 top-10 的比例。
- 召回 `limit=50`,在其中找 golden 排名。

### 2.3 三向量同库对比

一个 Qdrant collection 放三种向量,贴近生产实现来对比:

| 向量 | 模型 | 相似度 |
|---|---|---|
| `dense` | Qwen3-VL-Embedding-8B(MRL 1024) | COSINE |
| `bm25` | 我们的 `doc_sparse`(jieba+正则) | Qdrant `Modifier.IDF`(算 BM25) |
| `bgem3` | BGE-M3 `lexical_weights` | dot(权重已含重要性) |

`#1` 只比 sparse 单路(dense 是同一个,不影响 sparse 对比),但同时列出 dense 作参照。

## 3. 过程

### 3.1 流程

```
build_corpus_chunks.py  → chunks.jsonl(1067) + queries_exact.jsonl(25 程序化)
select_semantic.py      → candidates.jsonl(47 候选)
[workflow: 4 agent]     → queries_semantic.jsonl(31,agent 生成、避原词)
index_eval.py           → index 1067 chunk 到 Qdrant(dense+bm25+bgem3)
eval.py                 → #1 单路召回 + #2 RRF 权重扫描
retok.py                → #3 tokenizer 重标(独立流程:全量 2927 chunk 过 Qwen3-VL tokenizer 测 char/token)
```

### 3.2 过程中的工程坑(诊断记录)

跑 index 时连续遭遇假死,**完整走了诊断→修复→验证**,没有盲目重跑:

- **表象**:进程假死(GPU util 0%、CPU 时间不增长、卡 `wchan=pipe_write`),后续重跑又见 `exit 9`、`RuntimeError: ... freeze_support()`。
- **诊断手段**:`nvidia-smi util` + `ps -o time` 多次采样(确认 blocked 非计算)→ `cat /proc/<pid>/wchan`(等待点)→ `pgrep -P`(子进程,发现 defunct + resource_tracker)→ 读日志(发现 **dense 进度出现两次** = 脚本被跑了两遍)。
- **根因**:BGE-M3 检测到双 GPU(4090+5070)自动起 multiprocessing pool,spawn 子进程 **re-import 脚本当 `__main__`**(脚本无 `if __name__` 保护)→ 整脚本重跑 + 嵌套 spawn;多子进程争抢 stdout 管道 → `pipe_write` 假死。
- **修复**:① 全部逻辑移进 `main()` + `if __name__=='__main__'`;② BGE-M3 `devices='cuda:0'` 锁单卡(单卡不起 pool)。
- **附带教训**:长任务别用 `python ... | grep | tail`(tail 等 EOF 不消费管道、缓冲填满 → `print` 卡死),改 `python -u > file` 直接写文件再读。

(沉淀为仓库记忆 `reference_bge-m3-multiprocess-reentry`。)

## 4. 结果

### 4.1 #1 单路召回(MRR / Recall@10)

| route | 精确词(25) | 语义(31) | 整体(56) |
|---|---|---|---|
| **bm25**  | **0.738** / 0.96 | 0.210 / 0.45 | 0.446 / 0.68 |
| **bgem3** | 0.584 / 0.88 | 0.347 / 0.61 | 0.453 / 0.73 |
| **dense** | 0.149 / 0.24 | **0.794** / 0.94 | 0.506 / 0.62 |

- **精确词**:bm25 ≫ bgem3 ≫ dense。BM25 在 sparse 主场(编号/型号/金额一字不差)**明显赢** BGE-M3(0.738 vs 0.584);dense 在精确数字上很弱(0.149),符合预期。
- **语义**:dense ≫ bgem3 > bm25。语义是 dense 的活(0.794 碾压两 sparse);BGE-M3 lexical 学到一点语义(0.347 > bm25 0.210),但远不及 dense。
- **整体**:dense 最高(0.506);两个 sparse 几乎打平(0.453 vs 0.446)。

### 4.2 #2 RRF 权重(dense + bm25 客户端加权扫描)

| w_dense | 整体 MRR | 精确词 | 语义 | |
|---|---|---|---|---|
| 0.00 | 0.449 | 0.738 | 0.217 | 纯 sparse |
| 0.25 | 0.496 | 0.771 | 0.275 | |
| **0.40** | **0.541** | 0.781 | 0.348 | ← 峰值 |
| 0.50 | 0.499 | 0.627 | 0.396 | 等权 |
| 0.60 | 0.450 | 0.481 | 0.425 | |
| 0.75 | 0.431 | 0.384 | 0.469 | |
| 1.00 | 0.510 | 0.158 | 0.794 | 纯 dense |

- **hybrid 优于任何单路**:峰值 0.541(w_dense≈0.4)> 纯 dense 0.510 > 纯 sparse 0.449。dense 管语义、sparse 管精确词,互补成立。
- 最优 w_dense≈0.4(sparse 略重),但**强依赖查询的 exact:semantic 比例**(本集≈1:1);等权(0.50)0.499 已接近峰值。

> 注 1:#1 的纯路 MRR(dense 0.506 / bm25 0.446)与 #2 的端点(w=1.0 即 0.510 / w=0.0 即 0.449)有 ±0.004 差异——#1 直接对 top-50 列表算 MRR,#2 用客户端 RRF 打分(k0=60)重建排名,并列项重排所致,属正常。
> 注 2:#2 的权重扫描是 `eval.py` 里**客户端复刻的加权 RRF**(k0=60),用于探查最优比例;生产 store 用 **Qdrant 原生 `FusionQuery.RRF`(服务端等权)**,两者在等权点行为一致、加权点仅用于探查。

### 4.3 #3 est_tokens 重标(Qwen3-VL 真 tokenizer,2927 真实 chunk)

> 注:#3 用**全量 2927 chunk**(覆盖面大于 #1/#2 的 14 文档/1067 chunk 子集),两者语料范围不同。脚本 `retok.py`。

| | 实测 char/token | 当前启发式 | 偏差 |
|---|---|---|---|
| 中文(431) | 1.513 | 1.7 | token 低估 11% |
| 英文(2496) | 4.340 | 4.0 | token 高估 8% |

按文档类型(char/token):academic **3.83** / law **3.87** / financial_report_en **5.08** / government **5.35** / financial_research_zh 1.51。

- **真正的分界是"散文 vs 数字密集",不是中/英**:英文散文(academic/law)≈3.85,几乎正好 = 现值 4.0(误差<4%);偏差全来自数字/表格密集文档(财报/政府 5.0+)。
- 单值系数无法兼顾两簇;改成总体均值(en 4.34)反而会害最常见的散文文档。
- 两个误差方向都被兜住:高估 token → 块偏小 → small-to-big 检索期补(不丢数据);低估 → 块偏大,但远未及 Qwen3-VL 32k 上限。32k 仅护 dense;生产 sparse=**BM25**(jieba 分词无窗口上限,不受块偏大影响——评估里 BGE-M3 的 512 窗口不进生产)。

### 4.4 #4 rerank 精排(Qwen3-VL-Reranker-8B,对 hybrid 召回 top-50 重排)

cross-encoder 第二阶段精排,同 56 query(脚本 `eval_rerank.py`):

| | 精确词 MRR | 语义 MRR | 整体 MRR | R@5 |
|---|---|---|---|---|
| hybrid(召回) | 0.544 | 0.584 | 0.566 | 0.82 |
| **+rerank(精排)** | **0.924** | **0.821** | **0.867** | **0.93** |
| 提升 | **+70%** | +41% | **+53%** | +0.11 |

- **cross-encoder 精排大幅提质**:整体 MRR 0.566→0.867(正确 chunk 从平均第 ~1.8 名提到 ~1.15 名)。原理:它把 `(query, chunk)` 整个喂进模型看**逐词交互**,精度天花板远高于 bi-encoder(dense 各自编码再算距离,看不到交互)。
- **诚实打折**:语义那列(+41%)可能偏乐观——语义查询同源偏置(§6)cross-encoder 也吃;但**精确词那列(+70%)是干净的**(程序化挖、无同源),提升实打实。
- 注:此处 hybrid 基线 0.566 用 **Qdrant 原生 RRF** 的 top-50(与 §4.2 客户端复刻 RRF 口径不同);rerank 对比用同一基线,公平。

## 5. 结论与决策

| 项 | 决策 | 依据 |
|---|---|---|
| **sparse 选型** | **BM25** | **主因**:整体两 sparse **打平**(0.453 vs 0.446)而 BM25 **零模型/零 GPU/零维护** vs BGE-M3 2.3G+GPU。**加分**:hybrid 里 dense 已包语义(BGE-M3 语义优势冗余),sparse 只需补精确词,而精确词正是 BM25 主场(0.738 vs 0.584) |
| **RRF 权重** | **保持 Qdrant 标准 RRF(等权)** | hybrid 确优于单路;等权 0.499 已接近峰值 0.541,0.04 提升不值客户端融合复杂度,且最优 0.4 依赖查询分布会漂移。原则:**sparse 权重别给太低** |
| **est_tokens** | **不重标(保持现值)** | 散文准、数字密集偏但被 small-to-big + 32k 上限兜住;单值无法兼顾两簇,改均值反害散文 |
| **rerank** | **接入(默认关、按需开)** | cross-encoder 精排把 MRR 0.566→0.867(精确词 +70% 干净);但 +16G 显存、+每查询数秒 —— `Retriever.search(rerank=False)` 默认关、显式开,质量/成本由调用方按场景权衡 |

## 6. 局限(诚实说明)

- **合成评估集**:精确词程序化挖、语义 agent 生成,非真实用户查询分布。
- **规模小**:56 query / 14 文档,趋势可信,绝对值仅供参考。
- **语义 golden 单一**:每个语义查询只标 1 个 golden,可能有别的相关 chunk 被算"错"——但对三路同样苛刻,**对比公平**。
- **语义查询同源偏置**:语义查询由 agent 读源 chunk 反推生成,golden 与查询同源(措辞虽避开原词,语义结构仍贴源 chunk),会**系统性高估 dense 的语义召回**——0.794 应理解为上界,非真实场景期望值。
- **RRF 最优权重依赖查询分布**:本集 exact:semantic≈1:1,真实分布不同则最优 w 不同。

## 7. 复现

```bash
cd eval/component_retrieval
python build_corpus_chunks.py && python select_semantic.py   # 语义查询需 agent 生成,见 README
python index_eval.py && python eval.py    # #1 单路召回 + #2 RRF 权重
python retok.py                           # #3 tokenizer 重标(独立,无需 index)
```

需 WSL `navikb` 环境(GPU=4090)+ Qwen3-VL + BGE-M3 模型 + `parsed/` 真实数据。详见 [`eval/component_retrieval/README.md`](../../../eval/component_retrieval/README.md)。
