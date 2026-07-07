# 03 向量检索与混合召回

> **本篇导读**
> 这篇讲 pharos 的检索子系统(embedder):Qwen3-VL 图文同空间稠密向量、BM25 稀疏路、RRF 混合召回、cross-encoder 精排、以及"索引小块、交付大块"的 small-to-big 查询期组装。
> **面试权重:极高**——"你的 RAG 检索怎么做的"几乎必问,且本篇每个决策都有实测数据背书,是整套文档里最容易讲出深度的一篇。
> 前置阅读:切块篇(chunk / section / acl_index 概念,亦可直接读 [chunker INTEGRATION](../components/chunker/INTEGRATION.md));评测口径详见 [07 评估方法论](07-evaluation.md)。

---

## 1. 概念底座:检索为什么是 RAG 的天花板

生成模型只能基于"喂进上下文的内容"作答。切块决定了知识以什么形状存在,而**检索决定了哪些形状能被找回来**——召回不到,后面 prompt 工程做得再精细都是零。所以任何 RAG 系统的质量上限首先卡在检索,这一层要同时回答四个独立的问题:

**问题一:语义匹配与精确匹配是两种能力,一个模型给不全。**

- **稠密检索(dense / bi-encoder)**:query 和文档各自过一遍编码器变成向量,用余弦相似度衡量语义接近。优势是泛化——"营收下滑的原因"能召回写着"收入同比减少系……所致"的段落;盲区是**精确串**:法条编号、产品型号、金额这类字符,在向量空间里"第 42 条"和"第 43 条"几乎重合,一字之差检索不出差别。
- **词法检索(sparse,代表是 BM25)**:按词条倒排,词频 × 逆文档频率打分。优势恰好互补——`Section 6.1`、`v1.2`、`$1196` 一字不差命中;盲区是同义改写:用户换个说法就查不到。还有一支"学习型 sparse"(SPLADE、BGE-M3 lexical),用模型给词条赋权,介于两者之间。

主流做法是 **hybrid:两路都跑,再融合**。融合有两个流派:分数加权(需要先把余弦分和 BM25 分归一到可比量纲,脏活)和**名次融合 RRF**(Reciprocal Rank Fusion:只看每路的排名,`score = Σ 1/(k+rank)`,天然回避量纲问题)。

**问题二:召回和精排是两个阶段,精度和成本不可兼得。**

bi-encoder 各自编码,看不到 query 和文档的逐词交互,精度有天花板;**cross-encoder** 把 `(query, doc)` 拼在一起整个喂进模型打相关分,精度高得多但每对都要跑一次前向,不可能扫全库。工业界标准做法是两阶段:粗召回取 top-N(快),cross-encoder 只对这 N 个重排(慢而准)。

**问题三:多模态语料里,图怎么被文字 query 找到?**

两条路线:① 把图文字化(OCR / 生成 caption),再当纯文本检索——链路简单但信息有损;② **图文同空间编码**(CLIP 范式):文本和图像编码进同一向量空间,文字 query 直接跨模态召回图像——不需要 caption 兜底,但要求编码模型天生支持。

**问题四:检索要小块,LLM 要大块——索引粒度的根本矛盾。**

小块向量语义聚焦、检索准;但交付给 LLM 时,孤零零一小段常常缺上下文。业界解法叫 **small-to-big**(LlamaIndex)或 **parent document retriever**(LangChain):**索引时切小块、命中后按结构把所在小节原文取回来交付**。这就要求系统在向量库之外还保留原始文档结构。

pharos 对这四个问题各给了一个答案,且每个答案都有"被否决的备选 + 实测数据"。下面逐个拆。

---

## 2. Pharos 怎么做

先看一次查询的完整旅程:

```
                     索引期                                    检索期
 chunker ChunkResult                              query
    │                                               │
    ├─ image_only 纯图 ──> 图像向量(跳过 sparse)      ├─ encode_query(LRU 缓存 + instruction)──> dense 向量
    ├─ 其余(text/table/带caption图)                  ├─ query_sparse(jieba+正则)──────────────> BM25 向量
    │    ├─> 文字向量(Qwen3-VL, MRL 截 1024)          │
    │    └─> BM25 sparse(jieba+正则+FNV-1a)          ▼
    ▼                                        Qdrant hybrid_search:两路 Prefetch(各带 ACL filter)
 Qdrant(向量+payload) + sidecar JSON(原文结构)      └─> RRF 服务端融合 ──> 出口 ACL 复核 ──> Hit 列表
                                                    │
                                                    ├─(可选)cross-encoder rerank,失败降级
                                                    ├─ 按 section 去重(资产块豁免)
                                                    ├─ small-to-big:sidecar + assemble_big 组装大块
                                                    └─ 每条标 context_status ──> 交付 agent/generator
```

### 2.1 图文同空间:一个向量空间装下文本和图表

dense 层用 Qwen3-VL-Embedding-8B,**复用模型自带的官方脚本**(last-token pooling),不自写 pooling/forward——自写与官方数值漂移的风险不值得冒([src/embedder/dense.py:58-63](../../src/embedder/dense.py#L58))。文本与图像编码进同一个 4096 维空间。

索引期按 chunker 打的 `image_only` flag 分流([src/embedder/embed.py:70-85](../../src/embedder/embed.py#L70)):

- **纯图 chunk**(无 caption 无正文)→ `encode_image` 得图像向量,**跳过 sparse**——纯图没有可检索文字,BM25 无意义;召回全靠图文同空间,文字 query 直接跨模态命中;
- **text / table / 带 caption 的 image / chart** → 一律 `encode_text`(caption 文字向量)+ BM25 sparse——文字信号更全,不走图像向量(这是对旧设计的否决,见 §3);
- 图像路径缺失或失效的纯图,计入 `skipped` 明确上报,不静默假装索引成功。

跨模态召回不是"理论上应该行",做过最小实测:准确文字描述对"对应图"的相似度 0.74 / 0.49,对"非对应图"只有 0.37 / 0.17,无关描述(金毛在沙滩)对两图仅 0.07–0.11——分离度足够支撑"文字 query 召回纯图 chunk"这条路(两图两描述的 spot-check,口径见 §8)。

### 2.2 MRL 截维:1024 维的性价比

Qwen3 embedding 系列支持 Matryoshka(MRL):取全维向量的前 N 维重新 L2 归一化,仍是合法向量。pharos 默认截到 1024 维([src/embedder/config.py:17](../../src/embedder/config.py#L17)),比 4096 省 4 倍存储与检索开销;spot-check 显示几乎无损(对角线相似度 0.7388/0.4930 → 0.7363/0.4483,分离度健康)。

截维实现里有个数值刀刃([src/embedder/dense.py:65-74](../../src/embedder/dense.py#L65)):**先 `.float()` 把 bf16 转 fp32,再截维 + normalize**。bf16 只有 8 位尾数,直接在 bf16 上归一化,转 fp32 后 norm≈1.002 而不是 1.0——这个偏差曾让 local 建库向量与 remote 查询向量数值不一致(remote 路径是纯 numpy fp32,[src/embedder/remote.py:160-173](../../src/embedder/remote.py#L160)),修复后 GPU 实测两路 cosine=1.0000000、maxdiff 2.98e-08。这段故事(含"假绿测试"教训)在面试里是加分项,见 §6 问 3。

### 2.3 BM25 sparse:零模型、纯 CPU,但有三个刀刃

sparse 路的实现原则是"客户端只做分词和词频,打分交给 Qdrant 服务端的 `Modifier.IDF`"。三个容易做错的细节:

1. **精确串保全**([src/embedder/sparse.py:39-43](../../src/embedder/sparse.py#L39)):jieba 会把 `GPT-4`/`v1.2`/长编号切碎,而精确串恰是选 BM25 的初衷。所以用正则 `[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*` 完整保留字母数字串——但**只补 jieba 没完整切出的**。否则字母数字 token 被 jieba 和正则各 append 一次,doc 端 tf 双计,而纯中文 token 不匹配正则不会双计,BM25 权重被系统性扭曲(对抗审查 seal#5 抓的)。
2. **稳定 hash**([src/embedder/sparse.py:21-26](../../src/embedder/sparse.py#L21)):token → uint32 index 用 FNV-1a,明确不用 Python 内建 `hash()`——后者跨进程随机化,建库进程和查询进程的同一个词会 hash 到不同 index,doc/query 永远对不上。这类 bug 静默到只会表现为"sparse 路怎么召回这么差"。
3. **两端 values 语义**([src/embedder/sparse.py:47-60](../../src/embedder/sparse.py#L47)):doc 端 values=词频 tf,query 端 values=1.0,BM25 分由文档侧 tf × 服务端 IDF 决定。无有效 token 返回 None,该 chunk/query 只走稠密路。

### 2.4 hybrid RRF:结构、选路与分数语义

hybrid 用 Qdrant `query_points` 的两层结构([src/embedder/store.py:96-124](../../src/embedder/store.py#L96)):dense/sparse 各一个 `Prefetch`(每路召回 `prefetch_limit=50` 个候选),顶层 `FusionQuery(RRF)` 服务端融合——名次融合天然回避"余弦分和 BM25 分量纲不可比",且零客户端融合代码。image_only chunk 无 sparse 向量、query 也可能无有效 token,此时 sparse prefetch 整个省略。

两个容易被忽略的设计点:

- **ACL 硬过滤下推到每个 Prefetch**([src/embedder/store.py:118-120](../../src/embedder/store.py#L118)):实测发现嵌入式 QdrantLocal 在 fusion 模式下会**静默丢弃顶层 query_filter 的 should 子句**,权限过滤退化成 fail-open。修复是把 filter 放进每个 prefetch,fusion 只融合已过滤结果;返回前还有逐条 `acl_admits` 出口复核([src/embedder/store.py:126-129](../../src/embedder/store.py#L126))。ACL 的三道闸设计是另一篇的主角,此处只需记住:**检索层的权限过滤必须发生在召回层,否则 limit 会被无权结果污染**。
- **score_kind 显式标注分数量纲**([src/embedder/types.py:16-26](../../src/embedder/types.py#L16)):agent 可用 `strategy` 参数选路——dense/sparse 单路直查拿原生分(cosine/bm25,可解释、可跨查询比较),hybrid 走 RRF 拿融合分(只随名次,**不可跨查询比较**,嵌入式与 server 的 RRF k 常数还不同,绝对值差一个量级)。rerank 之后 score 与 score_kind 一起改写,防止"名次按 rerank、分数还是 RRF"的脱节误导 agent 的置信判断。

### 2.5 可选精排:cross-encoder 与"非对称失败"

Qwen3-VL-Reranker-8B(同源 cross-encoder)把 `(query, doc)` 整个喂进模型,yes/no token logits + sigmoid 出 0~1 相关分([src/embedder/rerank.py:61-72](../../src/embedder/rerank.py#L61))。接入方式([src/embedder/retrieve.py:106-116](../../src/embedder/retrieve.py#L106)):`search(rerank=True)` 时先多召回 `max(rerank_top_n, top_k)` 个候选,精排后取 top_k;分数用 `dataclasses.replace` 写回新 Hit(不原地改共享对象,[src/embedder/rerank.py:74-83](../../src/embedder/rerank.py#L74))。

失败处理是**有意的非对称**:dense 编码失败 fail-loud(没有向量就没有检索,必须响);rerank 失败(OOM/模型缺失/推理服务不可用)`except` 降级返回 hybrid 召回原序——rerank 是增强信号,降级安全,不让精排拖垮基础检索。reranker 还是 lazy 加载 + single-flight:不开 rerank 就不加载第二个 8B(local 模式 +16G 显存)。

### 2.6 small-to-big:sidecar 与查询期组装

索引粒度矛盾的 pharos 答案:**Qdrant 只存 chunk 向量 + payload;原始 elements/sections/banners/acl_index 按 doc_id 存 sidecar JSON**([src/embedder/embed.py:107-125](../../src/embedder/embed.py#L107))。为什么不塞进 Qdrant payload?因为查询期组装需要**全文档**的元素与小节结构,而 payload 是单点数据——组装是文档级操作,不是命中点级操作。

检索命中后,`_assemble` 用 payload 重建一个轻量 shim,按 doc_type 取预算(slides/policy 这类文档整节保留),调 chunker 的 `assemble_big` 把命中块所在小节的原文组装成 big-block 交付([src/embedder/retrieve.py:186-197](../../src/embedder/retrieve.py#L186))。取材只取与命中块**同 ACL** 的元素,使 big-block 的 ACL 天然等于命中块 ACL、出口可校验。

sidecar 的可靠性也有讲究:写侧原子写(`.tmp` + fsync + `os.replace`,防崩溃留半截 JSON 让组装永久崩)+ `SIDECAR_VERSION` 盖章;读侧**错误分层**——缺文件/坏 JSON 是单 doc 瞬态,降级返回裸 hit;版本不符是系统性 schema 漂移,故意不进降级 catch,响亮失败提示整体重建([src/embedder/retrieve.py:60-95](../../src/embedder/retrieve.py#L60))。"静默错"变"响亮失败"是这个子系统反复出现的母题。

### 2.7 context_status 状态机与三层去重

`search_with_context` 是交付前的最后一站([src/embedder/retrieve.py:119-184](../../src/embedder/retrieve.py#L119)),每条结果带一个 `context_status`,八种状态构成一个小状态机,让 agent 知道自己拿到的是什么:

| 状态 | 含义 | agent 该怎么用 |
|---|---|---|
| `full_section` | 完整小节 | 直接用 |
| `climbed_N` | 向上爬了 N 层的完整父节 | 直接用,知道上下文更宽 |
| `section_window` | token 受限的窗口片(**非完整小节**) | 需要更全可对 chunk_id 调 expand |
| `asset_no_prose` | 资产页无散文,数据在命中块 content_raw | 读 content_raw,别等散文 |
| `single_chunk_degraded` | sidecar 丢/损,降级裸 hit | 内容可用,组装不可用 |
| `single_chunk_acl` | big-block 出口 ACL 未过 | 只能用裸 hit |
| `deduped` | 与前一命中组装出同一 big-block,折叠 | 上下文看前一条 |
| `concise` | 调用方主动跳过组装(省 token) | 按裸 hit 用 |

去重有三层,每层都是被真实 bug 教育出来的:

1. **同 section 去重**([src/embedder/retrieve.py:136-146](../../src/embedder/retrieve.py#L136)):top_k 命中常聚在同一小节,组装出的 big-block 相同,重复喂 LLM 纯烧 token——同 `(doc_id, section_id)` 只留首个。但 `section_id=None` 是"无小节"不是"同一小节",退化按 chunk_id 去重(seal#6)。
2. **资产块豁免**([src/embedder/retrieve.py:138-142](../../src/embedder/retrieve.py#L138)):chart/table 的数据在自身 content_raw,不在散文 big-block 里。若图表与散文兄弟同小节被折叠,表里的数彻底丢失——eval 里实证过"召回到却答不出表里的数"。去重规则必须按**内容承载位置**分型。
3. **big-block anchor 去重**([src/embedder/retrieve.py:161-171](../../src/embedder/retrieve.py#L161)):不同小节的命中 climb 到同一个父节也会产出重复 big-block,按组装结果的 anchor 折叠;窗口块按 resolved_section 折叠。

同节折叠掉的条数记在 `SearchResults.section_folded_n`([src/embedder/retrieve.py:18-21](../../src/embedder/retrieve.py#L18))外露给 agent——这是本轮审查刚补的信号,故事见 §4。

### 2.8 查询向量 LRU

agentic RAG 多跳常重发相同 query,`encode_query` 做了 256 容量的 LRU([src/embedder/dense.py:89-105](../../src/embedder/dense.py#L89)),免重复 8B 前向。值得记的是锁结构:**双段锁**——读段只包字典查找、写段只包写入淘汰,真正昂贵的 encode(GPU 前向或 HTTP+重试退避)在锁外。两线程同 query 同时 miss 会各算一次、后写覆盖,MRL 确定性保证结果相同,这是**可接受的良性竞态**——绝不为消除它把整段包锁,否则推理服务预热期一个查询的退避会阻塞所有查询的缓存命中(这教训来自一次真实的"修可用性的补丁制造更大可用性事故",详见并发/扩展篇)。

---

## 3. 为什么这么设计:被否决的备选与数据

> **口径警告**:本节数据来自组件级检索评测(56 query / 1067 chunk / 14 文档,[EVALUATION.md](../components/embedder/EVALUATION.md)),与端到端 88 题评测([07 评估方法论](07-evaluation.md))是**两条不同的评测线,数字不可混引**。56 query 中 25 条精确词(程序化挖 df==1 稀有串,无偏)+ 31 条语义(agent 生成、刻意避原词,有同源偏置)。

### 3.1 sparse 选 BM25 而不是 BGE-M3:数据驱动的否决

| route | 精确词 MRR(25) | 语义 MRR(31) | 整体(56) | 成本 |
|---|---|---|---|---|
| **bm25(ours)** | **0.738** | 0.210 | 0.446 | 零模型、纯 CPU |
| bgem3 lexical | 0.584 | 0.347 | 0.453 | 2.3GB 模型 + GPU |
| dense(参照) | 0.149 | **0.794** | 0.506 | — |

判决逻辑分两步:**整体打平**(0.453 vs 0.446)时,零模型零 GPU 的 BM25 赢在成本;更关键的是**hybrid 语境下的角色分工**——dense 已经把语义包了(0.794),sparse 在混合里只需要补精确词,而精确词恰是 BM25 主场(0.738 完胜 0.584)。BGE-M3 学到的那点语义(0.347)在 hybrid 里是冗余能力,为冗余能力付 GPU 成本不值。SPLADE 类学习型 sparse 未单独进对比——BGE-M3 已代表该路线,且其中文能力偏弱(MIRACL-zh 36.3,材料口径)。

这个表还顺手证明了 §1 的"盲区"论断:dense 在精确词上 0.149,几乎瞎;BM25 在语义上 0.210,同样瞎——**hybrid 不是锦上添花,是互相补盲**。

### 3.2 RRF 等权而不是调权:实测后拒绝优化

客户端加权 RRF 扫描(k0=60):峰值在 w_dense≈0.4,整体 MRR **0.541 > 纯 dense 0.510 > 纯 sparse 0.449**——hybrid 确优于任何单路。但等权点 0.499 已接近峰值,0.04 的提升要付出的代价是:放弃 Qdrant 服务端原生融合、自己维护客户端融合代码,而且最优权重强依赖查询集的 exact:semantic 比例(本集≈1:1),真实分布一漂移,调好的权重就失效。

**决策:保持 Qdrant 服务端等权 RRF,只保留一条运维原则"sparse 权重别给太低"。** 这是面试里很好讲的"实测后拒绝优化"案例:优化收益、维护成本、参数脆弱性三方权衡,数据齐了反而更有底气说不。

### 3.3 rerank 接入但默认关:0.566 → 0.867 也不白给

同 56 query,对 hybrid 召回(Qdrant 原生 RRF,基线口径与 §3.2 的加权扫描不同)top-50 精排:

| | 精确词 MRR | 语义 MRR | 整体 MRR | R@5 |
|---|---|---|---|---|
| hybrid 召回 | 0.544 | 0.584 | 0.566 | 0.82 |
| **+rerank** | **0.924** | **0.821** | **0.867** | **0.93** |
| 提升 | **+70%** | +41% | +53% | +0.11 |

正确 chunk 从平均第 ~1.8 名提到 ~1.15 名。诚实打折:语义列的 +41% 可能偏乐观(语义查询同源偏置,cross-encoder 同样受益);**精确词列的 +70% 是程序化挖题、无同源偏置,最干净**。但代价是 local 模式 +16G 显存、每查询数秒——所以默认关、`rerank=True` 显式开,质量/成本由调用方按场景权衡。"默认开启"这个备选被成本否决。

### 3.4 其他被否决的备选

| 备选 | 否决理由 |
|---|---|
| 带 caption 的图走图像向量(DESIGN 旧稿) | caption 文字向量 + BM25 双路信号比单一图像向量更全,实现最终选文字路 |
| 自写 pooling/forward | 与官方脚本数值漂移风险;复用官方 `Qwen3VLEmbedder` |
| 原文全塞 Qdrant payload | 组装需要**全 doc** 的 elements/sections 结构,payload 是单点数据;sidecar 按 doc 存整体结构 |
| small-to-big 取材用 admit=(取一切 user 可见元素) | big-block 会混入比命中块更严 ACL 的内容而 ACL 标注低报(seal#2);改为只取与命中块同 ACL 的元素 |
| rerank 分数与 RRF 分数并存 | 名次按 rerank、分数还是 RRF,误导 agent 置信判断;score+score_kind 必须一起改写 |
| Python 内建 hash() 做 token index | 跨进程随机化,doc/query 对不上;FNV-1a 稳定 hash |

---

## 4. 实战复盘:一轮对抗审查在检索子系统抓到什么

写这套文档前,对 embedder 做了一轮"分析师深读 → 独立验证员先试图反驳"的对抗审查(产物即本轮的 fixes_applied / deferred 清单)。embedder 簇 6 项 confirmed 全部落地,另有 2 项 confirmed 但**有纪律地延期**。修复前基线 224 passed,修复后 259 passed(全仓口径,新增 36 用例)。

### 4.1 已修:六个"确认即改"的健壮性缺陷

**① index_document 删旧时机(medium,本轮最重)**
- **症状**:重索引 = 先 `delete_by_doc` 再逐 chunk 编码再 upsert。编码中途失败(remote 模式推理服务滚动重启且重试耗尽 / local 模式错卡缓存了 GPU 错误),旧向量已删、新向量未写——该文档**持久脱库**,失败窗口是编码全程(大文档分钟级)。验证员还发现更狠的连锁:local 错配置下批量重跑会"每篇先删再快失败",一次错配置能把整个索引串行删空。
- **根因**:把"删旧防孤儿"(重索引变短时旧高编号 point 残留)实现成了"先删",但防孤儿只要求删在 upsert 前,不要求删在编码前。
- **修法**:重排为**编码 → sidecar tmp 备好 → delete → upsert → 原子 replace**([src/embedder/embed.py:86-104](../../src/embedder/embed.py#L86))——编码与写盘全是"纯准备、失败无副作用",失败窗口从分钟级缩到毫秒级;delete 之后的失败响亮告警"已脱库需重跑",indexer 汇总失败清单并非零退出(替代静默"跳过")。
- **测试**:[tests/test_review_fixes.py:406](../../tests/test_review_fixes.py#L406) `test_index_document_encode_failure_keeps_old_index`——第 2 个 chunk 编码时炸,断言旧 point 数原样(修复前=0)、旧 sidecar 一字未动、无 tmp 残留。

**② list_documents 的 truncated 信号(medium)**
- **症状**:`limit=10000` 的语义是"chunk 扫描上限"而非文档数,库超一万 chunk 时文档清单静默截断,agent 的覆盖判断建立在残缺清单上且无从得知。
- **修法**:返回 `(docs, truncated)`,用 scroll 的 `next_page_offset` 语义天然区分"扫完全库"与"撞上限提前停"([src/embedder/store.py:131-158](../../src/embedder/store.py#L131)),toolcore 把 truncated 透传进返回 dict 并附 hint。
- **测试**:[tests/engine/test_store.py:100](../../tests/engine/test_store.py#L100)。教学点:**limit 参数必须想清楚"限的是什么"**,以及截断必须给信号——同仓 get_document 早有 truncated 惯例,这里是漏网。

**③ remote 模型握手校验(medium)**
- **症状**:推理服务换了模型而库没重建时,只要新模型全维 ≥ dense_dim,客户端照常截维成功——查询向量与库内向量来自**不同语义空间**,召回质量静默崩塌,唯一发现手段是 eval 指标事后下跌。
- **修法**:首查前一次性 GET /healthz,校验服务端 `model_dense` == 客户端模型 basename、`full_dim ≥ dense_dim`,不符 RuntimeError fail-loud([src/embedder/remote.py:115-144](../../src/embedder/remote.py#L115))。边界处理见真章:预热中(full_dim=None)/探活不可达**不阻断**——瞬态留给既有重试链,只有"拿到了字段且不符"才 loud,配置错配不该被重试吸收。
- **测试**:[tests/engine/test_remote.py:314-341](../../tests/engine/test_remote.py#L314) 四条(错配 loud / 维度不足 loud / 验过一次不再发 GET / 预热不阻断)。

**④ ensure_collection 幂等补索引**:payload index 原来只在创建分支建——server 模式下"create_collection 成功、索引建完前崩溃"的半初始化 collection 会永久缺索引(过滤退化全扫)且无自愈;未来新增索引也补不到存量库。修法:索引循环提到分支外幂等补建,每次启动自愈([src/embedder/store.py:51-58](../../src/embedder/store.py#L51));真 Qdrant server 实测通过。

**⑤ RemoteReranker.score 签名对齐**:缺 `instruction` 形参,违反基类替换性(LSP 子类收窄输入域)——remote 后端永远只能用 cfg 默认 instruction。修法一行:补参数,None 回落 cfg 与旧行为逐字节一致([src/embedder/remote.py:190-199](../../src/embedder/remote.py#L190))。

**⑥ section_folded_n 折叠计数(收窄版)**:同节去重折叠后交付数 < top_k 而 agent 无信号区分"库存尽"与"被折叠"。落地的是**信号的一半**:`SearchResults`(list 子类)携带计数,调用方按普通 list 消费零改动,toolcore 读进 meta([tests/engine/test_tools.py:219](../../tests/engine/test_tools.py#L219));**超采回填的另一半明确不做**,见下。

### 4.2 延期:两个"确认了但不能马上改"

这两条是工程判断的教学点——**confirmed 不等于该立即修**,判断标准是"改动是否改变输出内容/是否需要当前没有的验证环境"。

**embedder#3 索引期逐 chunk 编码、无批处理**(deferred 清单;证据:[src/embedder/embed.py:76-79](../../src/embedder/embed.py#L76) 里 `encode_text([ch.text])[0]` 单元素调用,官方接口天然吃批)。为什么不马上改:本仓自己立过等价性标准(bf16 norm 1.002 都要修),而**批前向 vs 单条前向在 padding/attention 下存在 bf16 数值差异风险**——没做 GPU 实测批/单编码 cosine 等价之前合入,新建库向量可能与既有库系统性微偏。且产品建库入口(indexer)walk 的是 local 路径,最坏的"每 chunk 一次 HTTP"场景当前不可达,SCALE_OUT P2-4 已登记为接受的取舍。**延期不是懒,是等价性纪律优先于吞吐优化。**

**embedder#6 去重后不回填**(超采回填半边)。同节折叠有理(big-block 相同),但请求 top_k=8 可能只拿到 2-3 条独立证据,跨小节的补位候选明明在 prefetch 的 50 个里却被 store 层 top_k 截断。修法草图已备(fetch_k=2×k 超采、去重循环够 k 即 break)。为什么不马上改:**超采回填改变检索交付内容**,eval 走同一函数,已发布的评估基线会失真——须 GPU 重跑 eval 验证后才能合入。所以拆成两半:不改变行为的"折叠计数信号"先落地(§4.1⑥),改变行为的"回填"排队等验证窗口。

---

## 5. 面试怎么讲

**30 秒版(电梯稿)**

> 检索层我做的是 hybrid + 可选精排:稠密路用 Qwen3-VL 图文同空间向量,MRL 截 1024 维省 4 倍存储;稀疏路用 BM25——选它是数据驱动的,精确词查询上 BM25 的 MRR 0.738 明显赢 BGE-M3 的 0.584,而语义反正由 dense 负责,整体打平时零模型零 GPU 就是胜负手。两路在 Qdrant 服务端做 RRF 名次融合,hybrid 整体 0.541 优于任何单路。之上有可选的 cross-encoder 精排,MRR 从 0.566 提到 0.867,但 +16G 显存所以默认关。交付端做 small-to-big:索引小块保检索精度,命中后按文档结构组装完整小节交付,每条结果带 context_status 状态告诉 agent 拿到的是完整节、窗口片还是降级裸块。

**3 分钟版(结构化展开)**

1. **先立问题**:检索是 RAG 上限;语义与精确匹配是两种能力(dense 精确词 MRR 只有 0.149,BM25 语义只有 0.210——各自的盲区实测就是瞎),所以必须 hybrid。
2. **sparse 选型讲数据驱动**:构造两类查询防评估偏置——25 条程序化挖的精确串(无偏)+ 31 条 agent 改写的语义题(自知有同源偏置)。BM25 vs BGE-M3 整体打平,但 hybrid 语境里 sparse 只需补精确词,BM25 主场胜出且零 GPU。顺带讲一个工程细节:jieba 会把 GPT-4 切碎,正则补精确串时"只补没切出的"防 tf 双计;token hash 用 FNV-1a 不用 Python hash(),因为后者跨进程随机化。
3. **融合讲 RRF + 拒绝优化**:名次融合回避量纲问题;加权扫描峰值 0.541 只比等权高 0.04 且最优权重依赖查询分布,决策是保持服务端等权——实测之后拒绝优化,比优化本身更能体现判断力。
4. **精排讲两阶段与非对称失败**:cross-encoder 看逐词交互,0.566→0.867,精确词 +70% 那列无同源偏置最干净;默认关是成本决策;dense 失败 loud、rerank 失败降级返回召回原序——增强信号与必要信号的失败语义应该不同。
5. **交付讲 small-to-big**:检索要小块、LLM 要大块;原文结构存 sidecar(组装是文档级操作,不塞 payload),原子写 + 版本盖章 + 错误分层(瞬态降级、系统性响亮);三层去重里资产块豁免——表格的数在 content_raw 不在散文里,折叠了就"召回到却答不出"。
6. **收尾丢一个反差点**:最近一轮对抗审查在这个"已封板"的子系统还抓出 6 个确认缺陷——最重的是重索引先删后编码、失败即持久脱库;修完之外还有两条 confirmed 被有纪律地延期,因为它们改变检索交付内容,必须等 GPU eval 重跑验证。质量不是一次达成的状态,是持续对抗出来的。

---

## 6. 追问预演

**Q1:为什么用 RRF 而不是分数归一后加权融合?**
要点:余弦分与 BM25 分量纲不同分布也不同,归一化方法(min-max/z-score)本身引入超参且对 outlier 敏感;RRF 只用名次,零超参(k 常数不敏感)、Qdrant 服务端原生支持。加权 RRF 实测过:峰值仅高 0.04,且最优权重绑定查询分布。反问自己也能答:RRF 的代价是丢掉分数强度信息——第 1 名比第 2 名好多少不可知,所以 pharos 用 score_kind 显式告诉 agent"RRF 分不可跨查询比较"。

**Q2:BM25 赢 BGE-M3,什么时候会反过来?**
要点:胜负是"hybrid 语境下的角色分工"决定的,不是绝对能力。如果系统没有 dense 路(纯 sparse 检索),BGE-M3 整体 0.453 略优且语义能力(0.347 vs 0.210)有价值;查询分布偏口语化改写、语料词汇不匹配严重时学习型 sparse 更有优势。另外 BGE-M3 有 512 token 窗口约束而 jieba+BM25 无窗口上限——长 chunk 场景又是一票。关键词:互补性、角色冗余、成本归因。

**Q3:MRL 截 1024 维,怎么确认没伤召回?数值上有什么坑?**
要点:MRL 是训练时就套娃的(前缀维度本身被优化过),不是事后 PCA;spot-check 对角线相似度几乎不变。坑是 bf16:8 位尾数上做 L2 normalize,norm 偏差到 1.002,修法是先 .float() 再截+归一。进阶谈资:守护测试第一版是假绿的——它拿到的输入已经是 fp32,删掉修复也不红;后来钉在"bf16 张量直接进 _mrl"这个真实建库入口上,纯 CPU 进 CI。**"每个已修都要有能变红的测试"**。

**Q4:rerank 提升这么大,为什么不默认开?rerank_top_n 怎么选?**
要点:成本三项——第二个 8B 模型 +16G 显存、每查询数秒延迟、吞吐骤降;RAG 里不是所有查询都需要精排(agent 可按任务开)。top_n 是召回深度与精排成本的平衡:太小则正确答案没进候选池(rerank 救不了没召回的),太大白烧;pharos 默认 50 = prefetch_limit,精排池上限就是召回池。失败语义:rerank 挂了降级返回 hybrid 原序,score_kind 保持 rrf 量纲,上层能据此标 rerank_degraded。

**Q5:图文同空间听起来很美,怎么验证它真的 work?纯图检索有什么已知短板?**
要点:最小实测——准确描述对对应图 0.74/0.49、对非对应图 0.37/0.17、无关描述 0.07–0.11,分离度支撑跨模态召回。已知短板(主动交代):纯图 chunk 检索期只有占位 text,cross-encoder rerank 按 text 重排会低估它——评估语料 image_only=0 所以没影响验证,生产要用需补检索期图路径方案(代码里是显式 TODO,[src/embedder/rerank.py:6-8](../../src/embedder/rerank.py#L6))。

**Q6:small-to-big 为什么不干脆索引大块?或者小块大块都索引?**
要点:索引大块 = 向量语义稀释,多主题混一个向量,检索精度掉;双索引 = 存储翻倍 + 两套索引一致性维护。查询期组装的成本(读 sidecar + 拼接)只发生在 top_k 条上,便宜。加分点:组装不是简单取父块,是 ACL 感知的——只取与命中块同 ACL 的元素,预算按 doc_type 分型,组装结果还有出口 ACL 校验。

**Q7:检索返回的 score 能直接当置信度用吗?**
要点:分口径答。cosine 可跨查询粗略比较;BM25 分依赖词频文档长度,横比弱;RRF 分只随名次,**绝对值无意义**且嵌入式与 server 的 k 常数不同,差一个量级;rerank 分是 sigmoid 0~1,最接近"置信度"语义。所以 Hit 上带 score_kind,rerank 后 score 和 kind 一起换——防止 agent 拿 RRF 分做阈值判断这种静默错误。

**Q8:你的评测数字可信度如何?**
要点:主动打折——56 query 规模小,趋势可信、绝对值仅供参考;语义查询由 agent 读源 chunk 反推生成,有同源偏置,dense 0.794 应理解为上界;精确词是程序化挖的,最干净;每个语义 query 只标 1 个 golden,对三路同样苛刻所以对比公平。这题答得好坏的分水岭在于**是否主动说出偏置**,而不是等追问。

---

## 7. 动手实验

**Lab A:BM25 分词与 tf 双计防护(CPU,已实测可跑)**

前置:仓库依赖环境(WSL `navikb` conda 环境,或任何装了 `jieba`/`qdrant-client`/`numpy` 的 Python);仓库根目录执行。

```bash
PYTHONPATH=src python -c "
from embedder.sparse import tokenize, doc_sparse, _tok_id
print('tokenize:', tokenize('第42条 GPT-4 营收 42000000 v1.2'))
d = doc_sparse('gpt-4 gpt-4')
print('gpt-4 tf =', d.values[list(d.indices).index(_tok_id('gpt-4'))])
"
```

预期输出(实测):`tokenize` 里 jieba 把 gpt-4 切碎成 `gpt`/`4`、把 `v1.2` 切碎,而正则补回了完整的 `gpt-4`/`v1.2`,`42000000` 完整保留;`doc_sparse('gpt-4 gpt-4')` 中 gpt-4 的 **tf=2.0 而非 4.0**——亲眼看到 [src/embedder/sparse.py:39-43](../../src/embedder/sparse.py#L39) 那条"只补 jieba 没完整切出的"规则如何防止字母数字 token 双计。可以再自己动手:把 `seen_jieba` 判断删掉重跑,观察 tf 变 4.0——这就是 seal#5 修的加权扭曲。

**Lab B:检索链路的失败模式矩阵(CPU,mock HTTP 零网络,已实测 12 passed)**

```bash
pytest tests/engine/test_remote.py -v -k "retry or backoff or readiness or degrade"
```

预期看到完整的失败谱:503/502/504/500 重试、4xx 零重试立即抛、ConnectTimeout/ReadError/RemoteProtocolError(docker kill 的真实断连形态)被 TransportError 谱吸收、耗尽抛 InferenceUnavailable、退避序列恰为指数 [0.5, 1.0]、以及 **rerank 失败降级返回 hybrid 原序且 score_kind 仍为 rrf**(dense loud / rerank 降级的非对称)。再跑 `pytest tests/engine/test_retrieve.py tests/engine/test_store.py -q`(实测 27 passed)可覆盖本篇 §2.7 的全部去重/状态机规则与 §4 的 truncated/幂等索引回归。

**Lab C(可选,GPU+WSL):复现 rerank 收益**

前置:WSL `navikb` 环境、4090、Qwen3-VL 双模型、`parsed/` 真实数据(见 [EVALUATION.md §7](../components/embedder/EVALUATION.md))。

```bash
cd eval/component_retrieval && python eval_rerank.py
```

预期复现 MRR 0.566→0.867(+53%)、精确词 +70%。

---

## 8. 诚实边界

面试中主动承认,比被问出来体面得多:

1. **评测规模与偏置**:56 query / 14 文档的组件级评测,趋势可信、绝对值仅供参考;语义查询同源偏置使 dense 0.794 与 rerank 语义列 +41% 都偏乐观(精确词列干净)。跨模态召回只有两图两描述的 spot-check,没有成规模的图检索基准。
2. **纯图 rerank 是已知 TODO**:image_only chunk 检索期缺绝对图路径,cross-encoder 按占位 text 重排会低估;评估语料恰好无纯图所以未暴露,生产接多模态语料前必须补。
3. **去重不回填已确认未修**(deferred embedder#6):top_k=8 可能只交付 2-3 条独立证据;折叠计数信号已给,但补位候选仍被截断——等 GPU eval 窗口。
4. **索引吞吐**:逐 chunk batch=1 前向,吞吐低于 8B 模型批前向能力(deferred embedder#3)——延期理由是批/单前向的 bf16 数值等价未实测,等价性纪律优先。
5. **local↔remote 的 E2 缺口**:encode 逐元素等价(cosine=1.0000000)已实测,但"等价 ⇒ hybrid top-k 一致"在逻辑上**不成立**(HNSW 近似 + RRF 名次融合,近重复段的分差可小于 maxdiff 而翻转排序),端到端 top-k 一致性未验证——文档里把它从"逻辑保证"诚实降级为"未验证缺口"。
6. **RRF 权重结论绑定查询分布**:exact:semantic≈1:1 的集上等权近优;真实分布严重偏斜时应重扫,唯一带走的原则是"sparse 权重别给太低"。

---

*本篇锚点均已按 2026-07-07 修复落地后的代码实读核验;工程细节见 [embedder DESIGN](../components/embedder/DESIGN.md) 与 [EVALUATION](../components/embedder/EVALUATION.md),扩展/多副本形态见 [SCALE_OUT](../SCALE_OUT.md)。*
