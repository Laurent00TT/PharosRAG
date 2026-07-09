# 01 RAG 全景与 Pharos 架构

> **本篇导读**:这是整套学习文档的总纲。先讲 RAG 为什么存在、朴素 RAG 会在哪些地方摔跤(问题谱系),再把 Pharos 的六个环节逐一对到谱系上,最后走一遍"一条请求的完整生命线"。
> **面试权重:★★★★★**——"介绍一下你的项目"是必考第一题,这篇就是答案的骨架。
> **前置阅读**:无。读完本篇再按兴趣进各分篇:[02 解析与切块](02-parsing-chunking.md) · [03 检索](03-retrieval.md) · [04 ACL 安全](04-acl-security.md) · [05 生成与 Grounding](05-generation-grounding.md) · [06 Agentic 与 MCP](06-agentic-mcp.md) · [07 评估方法论](07-evaluation.md) · [08 服务化](08-service-architecture.md) · [09 扩展性](09-scale-out.md) · [10 方法论与故事集](10-methodology-stories.md)。

---

## 1. 概念底座:为什么需要检索增强

### 1.1 LLM 的三个结构性缺陷

抛开具体项目,任何 RAG 系统都在回答同一个问题:**LLM 的参数化记忆不能当数据库用**。拆开就是三个缺陷:

1. **知识有截止日期,且看不见私有数据**。你公司上周的财报、内部规章,不在任何模型的训练集里。
2. **参数化记忆不可寻址、不可溯源**。模型"记得"某件事,但你无法问它"这句话的依据是哪份文件第几页"——错了也无从核对。
3. **无据时倾向编造(幻觉)**。而企业场景里,一个编造的数字比"我不知道"贵得多。

RAG(Retrieval-Augmented Generation)的本质动作只有一个:**把"靠参数回忆"换成"现场检索 + 现场阅读"**——回答前先从外部知识库里检索证据,把证据塞进 prompt,让模型基于证据作答。这样知识可以随时更新(重建索引即可)、答案可以溯源(引用指回原文)、无据可以拒答(证据为空就不答)。

### 1.2 朴素 RAG 与它的问题谱系

最朴素的 RAG 管道五步:**文档切块 → 向量化入库 → 用户问题向量化 → 取 top-k 相似块 → 塞进 prompt 生成**。十行 LangChain 代码就能跑通,但在真实语料上会系统性地摔跤。值得把坑位列成谱系——因为**一个 RAG 系统的成熟度,就是它对这张谱系表的覆盖率**:

| # | 问题 | 症状 |
|---|------|------|
| P1 | **解析失真** | PDF/扫描件/pptx 不是纯文本;页眉页脚水印混进正文,表格变乱码 |
| P2 | **切块破坏语义** | 固定窗口切断表格、条款、公式;更深层:**检索要小块(语义聚焦),生成要大块(上下文完整)**,一个粒度不可能同时满足两端 |
| P3 | **单路召回盲区** | 纯向量检索丢精确串(法条编号、型号、金额);纯关键词检索丢同义改写 |
| P4 | **多模态盲区** | 图表、纯图页根本不可检索,而财报/论文的关键数据恰恰在图表里 |
| P5 | **权限盲区** | 企业库里不是人人能看所有文档;检索完再过滤,等于无权内容已经污染了 top-k |
| P6 | **生成不忠实** | 召回对了照样编;引用编号是装饰,点开对不上原文;被索引的文档本身还可能是 prompt 注入的通道 |
| P7 | **消费形态单一** | 一问一答管道没法多跳;可反过来,agent 自由检索到底比管道好还是差?多数项目从来没量过 |
| P8 | **不可测量** | 改了 chunk 大小、换了检索策略,到底变好了吗?没有评估闭环,一切优化都是玄学 |

主流演进路径也可以按这张表理解:

- **naive RAG**:五步管道,啥都不管——上表全中。
- **advanced RAG**:在管道两端加料。检索前(query 改写/扩展)、检索中(hybrid 双路、metadata 过滤)、检索后(rerank 精排、small-to-big / parent-document 扩context)。主要对付 P2/P3。
- **modular / agentic RAG**:检索不再是固定管道的一环,而是暴露成工具,由 LLM agent 自己决定何时检索、检索什么、要不要多跳。对付 P7——但注意,"agent 一定更好"是个未经检验的直觉,后文会用数据打它。
- **评估侧**:ragas、TruLens 这类框架给了指标词汇表(faithfulness / answer relevancy / context recall),对付 P8;但有两个深坑大多数团队不碰:一是**裁判和被测系统同厂**带来的循环偏差,二是**评估管道自身的 bug** 会凭空造出假结论(Pharos 真踩过,见 [07 篇](07-evaluation.md))。

### 1.3 顺手回答一个高频面试题:为什么是 RAG,不是微调或长上下文?

三者不是竞争关系,是不同问题的答案,面试里值得一句话说清边界:

- **微调**改变的是模型的**行为与风格**,不适合注入频繁更新的事实(每次更新都重训,且事实仍不可溯源、照样幻觉);
- **长上下文**(把整个库塞进 prompt)有两道硬约束:成本,和"大海捞针"式的衰减——77 篇文档几百万 token,每问都得全量重付,中段信息的召回率还会掉;
- **RAG** 是"外挂可寻址记忆":更新=重建索引,溯源=引用指回 chunk,权限=检索层过滤。代价是引入了一整条需要工程化的管道——这条管道正是本项目的全部内容。

**Pharos 的定位**:一个自包含单仓的多格式 agentic RAG 系统。它对 P1–P8 每个坑都给出一个**有实测背书**的答案,并在本机 4090 上、拿 77 篇真实文档(7652 chunk)跑成一个带多身份鉴权、可观测、systemd 托管的团队服务。它不是玩具管道,也不是框架堆叠——每个关键决策都留有被否决的备选和数据。

---

## 2. Pharos 怎么做:六环节地图

### 2.1 全景数据流

```
索引侧:  文件 ─parse(MinerU)→ Element[] ─chunk→ Chunk[] + heading骨架 + ACL盖章 ─embed→ Qdrant + sidecar
查询侧:  问题 ─encode→ hybrid召回(dense+BM25, RRF, ACL硬过滤) ─→ rerank(可选)
              ─→ small-to-big(assemble_big 取包围区) ─→ generate(grounding + [cite:n]) ─→ 带引用答案
消 费:  闭管道(HTTP /v1/ask) ⊕ agentic(MCP 6工具, agent 自驱)
度 量:  eval 闭环(合成gold → 跑真系统 → 异厂裁判 → 五指标 + 归因)
```

六环节与问题谱系一一对应:

| 环节 | 组件 | 回答哪个坑 | 一句话答案 | 深入篇 |
|------|------|-----------|-----------|--------|
| parse | MinerU 客户端 + adapter | P1 | 5 种格式统一解析成 `Element[]`,换解析器=换 adapter 不动核心 | [02](02-parsing-chunking.md) |
| chunk | `chunker` | P2 | heading 树重建 + eager 骨架切小块;查询期 small-to-big 组装大块 | [02](02-parsing-chunking.md) |
| embed + retrieve | `embedder` | P3/P4/P5 | Qwen3-VL 图文同空间 + BM25,RRF 融合;ACL filter 下推到召回层 | [03](03-retrieval.md) / [04](04-acl-security.md) |
| generate | `generator` | P6 | grounding 约束 + `[cite:n]` 引用协议 + 零召回确定性拒答 | [05](05-generation-grounding.md) |
| consume | `src/pharos` | P7 | 守护进程双出口:HTTP 闭管道 + MCP agentic,共用同一 toolcore | [06](06-agentic-mcp.md) / [08](08-service-architecture.md) |
| measure | `eval/` | P8 | 去偏评估闭环:异厂裁判、五指标、paired 归因 | [07](07-evaluation.md) |

下面按环节过一遍关键机制(细节在各分篇,这里只建立地图)。

### 2.2 parse:统一到 Element 接缝

五种格式(PDF / 扫描件 / docx / pptx / xlsx)全走 MinerU 解析,产物经 adapter 转成统一的 `Element[]`——这是刻意设计的接缝:换解析器等于写一个新 adapter,核心切块逻辑一行不动([src/chunker/adapters/mineru.py:1](../../src/chunker/adapters/mineru.py#L1))。为什么选 MinerU 而不是自研或 Tika?靠的是三方覆盖率对比(公平口径下 MinerU 95.1% 反超,详见 02 篇——那里还有一个"覆盖率提取器自身有 bug 导致最初结论反了"的测量学故事)。

### 2.3 chunk:eager 骨架 + 查询期 small-to-big

这一步正面回答 P2 的"检索粒度 vs 生成粒度"矛盾,分两拍:

- **索引期**:多信号重建 heading 树(text_level 为主 + 编号校正,[src/chunker/core.py:158](../../src/chunker/core.py#L158) `heading_level` / [core.py:179](../../src/chunker/core.py#L179) `build_sections`),在树上切 leaf chunk([core.py:243](../../src/chunker/core.py#L243) `Chunker`)。每个 chunk 带 breadcrumb、section 锚点,并**盖 ACL 章**——默认 RESTRICTED、fail-closed([core.py:33-36](../../src/chunker/core.py#L33))。全树构建实测 66.6ms,所以是 eager 而非 lazy(早期设计叫 "Lazy Heading-Tree",被实测否决,文档名留作化石证据)。
- **查询期**:命中小块后,`assemble_big` 按 token 预算取包围区——小节太小就向上爬父节、超预算就在祖先内开窗拉兄弟([src/chunker/retrieve.py:109](../../src/chunker/retrieve.py#L109))。原始 elements 存在 sidecar 里,取材全程 ACL 感知。

即:**索引小块保检索精度,交付大块保生成上下文,两个粒度解耦**。

### 2.4 embed + retrieve:同空间多模态 + hybrid + ACL 硬过滤

- **多模态(P4)**:Qwen3-VL 把文字和图编进同一向量空间,纯图 chunk 能被文字 query 直接召回(描述↔对应图相似度 0.74/0.49);表格块额外合成检索信号文本(列头/行标签),数据本体存 `content_raw`。索引入口在 [src/embedder/embed.py:56](../../src/embedder/embed.py#L56) `index_document`。
- **hybrid(P3)**:dense(语义)+ BM25(精确串)双路召回,Qdrant 服务端 RRF 融合——RRF 只用名次不用分数,天然解决两路分数纲量不可比([src/embedder/store.py:96](../../src/embedder/store.py#L96) `hybrid_search`)。可选 cross-encoder 精排,失败自动降级不崩查询。
- **ACL(P5)**:filter **下推到每一路 prefetch**([store.py:118-123](../../src/embedder/store.py#L118))——无权内容根本不进候选集,而不是召回后再筛。这里有个著名的坑:嵌入式 Qdrant 在 fusion 模式下会丢顶层 filter 的 should 条件,只有 prefetch 级下推才真正生效([store.py:103-105](../../src/embedder/store.py#L103) 注释留痕)。出口处还有第二道 `acl_admits` 复核([src/embedder/retrieve.py:157-160](../../src/embedder/retrieve.py#L157)),三道闸的完整推演见 [04 篇](04-acl-security.md)。
- **查询期编排**:`search_with_context` 串起召回→section 去重→small-to-big→出口校验,每条命中带 `context_status` 状态机(full_section / section_window / asset_no_prose…),把"上下文完整性"变成 agent 可编程的显式信号([src/embedder/retrieve.py:119](../../src/embedder/retrieve.py#L119))。

### 2.5 generate:闭管道与 grounding

`Generator.answer` 是闭管道的心脏:检索 → context 装配 → prompt → LLM → 引用解析([src/generator/generate.py:35](../../src/generator/generate.py#L35))。三个关键机制:

- **`[cite:n]` 引用协议**:与正文裸 `[n]` token 隔离,防止检索正文里的脚注编号被误抓、防止恶意 chunk 伪造引用;越界编号直接丢弃而非映射错来源([generate.py:116-128](../../src/generator/generate.py#L116))。
- **零召回确定性拒答**:context 为空时代码直接返回"信息不足",不把作答权交给 LLM([generate.py:107-109](../../src/generator/generate.py#L107))——grounding 的底线不靠模型听话。
- **资产数据补喂(③修复)**:表格/图表命中时把 `content_raw` 补进 context([generate.py:75-80](../../src/generator/generate.py#L75)),修掉"表格里的数召回到却答不出"——这是评估闭环揪出的真 bug,完整诊断故事在 [05 篇](05-generation-grounding.md)。

### 2.6 consume:一个守护进程,双消费模式

所有消费都汇到 `pharos serve` 守护进程——系统里**唯一**碰嵌入式 Qdrant 与 GPU 模型的进程([src/pharos/service.py:1-16](../../src/pharos/service.py#L1))。为什么必须是守护进程:嵌入式 Qdrant 单客户端独占锁 + 8B 模型加载 1-2 分钟,"每个消费方各开一个进程"既抢锁又每次重付加载。

两种消费模式,共用同一检索引擎:

- **闭管道**(`POST /v1/ask`,[service.py:312](../../src/pharos/service.py#L312)):检索→grounding→DeepSeek→带引用答案。一问一答、确定性、可评估,**默认推荐**。
- **agentic**(MCP 6 工具:retrieve / list_documents / get_document / get_outline / expand / retrieve_grouped):检索引擎暴露成工具,agent(如 Claude Code)自己决定何时检索、怎么改写、要不要多跳。工具语义(校验/去重/预算/错误映射/使用契约)全部收在 transport 无关的 `toolcore` 单一来源([src/pharos/toolcore.py:1-13](../../src/pharos/toolcore.py#L1)),stdio 与 HTTP 两种绑定共用,契约不漂移。

MCP 有三入口:`pharos serve`(HTTP)、`pharos mcp`(stdio→HTTP 薄适配器,零 GPU、毫秒启动,[src/pharos/mcp_adapter.py:1-12](../../src/pharos/mcp_adapter.py#L1))、`pharos mcp --direct`(无守护进程时的 stdio 直连兜底),路由在 [src/pharos/cli.py:40-47](../../src/pharos/cli.py#L40)。

### 2.7 闭管道 vs agentic:一个引擎,两种消费哲学

这不只是"两个 API",是两种关于"谁掌握控制流"的哲学,值得单独对齐:

| 维度 | 闭管道(`/v1/ask`) | agentic(MCP 工具面) |
|------|-------------------|---------------------|
| 控制流 | 系统固定:检索一次→生成一次 | agent 决定:何时检索、改写几轮、要不要 expand/多跳 |
| 确定性 | 高——同 query 同库,行为可复现 | 低——依赖 agent 的编排策略 |
| 可评估性 | 强,eval 闭环直接量 | 只能整体量(被测 agent 也是变量) |
| 防幻觉手段 | grounding SYSTEM + 零召回代码接管 | toolcore `_INSTRUCTIONS` 使用契约(grounding/引用锚/何时停,[toolcore.py:20-43](../../src/pharos/toolcore.py#L20))——等价物但靠 agent 遵守 |
| 交互形态 | 一问一答,适合脚本/前端 | 多轮探索,适合"帮我梳理这批文档"类开放任务 |
| 实测正确性 | 单跳 0.97(72 题口径) | 净负 Δ−0.097(同口径;幅度待重评,见 §8) |

两个设计不变量把两种模式钉在同一套语义上:**工具契约单一来源**(toolcore,stdio 与 HTTP 零漂移)、**ACL 身份服务端权威**(agent 无法经参数改身份)。而闭管道自身的"智能"被刻意限制成**失败驱动**(smart-ask:只有数值题拒答才触发表格补检腿)——因为一旦在成功路径上"帮忙",闭管道就退化成隐形 agent,而 agent 编排已被实测判为净负(幅度待重评,见 §8)。这条"有界智能"原则展开见 [08 篇](08-service-architecture.md)。

### 2.8 measure:评估闭环

`eval/` 是整个项目的尺子:合成 gold(88 题 = 72 散文 + 16 表格)→ 跑真系统 → 程序化四指标(检索召回 / 全召回 / MRR / 引用召回)+ 裁判两指标(忠实度 / 正确性),裁判分 Tier1(DeepSeek,可复现)与 Tier2(双 Claude,权威、异厂去偏)([eval/run_eval.py:1-16](../../eval/run_eval.py#L1))。核心纪律:**裁判与被测系统异厂**,否则数字有循环偏差。这套闭环不仅打分,还做 paired 归因(single vs agentic vs decompose)——上一节的 Δ−0.097 就是它裁决出来的(幅度待重评,见 §8:eval 的 agentic 组装路径少装了两处生产修复)。

Tier1 基线一组数记住量级即可(deepseek 裁判,88 题,闭管道 single):检索召回 0.818 / MRR 0.627 / 引用召回 0.767 / 忠实度 0.977 / 正确性 0.818。Tier2 权威口径(双 Claude,72 题)忠实度 ≈1.000、正确性 0.847。**两套口径并存、不可混比**——为什么会有口径断代、如何诚实处理,是 [07 篇](07-evaluation.md) 的重头戏。

### 2.9 一条请求的完整生命线

面试时最能证明"这系统是你写的"的,是能不看资料走完一条请求。以 `POST /v1/ask {"query": "X 公司 2023 年分部营收是多少?"}` 为例:

1. **鉴权与身份**:X-API-Key 解析成身份(name+tenant+principals);keys/legacy/open 三模式自动选;非回环绑定强制 keys 模式,拒绝把整库裸奔到局域网([service.py:93-95](../../src/pharos/service.py#L93))。
2. **构建 Generator**(per-thread 惰性,[service.py:304-310](../../src/pharos/service.py#L304)),进入 `gen.answer`。
3. **query 编码**:dense 向量(Qwen3-VL,LRU 缓存)+ BM25 sparse 向量。GPU 前向在资源锁内,LLM 网络调用在所有锁外——锁模型见 [src/pharos/engine.py:6-12](../../src/pharos/engine.py#L6)。
4. **hybrid 召回**:两路 prefetch(各自带 ACL filter)→ 服务端 RRF 融合 → 出口 `acl_admits` 复核([store.py:96-129](../../src/embedder/store.py#L96))。
5. **section 去重 + small-to-big**:命中小块 → sidecar 读原始 elements → `assemble_big` 爬树/开窗组装大块,每条标 `context_status`([embedder/retrieve.py:119](../../src/embedder/retrieve.py#L119))。
6. **context 装配**:表格命中补 `content_raw`;section_path 面包屑并进 source 行(否则模型不知道数字是分部数据,实测会把分部营收错当公司总营收)([generate.py:75-89](../../src/generator/generate.py#L75))。
7. **prompt → DeepSeek**:SYSTEM 声明 passage 是不可信数据、要求 `[cite:n]`;passage 内已有的 `[cite:n]` 被中和(防索引文档注入)。
8. **引用解析**:答案里的 `[cite:n]` 映射回来源,越界丢弃([generate.py:116-128](../../src/generator/generate.py#L116))。
9. **smart-ask 失败驱动重试**:数值题拒答时,带 `kind=table` 检索腿重问一轮,**只采用完整答出的重试**——部分回答会夹带错误的"未提供 X"声明,实测忠实度 1.0→0.93,宁可保留诚实拒答([service.py:331-353](../../src/pharos/service.py#L331))。
10. **响应与观测**:HTTP 200 + status 字段(领域错误不用 HTTP 码表达,agent 程序化消费);请求日志记身份名不记 key。

这条线上每个环节都有对应分篇;把它讲熟,面试里任何"深挖某一层"的追问都有落点。

---

## 3. 为什么这么设计:被否决的备选

架构叙事的说服力来自"考虑过什么、为什么不"。Pharos 的关键否决(数据口径随注):

| 决策 | 被否决的备选 | 否决理由与数据 |
|------|------------|--------------|
| **守护进程独占资源** | 每个 agent 会话 stdio 直连、各开索引 | 嵌入式 Qdrant 单客户端独占锁,第二个进程直接报错;8B 模型加载 1-2 分钟,每会话重付。守护进程 + MCP 薄适配器后,会话接入毫秒级 |
| **闭管道为默认** | agentic(agent 自驱多跳)为默认 | paired 归因实测(72 题口径):single→agentic 正确性 Δ**−0.097**,→decompose Δ−0.014。agent 每跳都 ≤ 单跳(多轮检索 = 干扰块稀释 context),只在跨文档题微弱占优。**agentic 是能力选项,不是默认路径**——这是全项目最反直觉、也最有数据底气的结论(注意:本轮审查发现 eval 的 agentic 路径少装了两个生产侧修复,净负的**幅度**存疑待重评,方向暂无推翻证据,见 [06 篇](06-agentic-mcp.md)) |
| **eager 建 heading 骨架** | lazy(查询期才建树,听起来更聪明) | 77 篇实测全树构建仅 66.6ms——"懒"省下的是不存在的成本。教训:性能优化先测再做 |
| **hybrid RRF** | 单路 dense 或单路 BM25 | 组件级评估(合成集,趋势可信绝对值仅参考):hybrid 0.541 > dense 0.510 > sparse 0.449;开 rerank 后 0.566→0.867 |
| **自研 chunker** | chonkie 等现成切块库 | 三方对比 ours 79.5% vs chonkie 74.9%(证据保真口径);且护城河不在切块边界算法,在工程集成——source_indices 溯源、heading 树、ACL 盖章,这些库给不了 |
| **MinerU 解析** | 自研解析器 / Tika | 覆盖率打平(公平口径 MinerU 95.1% 反超)前提下,content_list schema 切块即用、五格式管线统一、无 JVM |
| **prompt grounding + 代码接管拒答** | 引入 NLI/事实校验模型 | 简单零依赖;可确定判定的分支(零召回)直接代码接管,不赌模型听话。Tier2 实测忠实度 ≈1.000(72 题口径),暂无引入重型校验的必要 |
| **嵌入式 Qdrant 起步** | 一步到位 Qdrant server 集群 | 渐进路线:单机零依赖先跑通、先评估;到阶段 F 才真拆三层(inference / pharos 多副本 / qdrant server),因为那时才有多副本需求。详见 [09 篇](09-scale-out.md) |
| **HTTP 200 + status 字段表达领域错误** | REST 式 4xx/5xx | agent 是程序化消费方,靠解析自然语言错误不可靠;结构化 status + retriable 标志让 agent 知道该重试还是换策略 |

注意口径纪律:上表 72 题与 88 题两套数字**不可混比**(88 题是表格题扩充后的口径,两者并存、各自标注)。这个"口径断代"本身就是评估方法论的教学点,见 [07 篇](07-evaluation.md)。

---

## 4. 实战复盘:写作前的全库对抗审查

这套学习文档动笔前,先对六个子系统做了一轮并行深读 + 对抗审查(6 个分析师 agent 产出知识地图,同时收集 35 条"疑似设计问题",每条派独立验证员**先试图反驳**)。结果:**34 confirmed / 1 refuted**(refuted = service#0 readyz 绕 Store._lock);34 confirmed 里有 2 条是跨层重复报告(deploy 复审重报 embedder#5 / service#4,已随修),去重后 **32 个不同问题**。这轮审查本身是最好的架构复习材料:

**已修(17 项直接落地)**:全部是"行为无关的健壮性修复"。验证证据:修复前基线 `pytest tests -q` = 224 passed,修复后 = 259 passed(新增 36 用例)。挑两个有全景教学价值的:

- **index_document 删旧时机重排**(embedder#11):原实现"先删旧向量→再逐 chunk 编码",大文档编码要几分钟,中途失败=该文档**持久脱库**。修法把顺序重排为"编码 + sidecar tmp 全部备好(纯准备,失败无副作用)→ delete → upsert → 原子改名",危险窗口从分钟级缩到毫秒级([src/embedder/embed.py:56](../../src/embedder/embed.py#L56) docstring 记录了完整失败面分析)。教学点:**索引管道的失败面要按"哪一步之后失败会留下不可恢复状态"来排序操作**。
- **healthz 信息收敛**(service#8):未鉴权的 /healthz 泄漏 collection 名与 llm_model。单看无害,但与 readyz 的加固不一致——安全边界的价值取决于最松的那块板。

**延期(15 项,confirmed 但不动)**:凡改变切块/检索/生成**输出内容**、或需 GPU/compose 环境验证的,一律延期并写明修法草图。为什么"确认了还不修"是正确的工程判断:

- 切块改动会导致 chunk id 平移(仓库实测过幽灵表格块让全库 7652→7675 的教训),旧索引全部作废;
- 检索交付内容的改动会让已发布的评估数字失真——**改动必须与"重建索引 + 重跑 GPU eval"绑定成一个原子动作**,否则数字和代码就脱节了。

典型如 chunker#0(ACL 感知路径下 big-block 系统性丢失全部标题文本,LLM 收到的 context 缺小节边界信号):根因清楚、修法草图齐备,但必须 bump SIDECAR_VERSION + 重建索引 + 重跑 eval 才能落——细节见 [02 篇](02-parsing-chunking.md)。各簇修复/延期的完整清单散在对应分篇的实战复盘节。

---

## 5. 面试怎么讲

### 30 秒电梯稿

> 我做了 Pharos,一个多格式 agentic RAG 系统:把 PDF、扫描件、Office 文档变成能问答、带企业级权限控制、可溯源引用的本地知识库,在单卡 4090 上服务小团队。它有两种消费模式——HTTP 闭管道问答和 MCP 工具面(给 Claude Code 这类 agent 自驱检索)。区别于一般 RAG demo 的是三点:权限过滤下推到召回层做到 fail-closed;有一套异厂裁判的去偏评估闭环,实测忠实度接近 1.0;所有关键决策都有被否决的备选和数据——包括用实测推翻了"agentic 一定更好"的直觉。

### 3 分钟结构化版

1. **问题**(30s):团队文档库要能问答,但朴素 RAG 有一串坑——解析失真、切块破坏语义、单路召回盲区、图表检索不到、权限裸奔、生成编造、改了不知好坏。我按这个问题谱系逐个给答案。
2. **架构**(60s):索引侧 MinerU 统一解析五种格式,chunker 重建 heading 树、切小块、盖 ACL 章;查询侧 dense+BM25 双路召回 RRF 融合、ACL filter 下推到每路 prefetch、small-to-big 组装大块喂 LLM、`[cite:n]` 引用协议保溯源。整个跑在一个守护进程里(嵌入式 Qdrant 独占锁 + 8B 模型加载 1-2 分钟决定了必须常驻),双出口:HTTP 闭管道 + MCP 工具面,工具语义单一来源不漂移。
3. **数据**(45s):生产库 77 篇真实文档 7652 chunk;88 题评估集,程序化四指标 + 异厂裁判两指标。忠实度 ≈1.0——系统宁可拒答不编造;单跳正确性 0.97(72 题口径)。最反直觉的结论:paired 归因显示 agent 自驱多跳比闭管道正确性低约 0.1,所以闭管道才是默认,agentic 是选项。
4. **方法论**(45s):每个组件封板前做对抗 review,全栈再做了五批系统性复审,揪出过评估管道自身的 bug——裁判 context 被截断,凭空造出"17% 幻觉"的假结论,重判后订正到 ≈1.0。这件事让我对"评估基建先于优化"有了肌肉记忆。

(收尾留钩子:权限、评估、扩展性三个方向都有完整的深挖故事,面试官挑哪个都能接。)

---

## 6. 追问预演

1. **"为什么不用 LangChain / LlamaIndex?"**
   答法:不是抵触框架,是这个项目的价值恰恰在框架帮不了的地方——heading 树重建、ACL 下推到 prefetch、small-to-big 的 ACL 感知取材、评估闭环,这些都要贴着存储引擎和语料写。框架的抽象在"每个环节都要可测可控"的目标下反而是负资产。可以补一句:接缝都留了(Element adapter、LLM 可插拔、retriever duck-typing),不排斥在外围用框架。

2. **"为什么闭管道是默认,不是让 agent 自由检索?"**
   关键词:paired 归因、Δ−0.097、干扰块稀释。答法:直觉上 agent 多跳更强,但同一套 gold 上 paired 对比,agentic 正确性净负约 0.1——多轮检索带进更多干扰块,每跳都不如单跳干净。只在跨文档题上 agent 微弱占优。所以默认闭管道,agentic 留给真需要交互探索的场景。诚实加注:后来发现 eval 的 agentic 路径少装了两个生产修复,幅度要重评,方向暂无推翻证据——主动说这个反而加分。

3. **"hybrid 检索为什么用 RRF 不用加权分数融合?"**
   关键词:分数纲量不可比。dense 的 cosine 和 BM25 的分数不在一个量纲上,加权要调超参且脆;RRF 只用名次,零超参,Qdrant 服务端原生支持。数据:hybrid 0.541 > dense 0.510 > BM25 0.449(合成集,趋势口径)。

4. **"权限怎么做的?检索完再过滤不行吗?"**
   关键词:fail-closed、下推、三道闸。检索后过滤有两个洞:无权内容占掉 top-k 名额(结果质量泄漏),以及代码路径一多必有漏筛。Pharos 把 ACL filter 下推到每路 prefetch(无权内容根本不进候选),出口再复核一道,small-to-big 取材也按 ACL 等价类门控。杀手锏证据:回归测试专门**禁掉出口闸**重测,跨租户仍 0 召回——证明第一道闸自身有效,不是靠兜底装出来的安全。细节引到 [04 篇](04-acl-security.md)。

5. **"怎么证明你的系统不编造?"**
   关键词:忠实度裁判、异厂、评估 bug 故事。答法:忠实度指标 = 答案里每个论断是否被喂进去的 context 支持,由异厂模型(Claude)裁判,实测 ≈1.0。最值得讲的是曾经报出 0.83:自审发现是裁判的 context 被截断、只看到 40% 段落——评估管道自己的 bug 凭空造出假结论。喂全 context 重判后订正。这个故事同时回答了"你的数字为什么可信":因为评估管道本身也被对抗审查过。

6. **"单机 4090,怎么扩展?瓶颈在哪?"**
   关键词:三层拆分、吞吐上界。已完成阶段 A–F 改造:GPU 前向拆成独立 inference 服务、应用层脱 torch 可 `--scale pharos=N` 多副本、嵌入式 Qdrant 转 server 模式、nginx 前置。诚实结论主动给:**吞吐上界 = 单卡 GPU 前向串行,不随副本数上升**;多副本扩的是非 GPU 并发、崩溃隔离、滚动升级。突破上界的路径是 vLLM continuous batching,已有等价性 probe 和 go/no-go 门。细节在 [09 篇](09-scale-out.md)。

7. **"表格里的数字答不出来,你怎么排查的?"**
   关键词:诊断纪律、根因链。评估暴露症状(召回对了答不出)→ 初判"content_raw 没喂"被逐题实跟推翻 → 真根因是资产块被 section 去重折叠 + big-block 组装排除双重丢失 → 修两处(资产块免去重 + generator 补喂 content_raw)→ 4 道表格题从"信息不足"变答对,单跳正确性到 0.97。完整故事在 [05 篇](05-generation-grounding.md)。

8. **"88 题够吗?gold 是模型合成的,可信吗?"**
   关键词:诚实 caveat、分层可信度。不回避:量确实小,跨文档题只有 5 道(所以 cross-doc 0.00 只当方向信号);合成 gold 有同源偏置(问题措辞与 golden chunk 词汇重叠,检索偏容易),缓解手段是 prompt 禁指代、多跳题由异厂模型跨 chunk 构造、表格题加程序化 QC 闸门。评估的定位是**趋势尺**而非绝对分——同一把尺子量改动前后,趋势可信。

---

## 7. 动手实验

### 实验 1(CPU,零 GPU/网络):用 fake 层走通产品面

产品层测试全部跑在 fake retriever + MockLLM 上,是理解"服务层消费什么契约"的最快路径:

```bash
cd projects/pharos
pip install -e .[dev]                       # src-layout 可编辑安装
python -m pytest tests -q --ignore=tests/engine   # 产品层:HTTP 服务/身份/会话/适配器
python -m pytest tests/engine/test_generate.py tests/engine/test_tools.py -q  # 闭管道与工具面脚手架
```

然后读 [tests/_fakes.py](../../tests/_fakes.py):`FakeRetriever` 用 duck-typing 对齐真 `embedder.Retriever` 的方法面,`make_hit`/`make_res` 伪造命中与 big-block。**练习**:改 `make_res` 的 `status` 为 `section_window`,跑 `tests/test_service.py`,观察 toolcore 如何把 context_status 透传给 agent——这就是 2.4 节状态机的可触摸版本。

### 实验 2(GPU/WSL,前置:WSL `navikb` 环境 + 4090 + 已建索引 `~/rag_real`):走一遍真生命线

```bash
conda activate navikb
python -m pharos health                      # 确认守护进程在(systemd 托管)
python -m pharos ask "库里关于 X 的内容有哪些?"   # 闭管道:观察 citations 与 finish_reason
```

对照第 2.8 节的十步生命线,在 `journalctl -u pharos -f` 里看请求日志(注意:记身份名不记 key)。想量化任何改动,评估闭环入口是 `eval/run_eval.py`(需先 `systemctl stop pharos` 释放嵌入式锁的场景与坑,见 [07 篇](07-evaluation.md) 与 [../OPERATIONS.md](../OPERATIONS.md))。

---

## 8. 诚实边界

面试中主动承认弱点比被挖出来强得多。本系统的已知边界:

1. **跨文档综合是硬伤**:cross-doc 正确性 0.00(n=5,72 题口径)。拿到两篇的块也合不出对比结论——这是综合能力问题不是检索问题,判断为研究性难题、边际收益低,**明确决定不追**。话术:"我知道它答不了什么,并且把这个边界写进了用户文档——弱项自己交叉核对。"
2. **agentic 净负的幅度存疑**:本轮审查确认 eval 的 agentic/decompose 路径绕过了生产 Generator 的两个修复(资产 content_raw 补回、面包屑),对 agentic 系统性不利。方向(不占优)暂无推翻证据,但 Δ−0.097 这个数不能再当铁数引用,待重跑。
3. **合成 gold 的同源偏置**:出题人见过 golden chunk,检索指标天然偏乐观。缓解手段有,但根治要真实用户查询日志——小团队场景暂时没有。
4. **评估量小、口径多**:88 vs 72 两套口径并存不可混比;组件级评估绝对值仅参考。这是刻意保留的诚实,不是疏忽。
5. **吞吐上界没突破**:多副本改造不增加 GPU 前向吞吐,vLLM 路线还在 go/no-go 门前。
6. **启发式的残留误差**:est_tokens 按字符除数估 token,数字密集文档误差大(财报 5.08 vs 假设 4.0),两个误差方向都有兜底但不是精确契约。
7. **本轮审查还有 15 项 confirmed 未修**:全部有修法草图和延期理由(绑定索引重建/GPU eval),不是技术债失控,是变更纪律——但"确认了还没修"这句话要自己先说。

---

*下一篇:[02 文档解析与切块](02-parsing-chunking.md)——heading 树多信号重建与"lazy 命名被实测反转"的方法论故事。*
