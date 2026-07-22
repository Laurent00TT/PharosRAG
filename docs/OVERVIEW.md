# Pharos — 多格式 agentic RAG 系统总览

> 顶层入口文档。系统全貌、组件关系、核心设计、评估结论与当前状态收在一处。新人从这里读起。
> 深入细节:[DESIGN](DESIGN.md) · [IMPLEMENTATION](IMPLEMENTATION.md) · [API](API.md) · [OPERATIONS](OPERATIONS.md) · [TESTING](TESTING.md) · [PROVENANCE](PROVENANCE.md)。
> 组件细节:[chunker](components/chunker/ARCHITECTURE.md) · [embedder DESIGN](components/embedder/DESIGN.md) · [generator DESIGN](components/generator/DESIGN.md) · [mcp-server](components/mcp-server.md) · [eval](../eval/README.md)。
> 评审全过程:[methodology/REVIEW_PLAN.md](methodology/REVIEW_PLAN.md)。

## 1. 这是什么

Pharos 是一个**自包含单仓的多格式 agentic RAG 系统**:把 PDF / 扫描件 / docx / pptx / xlsx 变成**能问答、带企业级 ACL、可溯源引用**的本地知识库(本机 4090)。不是切块实验——每个关键决策都有实测背书,整套经过 88 题去偏评估 + 多轮对抗评审加固,并已在 77 篇真文档上跑通。检索引擎与面向小团队的部署形态(双出口 + 多身份 + 可观测 + systemd 托管)现已**合并在同一个仓库**里,一次 `pip install -e '.[dev]'` 装齐(src-layout,editable)。

**两种消费模式,共用同一检索引擎:**
- **闭管道**(`generator` + DeepSeek):一问一答、确定性、可评估 —— **默认推荐**(评估证明它最好最省)。
- **agentic**(MCP 三入口,共用同一 `toolcore`):检索引擎暴露成 MCP 工具,agent(如 Claude Code)自驱多跳、交互强。

## 2. 全景:索引侧 + 查询侧 + 一条贯穿线

```
索引侧:  文件 ─parse(MinerU)─► Element[] ─chunk─► Chunk[] + heading 骨架 ─embed─► Qdrant + sidecar
查询侧:  问题 ─encode─► hybrid 召回(dense + BM25, RRF) ─► rerank(可选) ─► ACL 硬过滤
              ─► 查询期 small-to-big(assemble_big 取包围区) ─► generate(LLM + grounding + [cite:n]) ─► 带引用答案
贯穿:    ACL fail-closed 从 chunk 盖章一路守到生成出口;多模态图文同空间;评估闭环量化每一步
```

## 3. 五个组件(全部 设计→实现→对抗封板→评估;R1-R5 复审加固)

| 环节 | 组件 | 干什么 | 状态 |
|---|---|---|---|
| **parse** | MinerU 客户端 | 5 格式 → 统一 content_list/layout | ✅ |
| **chunk** | `chunker` | Element[] → heading 树切块;查询期 small-to-big | ✅ 封板 + 评估 |
| **embed + retrieve** | `embedder` | Qwen3-VL dense + BM25 → Qdrant hybrid + ACL 硬过滤 + 可选 rerank + small-to-big | ✅ 封板 + 评估 |
| **generate**(闭管道) | `generator` | 检索 → prompt → LLM → 答案 + `[cite:n]` + grounding;LLM 可插拔(DeepSeek V4 Flash) | ✅ 端到端评估(88 题) |
| **consume**(agentic) | `src/pharos`(`toolcore`) | retriever 暴露成 **6 个 MCP 工具**(retrieve/list_documents/get_document/get_outline/expand/retrieve_grouped),经三入口(`pharos serve` / `pharos mcp` / `pharos mcp --direct`)对外;ACL 身份启动绑定不可篡改 | ✅ 工具单测 + 契约无漂移测 + 真索引连通 |
| **measure** | `eval` | 去偏评估闭环:合成 gold → 跑系统 → 裁判(Tier1 deepseek 可复现 / Tier2 双 Claude 权威)→ 五指标 + 双层归因 + ACL 回归 | ✅ 见 §8 |

**单一 pytest 套件**(产品面 + 引擎面,后者收在 `tests/engine/`;基数见 [TESTING.md §1](TESTING.md))。CI 门槛拆两级:CPU CI = pytest(含 embedder `test_acl.py` ACL 谓词);GPU 发版前 = `eval/acl_regression.py`(WSL+4090,端到端 0 泄漏,不进 CPU CI)。

## 4. 亮点①:eager 廉价骨架 + 查询期 small-to-big

> **命名澄清**:早期叫 "Lazy Heading-Tree",经 77 篇/5337 标题对抗实证**否决了"懒"**——eager 全树才 66.6ms。真实架构是 **eager 骨架 + 查询期只组装 big-block**。

- **索引期 eager 建骨架**:多信号(text_level + 编号校正 + TOC 剥离)重建 heading 层级,切 leaf chunks,breadcrumb + `section_anchor` 挂进每个 chunk。亚毫秒、确定性、零 LLM。
- **查询期只组装 big-block**(`assemble_big`):命中 leaf → 读挂好的 breadcrumb + 按 token **取包围区**(过小向上 climb 父 section,超 max 在祖先内开窗拉相邻兄弟),从原始 elements 按 idx **ACL 感知**取材。sidecar 存原始 elements/sections。
- **返回状态自陈**(`context_status`):full_section/climbed_N=完整小节可直接用;**section_window**=token 受限窗口(非完整,可 expand);**asset_no_prose**=资产页数据在 content_raw;single_chunk_*/deduped/omitted_budget/already_returned 各有语义。

## 5. 亮点②:ACL 端到端 fail-closed

权限全程贯穿:chunk 盖章 → embed 拆 4 字段 → 检索硬预过滤(Qdrant filter **下推每个 prefetch** —— 嵌入式 fusion 会丢顶层 should 的坑)→ small-to-big 不跨 ACL 取材 → 出口二次校验(acl=None 也拒)→ generate 只喂授权 context。**无权文档"根本检索不到"而非"藏起来"**。回归:`eval/acl_regression.py` 44+ 断言全过,含 **"禁掉出口 acl_admits 后跨租户仍 0 召回"**——证明 RRF fusion 的 prefetch 下推**本身**就挡越权(非靠出口兜底)。

## 6. 亮点③:表格/图表数值 grounding(③ 修复)+ 多模态 + hybrid

- **③ 表格/数值 grounding**:评估发现"表格里的数召回到却答不出"——真根因是资产块的 `content_raw`(表 HTML/图表数据)被 section 去重折叠、又被 big-block 组装排除。修:retrieve 资产块不进 section 去重 + generator 资产命中补 content_raw。实测 4 道表格题从"信息不足"→答出正确数,single-hop 正确性达 0.97。
- **图文同空间**(Qwen3-VL):文字与图编进同一向量空间(描述↔对应图 0.74/0.49),纯图 chunk 能被文字 query 召回。
- **hybrid**:dense(意思)+ BM25(词)RRF 融合;可选 Qwen3-VL-Reranker 精排(实测 hybrid 0.566→rerank 0.867)。

## 7. 评估:组件级 + 系统级两层

**组件级(chunking / retrieval,合成集,趋势可信绝对值仅参考):** 自研切块护城河=工程集成(source_indices/heading-tree/ACL)非边界算法(ours 79.5% vs chonkie 74.9%);hybrid+RRF > 单路(0.541>0.510>0.449);rerank 大幅提质(→0.867);sparse 选 BM25。评估分三线:`eval/`(端到端五指标,皇冠)· `eval/component_retrieval/`(BM25/BGE/RRF 检索组件)· `eval/component_chunking/`(切块证据保真)。详见 [components/embedder/EVALUATION.md](components/embedder/EVALUATION.md)、[methodology/CHUNKING_EVALUATION.md](methodology/CHUNKING_EVALUATION.md)。

**系统级(端到端 RAG,`eval/`):gold = 88 题(72 散文 + 16 表格)**,hop 拆分 single 54 / multi_intra 29 / multi_cross 5,表格题全为 single-hop。可复现性分两档:**Tier1**(`--judge deepseek`,仓内可复现、同厂趋势)vs **Tier2**(双 Claude 权威,不进仓内可复现——需外部 Claude-Code 编排产出 `verdicts.json`)。

**Tier2 权威口径(双 Claude,R5 订正,闭管道 single,72 题口径):**

| 指标 | single(闭管道) | agentic | decompose |
|---|---|---|---|
| **忠实度** | **≈1.000** | 1.000 | 0.972 |
| 正确性 | 0.847 | 0.750 | 0.831 |
| 检索召回 | 0.854 | 0.840 | 0.852 |

**Tier1 基线(deepseek 裁判,88 题,闭管道 single):** 检索 0.818 / MRR 0.627 / 引用 0.767 / 忠实 0.977 / 正确 0.818。

> ⚠ **口径断代**:上表 72 题口径早于表格扩充,88 vs 72 **不可直接比较**——两者并存、各自标注。按 hop 正确性(72 口径):single-hop 0.97 / 单篇多跳 0.83 / 跨文档多跳 0.00(n=5)。双层归因(paired):single→agentic Δ**−0.097**、→decompose Δ**−0.014**。

**三条硬结论:**
1. **忠实度 ≈ 1.0,grounding 几乎滴水不漏**——系统宁可答"无相关信息"也不编,`[cite:n]` 溯源可靠。
   > ⚠ 早先曾报"忠实度 0.83 / 17% 无据论断",R5 自审查出那是 **eval bug**(裁判 context 被截断只看到 40%),喂全 context 重判后订正到 ≈1.0。**教训:评估管道自身的 bug 能凭空造出一个假"结论"。**
2. **agentic / decompose 净负** → **闭管道应为默认**:agent 编排每个 hop 都 ≤ 单跳(多检索=干扰块稀释),只在跨文档微弱占优,整体不及。
3. **正确性瓶颈只剩跨文档综合**(0.00,n=5):拿到两篇块也合不出对比,是综合不是检索;研究性难题,边际收益低,未追。

## 8. 对抗评审 R1-R5(全库系统性复审,[methodology/REVIEW_PLAN.md](methodology/REVIEW_PLAN.md))

| 批 | 范围 | 结果 |
|---|---|---|
| R1 | ACL / 安全闭环 | **0 confirmed**(干净;fusion 下推经"禁出口闸"证实本身挡越权) |
| R2 | 检索正确性 + ③ | 7 confirmed,修 6(窗口块错标 full_section 等) |
| R3 | generator + 引用 + prompt | 9→6,修 6(passage 注入中和、finish_reason 透出、thinking 按后端门控) |
| R4 | MCP 工具面 | 15→7,修 7(content_raw 绕过预算+降级不清、section_window 跨调用去重) |
| R5 | eval 方法论 | 16→6(2 HIGH),修 6 + 重判(**揪出上面的忠实度 bug**) |

**跨批元教训:一个修复(③)的涟漪会在下游层(生成/MCP/评估)冒出新回归——只有系统性、跨批的对抗复审逮得住**(R4 逮到 R2 的后遗症、R5 逮到评估 bug)。对抗验证一路驳回 ~40 条夸大/误报(挡噪声、只留真问题)。

## 9. 工程方法论

- **测试集驱动**:抽样 77 文档/1867 页归纳策略,评估闭环用数据定夺。
- **诊断→修复→验证纪律**:不盲改(③ 初判"content_raw 没喂"被逐题实跟推翻;忠实度 bug 靠 CTX_CAP 长度实测坐实)。任何"已修"附复现 + 改后对比。
- **对抗封板 + 系统性复审**:每组件封板前 red-team;全栈 5 批 R1-R5 复审。
- **每个决策有实测背书 + 诚实 caveat**(承认语义评估集同源偏置、cross-doc n=5 偏小、忠实度 bug)。

## 10. 当前状态 + 怎么用

**功能全部完成**:五组件 + MCP 三入口建好、单一 pytest 套件全绿(基数见 [TESTING.md §1](TESTING.md))+ 88 题去偏评估 + 5 批对抗加固,git 干净。当前版本 **v0.3.0**。

**已投产**:生产索引 `~/rag_real`(`PHAROS_INDEX_DIR`)≈ **77 篇真文档 / 7652 chunk**(14 类:论文/财报/研报/法规/政府/手册/幻灯/NASA 技报/新闻…);评估用库 `~/rag_eval_big`(evalbig)≈ 15 篇 / 1409 chunk。配置统一走仓根一个 `.env`(`PHAROS_*` 命名空间;`RAG_*`/`RAG_EVAL_*` 仅保留一版弃用别名),语料目录 `PHAROS_CORPUS_DIR`、索引目录 `PHAROS_INDEX_DIR`。

**怎么用(三入口,共用同一 `toolcore`)**:
- `pharos serve` —— HTTP 常驻守护进程,持有嵌入式 Qdrant 锁 + GPU 模型;团队多用户、可观测、systemd 托管的核心。
- `pharos mcp` —— stdio→HTTP 适配器,给 agent(如 Claude Code)接入常驻服务。
- `pharos mcp --direct` —— stdio 直连、无守护进程的回退路径。

上手:①(可选)重建索引指向你的语料;② 起 `pharos serve`(首查 lazy load 模型约 1-2 min),或 agent 侧配 `pharos mcp`;③ 直接问。
- ✅ **强项**:单篇事实、表格数值、单跳问答 —— 信 `[cite:n]` 引用(忠实度 ≈1.0)。
- ⚠ **弱项**:多篇深度综合/对比(cross-doc 0.00)—— 自己交叉核对。
- 想衡量任何改动:`eval/` 那套闭环就是尺子(运维/复现细节见 [OPERATIONS.md](OPERATIONS.md)、测试门槛见 [TESTING.md](TESTING.md))。

**已落地(曾经的非目标)**:早期作为个人单用户引擎时,把 "HTTP 常驻 / 多会话守护" 列为**明确不做**。合仓成 Pharos 后这些正是当前形态——`pharos serve` HTTP 守护进程 + 团队多身份(D10)+ 可观测(D11)均**已完成**,不再是非目标。

**仍明确不做(决策,非欠账,见 [ROADMAP.md](ROADMAP.md))**:跨文档综合提升(研究性难题、边际收益低)、MCP 取图工具(image_path 远端解不了)。

## 文档地图

```
docs/OVERVIEW.md                                 ← 本文(系统入口,新人先读)
docs/{DESIGN,IMPLEMENTATION,API,OPERATIONS,TESTING,ROADMAP}.md  设计 / 实现 / 接口 / 运维 / 测试 / 路线图
docs/{PROVENANCE,COMPONENT_NOTES}.md             溯源(来路/决策沿革)+ 组件笔记
docs/methodology/{LAZY_HEADING_TREE_DESIGN,MULTIFORMAT_IMPL,CHUNKING_EVALUATION,REVIEW_PLAN}.md   方法论 + 评审计划
docs/components/{chunker,embedder,generator}/*.md + components/mcp-server.md   组件文档
docs/archive/*                                   历史归档(引擎原型 DESIGN/IMPLEMENTATION、过程日志)
eval/README.md                                   端到端 RAG 评估闭环(去偏 / 双层归因 / ACL 回归 / Tier1·Tier2)
```
