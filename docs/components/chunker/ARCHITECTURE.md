# 架构 / 设计

## 1. 在管线里的位置

```
ingest ──► parse ──► [ chunker ] ──► embed ──► (vector store)
                        │  此组件
            from_mineru │                         查询期:
            (Element[]) ▼                         retrieve → hit chunk
                   Chunk[] + Section[]  ───────►  chunker.assemble_big(hit) → BigBlock → LLM
```

组件**只认归一化的 `Element[]`**(不绑 MinerU 的具体 JSON)。`adapters/mineru.py` 把 MinerU 的 `content_list.json` + `layout.json` 翻成 `Element[]`,是 parse 阶段的接缝——换 parser 只换适配器。

## 2. ingest 期数据流(`Chunker.chunk`)

```
Element[]
  │  1. 噪声过滤        丢 kind ∈ {header, footer, page_number}
  │  2. 标题检测+定级    text_level 为主 + 编号分段校正 + 年份守卫 + TOC/超长守卫
  │  3. 单调栈           while 栈顶.level >= 当前: pop → breadcrumb;同时收集 headings
  │  4. 节树            build_sections: 每节 [start_idx, 下一个 level<=自身的标题) + parent
  │  5. 资产原子化       table/image/chart 各成一 chunk(body=生成负载;检索文本=caption+脚注,
  │                     image/chart 另折 VLM 描述,table 另折面包屑+表头+行标签(封顶,数据单元格
  │                     不进)——只有 caption 时数字题的表格会被散文挤出 top-k,Pharos 实测后补)
  │  6. 文本组装        assemble_text: 同节连续文本块按 token 预算贪心累积(欠下限并、超上限句切)
  │  7. 挂 anchor       每 chunk 标 section_id + section_anchor([节的 idx 范围])
  ▼
ChunkResult(chunks, sections)
```

## 3. 多信号定级(核心,`heading_level`)

| 信号 | 用法 | 备注 |
|---|---|---|
| `text_level` | **主信号**(parser 免费给) | 基线;干净文档 parser 已正确,脏文档被压平到单一 level |
| 小数点编号 | **细分**:`2.1`/`3.4.2` 按 `.` 段数定深度(`2.1`→2) | 仅小数点提级——真章/节分层信号,无歧义 |
| 裸整数编号(**reset-aware**) | 仅当全文裸整数序列**单调递增**(真大纲)才提级为 L1;一旦出现**重启**(`1…9,1…` = 列表用法)就放弃,回落 `text_level` | **核心修复**(对抗审核 F1):防把 "1. 海外AI:" 这类循环编号列表项误升为顶级章节 |
| 项目符号守卫 | `-•●○▪◦` 等开头 → 不是标题(F3) | 防把 "- OpenAI…" 正文条目当成节 |
| 年份守卫 | 首段 `19xx/20xx` 不当编号 | 防 `2026 业绩` 被当 depth-1 |
| TOC / 超长守卫 | 点引导+尾页码、>120 字符 → 不是标题 | 防目录条目/正文段污染 |
| `law` SEC. | `SEC. N` → level 1 | 法条前缀 |

**为什么是融合而非赌单一信号**:真实语料(77 篇 / 5337 标题)只有 ~11% 标题能解析出编号;无编号是主体。所以 `text_level` 必须是主、编号是**有条件**校正——见 §6/§7。
**为什么 reset-aware 是必须的**:同一 `doc_type` 下编号语义会相反——某周报 "1. 海外AI:" 是新闻列表项(不该提级),某深度报告 "1. 端侧AI开启…" 是真章节(该提级)。唯一能区分的结构信号是**单调性**:列表会重启(`1…9,1…`),大纲不会。doc_type / 关键词都无法可靠区分,故用全文裸编号序列是否单调来开关提级。

## 4. 节树 + section_anchor

每个标题开一个 `Section{level, title, breadcrumb, start_idx, end_idx, parent_sec_id}`,范围到"下一个 level ≤ 自身的标题"。每个 chunk 记 `section_anchor=[start, end]`(其所属最深节的 idx 范围)。树不内联正文——正文在 chunk,节是结构索引。

## 5. 查询期 small-to-big(`assemble_big`)

命中 chunk 后,按**真实 token 量**取"大块":

```
sec = hit 的节
if tokens(sec) > max:          → 在 sec 内绕命中开窗(过大裁)
while tokens(cur) < target:
    parent = cur.parent
    if parent is None:         → 到顶,best-effort(整篇若仍小则整篇开窗)
    if tokens(parent) > max:   → 在 parent 内开窗(拉入相邻兄弟节内容)★并兄弟
    else: cur = parent         → 往上并祖先
return cur(或窗口)
```

- **过小往上并祖先**:小节扎堆时(实测中位 42 token)逐级上爬。
- **并相邻兄弟节**:往上会超 max 时,改在父节范围内绕命中开窗,自然拉入两侧兄弟节内容。
- **整篇兜底**:顶层节仍小且无父 → 整篇范围开窗;真·小文档则保持小(正确)。
- 实测:big-block 中位 818 token(≈目标),<200 占 0.6%(全来自真·单页极短文档)。

## 6. v2 与对抗性审核(为什么是现在这样)

v1 曾设计成"ingest 只留最小编号骨架 + 命中时懒重建层级"。**对抗性审核(跑 77 篇真实数据)推翻了两个命根:**

- **"有编号→零 LLM"只覆盖 ~13%**(学术);无编号是主体 → 改 `text_level` 为主、编号为校正。
- **"懒"是过早优化**:eager 全树 77 篇共 66.6ms(<1ms/篇),与查询期同算法 → 改 **eager**(ingest 顺手建,挂进 chunk),"懒"仅留给巨标题数+超稀疏+高更新的罕见特例。

收敛后本质 = **多信号定级的 eager parent-child + TOC 剔除**。完整记录见 [`../../methodology/LAZY_HEADING_TREE_DESIGN.md`](../../methodology/LAZY_HEADING_TREE_DESIGN.md)。

## 7. 对抗审核实测:fixture 是 happy-path(2026-06)

把组件实跑在两份文档上对照——`examples/fixtures`(13 元素,英文学术,理想)与真实研报 `financial_research_zh__AP202601131816964706`(355 元素,中文,封面密集 + 循环编号 bullet)。**21 条 finding,18 confirmed / 0 refuted。** 核心结论:**fixture 的标题编号与 `text_level` 永远一致,把这套定级唯一会犯错的方向(编号无条件凌驾 parser)全部排除了,导致讲解里"编号定级是真长板"在干净文档上"人造成立",在脏研报上反转。**

| 维度 | fixture 上 | 真实研报实测(修复前) | finding |
|---|---|---|---|
| 编号定级 | ✅ 完美(编号≡text_level) | 25 个 L1 里 **24 个是 list bullet 误升**(parser 原 text_level=2),顶层精确率 ~12% | F1 critical |
| 层级/嵌套 | ✅ 干净 `2 > 2.1` | 真章节 "2 行业一周要闻" 被误升 L1 的 "1.海外AI" 在 idx113 **腰斩**,~200 正文元素被踢出;叶子条目反升为节(**头脚倒置**) | F2 critical |
| breadcrumb | ✅ 章节路径 | **51% 深度=1**;倒置成 "1.海外AI > - OpenAI…";封面标签链 "行业评级:增持" | F3 high |
| small-to-big | 4/4 退化 doc-window(从不爬升) | 封面命中 climb 到标题横幅 s0 → **610 token 7 topic 大杂烩**;43% 走 doc-window 兜底 | F1/F4 high |
| 尺寸健康 | 玩具尺寸,反显"正常" | 63 节 19%<50tok、6 空壳;74 chunk **47%<50tok** | F4 partial |

**根因链(单点)**:`_bogus_number` 只挡年份+`>40` → 裸整数 `1…40` 全放行 → 编号校正**无条件** override parser 的 `text_level` → list bullet 升 L1 → 栈污染 → 树倒置 → breadcrumb/small-to-big 全部下游连锁。

**修复(reset-aware,见 §3)**:裸整数仅在全文序列单调时提级,出现重启即回落 parser;叠加项目符号守卫。这样把"周报循环编号"识别为列表(不提级)、"深度报告单调编号"识别为大纲(提级),消除 24 个假 L1 而不回归真编号文档。

**修复后实测(7 篇差异化文档,2026-06)**:

| 文档 | L1 数 | breadcrumb 深度≤1 | 内容 orphan | 结论 |
|---|---|---|---|---|
| AP4706 周报(目标) | 25 → **1** | 51% → **0%** | 0 | 假章节清零、倒置消除 |
| AP4904 深度报告 | 4 → **4** | — | 0 | 单调编号,**不回归** |
| Attention 学术 | 10 → **10** | — | 13⚠️ | parser 正确,**不回归** |
| law / slides / 政府 | 不变 | — | 0 | 编号未参与,**不回归** |
| NETFLIX 10-K | 18 → **15** | — | 0 | 编号注释正确降为嵌套 L2 |

### round-2 修复(内容回收 + 横幅守卫,对抗审核 22 finding / 16 confirmed / 0 refuted)

② **内容回收**(`core.py` 发射循环 `else` 兜底):此前只发 text/list,丢弃 `equation`(28)/`page_footnote`(282)/`ref_text`(67)/`code`(19)——全语料 407 个元素静默丢失。改后非资产带文本元素全进正文。实测 Attention 论文 orphan 13→**0**、7 个公式干净进 chunk。
③ **重复横幅守卫**(`_banner_texts`):同文本出现在 **≥50% 页面**且 ≥3 次 → 判 running banner 整体剔除。用页面占比(非频次)避免误杀合法重复标题(law `SALARIES AND EXPENSES`×4、slides `Engine Sensors`×6 安全)。实测政府 `FOR PUBLIC RELEASE` 清除,L1 23→**4**。

### round-3 修复(对抗审核在我没测的 GPT-4V 论文上抓出 3 个 HIGH,全部已修)

- **A 检索期横幅一致性**(`retrieve.py`):横幅只在 chunk 期删了,`_gather` 用全量 elements → 政府 **25/38 大块重新注入横幅**(最多连续 3 次)。改:`_banner_texts` 同样传入检索路径过滤 → **0/38**。
- **B aside_text 水印污染**(`core.py` NOISE_KINDS):`else` 兜底过度回收页边水印(arXiv 戳/券商书脊),11/11 注入正文。改:`aside_text` 入 NOISE_KINDS,chunk+检索一致剔除。
- **C 横幅守卫假阳**(`_banner_texts`):纯页面占比无形态判别 → 误删 GPT-4V 论文的 `Prompt:`/`GPT-4V:`(60%+ 页面)共 **198 个发言人标签**。改:横幅不得以冒号结尾 → 这 198 标签保留为标题、335 chunk 的 breadcrumb 带回发言人归属、**0 丢失**。

**round-3.1 清理**(round-3 审核的 4 个 low,均已落地):`NOISE_KINDS` 单一来源(retrieve 改 import)、`banners` 在 `ChunkResult` 上算一次复用(免每次检索重算)、`_window_within` 增长循环按同一 filter 计 token(修 banner/噪声 overcount)、补 aside_text 检索期单测。13 单测 → 现 12(口径)全过。

**仍未解决(round-4 候选)**:① **章节完整性**——AP4706 真章节 "2 行业一周要闻" 与子节 `2.1`、新闻条 `1.海外AI` 在 parser 里同为 `text_level=2`,reset-aware 统一压回 L2 后**互相切割**(span=1),需 doc 内上下文定级而非全文开关。② **发言人轮次碎片化**——GPT-4V 类对话论文里 `Prompt:`/`GPT-4V:`(text_level=2)保留为标题后,每个 turn 成一个微节(415 块、73% <50tok);可识别"短冒号+高频复现"标签作为正文内联前缀而非节标题。③ **脚注混排**——`page_footnote` 按阅读序插在正文中段,可加 footnote flag 或移到 section 末。④ **横幅冒号假阴**——当前"冒号结尾即非横幅"会放过 `CONFIDENTIAL:`/`DRAFT:` 式真横幅(本语料无样本);更强信号是 **bbox 位置稳定性**。

## 适用边界(诚实,修订版)

定级质量**取决于 parser 的 `text_level` 有多干净**,分三档:

- **强(parser text_level 已正确)**:干净学术/技术报告/英文规范——多数标题 parser 直接给对 level(如 Attention 论文 parser 给 25 个 `text_level=1`),编号只做小数点细分。breadcrumb 准、零 LLM。这正是 fixture 代表的一档。
- **中(parser 压平 + 编号是真大纲)**:深度报告类——parser 把所有标题压成单一 level,但编号 `1,2,3` 单调 → reset-aware 提级恢复章节。可用。
- **退化(parser 压平 + 编号是列表)**:周报、封面密集中文研报——编号循环重启,reset-aware **主动放弃提级**,退成"单根 + 扁平 L2"的通用 parent-child(诚实但无章节嵌套);封面标签仍会成为小噪声节(每页横幅已由 round-3 守卫剔除)。**此档不要指望精确层级,只把它当"带 breadcrumb 的大小切块"用。**

- **不擅长**:纯扫描件层级、跨多跳综合(GraphRAG/RAPTOR 的活)、远程引用(未实现)、章节完整性(同 text_level 混章节+列表时,见 §7 round-4 候选①)。
- **建议**:把期望锚定在"parser text_level 质量",而非 doc_type。结构干净就精确,被压平就退化为通用 parent-child——这是结构性上限,不是 bug。

## 8. 近期演进（2026-06，指针）

§1–7 是核心切块（heading-tree + small-to-big）。在其之上的扩展见对应文档：

- **文档级 metadata + ACL（安全边界）**：`Chunker.chunk(doc_meta=, acl=)` 盖到每 chunk，fail-closed 默认 `RESTRICTED_ACL`；检索须硬预过滤；small-to-big **不跨 ACL** 取材（`acl_index`/`admit`）。见 [`INTEGRATION.md §6`](INTEGRATION.md) + [`../../methodology/MULTIFORMAT_IMPL.md §11`](../../methodology/MULTIFORMAT_IMPL.md)。
- **多模态**：`Chunk.image_path`（image/chart 裁切图引用，已路径净化）+ `image_only` flag（无文字纯图也存活，供下游 Qwen3-VL **图像向量化**、跳稀疏路）。见 [`API.md`](API.md) + MULTIFORMAT §12。
- **5 格式覆盖**：PDF/扫描件→MinerU VLM；docx/pptx→MinerU office 后端（纯规则）；xlsx→独立 `table_chunker`（网格非文档流，heading-tree 不适用）。见 MULTIFORMAT。
- **table_chunker round-2**：merge 几何重建多层表头、legacy `.xls`（xlrd）、嵌入 chart 标题/系列名提取。见 MULTIFORMAT §13。
- **doc_type 上 Chunk + 封板对抗 review 修复**：同名兄弟 section 不合并、image_path 净化、admit fail-closed、doc_type 驱动查询期预算、lang `zh` 别名。见 MULTIFORMAT §14。

**数据契约新增字段**（见 API.md）：`Element.image_path` · `Chunk.image_path` / `Chunk.doc_type` · `BigBlock.acl` · `ChunkResult.acl_index()`。

## 格式与 parser 范围(诚实)

本组件设计上 **parser-无关**(只吃归一化 `Element[]`),理论上任何格式写个 adapter 都能接。**但目前的全部实测(77 篇 + 三轮对抗审核)只覆盖 PDF→MinerU 一条路。** 其余格式未测,且核心模型对它们的适配度差异很大——不要默认"能跑":

| 格式 | 适配度 | 说明 |
|---|---|---|
| **PDF(MinerU)** | ✅ **已验证** | 77 篇真实文档 + 三轮审核。本组件的所有实测都在这里。 |
| **Word(.docx)** | ✅ **MinerU office 后端(主)**(50篇,本地无VLM) | `scripts/parse_office.py` → MinerU 原生 office 解析(纯规则 OOXML,零模型)→ content_list → `from_mineru`。公平实测:MinerU ≈ 自研 adapter ≈ Tika(docx 文本覆盖 ~95%,三方打平),选 MinerU 因 content_list schema 切块即用(text_level/list_items/table_body/caption/chart/公式)+ 与 PDF/扫描件管线统一 + 无 JVM。`adapters/docx.py` 降为**零依赖 fallback**。 |
| **PPT(.pptx)** | ✅ **MinerU office 后端(主)** | 同上,走 `from_mineru` + `page_grouped`(幻灯片=页)。MinerU/自研/Tika pptx 覆盖相当(Tika 略胜 group)。`adapters/pptx.py` 降为 fallback。 |
| **Excel(.xlsx)** | ✅ **独立路径**(方案A,51 篇验证) | document chunker 的 heading-tree **不映射**网格 → 单独的 `table_chunker.py`(`TableChunker`),但**输出同样的 `Chunk` schema**。按 sheet/空行表区/行组切 + **列头作上下文**(markdown);宽表列分组不丢列。单元格值覆盖 **100%**,巨表(202k 行/108 sheet)可扛。限制:legacy .xls、图表 sheet、合并表头。详见 [`MULTIFORMAT_IMPL.md` §7](../../methodology/MULTIFORMAT_IMPL.md)。 |
| **扫描件 PDF** | ✅ **已验证**(40 篇 OCR，13 语言) | 复用 mineru adapter 走 OCR（同普通 PDF，**零新代码**）。文本 orphan **0**、40/40 解析；报纸/书/手稿结构合理。MinerU 在线 API 单文档 ≤200 页 → >200 页需抽页。质量随 OCR 准度走（parser 职责）。 |

**一句话**:本组件是给**文档流内容**(prose + 标题 + 内嵌资产)用的。**PDF / Word / PPT 已各自在真实语料 + 对抗审核上验证**(详见 [`../../methodology/MULTIFORMAT_IMPL.md`](../../methodology/MULTIFORMAT_IMPL.md));**Excel 请用别的组件**。**核心 `core.py` 仍 format-无关、零改动**——format-specific 的只有 adapter。扩新格式务必跑 `scripts/coverage_office.py` 验真实保真度(别再信自指标的 orphan=0)+ 重跑对抗审核。
