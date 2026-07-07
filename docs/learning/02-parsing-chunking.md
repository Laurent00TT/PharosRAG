# 02 文档解析与切块

> **本篇导读**:讲 pharos 的解析层(MinerU 统一入口 + Element 接缝)与切块层(heading 树多信号重建、doc_type 分型预算、资产原子化、查询期 small-to-big)。
> **面试权重:高**——切块是 RAG 里最能看出"做没做过真实语料"的一层,本篇有全项目最漂亮的两个故事:reset-aware 裸编号提级、66.6ms 实测杀死自己的 lazy 设计。
> **前置阅读**:无硬前置;评估数字的来源与口径见 [07 评估方法论](07-evaluation.md) 与 [../methodology/CHUNKING_EVALUATION.md](../methodology/CHUNKING_EVALUATION.md)。

---

## 1. 概念底座:切块在任何 RAG 里解决什么问题

在讲 pharos 之前,先把这一层的问题本身讲清楚——它与具体项目无关。

**为什么必须切块。** 三个硬约束叠加:
1. **embedding 的表征会随文本变长而稀释**——一个 30 页文档压进一个向量,任何具体问题都捞不准它;
2. **LLM 上下文有限且按 token 计费**——检索单元太大,喂进去的大部分是无关内容;
3. **chunk 是检索系统的"原子"**——引用溯源、权限控制、去重、计费,粒度全部锚定在 chunk 上。切错了,下游全错。

**核心矛盾。** embedding 想要**小而语义纯**的块(召回准),LLM 生成想要**大而完整**的上下文(不断章取义)。所有切块方案本质上都在调和这对矛盾。

**主流方案光谱**(从简单到结构化):

| 方案 | 思路 | 代表 | 弱点 |
|---|---|---|---|
| 固定窗口 | 每 N token 切一刀,可带 overlap | 早期教程 | 句子被腰斩、语义边界随机 |
| 递归规则切分 | 按分隔符层级(段落→句→词)递归下探 | LangChain / chonkie | 不懂文档结构,标题和正文平权 |
| 语义切分 | 相邻句 embedding 距离突变处切 | semantic-chunker 类 | 贵、不稳定、仍无层级 |
| 结构感知 | 按标题树/版面切,块携带层级信息 | Docling、自研 | 依赖解析质量,标题识别是硬骨头 |
| parent-child / small-to-big | 小块检索、大块(父节)喂 LLM | LlamaIndex 等成熟系统 | 需要额外的结构存储与查询期组装 |

**解析层同理有光谱**:纯文本抽取(PyPDF)→ layout-aware 解析(MinerU / unstructured,给出元素类型、bbox、标题层级)→ VLM 深度解析(表格转 HTML、图表抽内容)。切块的上限由解析的下限决定:解析层丢了表格结构,切块层无论多聪明都救不回来。

**一个常被忽略的视角**:切块的评价标准不是"块切得漂不漂亮",而是**"问题的证据切完之后还在不在一个可检索的单元里"**——证据保全。这是 pharos 评估切块的出发点(§3、[../methodology/CHUNKING_EVALUATION.md](../methodology/CHUNKING_EVALUATION.md)),也是理解它一切设计取舍的钥匙。

**切块层的输出契约**(一个生产级 chunk 应该携带什么,不只是文本):

- **检索文本**:embedding/BM25 消费的内容,可以与原文不同(资产块尤其如此);
- **生成载荷**:命中后喂给 LLM 的完整内容(表格 HTML、图表数据);
- **溯源**:回指原始解析元素的索引,支撑引用、审计、评估对齐;
- **结构**:面包屑(祖先标题链)+ 所属节的范围锚点,支撑查询期上下文组装;
- **治理元数据**:文档元信息(过滤/引用)+ 访问控制策略(权限粒度=chunk 粒度)。

把这五项当 checklist 去看任何切块方案,高下立判——固定窗口和递归切分只给第一项。

---

## 2. Pharos 怎么做

### 2.0 数据流总览

```
PDF/扫描件/docx/pptx ──MinerU 统一解析──> content_list.json (+ layout.json)
        │
        ▼ adapter 归一(from_mineru)
   list[Element]  ←—— 换 parser = 换 adapter,core 不动
        │
        ▼ Chunker.chunk()
   ① 噪声三件套过滤(NOISE_KINDS / 横幅 / 内容回收)
   ② heading 树多信号重建(text_level 为主 + reset-aware 裸编号)——eager,亚毫秒
   ③ leaf 切块(doc_type 预算 + 资产原子化 + page_grouped)
   ④ 盖章(chunk_id / section_anchor / doc_meta / ACL fail-closed)
        │
        ▼
   ChunkResult{chunks, sections, banners} ──> 向量库(chunk)+ sidecar(elements/sections/acl_index)
        │
        ▼ 查询期(命中后)
   assemble_big:small-to-big 三态(上爬 / 开窗 / 兜底),直接读 ingest 挂好的结构,不重建
```

xlsx/xls 走独立的 TableChunker(网格不是文档流,heading 树不适用),但输出同一 Chunk schema。

### 2.1 统一解析:MinerU + Element 接缝

pharos 把所有格式收敛到 MinerU 一个解析器:PDF、扫描件 OCR、docx/pptx(MinerU 原生 office 后端)产出同一份 `content_list` schema。适配层只有一个职责——把 parser 输出归一成 `Element[]`([src/chunker/types.py:22-41](../../src/chunker/types.py#L22-L41)):`idx/kind/text/text_level/caption/table_body/asset_content/merge_prev/image_path`。core 只吃这个接缝,**换 parser = 换 adapter,不碰核心**([src/chunker/adapters/mineru.py:45-65](../../src/chunker/adapters/mineru.py#L45-L65))。

一个值得注意的细节:跨页续接标志 `merge_prev` 不在 content_list 里,adapter 从 layout.json 的 para_blocks 按归一化文本前缀模糊匹配回填([src/chunker/adapters/mineru.py:31-38](../../src/chunker/adapters/mineru.py#L31-L38))——为什么不用 bbox 对齐?因为 content_list 与 layout 的 bbox 在**两个坐标系**(渲染图 vs PDF 点),这个坑在评估阶段还会再咬一次(§3.3)。这个模糊匹配本身有已确认的串扰缺陷,见 §4。

自研的 docx/pptx adapter 降级为零依赖 fallback,保留是因为它踩过的坑有教学价值(文本框、GROUP 形状递归、无样式标题推断——真实世界 64% 的 docx 不用标题样式)。

### 2.2 噪声三件套

解析产物里混着三类"看起来是内容的垃圾",各有一道针对性的闸:

1. **kind 黑名单**:`{header, footer, page_number, aside_text}` 直接丢([src/chunker/core.py:10](../../src/chunker/core.py#L10))。`aside_text`(页边水印/arXiv 竖排戳)是第三轮审查加的——11/11 语料案例确认是噪声。
2. **重复横幅守卫**:同一文本出现在 ≥50% 页面且 ≥3 次判为 running banner,整体剔除([src/chunker/core.py:115-133](../../src/chunker/core.py#L115-L133))。两个精细处:用**页面占比**而非绝对频次(否则误杀 law 文档里合法重复的 'SALARIES AND EXPENSES'×4);**冒号结尾的不算横幅**——对话类论文里 'Prompt:' / 'GPT-4V:' 在 60%+ 页面复现,但那是发言人标签,是内容。这条规则是在一篇没测过的 GPT-4V 论文上抓出 198 个发言人标签被误删之后加的。
3. **内容回收**:发射循环的 else 兜底把 equation/code/page_footnote/ref_text 等"带文本但非资产"的元素全部收进正文([src/chunker/core.py:319-328](../../src/chunker/core.py#L319-L328))——第二轮审查前这些被静默丢弃了 407 个元素。

横幅集存进 `ChunkResult.banners`,查询期 `assemble_big` 复用同一集合([src/chunker/retrieve.py:31](../../src/chunker/retrieve.py#L31))——**chunk 期和检索期必须一致剔除**。这不是洁癖:横幅只在 chunk 期删的话,big-block 从原始 elements 取材时会把它重新注入,实测某政府文档 'FOR PUBLIC RELEASE' 污染了 25/38 个大块,修复后 0/38(该文档的假 L1 标题也从 23 降到 4)。

### 2.3 heading 树多信号重建:reset-aware 是精华

这是本篇最值得吃透的机制。问题:MinerU 给了 `text_level`(标题层级提示),但它不总对;标题里的编号(`2.1`、`1.`)看似是更强的层级信号,但**同一个 '1.' 在不同文档里语义相反**——周报里 '1. 海外AI:' 是列表项,深度报告里 '1. 端侧AI' 是真章节。doc_type、关键词都区分不了。

pharos 的定级函数 [src/chunker/core.py:158-176](../../src/chunker/core.py#L158-L176) 按这个优先级融合信号:

- **text_level 为主**(parser 免费给的,是基线);
- **带点编号无条件细分**:`2.1` 按 '.' 段数给深度(dotted 编号几乎不会撒谎);
- **裸整数编号只在有"全文单调性证据"时才提级 L1**——这就是 reset-aware:chunk 前先跑一遍 pre-pass 收集全文裸编号序列([src/chunker/core.py:263-274](../../src/chunker/core.py#L263-L274)),`all(严格递增)` 才打开 `promote_bare` 开关。**关键洞察:列表会重启(1..9, 1..),大纲不会**。一旦序列出现重启,所有裸编号回落 text_level。
- **四道守卫**([src/chunker/core.py:136-155](../../src/chunker/core.py#L136-L155)):年份(19xx/20xx)不当编号;bullet 字形(`-•●○▪◦`)开头永远不是标题;TOC 形态(点引导+尾页码)或超 120 字符不是标题;law 文档 `SEC. N` 特判 L1。

定级完成后单调栈边扫边建树([src/chunker/core.py:277-290](../../src/chunker/core.py#L277-L290)):每个正文元素挂上 `_crumb`(祖先标题链)和 `_sec_head`(最深节的标题 idx),标题序列再一次性构建 Section 树。

**实证背书**(口径:77 篇 / 5337 个标题的真实语料):修复前,真实中文研报 AP4706 的 25 个 L1 里 24 个是循环编号列表项误升,树倒置、51% 的 breadcrumb 深度只有 1;reset-aware 落地后 25→1、浅 breadcrumb 归零,且 7 篇差异化回归文档(单调编号深报 4→4、Attention 论文 10→10、NETFLIX 10-K 编号注释正确降 L2)零回归。

### 2.4 eager 骨架:整棵树在 ingest 期建好

heading 树、breadcrumb、section_anchor 全部在 ingest 期一次算完,直接盖进每个 leaf chunk([src/chunker/core.py:331-350](../../src/chunker/core.py#L331-L350));查询期**直接读,不重建**。这个"理所当然"的设计其实是一次实证反转的产物,细节在 §3.1——它的设计文档至今叫 [LAZY_HEADING_TREE_DESIGN.md](../methodology/LAZY_HEADING_TREE_DESIGN.md),名字就是化石。

### 2.5 leaf 切块:token 预算分型 + 贪心装配

同节的连续文本块按 token 预算贪心累积([src/chunker/core.py:208-240](../../src/chunker/core.py#L208-L240)):单块超上限先句切;累积超 target 且非跨页续接(merge_prev)则 flush;第二遍把欠下限的小组往前合并。预算 `(min, target, max)` 按 doc_type 查表([src/chunker/core.py:22-31](../../src/chunker/core.py#L22-L31)),如 law=(300,700,1200)、中文研报=(250,550,900)。

两个反直觉的点:

- **预算数值各类型其实很接近,真正的分型差异是布局模式**:slides/policy 用 (0,9999,9999)——一页 slide / 一整节就是一个 chunk;slides 的分组键额外带 page(`PAGE_GROUPED`)。
- **分组键必须用标题实例的唯一标识(`_sec_head`,即标题元素 idx),而不是 breadcrumb 文本**([src/chunker/core.py:295-302](../../src/chunker/core.py#L295-L302))。同名兄弟节(表单/附录里重复的小节标题,约 15% 文档)如果按文本分组,不同节的正文会被合进一个 chunk 并错锚到第一个节——**不可逆地写错向量库**。这是封板 review 抓的三条 high 之一,教训通用:分组键要用实例标识,不用可重复的展示文本。

token 计数用 char/除数启发式(en/4.0、zh/1.7,[src/chunker/core.py:39-46](../../src/chunker/core.py#L39-L46)),不是真 tokenizer——为什么敢这么糙,§3.4 讲。

### 2.6 资产原子化:text 与 content_raw 分离

table/image/chart 各自成一个**原子 chunk**([src/chunker/core.py:372-402](../../src/chunker/core.py#L372-L402)),且做了一个关键的字段分离:

- `text`:**检索文本**,供 embedding 和 BM25;
- `content_raw`:**生成载荷**(表格 HTML / VLM 抽取内容),只在命中后喂给 LLM。

为什么分离?因为二者的最优内容完全不同。表格的检索文本由 `_table_signal` 合成([src/chunker/core.py:89-112](../../src/chunker/core.py#L89-L112)):前 2 行表头(财报常见双层表头)+ 其余每行首个非空单元格(行标签),**数据单元格刻意不进**——数字本身没有检索语义,只增噪声。图/chart 的检索文本由 `_asset_desc` 清洗([src/chunker/core.py:69-81](../../src/chunker/core.py#L69-L81)):mermaid 只留节点/边标签,丢 `graph TD`/`-->` 脚手架。

这个机制是被一次真实错答逼出来的(N7/N8):问 Netflix 总营收,系统把分部营收当总营收——检索侧根因是表格 chunk 的 text 原本只有 caption 一句,列头/行标签全锁在 content_raw 里不可检索,数字类问题的表格块被 MD&A 散文挤出 top-k。修复后动机案例直接答对,72 题回归正确性持平、忠实度 0.972→1.000(注意口径:这是 72 题基线,与后来的 88 题不可混)。

两个边界处理见功力:

- **纯图不丢弃**:只有 img_path、无任何文字的图,产一个带 `image_only` flag 的占位 chunk([src/chunker/core.py:393-394](../../src/chunker/core.py#L393-L394)),下游走 Qwen3-VL 图像向量化、跳过稀疏路。此前纯图直接丢,损失约 30% 的图(academic 类 38%)——而这些恰是 VL 向量化唯一能召回的对象。img_path 经 `_safe_rel` 净化(拒绝 `../`、绝对路径、URL scheme),因为它是 embed 链唯一的文件读取入口。
- **幽灵块门控**:表格 chunk 的存活条件是 `(cap | foot | body)` 至少一项存在([src/chunker/core.py:379-384](../../src/chunker/core.py#L379-L384))——面包屑**不得单独救活**无 caption 无表体的占位表格。没有这道门,实测全库 chunk 数 7652→7675,+23 个幽灵块令同文档后续 chunk id 全部平移、旧索引的 gold 集体错位。这个数字是 §4"为什么切块改动必须延期"的直接证据。

### 2.7 电子表格的独立路径:TableChunker

网格不是文档流:xlsx 没有阅读序段落,heading 树无从建起,所以 xlsx/xls 走独立的 TableChunker,但**输出同一 Chunk schema**,下游 embedder 零分叉。四个关键机制:

- **band 切分**:每 sheet 按空行切"带",但被竖向合并单元格覆盖的空行**不算分隔符**([src/chunker/table_chunker.py:81-101](../../src/chunker/table_chunker.py#L81-L101))——否则 5 行合并表头会被劈成没有列名的 header_only 碎片(red-team 在 hvs_vacancy 表上实锤过)。
- **真表头识别**:band 内首个"多列且 label-like"的行才是表头(数值占比 <0.34,[src/chunker/table_chunker.py:66-68](../../src/chunker/table_chunker.py#L66-L68));4 位年份算标签不算数值([src/chunker/table_chunker.py:49-53](../../src/chunker/table_chunker.py#L49-L53)),防 `['Race', 2010, 2020]` 被误判成数据行。表头之前的标题/单位行发独立文本块,永不丢。
- **merge 几何广播**:从 xlsx zip 直接解析 mergeCell 几何(不加载任何单元格,保住 read_only 流式,[src/chunker/table_chunker.py:141-178](../../src/chunker/table_chunker.py#L141-L178)),`_broadcast` 把左上角标签广播回整个合并区([src/chunker/table_chunker.py:181-200](../../src/chunker/table_chunker.py#L181-L200)),再逐列自上而下 join 出多层完整列名。
- **列切守恒 + chart 语义**:超宽表按列切片、每片自带 key 列,单元格零丢失([src/chunker/table_chunker.py:104-112](../../src/chunker/table_chunker.py#L104-L112));嵌入图表从 `xl/charts/chartN.xml` 抽标题+系列名+轴名成 chart chunk([src/chunker/table_chunker.py:231-265](../../src/chunker/table_chunker.py#L231-L265));legacy .xls 走 xlrd、复用同一套表头重建([src/chunker/table_chunker.py:213-228](../../src/chunker/table_chunker.py#L213-L228))。

为什么在表头上大动干戈?因为**列名↔数值绑定是这个组件的核心卖点**——修复前后对比:US Census 表的列名从 'CV for Retail Sales'(维度全丢)变成 'CV for Retail Sales 2023Q4 (p) Total E-commerce'(三个维度完整)。这背后"红队推翻自己判断"的故事在 §3.5。

实测数据:multiset 审计 584 万单元格坐实值覆盖 100%(续接带 deficit 21134→0);INSEE 的 .xls 从 0 chunk(硬崩静默漏掉)到 6767;EIA STEO 从 0 到 66 个 chart chunk;202k 行 / 127 sheet 的巨表 25s 流式扛住。

`chunk_result()` 额外为每 chunk 合成一个 Element、每 sheet 合成一个 Section([src/chunker/table_chunker.py:311-334](../../src/chunker/table_chunker.py#L311-L334)),让 assemble_big 能在 sheet 内回拼被行组/列切散开的同表片段——而正是这次"对齐 assemble_big 的 idx 语义"的字段覆写,埋下了 §4 的 chunker#4(xlsx 行级溯源丢失)。

### 2.8 查询期 small-to-big:三态组装

命中的是小块(语义纯,召回准),喂给 LLM 的是按真实 token 量"长大"的 big-block。`assemble_big`([src/chunker/retrieve.py:109-171](../../src/chunker/retrieve.py#L109-L171))读命中块的 `section_anchor`,走三态:

1. **命中节 > max** → 在节内绕命中开窗(`_window_within` 从命中种子向两端交替生长,[src/chunker/retrieve.py:68-87](../../src/chunker/retrieve.py#L68-L87));
2. **命中节 < target** → 沿 `parent_sec_id` 上爬;父节 > max 时改在父节范围内开窗,自然拉入相邻兄弟节内容(带 `seen` 集合防损坏 sidecar 的环形父指针死循环);
3. **无 section** → 命中页邻域开窗(防无标题多页文档拉整篇);顶层仍 < min 则整篇开窗兜底。

真实语料上**上爬是主路径而非兜底**:section 中位只有 42 token,过大裁窗只占 1%(77 篇实测);big-block 中位 818 token,贴着 target 800。`windowed=True` 标记"这是 token 受限窗口而非完整节"([src/chunker/types.py:107-122](../../src/chunker/types.py#L107-L122)),embedder 检索层把它映射成 `context_status="section_window"`([src/embedder/retrieve.py:172-182](../../src/embedder/retrieve.py#L172-L182)),agent 看到就知道可以对该 chunk_id 调 expand——上下文完整性从隐式变成可编程决策的显式信号。

安全维度(big-block 按 idx 从原始 elements 重取材,会跨 chunk 边界——"同文档≠同 ACL")由 `acl_index` 等价类门控解决([src/chunker/retrieve.py:90-106](../../src/chunker/retrieve.py#L90-L106)、[src/chunker/types.py:93-104](../../src/chunker/types.py#L93-L104)),真语料实测泄漏 3/79→0/79。这条线的完整故事属于 ACL 篇,本篇只需记住:**查询期取材的每个元素都过了与命中块同 ACL 的等价类判定,未知 idx fail-closed 排除**——这个"未知 idx 排除"恰好埋下了 §4 的 chunker#0。

### 2.9 sidecar:查询期取材的持久化底座

向量库只存 chunk;`assemble_big` 需要的原始 elements/sections/banners/acl_index 以每文档一个 JSON 存 sidecar([src/embedder/embed.py:107-125](../../src/embedder/embed.py#L107-L125)),写侧盖 `SIDECAR_VERSION` 章([src/embedder/config.py:10](../../src/embedder/config.py#L10)),读侧三重校验(缺失=单 doc 瞬态降级;版本不符=系统性 schema 漂移,响亮失败提示整体重建;elements 非密集有序=fail-closed 拒载,[src/embedder/retrieve.py:71-90](../../src/embedder/retrieve.py#L71-L90))。版本号的意义:把"旧 sidecar 被静默反序列化成错数据"变成"响亮失败"。

---

## 3. 为什么这么设计:被否决的备选与实测数据

### 3.1 lazy → eager:用 66.6ms 杀死自己的聪明设计(方法论金矿)

v1 的设计是"ingest 只留最小编号骨架 + 命中时懒重建层级"——听起来很聪明:大多数 section 永远不被命中,预建是浪费。对抗审核没有辩论这个直觉,而是**直接测**:eager 全树构建,77 篇总共 66.6ms,单篇不到 1ms,且与查询期重建是同一算法。结论瞬间清晰:懒加载**省不到任何东西**,反而把零成本工作搬进查询热路径,还引入缓存失效问题。当场翻转为 eager;"懒"降级为"巨标题数 + 超稀疏命中 + 高频更新"三条同时满足才考虑的特例——本语料 0/77 满足([LAZY_HEADING_TREE_DESIGN.md §6](../methodology/LAZY_HEADING_TREE_DESIGN.md))。

教学点:**性能优化必须先测再做**。文档名保留 "LAZY" 是有意的——它是"设计被数据推翻"的化石证据,面试讲这个故事比讲任何成功设计都有说服力。

### 3.2 "赌编号"被 13% 推翻:多信号融合的由来

v1 的另一个命根是"有编号 → 零 LLM 定级"。实测:77 篇 / 5337 个标题里只有 ~11% 能解析出编号,academic 之外(financial/law/policy/slides)几乎为零——**无编号是主体,不是尾部**。同时被否决的还有:

- **font_rank / bbox 高度分级**:跨文档不可比,10-K 的 L1/L2 字号相同;
- **LLM 消歧进热路径**:被定位成按 doc_type 路由的**离线可选增强**,永不进 ingest/查询主路径——成本可控性优先。

于是收敛到 §2.3 的方案:text_level 为主、编号做校正、reset-aware 做裸编号仲裁。这也是 [LAZY_HEADING_TREE_DESIGN.md §0](../methodology/LAZY_HEADING_TREE_DESIGN.md) 记录的第二次实证反转。

### 3.3 为什么是 MinerU、为什么不用现成切块库

**解析层**:MinerU vs Tika vs 自研 adapter。最初测出"自研 94% vs MinerU 87%"差点选错——后来发现是**测量 bug**(覆盖率提取器漏读 list_items),公平口径下 MinerU 95.1% 反超。三方覆盖率打平的前提下,MinerU 胜在 content_list schema 切块即用 + 与 PDF/扫描件管线统一 + 无 JVM(Tika 要 JVM + 63MB JAR)。

**切块层**:拿同一份证据保全 eval(43 篇 ground-truth / 356 问,MMDocIR 标注)做三方对比([CHUNKING_EVALUATION.md §8](../methodology/CHUNKING_EVALUATION.md)):

| 策略 | single%(ALL 口径)↑ | missing%↓ |
|---|--:|--:|
| **ours_exact**(带 source_indices 精确溯源) | **70.4** | **7.7** |
| chonkie(RecursiveChunker) | 62.1 | 21.9 |
| ours_fair(剥掉溯源,只比边界) | 58.2 | 26.0 |
| docling(HybridChunker,接 MinerU 输出) | 47.6 | 38.6 |

诚实的结论有三层:① Docling 接 MinerU 输出时 missing 是自研的 3-5 倍,直接出局;② **剥掉工程特性、只比切块边界,自研略输 chonkie**(TEXT-channel 口径 71.6% vs 74.9%)——边界算法不是护城河;③ 自研净胜靠的是 source_indices 溯源 + breadcrumb + per-chunk ACL + small-to-big 这套**工程集成**,chonkie/docling 都没有。"若哪天只要纯文本切块、不要这套工程特性,chonkie 是更省事的选择"——这句话写在工程文档里,面试敢原样讲出来,比吹赢更加分。

顺带一个方法论故事:这份 eval 首版结果惨不忍睹(72 题 unlocalized),差点得出"切块很差"的结论——根因是 content_list(渲染图坐标)与 ground-truth(PDF 点坐标)**坐标系错配**。逐文档文本配对推导缩放系数后,unlocalized 72→4,且严阈值下指标纹丝不动,证明旧版的阈值敏感只是坐标 bug 的症状。**先怀疑测量,再怀疑被测物。**

### 3.4 est_tokens:验证后决定"不改"

char/除数启发式够不够准?用 Qwen3-VL 真 tokenizer 对 2927 个真实 chunk 重标:英文散文 char/token≈3.85 vs 现值 4.0,误差 <4%;偏差集中在数字密集文档(财报 5.08 / 政府 5.35)与中文研报(1.51)。单值无法兼顾两簇,改成均值反而伤害散文主体;且两个误差方向都有兜底——高估→块偏小→small-to-big 补;低估→块偏大但远未及 32k 上限。**结论:保留现值**([src/chunker/core.py:39-46](../../src/chunker/core.py#L39-L46) 的注释就是这次验证的记录)。"验证之后决定不改"与"没验证就不改"是两回事,面试值得点破。

### 3.5 三次栽进同一个坑:自指标不可信

解析/切块层的验证极易陷入"用自己的输出验证自己"的循环。这个项目在同一类坑里栽了三次,值得单独立一节:

1. **orphan=0 假指标**(docx/pptx 扩展):自研 adapter 首跑报 orphan=0("内容零丢失"),但 orphan 检查只遍历 adapter **已经吐出**的 Element[]——对"提取之前就丢掉的内容"完全失明。真相:docx 文本框(`w:txbxContent`)从未被遍历(10/50 文件丢 390 段)、pptx 的 GROUP 形状不递归(16/50 文件丢 352 行,一份 harvard 讲义连讲师邮箱都丢了)。
2. **xlsx 覆盖率 100% 掩盖表头污染**:单元格值覆盖 100% 是真的(multiset 审计坐实),但"header=区首行"的盲取让约 33% 的区表头是标题/单位行而非真列名——**列名↔数值绑定坏了,而覆盖率这个指标根本量不到它**。红队实跑 5 个痛点,推翻了"表头问题占比低、不值得修"的判断;教训进了记忆库:severity ≠ occurrence,打穿核心卖点的低频缺陷必须修。
3. **覆盖率提取器漏读 list_items**:把 MinerU 的覆盖率判低 8 个百分点,"自研 94% vs MinerU 87%"的假对比差点让 parser 选型选错;修正后 MinerU 95.1% 反超。

三次的共同解法:**独立 ground-truth + 对抗测量**。为 office 路径专门建了独立指标(adapter 提取文本 vs 原始 OOXML XML 全量文本的词集包含度)——量的是 document↔chunker 的**保真度**,而不是 adapter↔chunker 的**一致性**;修复后 docx 词覆盖 95.6%、pptx 99.1%。方法论收获比数字值钱:**凡指标的分母来自被测系统自身的输出,它只能证伪、不能证真。**

---

## 4. 实战复盘:确认了,但一条都不能马上改

写这套学习文档前的对抗审查(2026-07)对 chunker 深挖出 **5 条 confirmed 缺陷(chunker#0-4),全部延期**——而同一轮审查里 generator/service/embedder/eval 有 17 项直接落地(全局口径:35 疑似 → 34 confirmed + 1 refuted,去重后 17 已修 + 15 延期,本篇 5 条即在这 15 条里)。为什么切块层特殊?

**因为切块的输出就是索引的 schema。** chunk_id 按文档内序号编(`{doc_id}#{n:04d}`),任何让 chunk 数量或边界变化的改动,都会让同文档后续所有 chunk id 平移——旧向量库、旧 sidecar、eval 的 gold 标注全部错位。这不是假设:§2.6 的幽灵表格块事件里,+23 个块就让全库 gold 错位,当场加门控才恢复。所以纪律是:**凡改变切块输出的修复,必须绑定 bump SIDECAR_VERSION + 重建索引 + 重跑 GPU eval,作为一个原子动作落地**;写作窗口内做不完整套,就一条都不动。"确认了但不能马上改"本身是工程判断,不是拖延。

五条延期项(均经独立对抗验证 confirmed,附最小复现):

| # | 缺陷 | 严重度 | 一句话根因 |
|---|---|---|---|
| chunker#0 | ACL 感知路径下 big-block 系统性丢失全部标题文本 | medium | 标题元素不进任何 chunk 的 source_indices → 不在 acl_index → fail-closed 被排除 |
| chunker#1 | CJK 无空格文本句切失效,超大中文段落突破 max 预算 | medium | 切分正则要求句末标点后有空白,中文句号后没有空格 |
| chunker#2 | merge_prev 回填的 12 字符前缀匹配,同页同前缀块串扰 | low | 模糊匹配无顺序消费、无唯一性要求 |
| chunker#3 | 单个裸编号列表项被硬造成 L1 假章节 | low | `all(空/单元素序列)` 真空真 → 无单调性证据也提级 |
| chunker#4 | xlsx 的 chunk_result 覆写 source_indices,丢行级溯源 | low | 合成 element 的 idx 语义与行号溯源复用同一字段 |

两条 medium 值得展开,它们各是一类教学案例:

**chunker#0 是"安全机制的正确方向产生质量副作用"的典型。** heading 元素在建树时被 `continue` 掉([src/chunker/core.py:277-287](../../src/chunker/core.py#L277-L287)),只进 Section 树、不进任何 chunk 的 `source_indices`,因此不在 `acl_index`([src/chunker/types.py:93-104](../../src/chunker/types.py#L93-L104))里;而 `assemble_big` 的默认门控对未知 idx fail-closed 排除([src/chunker/retrieve.py:102-105](../../src/chunker/retrieve.py#L102-L105))——生产默认路径(embedder 恒传 acl_index)产出的 big.text **不含任何标题行**。实测同一文档:legacy 模式 big.text 是 `Compensation\nPhilosophy\n正文…\nBenchmarks\n正文`,ACL 模式只剩两段正文直接相连。爬升到父节时尤其伤:兄弟子节正文连成一片、没有标题分隔,LLM 失去小节边界信号。耐人寻味的是,`get_document` 早就为同一根因打过补丁([src/embedder/retrieve.py:212-225](../../src/embedder/retrieve.py#L212-L225),对 own-section 可见的小节额外纳入其标题),但 assemble_big/expand 路径没同步——**同根因多出口,修一个出口不等于修完**。方向是 fail-closed(标题被漏出而非泄入),所以是质量缺陷不是安全漏洞;修法(把 Section.start_idx 按其节内 body 的 acl 补入 acl_index)会改变 sidecar 语义,必须 bump SIDECAR_VERSION 全量重建。

**chunker#1 是"主场语料反而没测到"的讽刺案例。** [src/chunker/core.py:197](../../src/chunker/core.py#L197) 的句切正则 `(?<=[。！？.!?])\s+` 要求标点后有空白——英文如此,中文正文句号后**没有空格**,re.split 原样返回整段。实测 3200 字中文段(est 1882 token)在 max=900 预算下产出单个 1882 token 的 chunk;人工加空格的对照组正常切成 3 片。预算契约失效 + embedding 语义稀释,而中文研报恰是语料占比最大的类型。查询侧 big-block 有 `_cap` 字符硬截兜底([src/chunker/retrieve.py:48-65](../../src/chunker/retrieve.py#L48-L65)),chunk 本体没有。修法已有草图(全角标点零宽切分 + 超限字符硬切兜底),但它直接改变切块边界——标准的"绑定重建 + 重跑 eval"延期项。

chunker#3 还有个设计哲学层面的看点:reset-aware 的设计初衷是"**要有单调性证据才提级**",而 `all()` 对单元素序列恒真意味着"单个样本零证据也提级"——代码与自己的设计注释相悖。缺陷往往藏在"设计意图与实现边界的缝隙"里。

顺带一提,同轮审查中与本篇底座相关、**已落地**的一项:embedder 的 `index_document` 重排(编码/sidecar 准备前置、delete→upsert→replace 收尾,脱库窗口从分钟级缩到毫秒级,[src/embedder/embed.py:56-105](../../src/embedder/embed.py#L56-L105))——它不改变切块输出,所以可以立即修。两相对照,"什么能马上改"的判据一目了然。

---

## 5. 面试怎么讲

**30 秒版(电梯稿)**:

> 我的切块层是结构感知 + small-to-big:MinerU 统一解析成归一化元素流,ingest 期用多信号(parser 层级为主、编号校正、全文单调性仲裁裸编号)重建标题树,把面包屑和节锚点直接挂进每个 chunk;表格图表原子化,检索文本和生成载荷分离;查询期命中小块后按真实 token 量上爬或开窗组装大上下文。评估不看切得漂不漂亮,看"证据保全"——用 MMDocIR 标注测证据是否还在一个 chunk 里,43 篇 356 问,单块保全 70%,对比过 chonkie 和 docling:边界算法我略输 chonkie,但溯源、面包屑、per-chunk ACL 这套工程集成是净胜项,这个结论我是敢写进文档的。

**3 分钟版(结构化展开)**:

1. **问题定义**(20s):切块要调和"embedding 要小而纯、LLM 要大而全"的矛盾;我的评价标准是证据保全,不是切块美学。
2. **架构**(40s):MinerU 统一解析 → Element 接缝(换 parser 只换 adapter)→ 噪声三闸(kind 黑名单 / 页面占比横幅守卫 / 内容回收)→ heading 树 → doc_type 分型预算 → 资产原子化 → 查询期三态组装。强调 eager:树在 ingest 建好,查询只读。
3. **一个深挖点——reset-aware**(60s):同一个 '1.' 在周报里是列表项、在深报里是章节,doc_type 和关键词都区分不了;唯一可靠信号是全文裸编号序列的单调性——列表会重启,大纲不会。真实研报 25 个假 L1 修到 1,7 篇差异化文档零回归。这一步之前,先讲 fixture 全绿但真实语料上树倒置的教训:**fixture 是 happy-path,对抗审核要拿真文档跑**。
4. **数据背书**(30s):77 篇/5337 标题定级语料;证据保全 43 篇/356 问 single 70.4%(ALL 口径);eager 全树 66.6ms/77 篇——顺势讲 lazy→eager 反转,"性能设计先测再做"。
5. **诚实收尾**(30s):边界算法单比略输 chonkie(71.6 vs 74.9,TEXT 口径);已知 CJK 句切缺陷和 ACL 路径丢标题,均已确认、修法有草图,延期是因为切块改动等于索引 schema 变更,必须绑定重建索引和重跑 eval 原子落地。

---

## 6. 追问预演

**Q1:为什么不直接用 LangChain/chonkie 的切块?**
要点:测过,不是感觉。同一证据保全 eval 三方对比;诚实承认纯边界质量 chonkie 略胜(74.9 vs 71.6,TEXT 口径);但 chonkie 没有 source_indices 溯源、没有 breadcrumb、没有 per-chunk ACL、没有 small-to-big 的节锚点——这些是下游(引用/权限/上下文组装)的硬依赖。关键词:**护城河是工程集成不是边界算法**。

**Q2:为什么不用语义切分(embedding 距离切)?**
要点:语义切分解决的是"边界划哪"的问题,而实测我的短板不在边界(single% 与 chonkie 相差 3pt 量级),在结构信息的保留与下游集成;语义切分贵(ingest 期每句 embedding)、不稳定(阈值敏感)、且产不出层级结构。反问式收尾:切块的钱应该花在"证据保全 + 结构挂载"上,不是花在把边界从 72 分优化到 75 分。

**Q3:标题层级为什么不用 LLM 判?**
要点:先给数据——只有 ~11% 标题有编号、无编号是主体,所以确实存在消歧需求;但 LLM 进 ingest 主路径意味着成本与不确定性进热路径。方案是分层:text_level 免费信号打底、编号做确定性校正、reset-aware 做零成本仲裁;LLM 定位为按 doc_type 路由的**离线可选增强**。关键词:确定性、亚毫秒、零 LLM 主路径。

**Q4:reset-aware 会误杀什么?**
要点:主动交代两个已知边界——①全有/全无开关:混合文档(真章节编号与重启列表共存)整篇只有一个 promote_bare,粒度是文档级不是局部;②已确认的 chunker#3:单个裸编号(len==1 序列)真空真提级,与"要单调性证据"的初衷相悖,修法是 `len>=2` 门槛,延期原因是改标题判级=改切块输出。能主动说出自己机制的失效模式,比机制本身更展示水平。

**Q5:small-to-big 为什么在查询期组装,而不是索引期直接存 parent 块?**
要点:①存两份(child+parent)是存储和一致性双开销,parent 边界还依赖预算参数,调参就要重建;②查询期组装能按**真实命中位置**开窗(命中种子向两端生长),parent 预存做不到;③实测组装的原料(sections/elements)读 sidecar 一次、树已 eager 建好,组装本身是纯 CPU 微秒级。附数据:section 中位 42 token,上爬是主路径,说明 parent 粒度不是静态可预存的——常常要爬多层。

**Q6:chunk 大小(target 800)怎么定的?token 估计不准怎么办?**
要点:预算按 doc_type 查表,但真正的分型差异是布局模式(slides/policy 整节不切);token 用 char 启发式,拿真 tokenizer 对 2927 chunk 验证过:散文误差 <4%,数字密集文档偏差大但两个误差方向都有兜底(高估→small-to-big 补;低估→远未及 32k)。关键词:**验证后决定不改**。同时主动承认 CJK 句切失效是真缺陷(不是估算问题,是切分器失效),已确认待修。

**Q7:线上索引已经建好,切块算法要升级,怎么办?**
要点:这正是我延期五条 confirmed 缺陷的原因。chunk_id 按文档内序号编,切块输出变化 = chunk id 平移 = 旧索引/gold/引用全错位(举幽灵表格块 +23 的实测);所以流程是:修复 + bump SIDECAR_VERSION(读侧响亮失败强制重建)+ 全量重建索引 + 重跑 eval 确认无回归,四步原子落地。可延伸:sidecar 版本校验把"静默错"变"响亮失败",缺文件(单 doc 瞬态)与版本漂移(系统性)区别对待。

**Q8:表格为什么单独成 chunk?数字为什么不进检索文本?**
要点:讲 N7 错答故事(分部营收当总营收)——表格检索信号原本只有 caption,列头/行标签锁在 content_raw 不可检索;修复是 `_table_signal` 抽表头+行标签,数据单元格刻意排除(数字无检索语义,徒增噪声);再讲幽灵块门控(面包屑不得单独救活占位表格)。数据:72 题回归持平、忠实度 0.972→1.000(72 题口径)。

---

## 7. 动手实验

**Lab 1:reset-aware 定级对照——同一批编号,重启 vs 单调两种命运**(CPU,~秒级)

```bash
cd C:/Users/11541/Desktop/projects/pharos
python -m pytest -q tests/engine/test_core.py -k "reset or monotonic" -v
```

两个测试对应两种命运:`test_reset_numbering_not_over_promoted`(裸编号序列 1,2,1,2 重启 → '1. 海外AI:' 保持列表项层级)与 `test_monotonic_numbering_still_promotes`(1,2,3 单调 → 提级 L1)。然后动手改:把重启测试 fixture 第二页的 '1. 影视:' 和 '2. 游戏:' **两条一起**改成 '3. 影视:' 和 '4. 游戏:',使全文裸编号序列变成严格递增(1,2,3,4),重跑观察同一批标题全部翻转为 L1——直观体会"全文单调性"这**一个**信号如何决定整棵树的形状。(反过来,若只改 '影视' 不改 '游戏',序列是 1,2,3,2 仍非单调,`all()` 守卫不会打开 `promote_bare`、标题也不会提级——这恰好复现了机制"一旦序列出现回落就全体回退"的本意。)改完记得还原。

**Lab 2:幽灵表格块门控——亲手复现 chunk id 平移**(CPU,~秒级)

```bash
cd C:/Users/11541/Desktop/projects/pharos
python -m pytest -q tests/engine/test_core.py -k "placeholder_table or table_retrieval_signal" -v
```

先看两个测试绿着:占位表格(无 caption/footnote/表体)被丢弃、正常表格的检索信号含表头+行标签。然后把 [src/chunker/core.py:379](../../src/chunker/core.py#L379) 的 `and (cap or foot or el.table_body)` 门控临时去掉重跑——`test_placeholder_table_still_dropped` 转红:面包屑单独把检索文本撑成非空,幽灵块复活,其后所有 chunk 的 id 平移。这就是"+23 幽灵块让全库 gold 错位"那个 bug 的最小模型,也是理解 §4 延期纪律的最短路径。改完还原。

**(可选,GPU/WSL)est_tokens 重标定**:在 WSL 的 `navikb` 环境加载 Qwen3-VL tokenizer(模型在 `~/models/Qwen3-VL-Embedding-8B`),对 sidecar 目录(`~/rag_sidecar`)里若干 doc 的 chunk text 算真实 token 数,与 `chunk.n_tokens` 散点对比、按 doc_type 分组看 char/token 比——复现"英文散文 ≈3.85-4.0、财报 5.08、中文研报 1.51,单值无法兼顾两簇但两个误差方向都有兜底"的那次"验证后不改"决策。

---

## 8. 诚实边界

面试中主动承认这些,比被追问出来强得多:

- **边界算法本身不是长板**:剥掉工程特性单比切块边界,自研 71.6% 略输 chonkie 74.9%(TEXT-channel 口径)。护城河在工程集成,这个定位是三方对比逼出来的,不是谦辞。
- **五条 confirmed 缺陷在册未修**(§4):CJK 句切失效(中文超大段突破预算)、ACL 路径 big-block 丢标题、merge_prev 前缀串扰、单裸编号假 L1、xlsx 行级溯源丢失。全部有最小复现与修法草图,延期系"切块输出变更须绑定重建+eval"的纪律,不是没发现。
- **横幅守卫的冒号规则是启发式**:它保住了 'Prompt:' 式发言人标签,但会放过 'CONFIDENTIAL:' 式真横幅;bbox 位置稳定性是更强的信号,已记为 deferred。
- **证据保全 eval 覆盖有偏**:仅 43/77 篇(MMDocIR 子集)有标注;**语料占比最大的 financial_research_zh 及 policy/form/tech_report 完全未被证据保全评估**——它们的切块策略只经统计画像,未经检索验证。law 的 100% 来自单篇文档,不可外推。
- **est_tokens 对数字密集文档偏差大**(财报 5.08 vs 假设 4.0):已验证、有兜底、决定不改,但它是启发式这件事要如实说。
- **promote_bare 是文档级开关**:混合文档(真章节编号与重启列表共存)无法局部仲裁,整篇一个开关的脆性是声明过的取舍。
- 一句可用的话术:"这一层我最有信心的不是某个算法,而是**每个数字都能报出口径、每个已知缺陷都有最小复现**——包括那些对我不利的。"

---

*本篇锚点均按 2026-07-07 代码实读核验(对抗审查修复落地后);评估数字口径:77 篇/5337 标题(定级语料)、43 篇/356 问(证据保全)、72 题(N8 回归基线,勿与 88 题混用)。*
