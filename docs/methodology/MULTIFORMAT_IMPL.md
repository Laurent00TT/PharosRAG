# 多格式扩展实施记录（docx / pptx / xlsx / 扫描件）

> 目标：把原本只验证过 **PDF→MinerU** 的 chunker 扩到其他文档类型。核心主张是组件 **format-无关**（只吃 `Element[]`），扩格式 = 写 adapter + 在各自语料上重跑对抗验证。本文件是这条线的干活记录(diagnose→decide→verify)。
>
> 状态：docx ✅ 已验证 / pptx ✅ 已验证 / 扫描件 ⏳ Phase 2 / xlsx ⏸ 单独决策(Phase 3)。日期 2026-06。

---

## 0. 为什么要这一步

之前的 chunker 全部实测在 PDF(MinerU)上,启发式都按 MinerU 的脾气调(reset-aware 补 text_level 压平、横幅守卫、aside_text 剔除)。用户问"是不是只针对 PDF"——架构上不是(`Element[]` 接缝),经验上是(只有 mineru adapter、只验证过 PDF)。本步要把这句话从"理论上能换"做成"实测能换"。

**关键洞察**:office 格式是 OOXML,本身结构化,**不需要 MinerU**——直接用 python-docx/pptx/openpyxl 读。而且 docx 的 heading 样式给的 `text_level` 比 MinerU 推断的更干净。扫描件仍走 MinerU(OCR)。Excel 是网格非文档流,是真正的分叉。

## 1. 语料采集(corpus_multiformat/)

从公网抓 docx/xlsx/pptx/扫描件各 ~50,逐文件 magic-bytes 校验 + 库打开验证 + sha256 去重 + 内容审核选差异性。结果(详见 `corpus_multiformat/MANIFEST.csv`):

| 类型 | 池→终 | 体积 | 语言 | 差异性 |
|---|---|---|---|---|
| docx | 75→50 | 7.7M | 7 种(en34/zh·fr4/de3/es·ja2/pt1) | 结构全谱:纯散文→99标题/52表/40图 |
| xlsx | 70→53 | 70M | 3 种(en49) | 1–127 sheet、7–24.5万行、图表/公式/合并表头 |
| pptx | 60→50 | 258M | 5 种 + RTL 阿拉伯语 + 2 纯图 | 6–229 slides |
| scanned | 80→50 | 494M | **13 种** | 1–626 页,竖排/Nastaliq/Fraktur/西里尔 |

源:archive.org(API)、政府开放数据、GitHub(`gh` 枚举)、edu。全部 gitignore(不提交二进制)。**约束**:MinerU 在线 API 单文档 ≤200 页 → Phase 2 扫描件要按页数过滤/抽样(池里有 626 页的)。

## 2. docx adapter(`chunker/src/chunker/adapters/docx.py`)

> ⚠️ **已降为 fallback** —— docx/pptx 的提交路径见 [§9](#9-决策修正docxpptx-正式改走-mineru-office-后端2026-06)。本节及 §3/§5 记录的自研 adapter 仍有效(零依赖 fallback),但**不再是主力**。

**设计**:
- 按 `doc.element.body` 子节点**真实阅读序**遍历(段落 `w:p` + 表格 `w:tbl` 交错)——`doc.paragraphs` 不含表格顺序,必须遍历 body。
- heading 级别用**语言无关的 `style_id`**(`Heading1..9`),回退 `style.name`/`Title`——非英文 docx 的样式显示名会本地化,但 style_id 不变。
- 表格 → `<table><tr><td>` HTML 进 `table_body`;内联图(`w:drawing`/`w:pict`)→ image 元素。
- `page=0` 全程(docx 可回流无固定页)→ page-grouped 与每页横幅守卫**自然失效**(正确:docx 无内联横幅,页眉页脚在独立 part,遍历 body 时天然排除)。

**诊断 → 决策:bold 标题推断(唯一新启发式)**

首跑发现 **32/50 docx 的 `sec=0`**(扁平)。诊断:这些文档**根本没用 Word Heading 样式**(作者手动加粗/字号充当标题)——这是真实世界 docx 的**常态(64%)**,不是个例。python-docx 看不到 style → `text_level=None` → 退化成无结构大块,反而比 MinerU(能从字体推断)给得少。

但 docx 有 MinerU 没有的信号:**直接字符格式**。量化:32 个无样式文档里有 **452 个"整段加粗+短"候选**(vs 全语料 420 个样式标题)——格式信号能把可恢复结构**翻倍**。样本确认是真标题("TAKE ACTION NOW!"、"Section 1. Purpose:"、德语"Jahresbericht 2017/18")。

**决策**:在 **adapter**(非 core,保持 format-无关)加保守回退——**仅当文档完全没有 Heading 样式时**,把"整段加粗 + ≤14 词 + 不以句末标点结尾"的段落推断为 level-1 标题。保守点:
- 统一 level-1(不赌字号分级——PDF 时代的对抗审核证明 font_rank 跨文档不可靠)。只求节边界 + breadcrumb,不求层级。
- 有样式的 18 个文档**不触发**推断(信任样式,不混淆)。

**验证(前后对比)**:

| 指标 | 推断前 | 推断后 |
|---|---|---|
| headings(text_level set) | 488 | **690** |
| sections | 472 | **674** |
| content orphans | 0 | **0** |

扁平→结构化实例:bylaws 0→22、legal_berkeley 0→23、lugov 0→29、葡语 ecdc 0→22。**仍扁平的**(mit_esp 纯散文 / cngov 中文表单 / CV / 财务表)是**真正无标题结构**的——推断没硬造,正确。

## 3. pptx adapter(`chunker/src/chunker/adapters/pptx.py`)

**设计**:幻灯片 = 页(`page=slide_idx`)→ 配 `Chunker(page_grouped={"pptx"})` 一片≈一文本块。标题占位符(`PP_PLACEHOLDER.TITLE/CENTER_TITLE`)→ level-1 heading;正文 text_frame 段落 → text;表→table HTML;图→image;chart→chart。演讲备注排除。标题优先排序。

**实测**:49/50 解析(`grinch` 纯图 deck → 0 元素,符合预期:58 张整页 PNG 无文本),**orphan=0**,page-grouped 生效。

**已知问题(非 bug,待 Phase 决策)**:`chunks/slides=2.04` 不是页分组失效,是**资产原子化**——每图/表/chart 单独成 chunk,图多的 deck(如 201 张图的 quotes deck)自然 >1。但放大了老问题:**captionless 图/chart → 0-token 占位 chunk**(检索无价值)。embed 阶段可跳 0-token;或后续给 pptx 图拉 alt-text 当 caption。

## 4. 核心主张验证

`core.py` / `retrieve.py` **零改动**(12 单测仍全过),docx + pptx 都跑通、内容零丢失。**"core format-无关"成立**——format-specific 的只有 adapter。PDF 调优的启发式在 office 上:横幅守卫因 `npages<4` 自动禁用(docx page=0);reset-aware 照常处理 office 标题编号;aside_text 在 office 不出现(无害)。

## 5. 对抗验证(workflow `w1046q938`:27 finding / 20 confirmed / 0 refuted)

**最大教训:我的 `orphan=0` 是假指标。** orphan 检查只遍历 adapter **已吐出**的 `Element[]`,对"提取前就丢的内容"完全盲——它量的是 adapter↔chunker 一致性,**不是** document↔chunker 保真度。"orphan=0 证明零丢失"是**错的**。对抗审核用真实语料把这个盲点和 3 个真内容丢失 bug 全挖出来了:

| # | 真 bug | 实测丢失 | 修复 |
|---|---|---|---|
| **F1** | docx **文本框** `w:txbxContent`(嵌在 w:drawing/mc:AlternateContent,非 body 子节点)从未遍历 | 10/50 文件、390 段、eu_easa 丢 232 段 | `_textbox_texts` 提取 + Choice/Fallback 去重 |
| **F2** | pptx **组合形状** GROUP 未递归(`slide.shapes` 不下钻) | 16/50 文件、352 行、harvard 丢讲师邮箱 | `_leaf_shapes` 递归 GROUP |
| **F4** | pptx `<a:br/>` 软换行被 run-join 吃掉 | 行内换行 | 改用 `para.text` |
| **F3** | captionless 图/chart → 1617 个 0-token 垃圾 chunk | chunks/slides 虚高 2.04 | core `_asset_chunk` 无 caption 无 content_raw 时返回 None |
| +sdt | docx **块级内容控件** `w:sdt`(body 子节点,tag 非 w:p/w:tbl)被跳过 | es_ucss 表单占位文本 | `_iter_blocks` 下钻 `w:sdtContent` |
| +A4-2 | page-grouped 检索:无节 slide 命中扩窗到整个 deck | slides small-to-big | `retrieve` 单页命中只在本页开窗 |

**建了真·保真度指标** `scripts/coverage_office.py`:adapter 文本 vs 原始 OOXML(`document.xml` 的 `<w:t>` / slide 的 `<a:t>`,排除页眉页脚 furniture)的**词集包含度**。这才能看见提取前丢的内容。

**修复后实测(真实 50+50):**

| 指标 | 修复前 | 修复后 |
|---|---|---|
| docx 正文词覆盖 | (orphan=0 假象) | **95.6%** |
| pptx 正文词覆盖 | (orphan=0 假象) | **99.1%** |
| **文本 orphan(真丢失)** | — | **0**(docx & pptx) |
| pptx chunks/slides | 2.04(含 1617 垃圾) | **1.23**(captionless 已清) |
| 资产 orphan(captionless 图,预期丢弃) | — | docx 115 / pptx 1926(无可检索文本) |

点名串已逐个验回:docx tamucc "Cultural Relativism" ✅、harvard 讲师邮箱 ✅。docx 词覆盖剩余 <97% 几乎全是 **CJK 文件**——经字符级核验是**度量噪声**(Word 把中日短语拆成多 w:t run,token 对不上;cngov 字符覆盖 **100%**、jpgov **99%**),非真丢失。`core.py` 仍**零改动主张成立的部分**:reset-aware/横幅/aside_text 未误伤;改的是 `_asset_chunk`(通用,利好 PDF)与 `retrieve` 单页开窗。12 单测全过。

**仍存在的真 gap(诚实,round-2 候选,均已确认非阻塞)**:
- docx **脚注/尾注**(`footnotes.xml` 独立 part)未捕获——同 PDF 的 page_footnote,可仿照补。
- **内联**内容控件 / 深层嵌套 SmartArt(pptx harvard 仍 88%)未完全覆盖。
- bold 推断的精修(枚举豁免已加;冒号标签 vs 字段标签的区分留待)。

## 6. Phase 2:扫描件(✅ 已验证,复用 mineru adapter,无新代码)

**40/50 篇 ≤200 页**经 MinerU VLM OCR 解析(1659 页,3 账号均衡 553 页/账号,**40 ok / 0 fail**),走**现有 `from_mineru` adapter**(扫描件→OCR→content_list→Element[] 同普通 PDF,零新 adapter)。10 篇 >200 页排除(MinerU 在线 API ≤200 页/文档约束;语言无损失)。

**实测(40 篇,13 语言)**:

| 指标 | 值 |
|---|---|
| 解析成功 | 40/40 |
| **文本 orphan(真丢失)** | **0** |
| 资产 orphan(captionless 扫描图) | 919(预期,无可检索文本) |
| heading rate | 8% of elements |
| 崩溃 | 0(含 RTL 阿拉伯/乌尔都、CJK 竖排、Odia/Tamil) |

**为什么 orphan=0 这次可信(无 office 那种自指标陷阱)**:`from_mineru` 是 content_list 的 **1:1 忠实映射**(无提取期逻辑会丢内容),所以"Element↔chunk 守恒"等价于"content_list↔chunk 守恒"=真守恒。OCR 保真度是 MinerU 的职责,不在 chunker scope。

**抽样核查的真实行为**:
- 报纸扫描(zh_xinminbao 28 L1 / ja_fujin 26 L1):L1 偏高**是正确的**——报纸每篇文章一个标题,不是过度切分。
- 干净书籍/年报:节树合理。手稿/索引扫描(ta_tdl/raggy_rastus 0 节):真无结构,扁平正确。
- chunker 结构质量**忠实反映 MinerU 的 OCR 层级**:OCR 准则准,劣质 OCR 则退化——这是 parser 的活。
- 语料标注修正:`portuguese__AlchemistPauloCoelho` 实为阿拉伯/乌尔都语(curate 标错),不影响解析。

**结论**:扫描件这条线**不需要 chunker 改动**,现有三轮验证过的 PDF 路径直接覆盖;内容零丢失,多语言/OCR 鲁棒。

## 7. Phase 3:xlsx 表格 chunker(方案 A,✅ 原型)

电子表格是**网格非文档流**,document chunker 的 heading-tree 整套不映射。所以做了一个**并列的表格 chunker** `chunker/src/chunker/table_chunker.py`(`TableChunker`),但**输出同样的 `Chunk` schema**,照样插进 ingest→chunk→embed。

**方案 A 设计**:检索单元 = **行组 + 列头作上下文**(渲染成 markdown,这样"加州的联邦建筑"能命中)。
- 每 sheet 按**空行切表区**,区首行 = 表头;数据行按 token 预算分组(target 600),每块自带表头。
- **宽表列分组**:>40 列时按列切片,每片带 key 列(首列)作行上下文 → 不丢任何单元格。
- `openpyxl(read_only,data_only)` 流式读值;breadcrumb=[工作簿, sheet];page=sheet。

**实测(51 篇 .xlsx,593 sheets)**:

| 指标 | 值 |
|---|---|
| **单元格值覆盖** | **100.0%**(宽表列分组后,beataml 49%→100%、govuk 58%→100%、boe 88%→100%) |
| chunks | 152,343(巨表驱动:boe 202k 行→64k 块、chicago 警员 114k 行→28k 块) |
| 巨表/多 sheet | 202k 行 / 108·127 sheets 全扛(25s) |
| 退化块(图表/封面 sheet) | 数据表 0% / 图表 sheet 3–8%(碎源注块,内容没丢) |

**干净数据表渲染**(检索就绪):`Sheet: Sheet1\n| Location Code | Asset Name | ... | State | Zip | Latitude |\n| --- |...\n| ... |`。

### 对抗验证(workflow `w3thlchnn`:23 finding / 21 confirmed / 0 refuted)

独立 **multiset 审计**(逐出现计数,584 万单元格)**坐实值覆盖 100%**(比我自报还干净)。但揪出核心缺陷:**自报"覆盖 100%"掩盖了表头污染**——`header=区首行` 盲取 + 纯空行切区,在政府统计表(boe/census/eurostat)上 **~33% 的区表头是标题行/单位行而非真列名**,直接打掉方案A 相对裸 dump 的增量价值。这正是 office 那个"自指标看着完美、核心已坏"的同型陷阱重演。外加:`_cell` 把 `|`→`/` **篡改真值**(8043 单元格含真作者分隔列)、`.xls` 硬崩 harness 静默漏、colgrp flag 算错。

**Round-2 修复(全部已落地 + 17 单测验证)**:

| 问题 | 修复 |
|---|---|
| 表头盲取区首行(~33%错) | 找**真表头行**(首个多列 label-like 行);**多行表头**前向填充合并;**跨带表头继承**(续接数据带复用上一带表头) |
| 标题/注释行被丢/当表头 | 表头前的标题/单位/注释行**单独发文本块**(永不丢,boe 目录全保留)→ 表头正确率 33%→~20%(残余多为"年份列头"误判) |
| `\|`→`/` 篡改真值 | 改 markdown 转义 `\|`(值保留),换行 `[\r\n]+`→空格 |
| 续接带更宽时丢列 | 续接用 `max(carried宽, 本带宽)`,multiset deficit 21134→**0** |
| .xls 硬崩+静默漏 | chunk() 显式抛 `ValueError`;harness glob `*.xls*` 把失败计入 bad 显示 |
| colgrp flag 算错 | 改记实际列区间 `cols:{first}-{last}` |

**修复后**:multiset 值覆盖 **100%**(boe/beataml/aces 全 deficit=0)、17 单测过、巨表仍可扛。

**已知限制(round-3 候选)**:① 2 篇 legacy `.xls`(需 xlrd<2 或转换);② 图表/dashboard sheet 仍渲成碎表/文本块;③ 表头检测在极不规则多行表头上仍 ~20% 不完美(政府表的固有难度);④ bool/大整数 float 渲染边界。

## 9. 决策修正:docx/pptx 正式改走 MinerU office 后端(2026-06)

> ⚠️ **本节推翻 §2/§3 把自研 adapter 当主力的结论。** 自研 adapter 降为零依赖 fallback;docx/pptx 的**提交路径 = MinerU 原生 office 后端**(`scripts/parse_office.py`)。

**触发**:发现 MinerU 有**原生 office 后端**(`mineru/backend/office/`,纯规则 OOXML 解析,**零 VLM/ML 模型**——广搜 torch/onnx/ocr/vlm 全路径零命中,`MagicModel` 只是规则分类器),不是我先前误以为的"转 PDF"。本地跑免配额、纯 CPU、输出 content_list(同 PDF/扫描件 schema)。

**第一次比较我得出"自研更强(94% vs 87%)"——这是测量 bug。** 对抗 review(`w2rjwb2df`,19 confirmed / 3 refuted)实锤:我的 MinerU 覆盖率提取器**只数 `text`+`table_body`,漏读 `list_items`**(MinerU 把笔记/编号/bullet 放这)。公平计入全字段:

| 口径 | 自研 adapter | MinerU |
|---|---|---|
| 我的错口径(漏 list_items) | 94.5% | 88% |
| **公平口径(全字段)** | 94.5% | **95.1%**(反超) |
| "adapter 领先≥8pct"文档 | — | 15/50 → **0/50** |

`edu_tamucc` 的 "Cultural Relativism" **MinerU 没丢**(在 list_items 块),98%→47% 纯属度量假象。逐词净账反偏 MinerU(多捕 157 vs 65)。自研还被查出真 bug:`_table_html` 不递归嵌套表(un_undf 丢正文)、内联 `w:sdt` 不下钻(es_ucss 丢西语表单)。

**Tika 头对头(同口径实测)**:docx 三方(自研/MinerU/Tika)文本覆盖**全打平 ~95%**;pptx Tika 因 group 处理略胜(harvard 100% vs MinerU 88%);xlsx 两者都给整表(不做网格切块)。

**为什么选 MinerU 而非 Tika**:覆盖打平的前提下,MinerU 的 content_list schema **切块即用**(text_level/list_items/table_body/image_caption/chart/OMML→LaTeX)、与 PDF+扫描件**管线统一**(同 `from_mineru`)、**纯 Python 无 JVM**(Tika 要 JVM+63MB JAR)。Tika 更适合"纯文本倒进搜索索引"的场景。

**端到端验证**:`parse_office.py` 100/100 docx+pptx 解析成功(写 `_manifest.csv` 供 pptx 路由 page_grouped),chunker 零失败(50+50,2733 标题,3487 chunks)。

**元教训(诚实记录)**:本次扩展我**三次栽在自造的"恭维自己代码"的指标上**——① orphan=0 自指标(office 验证戳穿)、② xlsx 覆盖率 100% 掩盖表头污染、③ 这次漏 list_items 把 MinerU 判低。共同模式:自指标会自欺,真相要靠独立 ground-truth + 对抗测量。**这本身就是"少手搓、多用成熟库 + 独立验证"的最强论据。**

## 10. 图描述进检索文本(让图可召回,2026-06)

**目的**:图在向量检索里是隐形空壳;把"最好的可得描述"折进图 chunk 的 embed 文本,让图可召回(设计 A:图各自成 chunk,描述当检索文本)。

**Part 1(core.py `_asset_chunk`/`_asset_desc`)**:图/chart 的检索 text = caption + footnote + `asset_content`。`_asset_desc` 对 mermaid **只抽节点/边标签**(`["Irrigation","Runoff"…]`),丢脚手架(`graph TD`/`-->`/`style X fill:#f9f`)+ 词边界截断 800。
**Part 2(parse_office.py)**:docx/pptx 从 `wp:docPr@descr`/`p:cNvPr@descr` 抽 alt → 填 image_caption,**强过滤**(文件名/路径/下划线ID/PPT样板/Office AI 自动 caption "with low confidence"/"A picture containing"),**只在图数对齐时应用**(否则跳过不错配)。

**对抗 review(`wmgwu4n4i`,4 finding confirmed;verify/synth 撞 session 限制未全跑)→ 已修**:
- mermaid 语法噪声 67 chunk → **0**(F2)。
- office 自动 caption 漏网 3/12 → **0**(part2-F1)。
- **数字诚实修正**(F1):非 1049 而是 **1586** 个图 chunk,其中**仅 47%(757)真有 VLM `asset_content`**,扫描件那半是 caption/footnote 来源(部分是 OCR 噪声如 `2011141981`)——之前把"VLM 抽取"与"caption 来源"混为一谈,已纠正。

**实测后**:PDF/扫描件 1585/1586 图带描述;office 9 个 alt(干净)。**三层**:Tier-0(alt)✅、Tier-1(MinerU 图 VLM,PDF/扫描件已有)✅、Tier-2(纯照片通用 captioning)未接。

**未验证/未解(session 限制 + 设计权衡,待后续)**:① **对齐正确性**(对齐文档里第 i 个 alt 是否真第 i 个图)+ ② **count-mismatch 跳过是否过保守**(brooklyn 41 真 alt 因 MinerU 只吐 35 被跳;可探 img_path=sha256(b64) 重构匹配救回)+ ③ **检索精度影响**(图描述会不会误召回)——这三个 review 维度因 session 限制未出独立 verdict。

## 11. 安全审计与修复(metadata + ACL 落地,2026-06)

> 目标:把文档级 **metadata + 访问控制(ACL)** 设计进 chunk 阶段(文档还没接权限,但先把边界做进去)。语义分层:**`doc_meta` = 便利**(payload 过滤 + 引用)、**`acl` = 安全边界**(检索硬预过滤 + fail-closed)。落地后跑了一次**聚焦安全的大 workflow** 做对抗审计。

**设计落地**:`chunker STAMPS,ingest EXTRACTS`——`extract_doc_meta`(office core.xml / PDF Info)抽 metadata,`Chunker.chunk(..., doc_meta, acl)` 把两者 **deepcopy 盖到每个 chunk**;`acl` 缺省 **fail-closed** 到 `RESTRICTED_ACL`(`unset=True`,空 allow)→ 未接权限的文档默认拒绝所有人,绝不意外公开。`TableChunker`(xlsx 第二条 Chunk 生产路径)同样盖章。

**对抗审计(workflow `wipax2btr`:19 finding / 17 confirmed / 0 refuted / 8 确认真泄漏)**。归并后 = 两类根因 + 一个 medium(TableChunker 系列在 review 期间已被修复任务合入,我验证为真):

| 项 | severity | 裁决 | 状态 |
|---|---|---|---|
| **`assemble_big` small-to-big 跨 ACL 取材**(S3 系列 + F2) | 🔴 critical | **上线前必修(阻断)** | ✅ 已修 |
| **`deny` 字段被静默忽略** | 🟠 high | **上线前必修 / 改契约(阻断)** | ✅ 已改契约 |
| `extract_doc_meta` override 无键白名单 | 🟡 medium | 建议修(非阻断) | ✅ 已修 |
| TableChunker fail-closed(裸 `{}` 绕过闸) | 🔴→✅ | 可接受;补回归测试 | ✅ review 期已修,已验证 |
| deepcopy 隔离 / public 受 tenant 约束 / doc_meta 不构成通道 | ✅ praise | 正确 | 保留 |

**核心结论(review 原话)——"生产闭合、检索敞开"**:chunker 两条生产路径(core + table)现在都 fail-closed 盖 `RESTRICTED_ACL`,但 `assemble_big` 在硬过滤**之后**用原始 elements 按 idx 重新取材,把"无权 chunk 根本检索不到"的保证整个架空。命中一个公开小节 → small-to-big 把同区间内被单独收紧(铁律3 per-chunk override)的兄弟小节明文一并捞回。**契约自相矛盾**:旧铁律4 声称"big-block 只在同文档内取材,语境不越权"——"同文档"≠"同 ACL",这句是**错误声明**。

**修复(全部已落地 + 真语料/单测验证)**:

| 问题 | 修复 |
|---|---|
| S3/F2 跨 ACL 取材 | `assemble_big` 加 `acl_index`({idx:acl},`ChunkResult.acl_index()`)+ `admit` 谓词;`_gather`/`_window_within`/`toks` 全程按 ACL 取材,默认**等价类**(只取与命中块同 acl,未知 idx **fail-closed 排除**);`BigBlock` 加 `acl` 字段带回校验依据;`Chunker.assemble_big` 自动建 acl_index → **便捷路径默认安全,调用方零改动** |
| deny 静默忽略 | 采纳 review 推荐 (b):§6 schema **删 `deny`** + 显式警告"硬过滤示例不执行 deny,收紧请从 allow 移除;确需黑名单须自己加 `AND NOT deny`" |
| meta override payload-confusion | `meta.py` 加 `_ACL_KEYS` 黑名单,拒绝把 `acl/tenant/allow/...` 抄进 doc_meta |
| 契约文档 | INTEGRATION §6 删错误铁律4 → 改"big-block 必须 ACL 感知" + 新增**铁律5 出口不变量**(任何返回给用户的文本都要能映射回 acl 复核);retrieve 示例改成构建并传 acl_index;API.md 签名全对齐 |

**真语料 before/after**(`government__21-00620-INLSR`,568 elements / 80 chunks;把一个 chunk 单独收紧后统计 small-to-big 把它独有 token 捞进多少 big-block):

| | 泄漏 big-block | small-to-big 是否还生效 |
|---|---|---|
| LEGACY(无 acl_index) | **3/79** | — |
| **修复后**(auto acl_index) | **0/79** | **78/79 块仍正常长大**(没被打回 hit-only) |

> 注:review 报 83/84 是"跨 chunk 取材的普遍度"=攻击面;此处 3/79 是"某一个被收紧 chunk 的明文实际漏进几个邻居窗口"。两者一致:跨 chunk 取材近乎普遍,任一被收紧块漏进覆盖它的那几个邻居;修复后归零且召回基本不变。**单测 24→27**(+跨 ACL 排除、+legacy 无防护锁约、+meta 白名单)。

**两点诚实交代**:
1. **等价类默认偏保守**:`acl_index.get(idx) == hit_acl` 判等,而 `allow` 是**列表**——`['a','b']` ≠ `['b','a']`。本应同权但顺序不同的 chunk 会被判为不同 → 兄弟内容被排除(**安全方向,牺牲召回**)。生产想精确"取调用者一切有权看的(跨不同但可见的 acl)",应传 `admit=lambda acl: acl_admits(acl, user)` 复用硬过滤同一谓词。
2. **拒绝采纳 review 一条 suggested_fix**:它建议 `MappingProxyType` 包 `RESTRICTED_ACL` 防篡改——但 `deepcopy(mappingproxy)` 在 Python 3.12 抛 `TypeError`,会把 fail-closed 默认路径变**崩溃**。现有 deepcopy 隔离(review 实测五类污染攻击全失败)已足够。

**依赖集成方(本仓边界外)**:`acl_index`/`admit` 是给集成方的契约;真正的硬过滤 SQL + `acl_admits` 谓词要在接向量库时落地——本仓只有文档契约,没有执行它的 query engine。落地时务必:① 保留 `(... OR public)` 括号(压平成 `AND` 会退化成跨租户 bypass);② deny 不自动生效。

**元教训(承接 §9 的同型反转)**:§9 记的是"自指标会自欺"——这次反过来,独立对抗 workflow **确认了我事前的诚实预测**(跑 review 前我就对用户点名 F1+F2 两个真漏洞,结果坐实),且我的自审防御(deepcopy 隔离)在五类攻击下守住。所以独立对抗验证**双向有用**:既能戳穿恭维自己的假指标,也能给诚实预测背书。另一条:**安全字段"写了不生效"(deny)是最毒的契约失败**——运维拿到正向确认(无报错、字段进 payload),控制却为空。

## 12. 多模态 embedding 接口：img_path 透传（2026-06-25）

**背景**：下游稠密 embedder 选定 `Qwen/Qwen3-VL-Embedding-8B`（多模态，能直接编码图像）。这**反转**了之前"图语义是已知缺口、只能靠 VLM 文字描述"的判断——图/图表可**直接对原始裁切图做向量化**，跳过"图→VLM 文字→text embedding"的有损链路。前提是 chunker 把 MinerU 的裁切图引用透传出来（之前全链路丢弃了 `img_path`）。

**逐类决策（实测 MinerU content_list 字段后定）**：

| 类型 | MinerU 给的 | 向量化路径 | chunker 带 image_path? |
|---|---|---|---|
| image / chart | 裁切图 `img_path`(images/*.jpg) | **图像向量化** | ✅ 暴露 |
| table | `img_path` **+** `table_body`(HTML) | HTML 文本（精确、结构化，优先） | ❌ None（走 content_raw） |
| equation | 仅 LaTeX(`text`/`text_format`)，**无 img_path** | LaTeX 文本 | ❌ 无图可带 |

**本次只做图片（image/chart）的图像向量化**；表格走 HTML、公式走 LaTeX，均不带图引用（避免下游误用表格/公式图）。

**全格式图像覆盖（B②，2026-06 修复）**：`img_path` 现覆盖 **PDF / 扫描件 / docx/pptx** 全格式。docx/pptx 走 MinerU office 后端，原 `parse_office.py` 传 `image_writer=None` 故不裁图（content_list `img_path` 全空，实测 0/567）；**改为传 `FileBasedDataWriter` 后 MinerU 自己裁图存盘 + 填 img_path**（重跑实测 **1097/1116** image/chart 带 path、870 张去重裁切图）。关键洞察：MinerU **内部做图↔元素关联**，绕过"按 OOXML media 顺序对齐"的坑——media 混母版/装饰/重复图（acl_gov 20 个 media 文件 vs 仅 3 张内容图），手动对齐必错配，而 MinerU 自己裁正好 3 张。差点去手写 OOXML 提图+对齐（脆弱大工程），核查 parse_office 发现只是一个被关掉的 `image_writer` 参数。（alt-text caption 增强是另一条独立线，仍保守对齐、count-mismatch 时跳过，不影响图像向量化——纯图靠图像向量召回，不靠 alt。）

**纯图存活（①，对抗 review 后修）**：`_asset_chunk` 原对"无 caption/footnote/VLM 内容"的图直接 `return None`（纯文本时代的卫生），实测会丢掉**约 30%** 的图（academic 38%）——而这些"没文字的图"恰是 VL 图像向量化唯一能召回的对象。已改为"**有 image_path 即视为可检索**"：纯图照常产出 chunk（text=占位符、n_tokens=0），打 `image_only` flag 供下游识别"走图像向量、跳过稀疏路"；仅"无图无文无 body"的真空 placeholder 仍丢弃。`img_path` 空串归一化为 `None`（③）。

**实现（贯穿三处，纯透传无 IO）**：
- `types.py`：`Element.image_path` + `Chunk.image_path`（均 `str|None`，加在 dataclass **末尾**不破坏位置参数顺序）。
- `adapters/mineru.py`：`from_mineru` 对每个元素 `image_path=el.get("img_path")` —— **忠实提取**（image/chart/table 都填，text/equation 自然 None）。
- `core.py`：`_asset_chunk` **只对 image/chart** 设 `image_path`（`el.kind in ("image","chart")`），table/text 为 None。

**契约**：`image_path` 是**相对 MinerU 输出根目录**的路径，**忠实透传、chunker 不碰文件系统**（保纯函数核）。绝对路径解析 + 喂 VL embedder 是**下游（embed/ingest）的事**。

**验证（diagnose→fix→verify）**：
- 端到端真实数据（`parsed/academic_paper__2309.17421v2`，136 image+6 chart+2 table）：Element 层 image/chart **142/142** 带 path、table 也忠实提取；Chunk 层 image/chart **107/107** 带 path（107<142 是无 caption 无内容被 skip）、table 与 text **全 None**。
- 单测 `test_image_path_passthrough`：锁 adapter 提取（Element image+table 都有、text None）+ core 透传（Chunk 只 image 有、table/text None）。**28 测全绿**。

**押后（Element 层已留口）**：表格/公式的图像向量化未做；若将来要表格视觉版面，Element 层已忠实保留 table 的 `img_path`，只需放开 `_asset_chunk` 的 kind 判断 + 下游加表格图路径解析。

---

## 13. table round-2 第 1 档：表头识别重写（merged-header + structure-misdetect，2026-06）

**起因**：对抗 red-team 实跑 5 痛点，推翻了我"占比低不值得做"的判断（教训见 memory `feedback_severity-not-occurrence`）。第 1 档修两个根因同源、且打穿组件核心卖点（"列名↔数值绑定"）的缺陷。

**修的三个机制**（`chunker/src/chunker/table_chunker.py`）：
1. **structure-misdetect（年份表头）**：`_numeric_frac` 把 4 位年份（1900-2100）算作**标签而非数值** → `['Race',2010,2020]` 不再被误判为数据行。
2. **merged-header（多层 merge 几何）**：`read_only`+`values_only` 丢 merge 几何（合并值只在左上角、其余 None）。新增 `_parse_merges`（从 xlsx zip **轻量解析 mergeCell、不加载 cell** → 大表性能不崩）+ `_broadcast`（左上角标签广播回整个 range）+ 逐列自上而下 join → 完整多维列名，替代只拼 2 行的 ffill。
3. **band 劈散（表头内空行）**：`_bands` 不再把"被跨行 merge 覆盖的空行"当分隔 → 多行表头（hvs 5 行）不被劈成 header_only 碎片。

**验证（diagnose→fix→verify）**：
- 回归单测 3 个：3 层合并表头（叶层不丢）、年份表头（识别为表头）、表头内空行（band 不劈）。
- 真实 US Census 文件 before/after：
  - `retail_mrts`：col2 列名 `'CV for Retail Sales'`（丢维度）→ `'CV for Retail Sales 2023Q4 (p) Total E-commerce'`（完整三维），9 列含 [E-commerce+2023]。
  - `hvs_vacancy`：United States 行 6.4 列名 **空白** → `'Rental Vacancy Rates First Quarter 2023'`；header_only 碎片 **6→2**。
- chunker **33 测全绿**。

**性能**：merge 几何走 zip XML 轻量解析，不触发 `read_only=False` 的全 cell 加载 → 127 sheet/24 万行大表仍流式。

**round-2 第 2 档（legacy-xls，已修 2026-06）**：`.xls` 原直接抛错 = 整文档 0 chunk（静默不可检索）；新增 `_read_xls`（xlrd reader，`formatting_info=True` 拿 merged_cells、half-open→inclusive 转换，**复用第 1 档表头重建**）。真实 INSEE：`population_ensemble.xls` **0→6767** chunk、`dossier28.xls` 0→17 chunk，red-team 提的事实（Ambérieu commune=14022）现可检索。

**round-2 第 3 档（嵌入 chart，已修 2026-06）**：Method A 只读 cell、丢 chart。新增 `_parse_charts`（zip 解析 `xl/charts/chartN.xml`：所有 `<a:t>` 标题/轴 + `<c:tx><c:v>` 缓存系列名；sheet 关联取 series 公式 `'Sheet'!$range`），每 chart 一个 `kind='chart'` chunk。真实 EIA STEO：**0 → 66 chart chunk**，red-team 的 `Historical spot price`/`NYMEX` 系列名现可检索。覆盖两种真实编码（Excel 缓存 `c:v` / openpyxl 裸 `a:t`）。

**round-2 第 4 档（oversplit）→ 留检索期**：red-team 自己定性"缓解即可、不重写切分"——本质是 retrieve 层用 `source_indices` 把列切/行组切散的记录回拼，属**向量库接入时的检索契约**，不是 table_chunker 的切分逻辑。**chunker 侧 round-2（第 1-3 档）完成。**

---

## 14. 封板对抗 review + 修复（进 embed 前，2026-06）

进 embed 组件前对整个 chunker 做封板全面对抗 review（7 维度 × red-team + 独立 verify，47 agent / 38 confirmed）。结论：**3 条 high 必修，修完封板就绪**。

**必修 3 条（已修 + 回归测试锁）**：
1. **同名兄弟 section 合并**（core.py `section_key`）：重复子标题（表单/附录/多主体申报，~15% 文档）的兄弟 section 因 crumb 文本相同被合进同一 chunk + 错锚第一个 section → 跨 section 正文混进同一 embedding（**不可逆写错向量库**）。修：`section_key` 从 crumb 文本改用 `_sec_head`（每 heading 实例 idx 唯一）。
2. **image_path 零净化**（mineru.py→core）：多模态 embed 链唯一文件读取入口逐字透传，污染产物（`../`/绝对/UNC/`://`）→ 任意文件读/SSRF。修：`_safe_rel` 在组件边界净化。
3. **admit 无 acl_index 时 big.acl 说谎**（retrieve.py `_make_admit`）：admit 提供但 acl_index 缺失时按 hit_acl 评估每元素（跨 ACL）却盖非 None acl（伪装已验证）→ 静默泄漏。修：admit 必须有 acl_index，否则 raise。

**一并做的 should-fix**：
- **doc_type 上 Chunk schema**：原 Chunk 无 doc_type → assemble_big 永远 DEFAULT_BUDGET（law/fin 预算查询期失效）。Chunk 加 doc_type + chunk 期 stamp + assemble_big 正确取（law=700 实测生效）。**稳定 embed payload schema**。
- **lang `zh` 别名**：est_tokens 接受 ISO `zh*`（原只认 `ch`，传 `zh` 静默走英文除数 → CJK 低估 2.35x 风险被 Qwen3-VL 截断）。
- 文档卫生：API.md flags 补 `image_only`/table flags + `doc_type` 行、INTEGRATION payload 补 `lang/page_end/doc_type`、删死代码 `_is_title_row`（引用未定义正则的 NameError 炸弹）、测试数刷新。

**接受的取舍（不阻塞封板）**：est_tokens 启发式（接 Qwen3-VL 后用官方 tokenizer 重标 BUDGETS）、promote_bare 全有/全无脆性、docx/pptx fallback 不填 img_path、原型 chunk_document.py 并存、acl_index 防御纵深项。

**验证**：chunker **38 测全绿**（新增同名兄弟/路径净化/admit-raise/doc_type-lang 4 项）+ 真实 academic doc 验证 section_key 改动无破坏（415 chunk/300 section/0 缺 anchor）。**chunker 封板就绪,可进 embed。** 第 4 档 oversplit + est_tokens 重标定留检索期/embed 期落地。

---

## 8. 待办

- **图 alt-text/caption(跨格式真缺口)**:MinerU/自研/Tika 对嵌入图都只到邻近题注(≈1%),要拿图内语义得单独接视觉模型(Tika 能抽 docx 的 `wp:docPr@descr` alt——可移植)。
- ~~benchmark Docling HybridChunker vs 自研 chunker core~~ ✅ **已做(2026-06,见 [EVALUATION.md §8](CHUNKING_EVALUATION.md))**:Docling 在 MinerU 输出上证据保全崩盘(missing 3–5×)→ 不换;自研护城河是 source_indices 精确溯源等工程集成、**非边界算法**(纯边界 chonkie 略胜 71.6%→74.9%)。组件版重测确认 R1–R3 修复对证据保全无感(价值在 breadcrumb/检索语义)。
- table chunker round-2:legacy .xls 支持、多行/合并表头、图表 sheet 识别跳过、过切修正。
- 自研 adapter(仅 fallback,优先级低):`_table_html` 递归嵌套表、内联 `w:sdt` 下钻。

## 7. 产物清单

- **docx/pptx 解析(主)**:`scripts/parse_office.py`(MinerU office → `parsed_office/<id>/*_content_list.json` + `_manifest.csv`)
- **xlsx**:`chunker/src/chunker/table_chunker.py`(方案A 表格 chunker)
- adapter(fallback):`chunker/src/chunker/adapters/{docx,pptx}.py` + 主 `adapters/mineru.py`
- harness:`scripts/review_{docx,pptx,xlsx,scanned}.py`、`coverage_office.py`
- 语料 + 清单:`corpus_multiformat/{docx,xlsx,pptx,scanned}/` + `MANIFEST.csv`、`parsed_office/`、`parsed_scanned/`(均 gitignore)
- **metadata + ACL(§11)**:`chunker/src/chunker/meta.py`(`extract_doc_meta` + ACL 键白名单);`core.py`/`table_chunker.py`(盖章 + fail-closed `RESTRICTED_ACL`);`retrieve.py` + `types.py`(`assemble_big` ACL 感知 + `BigBlock.acl` + `ChunkResult.acl_index()`);契约 `chunker/docs/INTEGRATION.md §6`(铁律1–5)、`API.md`;回归 `chunker/tests/test_{core,table}.py`(ACL 断言)
