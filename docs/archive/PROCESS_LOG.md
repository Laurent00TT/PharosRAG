# 过程记录 — 从问题到策略到实现（审计追踪）

> 按时间线记录每个阶段的**输入、决策、发现、纠错、产出**，使整个结论链可复用、可审核。
> 日期：2026-06-21。模型侧执行 + 用户决策点已标注。

---

## 阶段 0 — 概念问题（动机）

用户问："parser 解析出结构化 JSON 后，chunking 怎么做？要用 LLM 吗？"
**结论**：切割不需要 LLM——parser 已交付语义边界，LLM-切割是重复付费且不可复现；LLM 只该用于"增强"（表格摘要、补 caption），不该用于"切割"。这条成为后续所有设计的第一性原则。

---

## 阶段 1 — 真实 JSON 初探（单文档）

对象：本地 MinerU 跑出的一份券商研报（`AP2026...卫星通信`）。
**发现**：
- `content_list.json` 是切块友好视图；`block_list.json` 含 `is_discarded`、`mergeConnections`、稳定 `id`——是切块金矿。
- text_level 双向不可信：真标题被压平、非标题被提升；目录项混入。
- 表格=HTML(rowspan/colspan)、chart=VLM 估读数值(`~5`)、image=VLM mermaid（明显幻觉）。
**纠错点**：最初以为 `content_list.json` 就是"那个 json"，实测发现 `block_list.json` 才带关系信号 → 确立"双文件分工"。

---

## 阶段 2 — 数据集盘点与抽样设计

数据源 `knowledge-base/datasets`，4 集 / 880 PDF / 2.9GB。
**发现/决策**：
- `omnidocbench` 无 PDF（仅标注）→ 跳过。
- `pdf_corpus_v1/manifest.jsonl` 是跨集主清单（其 `benchmark_pdf` 312 条指向 mmdocir）→ 去重，得 14 类真实类型宇宙。
- 硬约束 = MinerU 1000 页/天/账号 → **不可能全量**，必须分层抽样（chunking 要的是类型×版式多样性，不是数量）。
- 各 manifest 带 `doc_type/layout_tags/page_count/language` → 抽样有据可依。

**用户决策点①**：给了 **3 个账号 token**，要求负载均衡分散解析；覆盖选**广度优先全类型均衡**。
→ 规模定 **77 文档 / 1867 有效页**（`PAGE_CAP=50`），3 key 各 ~620 页。

**纠错点**：
- mmdocir 磁盘文件名（描述名/哈希名混合）担心与标注 `doc_name` 对不上 → 实测 **313/313 精确匹配**，映射平凡。
- `pypdf` 未装 → 装好用于 mmdocir 页数统计（缓存到 `config/`）。

---

## 阶段 3 — MinerU 在线 API 学习

读 https://mineru.net/apiManage/docs。
**确认 schema**：
- `POST /api/v4/file-urls/batch` → `{data:{batch_id, file_urls[]}}`（url 顺序对应 files）。
- `PUT <file_url>` 二进制、**禁带 Content-Type**；上传完成**自动解析**。
- `GET /api/v4/extract-results/batch/{batch_id}` → `extract_result[]{file_name,state,full_zip_url}`。
- 限制：单批 ≤50 文件、单文件 ≤200MB/≤200 页、1000 页/天/账号、URL 24h。

---

## 阶段 4 — 仓库搭建与客户端

- 建 `chunk-test-repo`，`.gitignore` **先于** `.env` 写好（保护 3 个 token）。
- `select_sample.py`（分层抽样+归类+3key均衡）、`mineru_client.py`、`parse_batch.py`。
- **抽样结果**：77 篇 / 1867 页，14 类全达标；3 key 负载 A623/B622/C622、文件 26/25/26。

---

## 阶段 5 — 解析（冒烟 → 全量）

**冒烟**（1 篇 law，端到端）通过，并暴露**关键 schema 差异**：
- 在线 VLM 输出 **无 `block_list.json`**（本地版才有）。富结构改在 `layout.json.pdf_info`：
  - `discarded_blocks`（噪声，等价 is_discarded）
  - `para_blocks[].merge_prev`（跨页续接，比 mergeConnections 更细）
  - `lines[].spans[].score`（OCR 置信度，新增可用信号）
→ 据此调整分析器与 chunker 的信号来源。

**全量**：3 账号 6 批次（按 key×语言分组）并行；**77/77 全部 done**，退出码 0。

---

## 阶段 6 — 跨文档分析

`analyze_chunks.py` 逐文档算指标、按 14 类聚合。
**核心定量发现**（详见 [../analysis/CHUNKING_STRATEGY.md](../analysis/CHUNKING_STRATEGY.md) §1 表）：
- 噪声率 0→0.37；`discarded` 与 type 噪声几乎相等 → discarded 判定可信。
- 编号可恢复率：academic 0.64 vs law 0.02 vs financial 0.04 → 层级恢复必须分领域。
- 11/14 类正文块中位 < 75 token → 必须向上合并。
- 表 HTML 化 0.86-1.0 可靠；表/图 caption 覆盖 0.04-0.89 剧烈波动 → caption 不能假设存在。

**三个反常点核验（工程纪律：不只信聚合）**：
1. **law num%=0.02 失真**：章节是 `SEC. N`（字母开头，数字正则不匹配），深层 `(a)(1)(A)(i)` 藏在正文行首枚举符 → chunker 加 `LAW_SEC_RE` + 靠向上合并保条款树。
2. **form**：可填字段=空单元格 HTML 表，有值的是标签 → 资产特判保留整表。
3. **news 离群（标题1335/图381）**：`news_combined.pdf` 是拼接多文档 → 标记为特例。

---

## 阶段 7 — Chunker 实现与验证

`chunk_document.py` 实现 7 步流水线（噪声/缝合/分领域建树/资产特判/策略组装/元数据/parent-child）。
**产出**：7695 leaf chunks（77 文档）。
**审核通过**（抽样实测）：
- law `SEC. 2` 把 (a)(1)(2)(A)(B)(C) 整棵条款树保在一个 707-token chunk，未中切。
- 财报表格：`text`=caption+来源、`content_raw`=HTML，分离正确。
- 宣传册 21/21 无 caption 图全标 `captionless`+`vlm_content`。
- parent "1 本周行情回顾" 聚 12 子块/816 token，leaf↔parent 双向一致。
**记录的局限**：parent 按 breadcrumb 文本分组 → 同名 section 会跨页合并（见 DESIGN §7）。

---

## 阶段 8 — Ground-truth 评估（可审核闭环）

动机：不自说自话，用数据集自带标注检验 chunking。
对象：`mmdocir/MMDocIR_annotations.jsonl`，与样本**重叠 43 篇 / 356 问**（带 page+bbox+通道+答案）。
方法：`source_indices` 作桥，证据 bbox→content_list 元素→chunk；详见 [EVALUATION.md](EVALUATION.md)。

**两个测量 bug（按"诊断→修复→验证"处理，未把异常当结论）**：
1. **通道匹配 0%**：标注 `type` 是字符串 `"['Figure']"` 非真列表，成员判断恒假 → 用 `ast.literal_eval` 解析。诊断证据：DIAG2 对 asset 问题零输出。
2. **missing 虚高 55.6%**：单元素覆盖≥0.5 太严，且未区分"测量定位不到"与"chunking 丢了" → 改双向覆盖判据(交集/证据≥0.3 或 交集/元素≥0.5)收集证据元素集，并新增 `unlocalized`/`out_of_range` 排除项。
   - 同时诊断了页码对齐：offset 0 在干净文档上覆盖 0.7-0.79 确认正确；满页视觉文档的"偏移获胜"是噪声，**拒绝按文档 hack 偏移**（workaround）。

**修复后结果**：通道匹配 0%→**79.4%**、missing 55.6%→**8.2%**；证据单 chunk 保全 **65.8%**、split 25.9%。
与策略难度排序自洽（law 100%/government 92%/academic 87% 单 chunk 最好）。
**关键解读**：split 高的 research/brochure/slides 是"图+讨论文字"被资产原子化分开，parent-child 检索会重聚 → split 衡量"需回取 parent"，非"证据丢失"。

---

## 阶段 9 — 对抗性多 agent 审查 + 修复（最重要的质检）

用户要求"用对抗性 agent review 结果和产出"。编排 workflow：4 个独立怀疑者（代码/评估/策略/抽样）并行只找问题 → 每条 finding 派独立验证者反向核查。**33 agent，29 finding → 14 confirmed / 11 partial / 4 refuted。**

**确认并已修的真 bug（诊断→修复→验证）：**
- **F1 致命**：Step 2 跨页缝合 bbox-IoU **全程 0 命中**——content_list(渲染坐标 ~823×936) 与 layout(PDF 点 612×792) 坐标系不同。诊断：实测 261 个 merge_prev、merge_flag 命中 0；同标题两源 bbox 比例恒定(x1.63/y1.26)证明是缩放关系。**修复**：改文本前缀匹配 → 17 篇 112 chunk 命中（验证 >0）。
- **F2 高**：law `LAW_SEC_RE` 在长度检查前 return，把 `SEC. 2102. (a)(1)...长正文` 误判标题并丢弃。诊断：publ158 丢 16 个正文块。**修复**：长度/TOC 守卫前置 → 16/16 找回（验证）。
- **F6 高**：评估脚本同源坐标 bug（用 layout page_size 归一 content_list bbox）。**修复**：逐文档文本配对推导缩放 (sx,sy) 把 GT 变换进 content_list 空间。**效果验证**：unlocalized 72→4、single 65.8%→70.1%、且严阈值下仍 70.1%（**阈值敏感性本身就是坐标 bug 症状，修了就稳**）。
- **F3 高**：financial_zh TOC（空格+页码结尾）污染 breadcrumb。**修复**：加 TOC_TAIL_RE → 0/622（验证）。

**确认并已诚实改文档：** government "图表caption 0.11" 实为表格 caption（government 0 chart）；academic 图 caption 均值 0.88 实为图加权 0.22；"77 篇"实评 43 篇 + 加 distinct_docs 列；news/law 各单篇标"不可外推"；小样本(n≤2)标警告；"切割不用 LLM" 降级为设计假设（未做 A/B）；asset 匹配补 true-asset 严口径 83.2%。

**修复后核心数字（坐标修复版）：** 证据单 chunk 70.1% / split 22.5% / missing 7.4% / asset(真) 83.2%。

**判 refuted 的 4 条（对手也会错，不盲从）：** "策略无视 eval"（eval 是后做的且已标 11.2% 不可信）、"news 极端值干扰铁律"（已标离群）、financial 跨语言过度泛化（文档未如此声称）、"11.2% 被当正面证据"（已显式标为不可信）。

详见 `analysis/eval_report.json`、对抗审查产物。

---

## 决策汇总（可追溯）

| # | 决策 | 理由 | 备选/代价 |
|---|---|---|---|
| D1 | 切割不用 LLM | 结构已含边界、可复现 | LLM 切割：贵/不可复现 |
| D2 | content_list 为脊 + layout 为信号 | 各有独有信息 | 单用其一会丢信息；需 bbox 对齐 |
| D3 | 分层抽样而非全量 | 配额约束 + 多样性优先 | 全量：超配额、收益递减 |
| D4 | 3 账号按页数均衡 | 配额单位是页/天 | 按文件数：不反映真实成本 |
| D5 | 分领域层级规则 | 各领域编号体系不同 | 统一正则：law/财报大量漏切 |
| D6 | source_indices 全程保留 | 可审核硬需求 | — |
| D7 | 策略隔离为 `assemble_text` | 唯一有取舍处，便于调 | — |

---

## 复现入口

见 [IMPLEMENTATION.md §2](IMPLEMENTATION.md)。所有脚本确定性、可断点续跑。
