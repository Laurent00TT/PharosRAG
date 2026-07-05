# 设计文档 — 结构感知 Chunking 系统

> 目的：把 PDF → 可检索 chunk 的过程做成**可复用、可审核、可调**的工程系统。
> 本文讲"为什么这么设计"；实施细节见 [IMPLEMENTATION.md](IMPLEMENTATION.md)；过程与依据见 [PROCESS_LOG.md](PROCESS_LOG.md)；策略依据见 [../analysis/CHUNKING_STRATEGY.md](../analysis/CHUNKING_STRATEGY.md)。

---

## 1. 设计目标与非目标

**目标**
- 充分利用 parser（MinerU VLM）已交付的结构，**确定性**地切分（不依赖 LLM 决定边界）。
- 把 parser 输出的"碎块"**向上组装**成自包含、带上下文的 chunk。
- 每个 chunk **可溯源**（能追回原始元素），便于审核与回归。
- **分文档类型**差异化处理，而非一刀切。

**非目标**
- 不做 parsing（交给 MinerU）。不做 embedding / 检索（chunk 是输入，下游另议）。
- 不追求"最优"chunk 大小——大小是可调旋钮，随检索任务而定。

---

## 2. 总体架构与数据流

```
PDF 语料 ──select_sample.py──> corpus/<type>/        (分层抽样 + 归类)
        ──parse_batch.py────> parsed/<doc_id>/       (MinerU VLM, 3账号负载均衡)
                               ├─ *_content_list.json   语义单元(表格HTML/caption已组装)
                               └─ layout.json           pdf_info[]: 噪声/merge_prev/score/index
        ──analyze_chunks.py─> analysis/               (统计 → 策略依据)
        ──chunk_document.py─> chunks/<doc_id>.jsonl   (leaf chunks)
                               chunks/<doc_id>.parents.jsonl  (section 聚合, small-to-big)
```

**两个输入文件的分工**（关键设计前提，实测得出）：
- `content_list.json` = **已组装的语义单元**：表格的 `table_body`(HTML)+`table_caption`+`table_footnote` 捆在一个对象；图/图表同理。数组顺序≈阅读顺序。
- `layout.json.pdf_info[page]` = **关系型信号**：`discarded_blocks`(噪声)、`para_blocks[].merge_prev`(跨页续接)、`lines[].spans[].score`(OCR置信度)、`index`(阅读序)。

> 在线 VLM API **不产出本地版的 `block_list.json`**（无 `is_discarded`/`mergeConnections`）；等价信息在 `layout.json`。详见 PROCESS_LOG §解析。

---

## 3. Chunk 数据模型（schema）

leaf chunk（`chunks/<doc_id>.jsonl`，每行一个）：

| 字段 | 含义 | 设计理由 |
|---|---|---|
| `chunk_id` | `<doc_id>#0007` | 稳定唯一，便于引用/去重 |
| `doc_id` / `doc_type` / `language` | 来源与类型 | 检索过滤 + 分型策略 |
| `kind` | `text\|table\|chart\|image` | 下游对资产/文本分别处理 |
| `text` | **检索向量的输入文本** | 文本=组装正文；资产=caption+footnote(+可选摘要) |
| `content_raw` | 资产生成负载（表格 HTML / VLM content） | **检索与生成分离**：embed 摘要，喂原文 |
| `breadcrumb` / `section_path` | 标题面包屑 | 提供上下文 + 可引用，零 LLM 成本 |
| `parent_id` | 所属 section parent | small-to-big：命中 leaf→回取 parent |
| `page_start`/`page_end`/`bbox` | 定位 | 引用、re-rank、人工核对 |
| `n_tokens` | 估算 token | 预算控制与监控 |
| `trust` | `high\|low` | VLM 重建内容/低 OCR 标 low |
| `flags` | `captionless`/`vlm_content`/`cross_page_merged` | 暴露风险，供下游决策 |
| `source_indices` | 原 content_list 下标列表 | **可审核**：每个 chunk 追回原始元素 |

parent chunk（`*.parents.jsonl`）：`parent_id, section_path, child_ids[], text(聚合), pages[], n_tokens`。

---

## 4. 七步流水线（每步的设计与依据）

| 步 | 做什么 | 信号来源 | 依据（数据） |
|---|---|---|---|
| 1 噪声过滤 | 丢 `discarded_blocks` + type∈{header,footer,page_number} | layout + content_list | 噪声率 0→0.37，government/financial_zh 最重 |
| 2 跨页缝合 | `merge_prev=true` 的块拼回前块（**文本前缀匹配**对齐到 content_list；bbox-IoU 因坐标系不同会全失效，见 §7） | layout | academic 4.2 / form 13.8 / research 7.8 个/篇 |
| 3 分领域建树 | text_level 为主信号，数字编号定深度，law 用 `SEC.` 正则；剔 TOC、剔超长"伪标题" | content_list | text_level 双向不可信；law num% 0.02 |
| 4 资产特判 | 表/图/图表各成原子 chunk；caption→检索，body→生成；标 captionless/vlm_content | content_list | 表 HTML 0.86-1.0 可靠；图 caption 0.04-0.88 波动 |
| 5 文本组装 | 同 section 内按 token 预算贪心累积（**可调策略**，§5） | — | 11/14 类中位<75 token，必须向上合并 |
| 6 挂元数据 | breadcrumb/页/ bbox/token/trust/flags/source_indices | 全部 | 检索质量大头来自元数据 |
| 7 parent-child | leaf→section parent，聚合 parent 文本 | — | 有层级类型的最高 ROI 升级 |

---

## 5. 核心可调策略：`assemble_text()`（Step 5）

这是整个系统**唯一真正有取舍**的地方，被刻意隔离成一个纯函数，签名 `assemble_text(blocks, lang, cfg)`，`cfg=(min, target, max)`。

**当前策略（默认实现）**：
- 贪心累积同 section 的连续文本块到 `target`；
- `merge_prev=true` 的块**无条件**并入当前组（跨页续接优先于预算）；
- 单块超 `max` → 在句子边界切分；
- 结尾不足 `min` 的组**向前并入同 section 的上一组**（不跨 section 合并）。

**取舍（为什么是这条，代价是什么）**：
- 偏小 `target` → 检索精准但易丢上下文（靠 parent-child 兜底）。
- 偏大 `target` → 上下文足但稀释 embedding、挤占生成窗口。
- 不跨 section 合并 → 尊重结构、可解释；代价是**同名/单段 section 会留下小 chunk**（financial_zh 实测中位 124 token）——这是有意的，parent 层补偿。

> **这就是留给你调的旋钮**：改 `CONFIG[doc_type]` 的三元组，或改 `assemble_text` 的合并逻辑（如允许跨相邻 section 合并、改用语义相似度切分），即可改变检索颗粒度，其余六步不受影响。

**分类型预算 `CONFIG`（min/target/max，token）**：见 `scripts/chunk_document.py` 顶部；幻灯片/政策设为"整节不切"（一页/一节=一 chunk），其余依 [CHUNKING_STRATEGY.md §4](../analysis/CHUNKING_STRATEGY.md) 的 playbook。

---

## 6. 关键设计决策（决策记录）

1. **以 content_list 为脊、layout 为信号补充**，而非二选一。理由：content_list 已组装表格/caption（自己重建易错），layout 独有 merge/discarded/score。代价：需 bbox-IoU 对齐两者（已实现，阈值 0.5）。
2. **切割确定性、不调用 LLM**。理由：结构清洗后边界清晰；LLM 切割贵、不可复现、与已有结构冗余。LLM 仅保留给"资产摘要/补 caption"（接口预留，默认关）。
3. **资产原子化 + 检索/生成分离**（multi-vector）。理由：表格 HTML embedding 差，但生成需要原文。
4. **分领域层级规则**。理由：实测每个领域编号体系不同（academic `1.2` / law `SEC.`+`(a)(1)` / 财报不编号）。
5. **source_indices 全程保留**。理由：可审核是硬需求——任何 chunk 都能追回原始元素做回归比对。
6. **3 账号负载均衡按"页数"贪心分配**，而非按文件数。理由：配额是页/天，页数才是真实成本。

---

## 7. 已知局限（诚实记录，便于审核与改进）

- **parent 分组按 breadcrumb 文本**：同名 section（正文与附录都叫"1 本周行情回顾"）会被并入同一 parent，导致 parent 的 `pages` 不连续。改进：parent_id 纳入首次出现页或 section 序号。
- **merge_prev 对齐**：原 bbox-IoU 方案因 content_list(渲染坐标) 与 layout(PDF 点) 坐标系不同而**全程 0 命中**（对抗性审查 F1）；已改为**文本前缀匹配**（修复后 17 篇 112 chunk 命中）。纯无文本块仍可能漏缝。
- **超长块切片溯源退化**：单块超 `max` 经 `_sentence_split` 切成多片后，各片 `source_indices` 复用原块同一 idx，只能定位到原块、无法区分片（审查 F5）。需片级溯源时应附字符区间。
- **法律深层条款**只靠"块向上合并"保在一起，未显式建 `(a)(1)(A)` 树；若需按条款精确检索，需二级解析行内枚举符。
- **token 为字符估算**（en /4、zh /1.7），非真实 tokenizer。监控够用，计费场景需接真实 tokenizer。
- **VLM 的 chart/image content 数值不可信**（已标 `vlm_content`），但未自动校验；下游不应将其当事实索引。
- **news 类是拼接多文档**：当前按普通文档处理（1399 chunk），理想应先按文档边界分割。

---

## 8. 验证（against ground truth）

不只靠抽样自检——用 MMDocIR 标注问题（43 篇重叠 / 356 问）量化"证据保全度"，详见 [EVALUATION.md](EVALUATION.md)。
**坐标修复后**：证据保在**单 chunk 70.1%**（严阈值下相同→不依赖阈值）、split 22.5%、missing 7.4%、**asset 通道-kind 匹配 92.4%(any) / 83.2%(真资产严口径)**；答案串软召回 16.3%（仅参考，短词/改写假阴性高）。与策略难度排序自洽（academic/financial_report/government 较好）。
该评估是**调旋钮的客观标尺**：改 `CONFIG`/`assemble_text` 后看 `single%↑ / split%↓ / missing%≈0`。
**split 的正确解读**：research/brochure/slides 的 split 高，是"图+讨论文字"被资产原子化分到不同 chunk——属预期，parent-child 检索会重聚；它衡量"叶子层需回取 parent"，非"证据丢失"。
**覆盖局限**：ground-truth 仅覆盖 43/77 篇，financial_research_zh/policy/form/tech_report 未经此验证；law single=100% 仅来自单篇。

## 9. 扩展点

- 新文档类型：在 `CONFIG` 加预算 + 在 `heading_level()` 加领域正则即可（见 IMPLEMENTATION §扩展）。
- 启用 LLM 增强：在 `_mk_asset_chunk` 接入"表格一句话摘要 / 无 caption 图补述"，写入 `text` 并加 `trust=low` 标签。
- 接真实 tokenizer：替换 `est_tokens()`。
