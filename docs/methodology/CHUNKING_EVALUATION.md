# 评估文档 — 用 ground-truth 检验 chunking 质量

> 不自说自话：拿数据集**自带的标注问题**检验"chunking 是否把证据保在可检索的单元里"。
> 脚本 `scripts/eval_chunks.py`；结果 `analysis/eval_report.json` + `analysis/eval_by_doctype.csv`。
> **本版数字为坐标 bug 修复后的结果**（旧版被坐标错配污染，见 §2 与 [PROCESS_LOG 阶段9](../archive/PROCESS_LOG.md)）。

---

## 1. 为什么这样评

chunking 真正控制的是：**问题的证据，切完之后还在不在一个连贯的 chunk 里**（没被切碎、没被当噪声丢掉）。
所以我们不评 embedding/检索（那是下游），只评**证据保全度**——这是 chunking 能负责、且可被标注直接检验的部分。

**Ground truth**：`mmdocir/MMDocIR_annotations.jsonl`。每问题带 `answer`、`page_id`、`type`(通道：Figure/Table/Chart/Pure-text)、`layout_mapping[]`(证据的 page+bbox+page_size)。
与样本**重叠 = 43 篇 / 356 问**（仅 mmdocir 有标注；financial_research_zh/policy/form/tech_report 等**未被 ground-truth 覆盖**，见 §6）。

---

## 2. 方法（坐标处理是正确性关键）

**桥梁 = `source_indices`**：证据 bbox → 命中 content_list 元素下标 → 含该下标的 chunk。这正是 chunker 为可审核保留的字段。

**坐标系（修复后）**：实测 `content_list` 的 bbox 在**渲染图坐标系**，`layout.json`/GT 在 **PDF 点坐标系**，二者关系为 `content_list = layout × (sx, sy)`（同原点、同文档内常数，但 sx≠sy 且跨文档不同）。
- 旧版 bug：直接拿 GT(点) 与 content_list(渲染) 比，或用 layout 的 page_size 归一 content_list bbox → 系统性错配，把大量"其实能定位"的题误判为 unlocalized。
- 修复：逐文档**文本配对**（content_list 元素 ↔ layout para_block 同文本）推导中位 `(sx, sy)`，把 GT bbox 缩放进 content_list 空间后再算覆盖。**效果：unlocalized 从 72 → 4**。

**证据元素判据(双向覆盖)**：元素计入证据，当 `交集/证据面积 ≥ 0.3` **或** `交集/元素面积 ≥ 0.5`。
**`type` 是字符串** `"['Figure']"`，需 `ast.literal_eval` 解析。

---

## 3. 判定与指标

verdict 把"测量做不到"和"chunking 没做好"分开：

| verdict | 含义 | 计入质量分母? |
|---|---|---|
| `out_of_range` | 证据页超出 `PAGE_CAP=50` | 否（测量限制） |
| `unlocalized_zero` | 在范围内但**零元素重叠**（解析层无内容/版面差） | 否（更接近解析缺失） |
| `unlocalized_threshold` | 有重叠但低于阈值 | 否（阈值边界，修复后为 0） |
| `missing` | 定位到元素但**没被任何 chunk 保留**（当噪声/标题丢了） | 是 |
| `single` | 证据元素全落在**一个** chunk | 是（好） |
| `split` | 证据元素跨**多个** chunk | 是（碎片风险） |

另：**asset 通道-kind 匹配**分两口径——`any`(命中集合含期望 kind 即算，混合通道可被文本救回) 与 **`TRUE-asset`(真资产题必须命中 table/chart/image chunk)**，后者更严更诚实。

---

## 4. 结果（坐标修复后；77 篇中 43 篇有 ground truth）

整体（localized=311 / 356；excluded：out_of_range 41、unlocalized 4）：

| 指标 | 值 |
|---|--:|
| 证据保在**单个** chunk | **70.1%** |
| ↑ 严阈值(0.5/0.7)下 | **70.1%**（**与默认相同 → 不依赖阈值**，旧版的阈值敏感是坐标 bug 症状） |
| 证据 **split** 跨 chunk | 22.5% |
| 证据 **missing** 被丢 | 7.4% |
| **asset 通道-kind 匹配 (any)** (n=118) | 92.4% |
| **asset 通道-kind 匹配 (TRUE-asset 严口径)** (n=95) | **83.2%** |
| 答案串软召回 (n=288，仅参考) | 16.3% |

分文档类型（**distinct_docs 列防止单文档冒充类型趋势**）：

| doc_type | 文档数 | n问 | single% | split% | missing% | 备注 |
|---|--:|--:|--:|--:|--:|---|
| academic_paper | 7 | 30 | 86.7 | 6.7 | 6.7 | |
| financial_report_en | 7 | 37 | 73.0 | 27.0 | 0 | |
| government | 5 | 13 | 69.2 | 7.7 | 23.1 | |
| slides_tutorial | 4 | 23 | 60.9 | 39.1 | 0 | |
| brochure | 4 | 21 | 42.9 | 52.4 | 4.8 | 图主导、split 高 |
| guidebook | 4 | 21 | 42.9 | 57.1 | 0 | 图多、split 高 |
| research_report | 4 | 21 | 33.3 | 66.7 | 0 | split 最高 |
| admin_industry | 2 | 6 | 50.0 | 33.3 | 16.7 | n≤2，仅参考 |
| news | 1 | 136 | 81.6 | 6.6 | 11.8 | **单文档拼接件，不可外推** |
| law | 1 | 3 | 100.0 | 0 | 0 | **单文档，不可外推；且未覆盖 F2 的 SEC 长正文场景** |

---

## 5. 怎么读这些数字（含 nuance）

- **结果与策略难度排序自洽**：academic 87%、financial_report 73%、government 69% 单 chunk 较好；图主导的 brochure/guidebook/research_report split 高。
- **split 高 ≠ chunking 坏**：research/brochure/slides 的证据常 = **图 + 讨论文字**，被**资产原子化**有意分到 image chunk 与 text chunk；**parent-child 检索会在同一 section parent 下重聚** → 召回不丢。split 衡量"叶子层是否需回取 parent"，非"证据丢失"。
- **asset 匹配 any 92.4% vs TRUE 83.2%**：约 9pt 来自混合通道题被文本 chunk 救回；**以 83.2% 为准**判断资产原子化效果。
- **单文档行(news/law)不可外推**：news 一篇拼接件贡献 136 问、law 仅 3 问，已在表中标注。

---

## 6. 局限（诚实记录）

- **ground-truth 覆盖偏**：仅 43/77 篇（mmdocir 子集）有标注；**占比最大的 financial_research_zh(12篇) 及 policy/form/tech_report 完全未被证据保全评估**——它们的策略只经统计画像，未经检索验证。
- **PAGE_CAP=50** → 41 题 out_of_range；长文档被头部截断，**"跨页/深层级"结论在被截断样本上不可下定论**（见 [PROCESS_LOG 阶段9 / 抽样局限](../archive/PROCESS_LOG.md)）。
- **答案串召回 16.3% 不可信**：答案多为短词/数字/改写，子串匹配假阴性高 → **仅参考，不作为 chunking 指标**；严谨需 embedding/LLM 语义判定。
- **law single=100% 来自单篇 mmdocir 文档**，未覆盖 pdf_corpus 法案里 F2(SEC 长正文)场景；law 的真实保全度需补该类 ground truth。
- **scale 推导依赖文本配对**：无文本/纯扫描件无法推导 → 那些题计 unlocalized_zero。

---

## 7. 复现

```bash
# 自研：组件定稿版（source of truth）用组件包重产 chunks/
python scripts/gen_chunks_component.py
python scripts/eval_chunks.py        # -> analysis/eval_report.json + eval_by_doctype.csv
# 注：scripts/chunk_document.py 是原型版，已与组件分叉、未跟进 R1–R3（见 INTEGRATION §4），仅历史参照
```
确定性。比较 `single%↑ / split%↓ / missing%≈0`——**这就是调旋钮的客观标尺**。脚本同时报严阈值 single% 做敏感性自检。

---

## 8. 三方对比：自研 vs chonkie vs docling（组件定稿版，2026-06）

**问的问题**：切块这层是否也该用成熟方案（解析层已确认用 MinerU、不自研）？拿**同一份证据保全 eval、同一批 43 篇 ground-truth 文档**，把自研组件、chonkie `RecursiveChunker(recipe=markdown)`、Docling `HybridChunker` 放一起比。脚本 `scripts/compare_chunkers.py` → `analysis/compare_report.json`。

**桥的两口径（关键，分离归因）**：
- **`ours_exact`**：用自研 `source_indices` **精确**桥（chunk 确知来自哪些 element）——这是自研在**真实 RAG** 里的表现。
- **`ours_fair` / `chonkie` / `docling`**：都用**同一套 text-substring 桥**——剥掉 source_indices，**只比切块边界本身**划得好不好。

**TEXT-channel（最公平，纯文本证据，localized=215）**：

| 策略 | single%↑ | split% | missing%↓ |
|---|--:|--:|--:|
| **ours_exact** | **79.5** | 10.2 | **10.2** |
| chonkie | 74.9 | 11.2 | 14.0 |
| ours_fair | 71.6 | 10.7 | 17.7 |
| docling | 56.3 | 10.2 | 33.5 |

**ALL evidence（含表/图/图表，localized=311）**：

| 策略 | single%↑ | split% | missing%↓ |
|---|--:|--:|--:|
| **ours_exact** | **70.4** | 21.9 | **7.7** |
| chonkie | 62.1 | 16.1 | 21.9 |
| ours_fair | 58.2 | 15.8 | 26.0 |
| docling | 47.6 | 13.8 | 38.6 |

**结论（双向、有立场）**：
1. **Docling HybridChunker 不该用（接 MinerU 输出）**：missing 全场最高（text 33.5% / all 38.6%，是 ours_exact 的 3–5 倍）。它对 MinerU 导出的 markdown 做 tokenizer-aware + 语义合并，大量证据被丢/无法定位。
2. **自研护城河是工程特性，不是边界算法更聪明**：`ours_exact` 全场第一靠 source_indices 精确溯源；但剥掉它、只比边界的 `ours_fair` **略输 chonkie**（71.6% vs 74.9%）。诚实讲——单看"切块边界划得聪不聪明"，chonkie 的递归切分略胜自研。自研净胜来自 source_indices + heading-tree breadcrumb + per-chunk ACL + small-to-big 这套**工程集成**，chonkie/docling 都没有。
3. **R1–R3 修复在这条指标上"无感"**：组件版 vs 原型版是两套不同切法（已验证块数/边界不同，`identical=False`），但证据保全几乎完全收敛（text_only 持平、ALL 微动 +0.6% single）。说明 reset-aware 定级 / 横幅守卫 / aside_text 剔除改善的是 **breadcrumb 准确性 + 检索语义纯净度**，不是证据聚合——这条指标测不出它们，要测得换 breadcrumb 正确性 / 检索相关性指标。

**最终判断**：自研**值得保留**，但护城河重新定位为**工程集成而非边界算法**。**若哪天只要纯文本切块、不要这套工程特性，chonkie 是更省事的选择**（边界质量相当甚至略好，零维护）。原型 baseline 存档 `analysis/compare_report_proto.json`。
