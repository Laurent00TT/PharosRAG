# 实施文档 — 运行、复现、扩展、审核

> *归档件：本文由引擎原型仓原样迁入，保留原文不改。所述脚本与路径属于原仓布局，
> 当前版本见 [docs/components/chunker/](../components/chunker/)。*

> 面向"拿来就能跑、能改、能查"的操作手册。设计动机见 [DESIGN.md](engine-prototype-DESIGN.md)。

---

## 1. 环境

- Windows + Python 3.12（实测）；`pip install -r requirements.txt`（requests / pypdf / python-dotenv）。
- MinerU 在线 API token：复制 `.env.example` 为 `.env`，填 `MINERU_TOKEN_A/B/C`（多账号负载均衡）。`.env` 已被 `.gitignore` 排除，**勿提交**。

---

## 2. 一键复现（端到端）

```bash
python scripts/select_sample.py     # ① 抽样 → corpus/ + sample_manifest.csv
python scripts/parse_batch.py       # ② MinerU 解析 → parsed/        (幂等可断点续跑)
python scripts/analyze_chunks.py    # ③ 统计 → analysis/per_doc_stats.csv + by_type.json
python scripts/chunk_document.py    # ④ 切块 → chunks/*.jsonl + _summary.csv
python scripts/eval_chunks.py       # ⑤ ground-truth 评估 → analysis/eval_report.json + eval_by_doctype.csv
```

冒烟（先验证单文档管道+API schema，再全量）：`python scripts/smoke_test.py`。

**幂等性**：`parse_batch.py` 跳过 `parsed/<doc_id>/` 已有 content_list 的文档；其余脚本覆盖重写。`select_sample.py` 无随机性，同输入同输出（`config/mmdocir_pagecount.json` 缓存页数）。

---

## 3. 模块逐一说明（scripts/）

| 文件 | 职责 | 关键函数/数据 | 输出 |
|---|---|---|---|
| `select_sample.py` | 读 3 个 manifest，归一化 14 类，分层抽样（按页数桶均匀展开），按页数 3-key 贪心均衡，复制到 `corpus/<type>/` | `TARGETS`(每类篇数)、`spread_pick`、`assign_keys` | `sample_manifest.csv` |
| `mineru_client.py` | MinerU v4 客户端 | `create_batch`/`upload`(PUT不带Content-Type)/`poll_batch`/`download_and_extract` | — |
| `parse_batch.py` | 按 (key,语言) 分批提交、并发上传、轮询、并发下载解压 | `POLL_SECS`/`MAX_POLL_MIN` | `parsed/`、`parse_results.csv`、`batches.json` |
| `analyze_chunks.py` | 跨文档统计（噪声/层级/表图/merge/OCR置信度），按类型聚合 | `analyze_doc`、`NOISE_TYPES`/`NUM_RE`/`TOC_RE` | `analysis/per_doc_stats.csv`、`by_type.json` |
| `chunk_document.py` | 7 步流水线切块 | `CONFIG`、`heading_level`、`assemble_text`(旋钮)、`chunk_doc` | `chunks/*.jsonl`、`*.parents.jsonl`、`_summary.csv` |
| `eval_chunks.py` | 用 MMDocIR 标注检验证据保全度（源桥 `source_indices`） | `region_elements`(双向覆盖)、`parse_listish`、verdict 分类 | `analysis/eval_report.json`、`eval_by_doctype.csv` |

---

## 4. 可调旋钮（最常改的地方）

1. **chunk 颗粒度** → `chunk_document.py` 顶部 `CONFIG[doc_type] = (min, target, max)`。
   - 想要更细（偏事实检索）：调小 `target`/`max`。想要更粗（偏综合）：调大。
2. **组装逻辑** → `assemble_text()`。当前不跨 section 合并；若要允许跨相邻同级 section 合并、或改语义切分，只改这一个纯函数，其余六步不变。
3. **抽样规模/构成** → `select_sample.py` 的 `TARGETS`、`PAGE_CAP`、`KEYS`。
4. **解析参数** → `mineru_client.create_batch` 的 `model_version`(默认 vlm)、`enable_formula/table`、`language`。
5. **轮询节奏** → `parse_batch.py` 的 `POLL_SECS`(15s)、`MAX_POLL_MIN`(45)。

---

## 5. 如何扩展一个新文档类型

1. `select_sample.py`：在 `MMDOCIR_DOMAIN_MAP` 或 `PDFCORPUS_TYPE_MAP` 把原始 doc_type 映射到你的归一化名，并在 `TARGETS` 给配额。
2. `chunk_document.py`：
   - `CONFIG` 加一行预算三元组；
   - 若其层级编号特殊（如 law），在 `heading_level()` 加领域正则；
   - 若按页成块（如 slides），把类型加入 `PAGE_GROUPED`。
3. 重跑 ④，看 `chunks/_summary.csv` 的 `ch/doc`、`median_text_tok`、`captionless` 是否合理。

---

## 6. 如何审核产物（可审核性）

**A. 追溯单个 chunk 回原文**
每个 chunk 带 `source_indices`（原 `content_list` 下标）。比对：
```bash
python -c "import json; d=[json.loads(l) for l in open('chunks/law__PLAW-118publ38.jsonl',encoding='utf-8')]; \
c=d[1]; print(c['source_indices']); print(c['text'][:400])"
```
拿 `source_indices` 去 `parsed/<doc_id>/*content_list.json` 对应下标核对内容是否一致、是否漏切/错并。

**B. 看汇总指标找异常**
`chunks/_summary.csv` 逐文档列 `leaf_chunks / text / table / image / captionless / median_text_tok`。排查：
- `ch/doc` 异常大（如 news 1399）→ 可能是拼接多文档；
- `median_text_tok` < `CONFIG.min` 大面积出现 → 组装/合并阈值需调；
- `captionless` 高 → 该类资产检索盲区大，考虑开启 VLM 补 caption。

**C. 抽样人工核对**（已在开发中跑过，命令可复用）
- 法律：`section_path` 是否为 `SEC. N`，条款 `(a)(1)(A)` 是否整体在一个 chunk；
- 表格：`text` 是否=caption+footnote、`content_raw` 是否=HTML；
- parent：`parents.jsonl` 的 `child_ids` 是否与 leaf 的 `parent_id` 双向一致。

**D. 回归**
重跑 ④ 后 diff `chunks/_summary.csv`；指标无关变动应为 0（确定性）。改 `CONFIG`/`assemble_text` 后只该看到目标类型的 chunk 数/token 变化。

---

## 7. 配额与限制（运维须知）

- 单文件 ≤200MB / ≤200 页；单批 ≤50 文件；**1000 页/天/账号**。
- 上传 URL 24h 有效；上传完成自动解析，无需另调提交接口。
- 触发限流/失败：`parse_results.csv` 记 `state`(done/failed/timeout) 与 `err`；失败文档重跑 `parse_batch.py` 会自动续传（幂等）。
- 首轮成本：77 文档 / 1867 有效页，3 账号各 ~620 页，单日内完成。
