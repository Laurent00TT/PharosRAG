# Pharos 测试文档

> 三层:CPU 单测(每改必跑)/ 引擎回归(接缝改动时跑)/ GPU 冒烟(投产前跑)。
> 本文所有"实测"均为 2026-07-02 在 WSL `navikb`(4090)真实运行结果。

## 1. CPU 单测(38 项,~2s,不碰 Qdrant/GPU/网络)

```bash
conda activate navikb && cd pharos && python -m pytest tests -q
```

| 文件 | 覆盖 |
|---|---|
| test_sessions.py | 同会话同 set / 跨会话隔离 / LRU 逐出有界 / touch 刷新 |
| test_service.py | healthz;六端点 wiring;**no_identity fail-closed**;bad_arg 走结构化不走 422;API key 门槛(healthz 豁免/错 key 拒);**per-session 去重隔离 + 无头不去重**;ask(引用映射/include_contexts/空 query/llm_unconfigured/ask_failed 不泄内部细节) |
| test_adapter.py | 六工具转发参数映射;backend-down→结构化 backend_unavailable(hint 指向 pharos serve);401/5xx/非 JSON 映射;会话头存在;instructions 与引擎同源 |
| test_review_fixes.py | 对抗评审 P1 修复回归:no_identity hint 指 PHAROS_TENANT;indexer 拒 restricted+空 allow;工厂异常降级 ask_failed;doc_id URL 编码/空参本地拒/3xx 结构化;.env 行内注释/引号/int 指名报错/覆盖路径 expanduser;**适配器与引擎六工具 docstring 逐字同文**(直接 exec 引擎 server.py 断言) |

工具语义本体(already_returned/omitted_budget/预算含资产/无权不泄存在性…)**不在 Pharos 重测**——
单一来源在引擎 toolcore,由引擎 test_tools.py(22 项)覆盖。

**实测**:`38 passed, 1 warning in 1.88s`(N3 检索过滤落地后 +2:ask 过滤透传 / bad strategy 结构化)。

## 2. 引擎回归(toolcore 拆分后)

```bash
cd chunk-test-repo
python mcp_server/tests/test_tools.py          # 22 项,一行未改 -> 拆分兼容的证据
python -m pytest embedder/tests/test_store.py -q
```

**实测**:`mcp_server tool tests OK` + `7 passed`。

## 3. GPU 冒烟(真库 ~/rag_real,77 篇 / 7652 chunk)

步骤与实测结果(全过):

| # | 步骤 | 实测 |
|---|---|---|
| 1 | `python -m pharos serve`(后台) | 秒级启动,日志确认独占打开索引 collection=real tenant=demo |
| 2 | GET /healthz | `{"status":"ok",…,"tenant_bound":true}` |
| 3 | GET /v1/documents | 77 篇,coverage 14 个 doc_type |
| 4 | POST /v1/retrieve "Netflix 2015 revenue"(首查) | **19.1s**(含 dense 模型加载),top1 命中 NETFLIX_2015_10K,status=ok |
| 5 | POST /v1/ask "What was Netflix total revenue in 2015?" | status=ok,grounded 回答 + 2 条引用(chunk_id/页码正确),finish_reason=stop;top_k=5 未捞到总营收数字时模型如实说"信息不足"(grounding 正常,不编造) |
| 6 | 同会话(X-Pharos-Session: smoke1)重复 retrieve | 3 条全 `already_returned`,already_n=3 |
| 7 | 新会话 smoke2 同 query | `section_window×2 + deduped`,already_n=0(**隔离确认**) |
| 8 | MCP 适配器(live daemon):list/retrieve/outline | 77 docs / retrieve ok / outline 88 sections |
| 9 | CLI:`python -m pharos ask "库里有哪些关于 DDoS 攻击防护的内容?"` | 中文 grounded 回答 + 来源(标题/页码/小节/chunk_id) |

复现命令见 git 历史与 [IMPLEMENTATION.md](IMPLEMENTATION.md) §6。

**N3(ask 检索过滤)落地后的追加实证**(2026-07-03,同真库;起因:用户实测"Netflix 2015 营收"闭管道拒答):

| 问法 | 结果 |
|---|---|
| 中文,默认参数(用户原试) | 诚实拒答(散文压表格,数字在 p.16 表内、库内验证存在) |
| 中文/英文 + top_k 12 + rerank(无 kind) | 仍拒答 |
| 中文 + `--kind table` + top_k 15(无 rerank) | ⚠ **错答**:分部营收 4,180,339 被当总营收(留档,见 TODO P2) |
| **英文 + `--kind table --rerank`** | ✅ **$6,779,511 千美元,引用 p.16 Selected Financial Data** |
| 中文 + `--kind table --rerank` | 诚实拒答(明说只见分部数据)——跨语言表格排名残留缺口 |

结论:数字/表格题推荐 `--kind table --rerank` + 文档语言关键词;跨语言增强列 TODO P2。

**数值范围错答修复(2026-07-03,诊断→修复→验证全程)**:

- **根因链**(两个都缺才错答):① prompt 无数值范围约束;② **范围证据不在 prompt 里**——分部信息只在
  section_path 元数据,表格块正文无一字"Domestic Streaming"(实测:只加约束 B 依旧错答,证据补进才生效)。
- **修复**(引擎 generator):SYSTEM 加窄靶数值范围约束(区别于 R3 被 revert 的全称句级收紧)+
  context 的 source 行并入小节面包屑(`标题 § FORM 10-K > Domestic Streaming Segment`)。
- **验证三关**:① B 复现:不再错答,明确标注"仅为部分业务数据"并拒引申;② C 复跑:总营收仍答对
  ($6,779,511,p.16),无误伤;③ **同 DeepSeek 裁判前后对比**(72 题,检索/索引/裁判全同):
  忠实度 0.972→**1.000**(+0.028),正确性 0.847→**0.847**(±0)。零回归。
  (基线判分产物:eval/baseline_single_prescope*.json,gitignored)

**表格块检索文本增强(2026-07-03,诊断→实现→重建→回归→裁决,引擎 `2bd97a5`)**:

- **根因**(源码级实锤):表格块可检索信号 = caption+footnote 一句话(body 只进 content_raw,
  不参与 embed/sparse;面包屑也不拼)——英文查询同样吃亏,跨语言只是放大器。
- **实现**:`_table_signal`(表头前 2 行 + 每行首个非空单元格,数据单元格不进)+ 面包屑拼入;
  **存在性门控**(cap|foot|body):无门控实测全库 7652→7675(+23 幽灵块,会平移 chunk id、
  作废 gold/旧引用),门控后精确回到 **7652**(id 稳定性的硬证据)。chunker 测试 43→44。
- **重建**:~/rag_real(77 篇/7652)与 ~/rag_eval_big(15 篇/1409)全量重建。
- **回归**(72 题,同裁判):正确性 **0.847 持平**;引用召回 +0.007;忠实度 0.986(−1 题,
  图表极值推断的边界判罚,已知难类);检索召回 0.854→0.833(−2.1pp)。
- **下降面逐题诊断**:全部来自 3 题双 gold 的 multi_intra 各被挤掉一个**冗余** gold
  (3×0.5/72=2.1pp 全对上),**3 题在新索引下全部仍答对**;0 题上升——因 gold 采样自散文块,
  表格受益题型在 gold 里近乎为零(测量盲区,已列 TODO)。
- **动机案例(决定性)**:中文原问法 + `--kind table`(免 rerank)从错答/拒答变为
  **直接答对 $6,779,511 千美元**(引用 p.18 合并业绩表)。
- **裁决:保留**。按破坏度:解锁的题类此前是错答/拒答(高破坏),位移的是答案无影响的
  冗余召回(零破坏),正确性持平。

## 4. 对抗评审(P1,已完成)

三视角(安全/正确性·并发/契约·文档)× 每发现 2 独立反驳者,39 agent。结果:
**2 条 confirmed + 13 条自核实属实(验证 agent 撞额度,主线逐条对照源码核实)→ 全部修复并钉回归测试**
(test_review_fixes.py,11 项);3 条被证伪留档。完整清单与处置见
[COMPONENT_NOTES.md §对抗评审 P1](COMPONENT_NOTES.md)。修复后全量回归:Pharos 36 + 引擎 22+7 全绿。

## 5. 未覆盖(诚实清单)

- 并发压测(多客户端同时打 /v1/ask + retrieve):锁模型保守(全串行),个人场景未压测;
- `pharos index` 全量真跑(~/rag_real 已由引擎 index_real.py 建成,indexer.py 是其参数化版,
  仅逻辑评审 + 锁冲突提示路径验证;下次换语料建库时顺带实测);
- MCP 适配器在 Claude Code 真会话里的端到端(需要你在 app 里连一次;工具逻辑已由
  adapter×live daemon 冒烟覆盖)。
