# Pharos 测试文档

> 两道门:CPU 单测(每改必跑,一套 pytest 全绿=179 项)/ GPU 冒烟 + ACL 回归(投产前跑)。
> 数字均为 WSL `pharos`(4090)真实运行结果,非估算。

## 1. CPU 单测(单仓一套 pytest = 179 项 = 产品 59 + 引擎 120,~数秒,不碰 Qdrant/GPU/网络)

```bash
conda activate pharos && cd pharos && python -m pytest -q
```

产品面 59 项在 `tests/`;引擎面 120 项在 `tests/engine/`(折叠进本仓后同一套 pytest 一并跑,
含 embedder `test_acl.py` 的 ACL 谓词单测)。CPU CI 门槛 = 这一套 pytest 全绿。

| 文件 | 覆盖 |
|---|---|
| test_sessions.py | 同会话同 set / 跨会话隔离 / LRU 逐出有界 / touch 刷新 |
| test_service.py | healthz;六端点 wiring;**no_identity fail-closed**;bad_arg 走结构化不走 422;API key 门槛(healthz 豁免/错 key 拒);**per-session 去重隔离 + 无头不去重**;ask(引用映射/include_contexts/空 query/llm_unconfigured/ask_failed 不泄内部细节) |
| test_adapter.py | 六工具转发参数映射;backend-down→结构化 backend_unavailable(hint 指向 pharos serve);401/5xx/非 JSON 映射;会话头存在;instructions 与引擎同源 |
| test_review_fixes.py | 对抗评审 P1 修复回归:no_identity hint 指 PHAROS_TENANT;indexer 拒 restricted+空 allow;工厂异常降级 ask_failed;doc_id URL 编码/空参本地拒/3xx 结构化;.env 行内注释/引号/int 指名报错/覆盖路径 expanduser;**适配器与 `mcp_stdio` 六工具 docstring 逐字同文**(仓内结构性断言:两侧 docstring 相等 + `_INSTRUCTIONS` 单一来源自 `toolcore`) |
| test_smart.py | smart-ask(D9):数值题拒答才触发表格腿重试(择优采用)/ 非数值不触发 / 显式 kind 尊重 / 开关 / 拒答 hints |
| test_team.py | 多身份(D10):keys 解析 fail-closed / 401 / 身份逐请求流到引擎 / 跨用户会话隔离 / stats admin 门控 / 非回环启动守卫 / name 唯一+禁`\|` / keys new 不裸抛 / 观测崩溃安全 / 结构化失败计 errors / 日志不落 key+截断可关 |

工具语义本体(already_returned/omitted_budget/预算含资产/无权不泄存在性…)的单一来源是
`src/pharos/toolcore.py`,由 `tests/engine/` 下的工具语义测试覆盖(与产品面同一套 pytest 一并跑)。

**实测**:产品面 `59 passed`(六个测试文件:sessions/service/adapter/review_fixes/smart/team);
连同 `tests/engine/` 的 120 项,单仓一套 pytest 合计 **179 passed**。

## 2. 引擎面测试(已折叠进本仓 `tests/engine/`)

引擎折叠进本仓后不再是独立仓、也无独立门:引擎面 120 项与产品面 59 项同一条命令
(`python -m pytest -q`)一并跑。契约不漂测试(适配器 vs `mcp_stdio` docstring 逐字同文、
`_INSTRUCTIONS` 单一来源自 `toolcore`)已是**仓内结构性**测试,不再需要跨仓 exec。

```bash
python -m pytest tests/engine -q               # 引擎面 120 项(可单独跑;通常与产品面合并跑)
```

**实测**:`120 passed`(引擎面)/ 全套 `179 passed`。

## 3. GPU 冒烟(真库 ~/rag_real,77 篇 / 7652 chunk)

> 投产前门槛除本节冒烟外,另有 `eval/acl_regression.py`(WSL+4090 端到端 0 泄漏);
> 因需 GPU,不进 CPU CI,与下方 GPU 冒烟同属投产前手动关。

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

**gold 补表格题 + 88 题新基线(2026-07-03,引擎 `2ccda93`)**:

- gen_gold_tables.py 定向出 16 道表格题(中英混合,程序化 QC 拦下 2 条出题幻觉);
  gold 72→88,**口径断代**(与历史 72 题聚合数字不可直接对比)。
- **新权威基线**(88 题,DeepSeek 裁判,闭管道默认参数):检索召回 0.818 / MRR 0.627 /
  引用召回 0.767 / 忠实度 0.977 / 正确性 0.818。
- **拆分**:散文 72 题(检索 0.833 / 正确 0.861 / 忠实 0.972)vs **表格 16 题
  (检索 0.750 / 正确 0.625 / 忠实 1.000)**——表格题不带 kind 过滤也有 75% 检索命中
  (增强前这类题近乎不可命中,无 before 数字:旧索引已重建覆盖,诚实留空);
  忠实度满分 = 4 题检索 miss 全部诚实拒答、零瞎编(数值范围约束在新题类上兑现)。
- **表格题的 6 个错**:4 题 = 检索 miss(拒答被判错,检索侧余量);2 题 = 检索到但表格读数错
  (大表行列对位,生成侧余量)。这就是下一步表格向工作的对称标尺。

**smart-ask 落地实录(2026-07-03/04,四轮 88 题实验定形,设计 DESIGN D9)**:

| 版本 | 表格16 正确 | 散文72 正确 | 忠实度(全卷) | 裁决 |
|---|---|---|---|---|
| 基线(无 smart) | 0.625 | 0.861 | 0.977 | — |
| ① 前置表格腿 | **0.875** | **0.792** ❌ | 0.966 | 否决:腿带进的相近数值误伤 5 道原本答对的散文题 |
| ② 失败驱动,rerank_top_n=30 | 0.750 | 0.847 | 1.000 | 假象:腿太浅(五年表在粗排 31-50 名,精排池装不进),旗舰手工案例失效 |
| ③ 失败驱动,top_n=50,无条件采用 | 0.625 | 0.833 | **0.932** ❌ | 否决:部分回答夹带"未提供 X"的错误缺失声明(X 在 context 里) |
| ④ **失败驱动 + 择优采用(终版)** | 0.688 | 0.833 | **0.977** | **采纳** |

终版归因(rows 带 retried/retry_kept 标记):未触发面 81 题与基线配对仅 2 题翻转
(均为五轮实验里反复横跳的已知不稳定题=噪声底噪 ±2 题);弃用路径 4 题正确性与基线完全一致
(零损失);采用路径 3 题中 1 题 ✗→✓。**旗舰手工案例(考卷未覆盖的多值跨年题型)默认参数
从拒答变全对**:中文"逐年净利润"五年全对(auto=table_leg_retry)、中文"总营收"答对。

方法论沉淀:① 默认行为的智能只作用于失败路径——成功路径上的任何"帮忙"都是风险(①的教训);
② 精排池深度必须 ≥ 对的块的粗排最差名次(②的教训);③ 重试从"全拒答"变"部分回答"时,
对缺失部分的错误断言是新失败面,择优门槛必须卡住它(③的教训);④ 单轮 LLM eval 噪声底噪
±2 题,这个粒度的横向比较必须配对归因(retried 标记已内建 run_eval)。

## 3b. 团队服务面实测(多身份 / 观测 / 压测 / 演练)

**CPU 单测 46→59**(test_team.py 新增:keys 解析 fail-closed / 401 / 身份逐请求流到引擎 /
跨用户会话隔离(伪造同 session id)/ stats admin 门控 / 非回环启动守卫 / 日志不落 key+截断+可关 /
name 唯一性与禁 `|` / keys new 不裸抛 / 观测崩溃安全 / 结构化失败计入 errors)。

**多身份 live 演示**(真库,alice=demo/admin、bob=other):无 key/伪造 key → 401;alice 见 77 篇;
**bob(other 租户)见 0 篇、检索 empty**(引擎 ACL fail-closed 兑现,身份只是"谁在问");
bob 读 stats → 403、alice → 200。

**并发压测**(bench.py,4090/77 篇):检索 p50 随并发线性升(1 并发 408ms → 10 并发 3.07s),
吞吐天花板 ~3.2 req/s,**零错误**;混合负载下 ask 在飞时检索 p50 仍 352ms(LLM 段不持锁,D8 实锤)。
完整表见 [OPERATIONS §4](OPERATIONS.md)。容量结论:≤10 人同时活跃体验可用。

**备份恢复演练**:停服→备份(27MB/3s)→恢复到备用目录→8788 端口备用实例→77 篇可见+检索 ok,
热模型 RTO 33s(冷启动另加模型 lazy load,已在 OPERATIONS 诚实标注)。

## 4. 对抗评审(P1,已完成)

三视角(安全/正确性·并发/契约·文档)× 每发现 2 独立反驳者,39 agent。结果:
**2 条 confirmed + 13 条自核实属实(验证 agent 撞额度,主线逐条对照源码核实)→ 全部修复并钉回归测试**
(test_review_fixes.py,11 项);3 条被证伪留档。完整清单与处置见
[COMPONENT_NOTES.md §对抗评审 P1](COMPONENT_NOTES.md)。修复后全量回归:单仓一套 pytest 全绿。

**团队服务面安全评审(T5)**:身份/会话/观测/运维四视角 × 每发现 2 反驳者,36 agent。
**0 confirmed 安全漏洞**——完成的 verify 确认核心 ACL 边界稳固(身份 name 只是展示标签 + 会话去重前缀,
真边界是 `_current_user` 建的引擎 User;跨租户去重碰撞的失败方向不越权)。⚠ 诚实说明:多数 verify
agent 撞服务端限流未跑完,**不依赖投票、逐条自核实**。核实后修一批**健壮性/文档/观测完整性**缺陷并
钉回归测试(test_team.py 相关项):观测崩溃跳过记录→try/finally;结构化失败(200)漏计 errors→按业务
status 补判;query 截断层耦合→下放 obs 层;name 唯一性 + 禁 `|`;keys new 裸抛→复用 load_keys 校验;
备份漏 .env / 缺 mkdir;RTO 冷热标注;若干文档字段不一致。回归:单仓一套 pytest(179)全绿。

## 5. 未覆盖(诚实清单)

- **冷启动 RTO** 未实测(演练是热模型;冷启动加模型 lazy load 20s-2min,已在 OPERATIONS 标注推算值);
- **>10 并发 / 长时压测**未做(已知吞吐天花板 ~3.2 req/s,更大规模走 v2 Qdrant server 模式);
- `pharos index` 全量真跑(~/rag_real 已由 `scripts/` 下的建库脚本建成,indexer.py 是其参数化版,
  仅逻辑评审 + 锁冲突提示路径验证;下次换语料建库时顺带实测);
- MCP 适配器在 Claude Code 真会话里的端到端(需要你在 app 里连一次;工具逻辑已由
  adapter×live daemon 冒烟覆盖)。
