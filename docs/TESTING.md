# Pharos 测试文档

> 三层:CPU 单测(每改必跑)/ 引擎回归(接缝改动时跑)/ GPU 冒烟(投产前跑)。
> 本文所有"实测"均为 2026-07-02 在 WSL `navikb`(4090)真实运行结果。

## 1. CPU 单测(25 项,~1s,不碰 Qdrant/GPU/网络)

```bash
conda activate navikb && cd pharos && python -m pytest tests -q
```

| 文件 | 覆盖 |
|---|---|
| test_sessions.py | 同会话同 set / 跨会话隔离 / LRU 逐出有界 / touch 刷新 |
| test_service.py | healthz;六端点 wiring;**no_identity fail-closed**;bad_arg 走结构化不走 422;API key 门槛(healthz 豁免/错 key 拒);**per-session 去重隔离 + 无头不去重**;ask(引用映射/include_contexts/空 query/llm_unconfigured/ask_failed 不泄内部细节) |
| test_adapter.py | 六工具转发参数映射;backend-down→结构化 backend_unavailable(hint 指向 pharos serve);401/5xx/非 JSON 映射;会话头存在;instructions 与引擎同源 |

工具语义本体(already_returned/omitted_budget/预算含资产/无权不泄存在性…)**不在 Pharos 重测**——
单一来源在引擎 toolcore,由引擎 test_tools.py(22 项)覆盖。

**实测**:`25 passed, 1 warning in 0.90s`。

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

## 4. 对抗评审

对新代码(service/adapter/sessions/engine + toolcore 拆分)做多 agent 对抗评审
(安全/正确性/契约一致性三视角 + 逐条反驳验证),confirmed 项修复后在此记录:

- 见 [COMPONENT_NOTES.md](COMPONENT_NOTES.md) 与 git 历史(评审轮次的修复各自成 commit)。

## 5. 未覆盖(诚实清单)

- 并发压测(多客户端同时打 /v1/ask + retrieve):锁模型保守(全串行),个人场景未压测;
- `pharos index` 全量真跑(~/rag_real 已由引擎 index_real.py 建成,indexer.py 是其参数化版,
  仅逻辑评审 + 锁冲突提示路径验证;下次换语料建库时顺带实测);
- MCP 适配器在 Claude Code 真会话里的端到端(需要你在 app 里连一次;工具逻辑已由
  adapter×live daemon 冒烟覆盖)。
