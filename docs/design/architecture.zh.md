<div align="right">

**中文** | [English](architecture.md)

</div>

# 架构

NaviKB 是四层结构,层与层之间有明确的契约。每一层都可以独立测试、独立替换。
本文描述形状本身,不涉及 API 表面和运维流程。

## 读者

本文面向两类人: (1) 评估 NaviKB 是否适合自己用例的工程师;(2) 思考"某个新
功能应该放在哪一层"的贡献者。假设读者熟悉向量检索、大语言模型 API、以及
Model Context Protocol (MCP)。

## 四层结构

```text
┌─────────────────────────────────────────────────────────────────┐
│  4. Serve     FastAPI tool server  +  MCP sub-app  +  REST API  │
│               Auth · Audit · Maintenance · Backup · GC          │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│  3. Retrieve  4 通道检索 (dense, sparse, vision, desc)           │
│               RRF 融合 → cross-encoder rerank → NavIndex         │
│               指针 enrich → evidence/hint 分区                   │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│  2. Storage   Qdrant (3 个版本化 collection,多模态)              │
│               SQLite (metadata, audit, users, maintenance)       │
│               本地 image store (页面渲染图)                       │
│               SQLite NavIndex (pointer-only 结构索引)            │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│  1. Ingest    MinerU 解析  →  分块  →  4 通道 embed              │
│               → metadata + nav 自动构建 →  后台 worker           │
└─────────────────────────────────────────────────────────────────┘
```

箭头方向是 ingestion 时的数据流。读路径方向相反: serve → retrieve →
storage。Layer 1 (ingest) **永远不读** Layer 4 (serve);两者之间唯一的
通信就是 SQLite job state + 跨进程 cache-epoch 计数器。

### Layer 1: Ingest

职责: 接收 PDF 路径(或其他支持的格式),产出检索层需要的所有 artifact。

pipeline 是确定性的(相同输入得相同输出,modulo 时间戳);它**不是**
流式处理器。每份文档在后台 worker 进程里独立处理一次。worker 是与
FastAPI server **独立的 OS 进程**,不是线程,不是 asyncio task —— 这是
全系统最重要的运维决策。一个决定换来三件事:

- **优雅崩溃恢复**: worker 挂掉不会拖垮 serving 层
- **独立资源预算**: worker 可以占 GPU 显存做 vision encoding,不影响 query 延迟
- **维护暂停不需要重启**: server 翻一个 SQLite flag,worker 在 claim 下一个
  item 之前观察到

ingest pipeline 每份文档产出:

1. **文本 chunk** + parent-chunk 引用(chunk size 和 overlap 由 chunker 调,
   形状的论证见 `docs/design/navigation-first.zh.md`)
2. **页面级 dense + 学习式 sparse 向量** —— dense 用 Qwen3-VL-Embedding(dim 4096),
   sparse 用 MILCO(doc 侧剪枝;与 dense 不同的另一个模型)
3. **页面级视觉 embedding**(Qwen3-VL,每页一个单向量 dense,与文本通道同 4096 维空间,
   不是多向量 MAX-SIM 模型)
4. **页面级 LLM 描述**(vision LLM,仅当页面包含视觉内容;纯文本页跳过)
5. **Metadata 行**(SQLite): doc_id、owner、status、version、effective_date、
   expiry_date、supersedes、deleted_at
6. **NavIndex 条目**: 从解析器拿到的 heading + page label 构建的 pointer-only
   结构索引

### Layer 2: Storage

三个互相独立的 store,每个按数据的访问模式选型:

| Store | 后端 | 存什么 | 为什么这样选 |
|---|---|---|---|
| **向量索引** | Qdrant(默认本地文件模式,可切到 server 模式) | 文本 chunk、vision 页、description —— 三个版本化 collection | 小团队最好的向量 DB:零配置本地模式,无痛切到 server,每通道命名向量(named vectors),Python 工程体感好 |
| **关系状态** | SQLite(一个文件多张表) | documents、users、audit_log、maintenance_state、cache_epoch、usage_feedback、ingestion_jobs | 两个进程(server + worker)需要共享状态;SQLite 是唯一零部署的真跨进程一致性方案。WAL 模式 + worker claim 路径上的 BEGIN IMMEDIATE |
| **图片 store** | 本地文件 `image_storage_path/<doc_id>/page_N.png` | vision 检索 + Agent fetch 时的页面渲染图 | 简单、可调试、便宜;软删除阶段文件就地留存,直到 GC 清掉 |

NavIndex 单独说一下: 它是**独立的 SQLite 库**(默认 `nav_index.db`),装的是
pointer-only 的 NavEntry 行。每个 NavEntry 知道自己所属文档、页码范围、
label、父子条目、`resource_uris` —— 但**绝不**持有原文或生成的描述。这种
切分正是"navigation, not retrieval" 成为主访问模式的根本: 索引可以被
serve、walk、expand,不需要 load 任何重的内容通道。

### Layer 3: Retrieve

检索特意做成正交阶段的小型组合:

```text
query
  │
  ├─► (可选) HyDE 扩展 + Multi-Query (query rewrite adapter)
  │
  ├─► 4 通道并行检索
  │     • text dense (Qwen3-VL-Embedding)
  │     • text sparse (MILCO 学习式稀疏)
  │     • vision (Qwen3-VL 单向量 dense)
  │     • description (Qwen3-VL-Embedding on LLM 描述)
  │
  ├─► RRF 融合(基于排名,不是分数 —— 不同通道的分数不可比)
  │
  ├─► cross-encoder rerank (只 rerank text 候选;vision-only
  │   命中以 RRF 分透传)
  │
  ├─► NavIndex enrich (每个 evidence 命中带上周围 NavEntry 的指针 ——
  │   章节、父节点、兄弟节点)
  │
  └─► evidence/hint 分区(解析得到的文本 → evidence_fields;
                          LLM 描述 → hint_fields)
```

两个不那么明显的选择值得点出:

- **RRF over 分数融合。** dense cosine = 0.78、sparse BM25-ish = 12.4、
  多向量 MAX-SIM = 3.1、description dense = 0.65 没有可比的尺度。基于排名
  的 RRF 是最便宜的正确答案;trained late-fusion 更好但带来训练依赖,
  跟"小团队阶段"明确避免的方向冲突。

- **Cross-encoder 只 rerank 文本。** NaviKB 早期版本把 LLM 生成的描述跟
  解析文本一起喂给 cross-encoder。那是在重复计算一个已经不确定的信号。
  现在的规则: 有 text_chunk 的候选 rerank;vision-only 命中以 RRF 分
  passthrough;纯 description 命中露出但被标记为 `result_type: hint`。

任何重模型都用**惰性初始化**。reranker + LLM adapter (HyDE、Multi-Query、
Agent final) 在首次使用时构造,不在 startup 加载;这让 server 冷启动保持
~5 秒,且不会为不用的路径预占 GPU 显存。

### Layer 4: Serve

serve 层把 retrieval engine 包成三张脸:

1. **FastAPI REST 端点** —— 给脚本和 ingestion 提交用的程序化访问
   (`POST /ingestion/jobs`、`DELETE /documents/{id}`、
   `POST /documents/{id}/restore`、`POST /admin/maintenance_mode`)
2. **MCP sub-app** —— 一等 Agent 表面,挂在 `/mcp`,暴露工具
   (`kb_search`、`kb_hybrid_search`、`kb_search_nav`、`kb_expand_nav_entry`
   等)和资源(`kb://documents/...`、`kb://nav/...`)。evidence/hint 分区
   和 navigation prompt 在 MCP 层最显著
3. **后台 worker** —— 独立进程,从 ingestion 队列消费,带公平 claim 语义
   (一个用户的批量提交不会饿死另一个用户的第一个 item)

跨切面关注点散布在 serve 层:

- **Auth**: 硬切到用户 token。任何请求(REST 或 MCP)都必须带
  `manage_users.py create` 发的有效 API key。users 表为空时 server 拒绝
  启动(bootstrap-refuse-start 不变量)
- **Audit**: 三种语义类 —— *单阶段强制* (用户管理操作)、*两阶段* (delete
  和 reassign)、*best-effort* (其他)。两阶段保证: 要么动作和它的
  `.completed` 审计行都成功,要么动作的效果可回滚(例如软删除时
  `.attempted` 行已落盘可作 trail)
- **Maintenance**: SQLite-backed flag,两个进程都读。写入方在 mutate
  之前观察;admin 端点等 active worker claim drain 完才返回 `drained:true`
- **软删除 + GC**: DELETE 永远不动 Qdrant 或图片;只翻状态 + 写
  `deleted_at`。Restore 是再翻回去。GC 在带外运行,清掉超过 retention
  的软删除文档
- **缓存失效**: SQLite cache-epoch 计数器解跨进程过期问题。worker 在
  ingest / deprecate 之后 bump;server 把 epoch 折进自己的搜索 cache key

auth / audit / maintenance / 软删除 / GC 的深入讨论会放在
`docs/design/security-model.zh.md`。

## 进程拓扑

```text
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  FastAPI tool_server          │         │  后台 worker                  │
│  - REST + MCP sub-app         │         │  - 从 ingestion_jobs claim    │
│  - 懒加载 reranker, LLMs       │         │  - 跑 ingest pipeline         │
│  - 读 cache_epoch              │         │  - 完成后 bump cache_epoch    │
└─────────┬─────────────────────┘         └─────────────┬────────────────┘
          │                                              │
          │      共享 SQLite (kb_metadata.db)            │
          │   ┌───────────────────────────────────┐     │
          ├──►│ documents · users · audit_log     │◄────┤
          │   │ maintenance_state · cache_epoch   │     │
          │   │ ingestion_jobs · ingestion_items  │     │
          │   └───────────────────────────────────┘     │
          │                                              │
          │            共享 Qdrant                       │
          │   ┌───────────────────────────────────┐     │
          ├──►│ text_chunks · vision_pages · desc │◄────┤
          │   └───────────────────────────────────┘     │
          │                                              │
          │            共享文件系统                       │
          │   ┌───────────────────────────────────┐     │
          └──►│ image_storage_path/<doc_id>/...   │◄────┘
              └───────────────────────────────────┘
```

两个进程**永远不直接通信**。所有协调都通过 SQLite。这在 2026 年是不流行的
—— 默认反应是去抓一个 message broker —— 但在小团队规模上是正确答案,
因为它消除了一类失败模式(broker 挂了、broker 慢、broker 路错),代价是
"worker 不能跟 server 跑在不同机器上"。对单机或局域网部署,这个代价是零。

## 设计预想到的失败模式

这张表对"会出什么问题"是诚实的,不是对"不会出什么问题"的吹捧。

| 失败 | 现象 | 缓解 |
|---|---|---|
| Worker 在 ingest 中途崩溃 | item lease 过期,下一次 claim 重新拿走 | `claim_next_item` 重领窗口;per-item attempt 计数器;退避 |
| 两个 worker 抢同一 item | SQLite BEGIN IMMEDIATE 写锁让一个赢;另一个看到行已 claim 就跳过 | worker claim 事务上显式 BEGIN IMMEDIATE |
| Ingest 后搜索缓存返回过期结果 | 最多 `search_cache_ttl_s` (默认 300s) 的过期 | cache-epoch 计数器: worker commit 后 bump,server 把它折进 cache key |
| 向量库和 metadata 不一致 | doctor `reconcile` 检出孤立向量、孤立图片、active 文档无向量 | `doctor.py` 深度 reconcile + `gc_deleted_docs.py` 清残留 |
| Delete 时 audit DB 不可用 | 两阶段: `.attempted` 行已落盘;主操作继续;`.completed` 失败回落到 JSONL 或对调用方返回 HTTP 503 | JSONL fallback 文件;admin 重建审计 trail |
| Server 处于 maintenance,用户提交写入 | 所有写路径(REST + MCP 写工具 + worker claim)都检查 flag 拒绝 | `require_no_maintenance` 依赖 + worker `maintenance_check` 回调 |
| 备份在写入中途拍摄 | maintenance drain + SQLite online backup API(捕获未 checkpoint 的 WAL) | drain loop 5 分钟上限;backup 脚本用 `sqlite3.Connection.backup()`,不是 `shutil.copy` |

## 故意不做的事

- **多机部署。** SQLite 作为跨进程协调把我们卡死在单机。要走多机就得
  换协调层;我们没这个计划。
- **LLM 驱动的 schema/ontology 归纳。** NaviKB 里的"图"就是文档自己的
  标题层级,不是模型对实体边的猜测。
- **托管 SaaS。** 我们打算坚持本地优先。想要托管服务的 operator 应该看
  别的项目。
- **流式 ingest。** 每份文档是端到端处理完才开下一份。对成千上万的小文档
  这会浪费;我们接受这个代价换失败模式简单。

## 接下来读什么

- [Status & 发布计划](../status.zh.md) —— 已做、测试中、阻塞中的事
- [`navigation-first.zh.md`](navigation-first.zh.md) —— NavIndex 为什么
  是这个样子
- [`security-model.zh.md`](security-model.zh.md) —— auth、audit、软删除
  契约
- [`comparison.zh.md`](comparison.zh.md) —— 与相关项目的明确对比
