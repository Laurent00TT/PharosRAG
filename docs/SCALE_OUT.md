# Pharos 水平扩展演化:拆 GPU 推理层 → 真多副本

> 状态:**设计定案 v2,实施未开始**。经"侦察 → 六维度设计 → 对抗审视 → 综合"产出,再经"四路对抗审查
> (准确性/完整性/可执行性/一致性)→ 主编修订"打磨。所有 `file:line` 已逐行核对源码(2026-07)。
> v2 修正了 v1 的一个核心施工陷阱(P0-1 病灶定位错到 service 层,实为 toolcore 兜底)+ 7 处 MUST-FIX。
>
> 本文把 [DESIGN.md](DESIGN.md) 标为**非目标(v2)** 的"水平扩展/多副本"具体化为可执行的**六阶段落地计划**。
>
> **落地纪律**:先提案后改、每阶段可独立验证、"已修"须附复现命令 + 改动前后实测对比。
> **环境前置(所有验证命令的先决)**:WSL `conda activate navikb && cd <repo> && pip install -e '.[dev]'`。
> 未装齐核心依赖(jieba/qdrant-client 等)直接 `pytest` 会 **collection error**,不是真失败。

---

## 0. 一句话

把系统里**唯一吃 GPU** 的部分(Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B 的前向)拆进独立推理服务,
让 `pharos serve` 变**无状态、不吃 GPU、不需要模型文件在本地** → 可多副本。但**多副本的真开关不是拆 GPU,
是把嵌入式 Qdrant 迁到 server 模式**(阶段 D);拆 GPU 只是必要非充分条件。

---

## 1. 背景与目标

### 1.1 为什么拆

`pharos serve` 当前在进程内加载两个 8B 模型到 GPU(见 D1)。这带来一个结构性天花板:**应用层绑死 GPU
= 无法水平扩展**。要多副本扛并发、要滚动升级不断服务、要崩溃隔离,前提都是应用层能无 GPU 地复制多份。

### 1.2 拆什么 / 不拆什么

| | 归属 | 理由 |
|---|---|---|
| `Dense.encode_*` / `Reranker.score` 的**纯 GPU 前向** | → 独立推理服务 | 唯一吃 GPU 的部分,拆出去应用层就脱 GPU |
| MRL 截维 / query LRU 缓存 / 排序写回 `Hit` / ACL / small-to-big | 留应用层 | 纯业务逻辑,无 GPU;留下来推理服务才能纯到可换 vLLM |
| chunker / generator / toolcore | 不动 | 本就不吃 GPU |

**边界原则**:画在"纯 GPU 前向"与"业务逻辑"之间。推理服务端点里**不出现任何 pharos 业务概念**
(no ACL、no Hit、no sidecar),只有 `texts→vectors`、`(query,docs)→scores`。这样将来能被 vLLM/TEI 无痛替换。

### 1.3 约束(不可违反)

1. **向后兼容**:`inference_url` 空 = local(进程内 GPU),默认行为**零改动**。回归基线 = 当前 CPU 测试套件全绿
   (基数以 navikb 环境 `pytest tests` 实测为准;`grep -c 'def test_'` = 185,含 4 个 remote 骨架测试 →
   "local 零破坏"应扣除这 4 个,即 181;⚠ [DESIGN.md](DESIGN.md):166 记的是 "179 passed",两处需对齐,见 §9)。
2. **引擎改动最小**:靠依赖注入 + 工厂(`make_dense`/`make_reranker`),不改 local 代码路径。
3. **等价性**:local 与 remote 产出的**最终检索向量数学等价** —— 同一个库能 local 建、remote 查,不错位。
   这是硬约束,靠"全维返回 + 客户端截维"结构性保证,并用测试证明(阶段 B)。
4. **贴近真实生产**:探针分离、readiness 导流、失败重试、容器编排都按生产做法,因为本项目的目的是**学生产运维**。

### 1.4 与现有设计的关系

- [DESIGN.md](DESIGN.md) §1 非目标:"水平扩展/多副本(嵌入式 Qdrant 天花板,规模驱动的 v2 才做)" ← **本文提上日程**
- [DESIGN.md](DESIGN.md) D1 否决备选:"Qdrant server 模式…这是 v2 的自然升级路径(EmbedConfig 换 url 即可)"
  ← **阶段 D 展开。⚠ 批注:"换 url 即可"仅指配置面**;实际还需 `store.py` 三分支 + `engine.py` 透传 +
  数据迁移 + **ACL server 模式重测(Q2)** —— 后两块最重,D1 的措辞过度简化了(§9 需纠正)。
- [TODO.md](TODO.md) v2 方向:"Qdrant server 化后才谈得上多 pharos 副本" ← **阶段 D→F 兑现**

落地完成后需回填这些文档(见 §9)。

---

## 2. 目标架构

### 2.1 目标拓扑:三服务,GPU 只在一处

```
                    ┌───────────────────────────────────────────────┐
   client ──────▶   │  nginx (负载均衡, 阶段 F)                        │
                    └──────────────┬────────────────────────────────┘
                          ┌────────┴────────┐
                          ▼                 ▼
                 ┌─────────────────┐ ┌─────────────────┐    无状态 / 无 GPU / 无 torch
                 │ pharos 副本 #1  │ │ pharos 副本 #2  │ …  可多副本 (阶段 D 解锁)
                 └───┬─────────┬───┘ └───┬─────────┬───┘
       /embed /rerank│         │gRPC     │         │
                     ▼         ▼         ▼         ▼
          ┌────────────────────┐   ┌────────────────────────┐
          │ inference (×1, GPU) │   │ qdrant (server 模式)    │
          │ 两个 8B 常驻, 全维返回│   │ 持久卷, 多副本共享 (阶段D)│
          └────────────────────┘   └────────────────────────┘

  建库:离线单例,走 local (inference_url 留空) 绕过逐-chunk HTTP 税(见 D4/F4)
```

### 2.2 三个决定性取舍

**取舍一 —— 边界画在"纯 GPU 前向"和"业务逻辑"之间。** 见 §1.2。推理服务纯到端点无业务概念 → 可换 vLLM。

**取舍二 —— 全维返回 + 客户端截维。** 推理服务用 `dense_dim=10**9`(`inference_server.py:33` `_FULL_DIM`)
构造 `Dense`,让其 `_mrl` **永不截维**、返回模型原始全维 normalized 向量;客户端 `RemoteDense._mrl_np`
(`remote.py:49`)按真实 `dense_dim` 截。**这让 local 与 remote 走同一份 `Qwen3VLEmbedder` 前向 +
数学等价的截维**,等价性是**结构性保证而非实测赌注** —— 这正是它优于直接切 vLLM(pooling 自实现、
等价性要赌)的硬理由。

**取舍三 —— dense 失败要"可重试的 loud",不是被宽兜底吞掉,更不是静默降级。** 推理服务预热(1-2 分钟)、
副本滚动重启是**正常运维态**。骨架当前的真实行为(**非 v1 所说的"裸 500"**):dense 抛异常经 `encode_query`
(不在 retrieve 的 try 内)冒泡到 `toolcore.py:216` 的 `except Exception`,被**吞成通用 `backend_unavailable`
(HTTP 200, retriable:true),无重试、无退避、丢失"推理不可用"语义**。定案:客户端有限重试 + 退避吸收瞬态,
重试耗尽抛语义异常 `InferenceUnavailable`,由 toolcore 细化成 `inference_unavailable`;dense **绝不降级到
空/纯 BM25**(query 向量是主召回信号,静默降级 = 用户拿到质量崩塌却无感,比失败更坏)。rerank 保持现有降级
(`retrieve.py:105` 已 `except Exception → hits[:k]`,增强信号降级安全) —— **dense loud / rerank 降级这个
非对称是有意的,须测试锁死(S10)**。

> 超时语义(F3 依据):`inference_timeout=120` 传 `httpx.Client(timeout=...)` 是**总** timeout。进程没起
> (connection refused)时 httpx **立即抛 `ConnectError` 不等 120s**;"等 120s"只在"连上了但服务端 hang/
> 预热阻塞"时成立。别把这两个场景混为一谈(阶段 A mock 时要分别覆盖)。

### 2.3 已落地骨架(现状,阶段 A 之前)

| 文件 | 已有 | 状态 |
|---|---|---|
| `src/embedder/inference_server.py` | FastAPI:`/embed` `/embed_image` `/rerank` + `/healthz` `/readyz`,后台预热,GPU 串行锁 | 骨架在,有 P0-2 缺陷 |
| `src/embedder/remote.py` | `RemoteDense`/`RemoteReranker`(继承基类,override 前向走 HTTP)+ `make_dense`/`make_reranker` 工厂 | 骨架在,有 P0-1/P2 缺陷 |
| `src/embedder/config.py` | `inference_url` / `inference_timeout` 字段 | 已加,阶段 A 补重试字段 |
| `src/pharos/config.py` | `inference_url`(`PHAROS_INFERENCE_URL`) | 已加 |
| `src/pharos/engine.py:63` | `build_retriever` remote 模式跳过本地 scripts 检查 | 已加 |
| `tests/engine/test_remote.py` | 4 测试:①工厂选路 ②`_load` no-op + `_mrl_np`↔`_mrl` 等价(同一函数 `:40`) ③encode_text + query 缓存(`:56`) ④reranker 契约(`:72`) | 已过 |

**评价**:骨架**核心是对的**(边界、全维返回、工厂开关都保留),但对抗审视挖出两个"每次重启必现"的运维 bug
(P0-1 无重试 / P0-2 加载失败假 loading)+ 一个"上生产 go/no-go"的等价性缺口 + 多副本的真正阻塞项(Qdrant)。以下逐条。

---

## 3. 关键决策表(吸收对抗审视)

### 3.1 服务边界与等价性

| # | 问题 | 选择 | 为什么 |
|---|---|---|---|
| D1 | 截维在哪端 | **全维返回 + 客户端截维** | 等价性结构性保证;一个推理服务能服务不同 `dense_dim` 的库;带宽同机房非瓶颈 |
| D2 | `dense_dim ≥ 全维` 时 `_mrl_np` 静默不截 | **加下界断言 fail-loud + `/healthz` 暴露 `full_dim` 启动校验** | 误配 `dense_dim` 会静默产错位向量;断言把"配置错"变"启动崩" |
| D3 | rerank instruction 谁控制 | **客户端唯一真相**;服务端 `/rerank` 用 `q.instruction`,缺省 fail-loud 不静默 fallback | 现状客户端传了 instruction 但服务端**忽略**、用自己 cfg(`inference_server.py:116` 调 `reranker.score(q.query, q.documents)` 未传 `q.instruction`),是沉默双源 footgun |
| D4 | image 编码跨进程 | **建库走 local / 查询走 remote**;base64 TODO 非阻塞 | `encode_image` 只建库用(`embed.py:72`),查询路径不调;建库本就独占单例、与 GPU 同机,跨机问题直接消失 |
| D5 | 纯文本查询够不够 MVP | **够**;查询副本只依赖 `/embed`+`/rerank`+`/readyz` | 查询路径全程不调 `encode_image` |
| D6 | 契约怎么演进 | **只增字段不改键名**;新增 `full_dim`/`model_dense` 指纹 | 向后兼容 + 支撑等价性断言与"连错服务"检测 |

### 3.2 失败模式(P0/P2)

| # | 问题 | 选择 | 为什么 |
|---|---|---|---|
| F1 | dense 503/超时 → 被 toolcore 宽兜底吞成通用 `backend_unavailable`、无重试(**P0-1**) | `_post_*` 有限重试 + 指数退避(只重 503/connect/read-timeout,4xx 立即抛);**修复落点在 `toolcore.py`**(细化 `inference_unavailable`,见 §5-A/M1);**dense 不降级空** | 预热/滚动重启是正常运维态;query 向量是主召回信号 |
| F2 | `_guard` 漏判 `state.err` → 加载永久失败仍报 "loading"(**P0-2**) | `_guard` 对齐 `readyz`:先判 `state.err`;**连带修 `/healthz`**(现也不看 `state.err`,见 S7) | 错卡/缺模型时运维看到"永远预热"而非"加载失败",误导排障 |
| F3 | 单值 120s timeout,服务端 hang 时也等 120s(**P2-2**) | 拆 `httpx.Timeout(connect=3, read=120, write=10, pool=5)` | connect 短→快速发现 hang;read 长→容忍长文本前向(注:connection refused 本就立即失败,见 §2.2 超时语义) |
| F4 | 建库 remote 逐 chunk HTTP 净负担(**P2-4**) | 默认建库 local;坚持 remote 则 `embed.py` 攒批 | 建库需 GPU 机 + 独占 Qdrant、无多副本需求,remote 零收益纯负担 |
| F5 | 两个 `httpx.Client` 各建连接池(**P2-1**) | dense/reranker 共享一个 client(模块级按 url 缓存);**close 落点明确**(`create_app` lifespan shutdown 或 `atexit`,见 §5-A/M8) | 打同一个推理服务,双连接池纯浪费 |

### 3.3 多副本前置条件(定案里最易被忽略的 go/no-go)

| # | 问题 | 选择 | 为什么 |
|---|---|---|---|
| Q1 | 多副本 pharos 起不来 | **嵌入式 Qdrant → server 模式**。`store.py` 现为 `:memory:`/`path` **二分支**,需扩成 `url`/`:memory:`/`path` **三分支**(`:memory:` 优先级必须保住,大量测试依赖它短路);`config.py` 加 `qdrant_url`;**`engine.py:70-73` 显式透传 `qdrant_url=cfg.qdrant_url`**(否则和当初 `inference_url` 同款坑:pharos 侧配了静默丢失) | 嵌入式单进程文件锁互斥,多 pharos 进程无法同开一个库;**这才是多副本真开关,不是拆 GPU**。代码已埋伏笔(`store.py` payload index 注释"local 无效但 server 需要") |
| Q2 | server 模式 ACL 语义变 | **新增 server-mode ACL 测试**(不是重跑 `:memory:` 版 —— 那还走嵌入式 fusion,对 server 模式零覆盖;见 M4):参数化 `Store` fixture 支持 `qdrant_url` → 对真 Qdrant server 建库 → 跑与 `test_acl_hard_filter` 同 5 条 fail-closed 断言 | 嵌入式 RRF fusion 曾静默丢顶层 `should` → 越权泄漏(安全铁律);server 模式 fusion filter 语义不同,这是**安全等价性**,与向量等价性同级 |
| Q3 | `LockedRetriever` 大锁去留 | **多副本 v1 保留进程内大锁**(单副本内检索串行是已知边界);要兑现"提升非 GPU 并发"须先审共享可变状态(`retrieve.py` 的查询作用域 `cache={}`、sidecar 读),再评估放宽锁粒度 + 并发压测 —— **落到 §5-F**,不盲拆 | server 模式 Qdrant 并发安全 + GPU 前向外移后大锁两个存在理由都没了;但拆前要确认无其他共享可变状态 |

### 3.4 继承 vs 独立客户端 & 推理器选型

| # | 问题 | 选择 | 为什么 |
|---|---|---|---|
| I1 | `RemoteDense(Dense)` 继承使基类内部实现变远程隐式契约,脆 | **保留继承 + 测试锁死不变量**(轻);抽纯函数去继承留到真出问题 | 去继承要动 local 代码,与"引擎改动最小"冲突;测试锁不变量成本低、立刻红 |
| I2 | pharos-slim 镜像能否脱 torch | **go/no-go 断言范围 = 整条 import 闭包**:slim 环境 `python -c "import pharos.engine"` + 构造 `RemoteDense/RemoteReranker` 后 `'torch' not in sys.modules`(不止 `dense.py`/`rerank.py` 两文件顶层,整条 `engine→Retriever→Store→qdrant_client` 链都不能拉 torch) | 若任一环顶层 `import torch`,slim 镜像 import 即崩;torch 应只在 `_load`/`_mrl` 内延迟 import |

**推理器选型定案:留自包 FastAPI,不切 vLLM/TEI/Triton。**

| 后端 | 判断 | 一句话理由 |
|---|---|---|
| **自包 FastAPI(现状)** | ✅ 现在留它 | 两端同一份 `Qwen3VLEmbedder`,等价性结构性保证;单机 4090 够用;探针/锁/等价性都是可迁移的生产模式 |
| **vLLM** | 规模上来再评估 | 唯一合格替代(continuous batching 是硬价值);**但 pooling 自实现,切前必须实测 cosine>0.9999** |
| **TEI** | ❌ 排除 | **pooling 自实现 → 破坏取舍二的结构性等价**(与排除 vLLM 同源);次要:无 vision-language,`encode_image` 也做不了 |
| **Triton** | 搁置 | Qwen3-VL 无开箱模型,自写 Python backend 逻辑量 ≈ 现在的 `inference_server.py`,过度工程 |

**换 vLLM 的唯一正当触发**:查询 QPS 上升到 GPU 前向排队明显(`/embed` 队列深度持续 >1),需要
continuous batching —— 不是"vLLM 更潮"。触发时迁移成本极低(端点契约不含业务概念)。

---

## 4. 对抗审视发现的缺口清单(P0/P1/P2/P3)

> 骨架能跑,但以下是它的真实缺陷。按严重度排序。**最该先修是 P0-1、P0-2** —— 运维正常态就触发。

| # | 级别 | 问题 | 位置 | 修法 | 阶段 |
|---|---|---|---|---|---|
| P0-1 | 高 ✅**A修** | dense 无重试 → 预热/重启窗口查询被吞成无重试的通用 `backend_unavailable`(不裸崩但瞬态不吸收) | `remote.py` `_post_retry`(原无重试);`retrieve.py:93,252` `encode_query` 不在 try 内;`toolcore.py:216/278` 宽 `except` 吞语义 | `_post_retry` 对**任意 5xx + 任意 httpx 超时(`TimeoutException`)+ `ConnectError`** 重试退避、抛 `InferenceUnavailable`;toolcore 两处 `except` 前 **duck-type**(marker,不 import embedder)分流 `inference_unavailable` | A |
| **M1** | 高(并发) ⏳**B前必修** | remote 重试退避 `time.sleep` 在 LockedRetriever **检索大锁内** → 预热/滚动重启期整副本查询串行停摆(P0-1 修复的副作用;阶段A对抗审查发现) | `remote.py:76` sleep;`engine.py:37-39` 锁全程包 search;`retrieve.py:93` 锁内 encode | 锁粒度重构:remote 后端 encode/HTTP 移出检索锁 + `dense._query_cache` 加细粒度锁(与 §3.3 Q3 大锁拆分同源)。**只在 remote 激活时触发,故归 B 开头(真起 remote 前),不塞进 A 草率改** | **B 开头** |
| P0-2 | 高 ✅**A修** | `_guard` 漏判 `state.err` → 加载失败仍报 "loading";`/healthz` 也不看 `state.err` | `inference_server.py` `_guard`/`readyz`/`healthz` | 抽 `_readiness` 纯函数,`_guard`/`readyz` 先判 `err`;`/healthz` 保持 liveness=ok 但暴露 `err` 字段(S7:liveness≠readiness) | A |
| P1-1 | 高 | `_mrl_np` 无下界断言,`dense_dim ≥ 全维` 静默不截 | `remote.py:49-58` | 加 `full.shape[-1] < d` fail-loud;`/healthz` 暴露 `full_dim`,客户端启动断言 | B |
| P1-2 | 高 | numpy/torch renorm 等价性**零测试**(eps 已核对均 `1e-12`,非不一致;真风险在 dtype) | `remote.py:56`(`np.clip(_,1e-12,_)`,fp32) vs `dense.py:60`(`F.normalize` 默认 eps=1e-12)`:61`(`bf16→.float()`) | GPU 等价性测试 `np.allclose(atol=1e-6)`;**重点验 bf16↔fp32 renorm 数值偏差**(非 eps) | B |
| P1-3 | 高 | 继承使基类内部实现变远程隐式契约,脆 | `remote.py:22,61` | 测试锁死"整条 import 闭包无 torch(I2)"+"`encode_query` 走 HTTP 不碰 `_model`" | C |
| P2-1 | 中 ✅**A修** | 两个 `httpx.Client` 各建连接池,且无 close | `remote.py:29,68` | 共享一个 client(模块级 `_CLIENTS` 按 url)+ atexit close | A |
| P2-2 | 中 ✅**A修** | 单值 120s timeout,服务端 hang 也等 120s | `remote.py:29,68` | 拆 `httpx.Timeout(connect=3,read=120,...)` | A |
| P2-3 | 中 | rerank instruction 双源(客户端传了但服务端忽略) | `inference_server.py:116` | 服务端用 `q.instruction`,收敛客户端唯一真相 | E |
| P2-4 | 中 | 建库逐 chunk HTTP 净负担 | `embed.py:72,75` | 默认建库 local;否则攒批 | D4/文档 |
| P3 | 低 | `Reranker` 无 `_gpu_error` 缓存,与 `Dense` 不对称 | `rerank.py:35-38`(裸 `_assert_gpu`) vs `dense.py:25,41-47` | `Reranker._load` 对齐加缓存 | F |

---

## 5. 六阶段落地计划(A→F)

> 原则:**先补 P0 失败模式(运维必现)→ 再补 P0 等价性测试(上生产 go/no-go)→ 再回归护网 → 才做多副本/
> 容器化**。顺序刚性:在错误/脆弱的单副本上水平复制 = 复制错误、放大故障面。每阶段"已修"须附复现命令 +
> 改动前后实测对比。**所有验证命令先满足文首"环境前置"**。

### 阶段 A —— 补 P0 失败模式(仍单副本;全 CPU-mock 可验)

**改动**(5 文件):
- `embedder/errors.py`(**新增,纯异常无 httpx 依赖**):`class InferenceUnavailable(RuntimeError)`。放中立文件,
  让 `toolcore` 只 import 一个纯异常类、不碰 httpx(保持 toolcore 薄)。
- `config.py`(embedder):加 `inference_connect_timeout=3.0` / `inference_retries=2` / `inference_backoff=0.5`
- `remote.py`:模块级 `_CLIENTS` + `_get_client`(按 url 共享连接池 + 拆分 `httpx.Timeout`,修 P2-1/P2-2);
  `_post_retry`(**任意 5xx + `httpx.TimeoutException` + `ConnectError`** 指数退避,`max(0,retries)` clamp,4xx 立即抛,
  耗尽抛 `InferenceUnavailable`,修 P0-1);`_post_vectors`/`score` 改用之;`atexit` 关 `_CLIENTS`。
- `inference_server.py`:抽 `_readiness(ready,err)` 纯函数,`_guard`/`readyz` 先判 `err`(修 P0-2);
  `/healthz` 保持 `status:ok`(liveness)但加 `error` 字段暴露(S7:配置错重启修不好,报不健康只会 crashloop)
- `toolcore.py`:`_retrieve_impl`(`:216`)、`_grouped_impl`(`:278`)的 `except Exception` **改 `as e` + duck-type 分流**:
  `if getattr(e,"inference_unavailable",False): return _err("inference_unavailable", retriable=True)`(**P0-1 真正落点**;
  用 marker 不 import embedder,保持 toolcore stdlib-only;service.py 不改 —— 异常到不了那层)
- `embed.py:45`(S4):`Dense(cfg)` → `make_dense(cfg)`(建库也走工厂;默认 `inference_url=""` 仍 local,行为不变;
  建库 remote 化的失败处理属阶段 D)

**验证**(`pytest tests/engine/test_remote.py` 18 测试 + 全量 `pytest tests` 199,全 CPU-mock;先满足环境前置):
- 重试:前 2 次 503 → 第 3 次成功(恰好 2 次重试);**502/504 网关瞬态**也重试;**500** 耗尽抛 `InferenceUnavailable`
- 不重试:4xx 立即抛(零重试)
- 超时:`ConnectTimeout`(TimeoutException,非 ConnectError)被捕获重试;退避是**指数** `[0.5,1.0]`;负 retries clamp 不 crash
- toolcore 分流:mock 持续 5xx → `_retrieve_impl`/`_grouped_impl` 都返 `inference_unavailable`(非 `backend_unavailable`);普通异常仍 `backend_unavailable`
- P0-2:`_readiness(False, err)` 返 `error`(非 `loading`);`_readiness(True, None)` 返 `ready`
- **非对称(S10)**:mock reranker 抛 `InferenceUnavailable` → `retrieve.search` **降级**返 hybrid(不 loud、不崩);dense 侧 loud(上一条)
- **client 共享(P2-1)**:同 url `d._client is rr._client`
- **回归**:`pytest tests` 199 全绿(local 零破坏;基线见 §1.3)

**里程碑**:推理服务重启/预热窗口内,`/v1/retrieve` 与 `/v1/retrieve_grouped` 都收到结构化 `inference_unavailable`
(可重试),而非无重试的通用 `backend_unavailable`;加载永久失败时 `/readyz`、`/embed` 报 `error`,`/healthz` 仍 `ok`(liveness)但 `error` 字段暴露。

**阶段 A 完成状态(2026-07)**:✅ 代码 + 测试落地,`pytest tests` **199 绿**(185→199,+14 remote 测试)。经四路对抗审查,
已修 M2(超时基类漏 `ConnectTimeout`)/ M3(非 503 的 5xx 不重试)/ M4(rerank 降级只有类型断言无行为测试)/ S1(负 retries)/
S6b(退避系数)。**遗留 M1(高·并发)**:remote 重试退避在检索大锁内 → 归 **B 开头**(真起 remote 前)修锁粒度;S2/S3
(测试卫生 / `_get_client` 无锁)与 M1 强耦合,一并在 B 开头改。**A 阶段 local 默认路径不触发 M1**(工厂选 local Dense、
不走 `_post_retry`),故不阻塞 A 的"local 零破坏";但 M1 是 remote 端到端的核心可用性属性,B 起 remote 前必修。

### 阶段 B —— P0 等价性测试(仍单副本;**必须 CUDA torch 环境 + 起真推理服务**)

**改动**(✅已落地):
- `remote.py:_mrl_np` 下界断言(P1-1);`dense.py:_mrl` **改 fp32 normalize**(P1-2 修复,见完成状态);
  `inference_server.py:/healthz` 暴露 `full_dim`/`model_dense`(D2/D6);客户端 dense_dim 校验用运行时 `_mrl_np` 下界断言兜底
- 新增 `tests/engine/test_equivalence_gpu.py`(`skipif` 无 GPU 自动跳过;只需 local 模型对同一真 bf16 全维跑 torch/numpy
  两条截维路径,不起服务、不 OOM)。端到端(起服务混建混查)分时脚本见 scratchpad,命令记于完成状态。

**怎么跑(可照抄;须 navikb 的 CUDA torch,非默认 CPU torch)**:
```bash
# 1) 起推理服务(占 GPU,后台预热 1-2 分钟)
conda activate navikb && python -m embedder.inference_server &   # 默认 0.0.0.0:8900
# 2) 等 /readyz 转绿
until curl -sf localhost:8900/readyz; do sleep 5; done
# 3) 跑等价性测试(指向真服务)
PHAROS_INFERENCE_URL=http://localhost:8900 pytest tests/test_equivalence_gpu.py -m gpu -v
```
- **E1 向量级**:同批 texts(中英/长短/特殊字符),`Dense(cfg_1024).encode_text` vs `RemoteDense.encode_text`,
  `np.allclose(atol=1e-6)` 且 `norm≈1.0`;image 同理
- **E2 混建混查**(⭐ **上生产 go/no-go**):local 建小库 → 切 remote 查同一 collection,top-10 doc_id 集合与
  "全 local 建全 local 查"一致(score 差 <1e-5)
- **E3 reranker**:同 query+docs,local `score` vs remote `/rerank` `allclose`

**里程碑**:E1/E3 实测等价 + **E2 混建混查端到端 top-k 一致**。⚠ E1 逐元素等价**不蕴含** E2(top-k 要过 HNSW
近似 + RRF **rank** 融合两层,近重复段分差可 < E1 maxdiff 而翻转排序);**E2 必须端到端实测,不能逻辑替代**。
**E2 不过则整个方案不上生产** —— 混建混查错位是静默数据损坏,最难查。

**阶段 B 完成状态(2026-07;经阶段B对抗审查修订)**:M1 锁下沉先行(`fe04256`);**E1/E3 实测等价,E2 未端到端验证**:
- **E1 encode 完全等价**:cosine=**1.0000000**、maxdiff **2.98e-08**、query cosine=1.0(分时实测 `scripts/equiv_gpu.py`)。
- **E3 rerank 完全等价**:maxdiff **0.00**。
- **修了 P1-2 真 bug**:`Dense._mrl` 原在 **bf16** 上 normalize → 转 fp32 后 norm≈**1.002**;统一 fp32(先 `.float()`
  再截+normalize)后 norm=**1.000**,与 remote `_mrl_np` 对齐。**守护**:`test_remote.py::test_mrl_fp32_normalize_on_bf16_input`
  (纯 CPU:bf16 张量进 `_mrl` → norm=1.0 + 反向守卫 bf16 normalize 会 >1.001;删 `.float()` 即红)。**旧库兼容**:截维前缀
  bf16→fp32 无损、方向不变,旧 bf16 库仍可 COSINE 查;重建可获数值一致。
- **⚠ E2 混建混查:未端到端验证的已知缺口**(审查 M2)。E1 逐元素等价(2.98e-8)**不能**逻辑推出 E2:默认 hybrid 走
  HNSW 近似 + RRF **rank** 融合,近重复 chunk 的 cosine 分差可 < 2.98e-8 → local/remote 排序翻转、RRF 放大。**上生产前
  须补实测**:~50 真实 query,local 建库后分别用 local/remote encode 查同 collection(hybrid),断言 top-k `chunk_id` 序完全一致。
- **image 等价**(审查 S2):`encode_image` 与 text 共用 `_mrl`,截维等价随 text 成立(bf16 守护已覆盖);但 **image 路径
  跨机一致性是独立未决项**(`remote.py` base64 TODO:建库机/查询机看同一路径可能解到不同图)。
- **显存约束(实测)**:inference(2×8B)占 4090 **33GB/48GB**,local Dense 再要 16GB → 双加载 **OOM**,故等价性实测**分时**
  (remote 存档 → 停服务 → local 对比)。这也印证"应用层脱 GPU":pharos 副本 0 GPU。
- **护栏措辞诚实化**(审查 S1):`_mrl_np` 下界断言(P1-1)是**运行时**兜底(首次 encode 才触发、只覆盖 dense_dim > full_dim);
  `healthz` 的 `full_dim`/`model_dense` 目前**仅暴露供人工排障,无客户端自动校验**(启动 fail-fast 的客户端 probe 是后续项)。
- **跑法**:CPU `pytest tests`(210 绿,含 remote/concurrency/bf16 守护);GPU `pytest tests/engine/test_equivalence_gpu.py`
  (navikb,自动 skip);端到端分时 **`python scripts/equiv_gpu.py --step remote|local`**(已入库,可复现审计)。

### 阶段 C —— CPU-mock 测试矩阵进 CI(仍单副本;常驻回归网)

**改动**:扩充 `tests/engine/test_remote.py` 成完整矩阵(全 CPU-mock,CI 常驻):
- 工厂选路 / `_mrl_np` 截维+renorm / rerank 契约 / query LRU 跨后端一致 / A 阶段的重试/超时/降级/非对称
- **脱 torch go/no-go(I2,S5)**:slim 路径 `import pharos.engine` + 构造 `RemoteDense`/`RemoteReranker` 后断言
  `'torch' not in sys.modules`(整条 import 闭包,不止两文件顶层)
- **I1**:image payload 原样传(非 base64)regression guard

> ⚠ 措辞精确(S4):是**被测应用层代码路径**(`remote.py`/`make_dense`/`pharos.engine`)不 import torch。
> **测试自身可用 CPU torch** 做 `_mrl`(torch)↔`_mrl_np`(numpy)对照(`test_remote.py:42` 就这么干,不能删)。

**里程碑**:`pytest tests/engine/test_remote.py` 全绿、且 `import remote` 后 `sys.modules` 无 torch。

### 阶段 D —— 多副本真开关:Qdrant server 模式(⭐ **这步之后才能真多副本**)

**改动**:
- `store.py`:`:memory:`/`path` 二分支 → **`url`/`:memory:`/`path` 三分支**(有 `qdrant_url` 用 `QdrantClient(url=...)`;
  **`:memory:` 优先级保住**,否则内存测试走进 url 分支崩)
- `config.py`(embedder + pharos):加 `qdrant_url`(`PHAROS_QDRANT_URL`)
- **`engine.py:70-73`**:`EmbedConfig(...)` 加 `qdrant_url=cfg.qdrant_url` 透传(漏了 pharos 侧配了静默丢失)
- **`tests/`**:参数化 `Store` fixture 支持 `qdrant_url`(为 Q2 的 server-mode ACL 测试铺路)
- 起一个 Qdrant server(docker `qdrant/qdrant`),把嵌入式库数据迁入

**验证**:
```bash
# (a) 向后兼容:不设 qdrant_url → 走嵌入式,行为零改动
pytest tests
# (b) 多副本锁解开(核心里程碑;不发查询→靠 dense lazy load 不占 GPU):
#     dense 是首查才 load,pharos serve 启动阶段只打开 Qdrant。故起两个 serve 即可验锁,无需 GPU。
docker run -d -p 6333:6333 qdrant/qdrant
PHAROS_QDRANT_URL=http://localhost:6333 PHAROS_PORT=8801 pharos serve &   # 成功
PHAROS_QDRANT_URL=http://localhost:6333 PHAROS_PORT=8802 pharos serve &   # 也成功(server 模式)
#     反证:两个都不设 QDRANT_URL、指同一 PHAROS_INDEX_DIR → 第二个在 Store 打开嵌入式 Qdrant 时报锁
# (c) ⭐ 安全 go/no-go:新增 server-mode ACL 测试(非重跑 :memory: 版)
PHAROS_QDRANT_URL=http://localhost:6333 pytest tests/engine/test_store.py -k acl_server
```
> ⚠ 两点精确(M3/M4):① D 里程碑**止于"两个 serve 启动成功"**(证明 Qdrant 文件锁解开);发实际查询要 load 模型,
> local 双开 = 4 个 8B 爆 4090 显存 —— **真 remote 双副本应答留到 E**(inference 容器起来、pharos 无模型时)。
> ② ACL 测试是**新增 server-mode 版**(参数化 fixture 指真 server),不是原样重跑硬编码 `:memory:` 的
> `test_acl_hard_filter` —— 后者还走嵌入式 fusion,对 server 语义零覆盖、会假绿放行。
> 先确认 `pharos serve` 是否支持 `PHAROS_PORT`(config 有 `PHAROS_PORT`,✓);若要 `--port` flag 则在本阶段补 CLI。

**里程碑**:两个 `pharos serve` 进程同时连同一个 Qdrant server 成功启动(嵌入式下第二个必报锁);
新增的 server-mode ACL 越权测试全绿(fail-closed 保持)。

**阶段 D 完成状态(2026-07)**:✅ 两个 go/no-go 都过,server 模式代码 + 回归测试落地。
- **代码(三分支透传)**:`store.py` `__init__` 改 **url > :memory: > path** 三分支(`:memory:` 优先级保住,大量测试依赖);
  `EmbedConfig.qdrant_url` + `PharosConfig.qdrant_url`(`PHAROS_QDRANT_URL`);`engine.py:build_retriever` 显式透传
  `qdrant_url=cfg.qdrant_url`(否则 pharos 侧配了静默丢失,同当初 `inference_url` 坑)。**向后兼容**:不设 `qdrant_url`
  走嵌入式,全量 211 passed 零破坏。
- **⭐ Q2 ACL 越权重测(安全 go/no-go)= GO**:Qdrant server v1.18.0(与 client 1.18 匹配)实测,**hybrid(RRF)/dense/
  sparse 三路都 fail-closed** —— 只回 public+有权(c1/c2),无权/跨租户/unset(c3/c4/c5)绝不召回。**嵌入式 RRF 丢顶层
  should 的铁律在 server 模式未复发**(store.py 的"ACL 下推每个 prefetch"在 server fusion 下同样生效)。
- **⭐ 多副本锁 = GO**:嵌入式同 path 第二个 client 报 `already accessed by another instance`(单进程独占,多副本
  起不来);server 模式两个 client 同开一库都成功。**server 解开了嵌入式文件锁,多副本前置成立**。
- **回归测试**:`tests/engine/test_store_server.py`(skipif server 不可达,CI 自动跳过;手动:起 `~/qdrant/qdrant` 后
  `pytest tests/engine/test_store_server.py`)——`test_server_mode_acl_fail_closed_all_strategies` + `test_server_unlocks_multi_client`。
- **数据迁移(真部署时)**:现有嵌入式 `~/rag_real` → server 用 `scroll`+`upsert`(或 Qdrant snapshot);**未迁真库**
  (D 用假向量小库验语义);迁移 + 真两进程 pharos serve 应答留 **阶段 E**(compose 起 inference + 多副本时端到端)。
- **Store._lock 现状**:仍保留(M1);server 模式 Qdrant 并发安全,此锁"保护嵌入式单 client"的理由消失,**放宽/删除
  留阶段 F/Q3**(先确认无其他共享可变状态)。Qdrant server 由 v1.18.0 二进制跑(非 docker;Docker Desktop 留阶段 E 容器化)。

### 阶段 E —— 容器化 / 编排(多副本真应答)

**改动**:
- `Dockerfile.inference`(`nvidia/cuda` base + `.[gpu]`)、`Dockerfile.pharos`(slim,**不装 torch**)、
  `docker-compose.yml`(inference×1 GPU 直通 + qdrant server + pharos×N)
- 顺带修 P2-3(rerank instruction 收敛客户端)
- compose 关键点(Windows/WSL2 特有坑):
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `NVIDIA_VISIBLE_DEVICES=GPU-<4090-UUID>`(FASTEST_FIRST index 会漂,锁 UUID)
  - 模型走 `-v ...:/models:ro`,放 WSL ext4(**别放 `/mnt/c`**,跨边界 IO 慢)
  - healthcheck `curl /readyz` + **`start_period: 300s`**(两个 8B 串行预热,给不够会无限重启)
  - pharos `depends_on: inference: condition: service_healthy`

**验证/里程碑**:
```bash
docker compose up -d
until curl -sf localhost:8900/readyz; do sleep 10; done       # inference <5min 转绿
curl -s localhost:8787/v1/retrieve -d '{"query":"..."}'       # pharos remote 应答(此处才是真 remote 双副本应答)
docker compose exec inference python -c "import torch; print(torch.cuda.get_device_name(0))"  # 含 "4090"
docker compose exec pharos    python -c "import torch"        # 应 ModuleNotFoundError(证明真脱 torch)
```

### 阶段 F —— 真多副本落地验证 + 收尾(学习高潮)

**改动/验证**:
- `docker compose up --scale pharos=3` + nginx 轮询;**`docker kill` 一个副本,in-flight 之外的查询无感**(亲眼验证水平扩展)
- **Q3 大锁边界**:多副本 v1 每副本内保留进程内大锁 → 单副本内检索仍串行(已知边界)。要兑现"多副本提升非 GPU
  并发",需审 `retrieve.py` 查询作用域 `cache={}` / sidecar 读等共享可变状态 → 评估放宽锁粒度 + 并发压测。**先讲清
  这个边界**,别默认"scale=3 就等于 3× 吞吐"(GPU 前向串行 + 进程内大锁是两重上限)。
- 鉴权(可选):`inference_server.py` 加 `INFERENCE_API_KEY`(现状 `0.0.0.0` 裸奔;单机/内网可先靠网络隔离)
- P3:`Reranker._load` 补 `_gpu_error` 缓存
- 文档回填(见 §9)

**里程碑**:nginx 后 3 副本轮询应答;`docker kill` 一个副本,查询无感;整套一键起停。吞吐边界(GPU 串行 + 大锁)在文档写清。

---

## 6. 等价性 & 安全 go/no-go gate(不过不前进)

| Gate | 阶段 | 判据 | 不过的后果 |
|---|---|---|---|
| **向量等价 E1** | B | `np.allclose(local, remote, atol=1e-6)` | numpy/torch 路径有偏差(重点查 bf16↔fp32) |
| **混建混查 E2** | B | local 建 remote 查 top-10 与全 local 一致 | **不上生产** —— 静默数据损坏 |
| **server-mode ACL 越权(新增)** | D | 参数化 fixture 指真 server,5 条 fail-closed 全绿 | **不迁 server** —— 安全回归,越权泄漏(重跑 `:memory:` 版会假绿) |
| **脱 torch** | C/E | slim 环境 `import pharos.engine` 后无 torch;pharos 容器 `import torch` 失败 | slim 镜像目标未达成 |

---

## 7. 风险表

| 严重度 | 风险 | 缓解 |
|---|---|---|
| 最高 | **E2 混建混查未端到端验证**:E1 逐元素等价**不蕴含** hybrid RRF+HNSW 下 top-k 不变 | 上生产前必补 E2 实测(§5-B);E1/E3/bf16 已实测 + CI 守护 |
| 高 | dense 无重试(现状):任何抖动/预热/重启 → 查询被吞成无重试的 `backend_unavailable`,瞬态不吸收 | 阶段 A,先修(落点 toolcore) |
| 高 | 嵌入式 Qdrant 文件锁:不迁 server,多副本根本起不来 | 阶段 D,多副本真 go/no-go |
| 中 | server 模式 fusion filter ACL 语义变 → 越权泄漏 | 阶段 D **新增** server-mode 越权测试(非重跑嵌入式版) |
| 中 | `dense_dim`/`qdrant_url` 跨副本无校验或漏透传 → 静默错位/配置丢失 | D2 断言 + `engine.py` 显式透传 + 建库落模型指纹到 sidecar |
| 中 | pharos-slim 脱 torch 未验证(整条 import 闭包) | 阶段 C 的 I2 闭包断言 |
| 中 | 多副本仍带进程内大锁,scale≠线性吞吐 | §5-F 讲清 GPU 串行 + 大锁两重上限;放宽锁需审共享可变状态 |
| 低 | image 跨机路径 TODO | 只建库用、建库同机,影响面小;文档写清约束 |

---

## 8. 已决策项 & Open Questions

**已拍板**(2026-07):
- **范围**:一路做到真多副本(A→F 全套,含 Qdrant server 迁移 + 容器化)
- **落地方式**:先给逐文件 diff 提案,确认后再改
- **建库/查询**:建库走 local(绕过逐-chunk HTTP 税)/ 查询走 remote(D4)
- **继承**:保留继承 + 测试锁不变量,不重构去继承(I1)
- **鉴权**:先靠内网/loopback 网络隔离,`INFERENCE_API_KEY` 留阶段 F 可选

**Open(遇到再定)**:
- Qdrant server 用 docker 单实例够不够,还是要持久化编排(先单实例,规模再说)
- nginx vs 直接 K8s Service(本地学习先 nginx + compose,K8s 是后续)
- 建库是否最终也要 remote 化(默认 local;若要运维一致性再做 `embed.py` 攒批)
- `pharos serve` 多端口:先用 `PHAROS_PORT` env(已支持);是否加 `--port` flag 待定

---

## 9. 落地后需回填的文档

- [DESIGN.md](DESIGN.md) §1 非目标:移除"水平扩展/多副本";D1 **纠正"换 url 即可"**为"配置面换 url,另需 store 三分支 + 数据迁移 + ACL server 重测";`:166` 测试基数 **179 → 与实测对齐**
- [TODO.md](TODO.md) v2 方向:"Qdrant server / 多副本"从候选移到已交付;同样纠正"换 url 即可"
- [OPERATIONS.md](OPERATIONS.md):补推理服务运维(启动/预热/失败降级/重试语义)、多副本运维(滚更/单副本故障/大锁边界)
- [README.md](../README.md):架构图改画"应用层无 GPU 多副本 + 独立推理服务"
- [TESTING.md](TESTING.md):补 remote 测试矩阵 + 等价性 gate + server-mode ACL 测试

---

## 附:溯源与修订

- **v1**:多智能体流程产出(侦察 2 + 六维度设计 3 + 对抗审视 1 + 综合 1),对已落地骨架逐条挑刺(P0-1…P3)。
- **v2**(本文):经四路对抗审查(准确性/完整性依赖/可执行性/一致性)+ 主编修订,吸收 8 条 MUST-FIX +
  13 条 SHOULD-FIX。**最重要的修正**:v1 把 P0-1 修复落点定在 `service.py`,实为死代码 —— 异常早被
  `toolcore.py:216/278` 的 `except Exception` 吞成 `backend_unavailable`,真正落点在 toolcore(M1)。
  其余:补 grouped 路径(M2)、D 里程碑砍成 local 双开证锁 + 显存约束(M3)、ACL 改新增 server-mode 测试(M4)、
  补 `engine.py` 透传 + `store.py` 三分支(M5)、环境前置(M6)、测试基线对齐 DESIGN(M7)、client 生命周期(M8)。
- 所有 `file:line` 已核对 2026-07 源码。
