# 09 扩展性演进:单机到多副本

> **本篇导读**
> 讲 pharos 从"单进程绑死 GPU + 嵌入式向量库"的单体,经六阶段(A–F)演进到"三层可扩缩:inference GPU ×1 / pharos slim ×N / qdrant server + nginx"的全过程:每一步在解决哪个结构性瓶颈、哪些备选被否决、实测数据说了什么。
> **面试权重:极高**——这篇是系统设计面试的主素材(服务拆分、等价性、失败模式、负载均衡、背压、探针工程、吞吐边界,全是高频考点),且每个结论背后都有本仓可复现的实测。
> 前置阅读:[../SCALE_OUT.md](../SCALE_OUT.md)(工程文档,本篇不复述其运维细节);评估口径见 [07 评估方法论](07-evaluation.md)。

---

## 1. 概念底座:为什么 RAG 服务的水平扩展不是"多起几个进程"

任何在线服务想扛更多并发,教科书答案都是"无状态应用层 + 负载均衡 + 多副本"。但典型的自托管 RAG 服务天生就是反面教材——它至少有三类耦合,把"复制一份进程"变成不可能:

1. **计算资源耦合**:embedding / rerank 模型在应用进程内加载。一个 8B 模型十几 GB 显存,副本数直接被显卡数钉死;更糟的是应用逻辑(检索编排、权限过滤、上下文组装)明明不吃 GPU,却陪着模型一起无法复制。
2. **状态耦合**:嵌入式向量库(Qdrant local / Chroma / LanceDB 之类)以文件锁独占方式打开。第二个进程连库都开不了——这才是多副本的硬互斥,比 GPU 更早把路堵死。
3. **生命周期耦合**:模型加载要几分钟预热,崩一个组件全进程陪葬,升级必须整体停服。滚动升级、崩溃隔离都无从谈起。

业界的解法光谱大致三档:

| 维度 | 常见方案 |
|---|---|
| 推理层拆分 | 自建 HTTP 前向服务 / TEI(text-embeddings-inference)/ vLLM(continuous batching)/ Triton |
| 向量库 | 嵌入式(单进程)→ server 模式(Qdrant/Milvus/Weaviate server,多客户端并发) |
| 应用层 | 无状态化 + LB(nginx/Traefik/K8s Service),配 readiness 探针、重试、背压、优雅停机 |

拆分之后又会冒出一整族"分布式税":两端向量必须**数学等价**(否则混建混查静默错位)、瞬态失败要重试但重试要有背压对偶、探针自己可能变成故障源、超时预算要跨层对齐。pharos 这次改造把这些税逐一交了一遍,而且每一笔都留了实测凭据——这正是它作为面试素材的价值:不是"我知道该拆",而是"我拆过,知道会在哪里流血"。

---

## 2. Pharos 怎么做:三个绑死 → 六阶段 → 三层

### 2.1 起点:单体的三个绑死

改造前 `pharos serve` 一个进程里同时住着:Qwen3-VL-Embedding-8B + Reranker-8B(GPU)、嵌入式 Qdrant(文件锁独占)、全部应用逻辑(ACL/RRF/small-to-big/生成编排)。三个绑死一一对应上一节的三类耦合。六阶段各拆一环:

| 阶段 | 解决什么 | 一句话 |
|---|---|---|
| **A** | 失败模式 | 拆出推理服务骨架 + 客户端有限重试/退避,预热与滚动重启不再吞查询 |
| **B** | 等价性 + 并发 | 锁下沉(删检索大锁);local↔remote 向量等价实测,顺手挖出 bf16 真 bug |
| **C** | 回归网 | CPU-mock 测试矩阵进 CI(含"整条 import 闭包脱 torch"断言) |
| **D** | **多副本真开关** | 嵌入式 Qdrant → server 模式(文件锁互斥才是硬瓶颈,拆 GPU 只是必要非充分) |
| **E** | 容器化 | 三容器 compose(inference GPU / pharos slim / qdrant),数据迁移,readiness 链 |
| **F** | 真多副本 | 先对 A–E 全量对抗审查(27 confirmed 修完)再上 nginx;`docker kill` 无感实测 50/50 |

改造后的拓扑:

```
 client ──▶ nginx :8080(唯一对外入口,动态 DNS 轮询)
              │
      ┌───────┴────────┐
      ▼                ▼
 pharos 副本×N(slim,无 torch,~250MB 镜像)
      │  /embed /rerank        │  REST :6333
      ▼                        ▼
 inference ×1(GPU 4090,     qdrant server(唯一有状态,
  两个 8B 常驻,12.7GB 镜像)    持久卷,单一真源)
      └── 副本共享 sidecar 卷(查询期只读访问,small-to-big 的命门)
```

### 2.2 第一层:inference :8900——"纯 GPU 前向"的边界铁律

系统里唯一吃 GPU 的东西是两个 8B 模型的前向,就只把它拆成独立 FastAPI 服务。端点里**不出现任何 pharos 业务概念**——没有 ACL、没有 Hit、没有 sidecar,只有 `texts→vectors`、`(query,docs)→scores`。MRL 截维、query LRU 缓存、排序写回、small-to-big 全部留在应用层。

- 服务端用 `dense_dim=10**9` 构造 Dense(强制 local,防递归),让 `_mrl` 永不截维、返回模型原始全维(4096)归一向量:[src/embedder/inference_server.py:73](../../src/embedder/inference_server.py#L73)(`_FULL_DIM` 定义在 [:34](../../src/embedder/inference_server.py#L34))。
- 客户端 `RemoteDense._mrl_np` 用纯 numpy 按真实 `dense_dim`(1024)截维 + L2 renorm,并带下界断言 fail-loud(配置错位不许静默灌短向量):[src/embedder/remote.py:160-173](../../src/embedder/remote.py#L160)。
- 应用层只要 `PHAROS_INFERENCE_URL` 非空,工厂就切到 Remote 后端,本进程不加载模型:[src/embedder/remote.py:203-211](../../src/embedder/remote.py#L203);remote 模式连本地模型文件都不需要:[src/pharos/engine.py:27-33](../../src/pharos/engine.py#L27)。

这就是**"全维返回 + 客户端截维"的等价性结构保证**:local 与 remote 走同一份 `Qwen3VLEmbedder` 前向代码 + 数学等价的确定性截维,所以"local 建库、remote 查询"的向量必然一致——等价性是结构性质,不是实测赌注。阶段 B 的 GPU 实测做了背书:E1 encode cosine=1.0000000、maxdiff 2.98e-08;E3 rerank maxdiff 0.00(`scripts/equiv_gpu.py` 分时跑——2×8B 占 33GB/48GB,双加载会 OOM,这个约束本身反过来印证了"应用层必须脱 GPU")。

等价性 gate 还逼出一个真 bug:`Dense._mrl` 原来在 **bf16** 张量上 normalize,转 fp32 后 norm≈1.002;而 remote 的 numpy 路径全程 fp32,norm=1.0。COSINE 距离下方向等价、检索结果看不出异常,纯靠肉眼永远发现不了。修复 = 先 `.float()` 再截维 + normalize:[src/embedder/dense.py:65-74](../../src/embedder/dense.py#L65);纯 CPU 守护测试连"反向守卫"都有(证明 bf16 上 normalize 确实 >1.001,删掉 `.float()` 测试即红):[tests/engine/test_remote.py:260](../../tests/engine/test_remote.py#L260)。

### 2.3 第二层:pharos slim ×N——脱 torch、锁下沉、失败非对称

**脱 torch 到 build 期**。核心依赖不含 torch(`[gpu]` extra 隔离),slim 镜像 `pip install .` 后加两道构建期断言:`find_spec('torch')` 必须为 None,且 import `pharos.engine`+`pharos.service`、构造两个 Remote 后端之后 `'torch' not in sys.modules`——断言打在**整条 import 闭包**(engine→Retriever→Store→qdrant_client 任一环顶层 import torch 都会崩),把污染从"运行时首查崩"提前到"build fail":[Dockerfile.pharos:16-27](../../Dockerfile.pharos#L16)。副本镜像 ~250MB vs inference 的 12.7GB——多副本复制的是应用层,镜像大小直接决定扩缩与滚动升级速度。

**锁下沉(M1)**。阶段 A 给 remote 客户端加了重试退避修可用性,但对抗审查发现退避的 `time.sleep` 发生在旧 `LockedRetriever` 检索大锁内——预热期一个查询持锁 sleep,整副本所有查询串行停摆:修一个可用性 bug 制造了更大的可用性雪崩。修法是结构性的:删大锁,下沉为资源类锁——`Store._lock`(嵌入式单 client 非线程安全)、`Dense._fwd_lock`(GPU 前向,仅 local)、`_load_lock`(single-flight)、`_cache_lock`(LRU 双段,encode 的 HTTP+退避在锁外):[src/pharos/engine.py:6-13](../../src/pharos/engine.py#L6)、[src/embedder/dense.py:29-31](../../src/embedder/dense.py#L29)、[src/embedder/dense.py:89-105](../../src/embedder/dense.py#L89)。Remote 后端把前向/加载锁 override 成 `nullcontext`(HTTP 天然并发):[src/embedder/remote.py:107-108](../../src/embedder/remote.py#L107)。还显式接受了一个良性竞态:两线程同 query 同时 miss 各算一次、后写覆盖——拒绝为消除它把整段包回锁里。

**失败非对称:dense loud / rerank 降级**。这是有意设计并用测试锁死的:

- dense 失败:`_post_retry` 重试耗尽抛 `InferenceUnavailable`——一个纯异常类,带 `inference_unavailable=True` 类属性作 marker:[src/embedder/errors.py:11-21](../../src/embedder/errors.py#L11)。异常冒泡到 toolcore 的宽 `except`,用 `getattr(e, "inference_unavailable", False)` duck-typing 分流成结构化 `inference_unavailable`(retriable:true):[src/pharos/toolcore.py:229-234](../../src/pharos/toolcore.py#L229)。**绝不降级成空结果或纯 BM25**——query 向量是主召回信号,静默降级等于用户拿到的结果质量已经崩了却毫无察觉,比直接失败更坏。
- rerank 失败:`retrieve` 的 try/except 吞掉,降级返回 hybrid 召回的 `hits[:k]`,上层标 `rerank_degraded`:[src/embedder/retrieve.py:113-116](../../src/embedder/retrieve.py#L113)。rerank 是增强信号,降级安全。

toolcore 保持 stdlib-only(不 import embedder),靠 marker 而非类型判断——这是"依赖边界"和"错误语义"两个设计约束的交点。

**重试谱的完备性**。`_post_retry`:任意 5xx 一律当瞬态重试(502/504 是网关摘后端的形态,硬编码 503 会漏),`except httpx.TransportError` 捕获全部传输层异常,4xx 走 `raise_for_status` 立即冒泡不重试:[src/embedder/remote.py:67-96](../../src/embedder/remote.py#L67)。超时拆分 connect=3s / read=120s,dense/reranker 按 url 共享一个带锁的 client 缓存:[src/embedder/remote.py:40-52](../../src/embedder/remote.py#L40)。为什么强调"完备":F 阶段审查发现首版只捕 `ConnectError+TimeoutException`,而 **docker kill 最典型的断连形态是 `RemoteProtocolError`(keep-alive 死连接复用)和 `ReadError`(对端 RST)**——漏这两个类型,请求绕过重试被吞成无重试的 `backend_unavailable`,"kill 无感"链条在客户端这半就断了。守护测试:[tests/engine/test_remote.py:290](../../tests/engine/test_remote.py#L290)、[:300](../../tests/engine/test_remote.py#L300)。

### 2.4 第三层:qdrant server——多副本的真开关

**多副本的真开关不是拆 GPU,是嵌入式 Qdrant 的单进程文件锁**(拆 GPU 只是必要非充分)。`Store.__init__` 按优先级 `url > :memory: > path` 三分支建 client(`:memory:` 必须排在 path 前,大量测试依赖它短路):[src/embedder/store.py:26-31](../../src/embedder/store.py#L26)。

这一步的教学点不在"加个字段",而在**每个生产出口都得真正用上这个字段**。`qdrant_url` 首版三个出口全漏——engine(查询)、indexer(建库)、mcp_stdio(agentic stdio),server 链从未真正通过;我曾宣称"改了 engine"却没改,被自己的核实纪律逮住。补齐后三处透传都有守护测试(mock QdrantClient,删透传行即红):[src/pharos/engine.py:34-43](../../src/pharos/engine.py#L34)、[src/pharos/indexer.py:59-65](../../src/pharos/indexer.py#L59)、[src/pharos/mcp_stdio.py:56-66](../../src/pharos/mcp_stdio.py#L56)。同款坑后来在 `inference_url` 上原样复发(mcp_stdio 漏透传,slim 环境首查 `import torch` 崩)——"配置字段的加入 ≠ 生效,必须逐出口验证消费"是这次改造沉淀的可迁移经验。

**换存储后端要重测的不只是数据,还有安全语义**。嵌入式 QdrantLocal 在 RRF fusion 下会丢弃顶层 `query_filter` 的 `should`(实测发现),所以 ACL filter 必须下推到每个 prefetch:[src/embedder/store.py:103-123](../../src/embedder/store.py#L103)。迁 server 后 fusion filter 语义不同,必须**新增 server-mode 越权测试**而不是重跑硬编码 `:memory:` 的旧测试(那还走嵌入式 fusion,对 server 语义零覆盖 = 假绿):`tests/engine/test_store_server.py` 里有一条原始探针绕过出口 `acl_admits`、直接断言 server RRF 的 `query_points` 原始输出不含越权点——测的层次决定测试证明的命题(prefetch 下推真生效,而非出口兜底掩盖)。还配了 `PHAROS_REQUIRE_QDRANT_SERVER=1` 强制不许静默 skip(否则 CI server 不可达全 skip 也是假绿)。

**数据迁移脚本按"防中间态"设计**:先拿源库独占锁、再校验源 collection 存在且非空——都在对 dst 产生任何副作用之前;建 collection 复用 `Store.ensure_collection`(7 个 payload index 绝不手抄,漏一个 ACL 索引则 server 端过滤静默失准);scroll 必带 `with_vectors=True`(默认不返向量,漏了迁成"无向量点"检索全崩不报错);count 校验 + uuid5 幂等可重跑;退出语明说"仅验点数,向量/ACL 未校验"不假宣告:[scripts/migrate_to_server.py:44-57](../../scripts/migrate_to_server.py#L44)、[:70-71](../../scripts/migrate_to_server.py#L70)、[:82-89](../../scripts/migrate_to_server.py#L82)。实测 7652→7652 count 一致。

### 2.5 入口:nginx 负载均衡与"kill 无感"这条链

nginx 是唯一对外入口(`127.0.0.1:8080`),pharos 改 `expose: 8787` 不发布宿主端口——否则 `--scale` 第二副本必端口冲突(E 阶段的 sidecar 验证就曾在单副本上假通过):[docker-compose.yml:128-129](../../docker-compose.yml#L128)、[:143](../../docker-compose.yml#L143)。

"docker kill 一个副本、客户端零感知"不是一个开关,是一条链,断任何一环整体承诺失效:

1. **动态 DNS 轮询**:`server pharos:8787 resolve` + `resolver 127.0.0.11 valid=10s` + `zone`——开源 nginx≥1.27.3 按 Docker DNS 返回的全部副本 IP 轮询,scale 增减 10s 内生效(规避"启动解析一次就固化"的经典坑):[deploy/nginx.conf:14-20](../../deploy/nginx.conf#L14)。
2. **跨副本重试**:`proxy_next_upstream error timeout http_502/503/504 non_idempotent` + `tries 3` + `connect_timeout 2s`(死副本快速失败)——`non_idempotent` 让 POST(retrieve/ask)也重试,连被 kill 副本上的**在途**请求都落到健康副本;代价(极端下 ask 重复触发 LLM)是显式接受的取舍:[deploy/nginx.conf:36-42](../../deploy/nginx.conf#L36)。
3. **客户端异常谱**:上文的 `TransportError` 全谱重试(kill 的断连形态被吸收)。
4. **优雅停机**:`stop_grace_period: 30s` + uvicorn `timeout_graceful_shutdown=25`(< 30s,SIGKILL 前干净 drain):[src/pharos/cli.py:34-37](../../src/pharos/cli.py#L34)、[docker-compose.yml:84](../../docker-compose.yml#L84)。

实测(2026-07-07,3 副本):18 请求轮询 **6/6/7** 近均匀;持续 50 请求中途 `docker kill` 一副本 → **50/50 全 200、0 失败**,流量重分布到存活两副本(34/34)。另一条反直觉实测:`restart: unless-stopped` **不会**拉起被手动 kill 的副本(Docker 视为人为停止,restarts=0)——kill 无感靠 nginx failover 而非容器自愈,OPERATIONS 的措辞据此诚实订正过。

各副本日志用 `$$HOSTNAME` 运行时取本容器名写子目录,防多副本 JSONL 交错:[docker-compose.yml:107](../../docker-compose.yml#L107)。

### 2.6 两个贯穿机制:readiness 工程学与背压

**readiness 工程学**——这次改造攒了三个"探针自己变成故障源"的实测案例:

- 探针与业务共享 FastAPI 默认 40 线程池 → 高负载下探针排队饥饿 → "负载越高,就绪信号越假" → 编排把正常干活的健康副本全摘掉 = 全局 outage。修:探针端点改 `async def`(纯内存读不进线程池),pharos `/readyz` 的 qdrant 阻塞调用 offload 到专用 `_PROBE_LIMITER`(8 线程):[src/pharos/service.py:40](../../src/pharos/service.py#L40)、[:224-256](../../src/pharos/service.py#L224)。
- inference 探活用默认 `httpx.Timeout(3)`,各阶段累计最坏 ~9s > healthcheck 5s 预算 → 自误判。修:显式 `Timeout(1.5, connect=1.0)`。
- nginx 容器配置 `:ro` 挂载 → 入口脚本无法追加 ipv6 listen → 容器内 `localhost` 先解析 `::1` → 假 unhealthy(真服务完全正常)。修:healthcheck 钉死 `127.0.0.1`:[docker-compose.yml:147-149](../../docker-compose.yml#L147)。

三例合成一条可迁移经验:**探针的资源路径、超时预算、名字解析都必须与业务隔离并显式钉死**。此外 liveness ≠ readiness 的区分做得很干净:`/healthz` 恒 ok(配置错重启修不好,报不健康只会 crashloop)但暴露 `error`/`full_dim` 字段;`_readiness` 是纯函数且 **err 优先于 loading**(加载永久失败要报 error,不是"永远预热中"),`/readyz` 与各端点 `_guard` 共用这一处避免三处漂移:[src/embedder/inference_server.py:42-52](../../src/embedder/inference_server.py#L42)。warmup 后台线程区分永久配置错(留 err 不自杀)vs 瞬态(退避重试,耗尽 `os._exit(1)` 让 restart 拉起):[src/embedder/inference_server.py:91-112](../../src/embedder/inference_server.py#L91)。compose 侧 `start_period: 600s` 的注释还订正过一个错误直觉——"给短了无害"是错的:超期后 3 次失败即判 unhealthy,`depends_on: service_healthy` 令整个 up 中止:[docker-compose.yml:69-73](../../docker-compose.yml#L69)。

**背压是客户端重试的对偶**。客户端有重试,服务端就必须有准入控制,否则重试把过载放大成雪崩:GPU 前向串行,无上限排队 = 客户端 read-timeout 放弃后请求仍在队里烧 GPU + 重试再入队 = 3× 无效功 + 拥塞正反馈。inference 用 `BoundedSemaphore(16)` 把"执行中 + 等 gpu_lock"的在途总数钉死,满则立即 503 overloaded 快速失败;客户端把 5xx 当 `InferenceUnavailable` 退避重试,天然兼容:[src/embedder/inference_server.py:86](../../src/embedder/inference_server.py#L86)、[:131-135](../../src/embedder/inference_server.py#L131)。这是简单版 load shedding,比引入真队列/优先级简单一个量级且够用。

### 2.7 诚实结论:吞吐上界 = 单卡 GPU 串行

多副本**真正扩什么、不扩什么**,三重上限按主导顺序:

1. **最硬上限:GPU 前向串行**。唯一 inference 容器,`gpu_lock` 串行单卡前向([src/embedder/inference_server.py:146](../../src/embedder/inference_server.py#L146))+ 信号量背压。无论多少 pharos 副本,dense/rerank 前向排同一条队。实测容量(2026-07-04,单机嵌入式形态,4090,7652 chunk,见 [../OPERATIONS.md](../OPERATIONS.md) §4):并发 1/2/5/10 → 2.5/2.9/3.0/**3.2 req/s**,p50 408ms→3.07s,排队线性、零错误;混合负载下 ask 在飞时检索 p50 仍 352ms(LLM 段不持锁)。
2. **进程内检索状态已基本消除**:检索路径副本间无共享可变状态(查询作用域 `cache={}` 是局部变量),所以多副本**真扩非 GPU 段并发**——ACL 过滤、RRF 融合、sidecar 读、small-to-big 组装、JSON 编解码。
3. **跨副本状态有意降级**:SessionRegistry 去重效果掉到 ~1/N、`/v1/stats` 只反映单副本(opt-in 便利,丢了不影响正确性)。唯一必须跨副本一致的是 **共享 sidecar 卷(查询期只读访问)**——副本命中却取空 sidecar 就是静默降级,正反证都实测过:2 副本共享卷 10 查询 0 降级 vs 控制组空 sidecar 19 降级:[docker-compose.yml:93-94](../../docker-compose.yml#L93)。(挂载模式的诚实边界:E 阶段整个 `/index` 是 RW 单挂载,sidecar 只是查询期只读*访问*;把 sidecar 单独 `:ro`、logs 单独 `:rw` 的纵深防御拆分是 F 阶段意图,目前只落在 compose 注释里 [docker-compose.yml:110](../../docker-compose.yml#L110),未实装。真正已落地的 `:ro` 挂载是明文 key 表。)

所以 `--scale pharos=N` 扩的是**非 GPU 并发 + 崩溃隔离 + 滚动升级**,**不是 QPS**(被单卡串行钉死在 ~3.2 req/s 量级)。突破它的唯一正当路径是 vLLM continuous batching,唯一正当触发是 `/embed` 队列深度持续 >1——不是"vLLM 更潮"。

### 2.8 vLLM:Plan B 并存 + 等价性从"保证"降级为"实验"

自建 FastAPI 的等价性是结构性的(两端同一份前向代码);换 vLLM 后 pooling 是它自己实现的,等价性从结构保证降级成**必须实测的命题**。所以方案是"不替换、先并存":vLLM 侧呈现完全相同的 `/embed /rerank /readyz` 契约(薄适配器套 chat 模板转发 `/v1/embeddings`),应用层零改动,compose profile 互斥切换 = 秒级回退——"端点无业务概念"的红利,到这里就兑现了。四道门顺序刚性(G0 可行 → G1 向量等价 → G2 混建混查 → G4 吞吐),过门才升 Plan A,详见 [../VLLM_PLAN.md](../VLLM_PLAN.md)。

实测(2026-07-07,`scripts/vllm_equiv_probe.py` + `scripts/vllm_g2_topk.py`):

- **G0 GO**:vLLM 0.22.1 以 pooling 加载 Qwen3-VL-Embedding-8B,出 4096 维已归一向量。关键参数 `max_model_len=8192`(模型 max_position_embeddings=262144 会让 vLLM 预留 36GB KV cache 必 OOM;embedding 单次前向无需长上下文):[scripts/vllm_equiv_probe.py:107-111](../../scripts/vllm_equiv_probe.py#L107)。
- **G1 边缘**:8 样本 cosine ∈ [0.99956, 0.99982],min 0.99956 < 0.9999 门槛。头号嫌疑是 transformers 4.57↔5.10 版本差;最短文本("a")最低,符合"短序列放大逐 token 数值差"。**边缘不拍板,交 G2 定生死**——因为"逐元素等价不蕴含 top-k 不变"(HNSW 近似 + RRF rank 融合会放大微差),判据要设在业务真实影响层(召回翻转)而非中间指标(cosine):[scripts/vllm_equiv_probe.py:147-154](../../scripts/vllm_equiv_probe.py#L147)。
- **G2 基本通过**:88 题真库(7652 点),top-1 一致 **87/88(98.9%)**、top-10 集合一致 97.7%、Jaccard@10 0.9959,分歧几乎全在 rank≥6 尾部近重复换序。等价侧 GO,升 Plan A 只差 G4 吞吐门(未跑)。

过程本身也是方法论素材:探针连跑三次假失败,每次表面结论都是"vLLM 不支持 Qwen3-VL pooling"——真根因分别是 ①`CUDA_VISIBLE_DEVICES` 给了 UUID(vLLM 只吃整数索引)、②KV cache OOM(缺 `max_model_len`)、③自己的 `grep -v` 把真实 Traceback 过滤掉了。三个全是配置/工具使用错,零个是架构问题。教训固化进了探针脚本:初始化失败时原样打出真实报错并注明"未必是架构不支持":[scripts/vllm_equiv_probe.py:113-115](../../scripts/vllm_equiv_probe.py#L113)。

---

## 3. 为什么这么设计:被否决的备选

| 备选 | 否决理由 |
|---|---|
| **直接切 vLLM/TEI 当推理层** | 两者 pooling 都要自实现,等价性从结构保证变实测赌注;TEI 另无 vision-language 支持(`encode_image` 做不了)。定案:自建 FastAPI 现在留,vLLM 走 Plan B 四道门(§2.8),触发条件是队列深度不是流行度 |
| **Triton** | Qwen3-VL 无开箱模型,自写 Python backend 逻辑量 ≈ 重写一遍 inference_server,过度工程 |
| **服务端截维**(而非全维返回) | 服务端必须知道每个库的 `dense_dim`,与"端点无业务概念"冲突;多库多维时要多实例。全维返回让一个推理服务同时服务不同维度的库,带宽(全维 JSON)同机房非瓶颈 |
| **dense 失败降级 BM25/空结果** | query 向量是主召回信号,静默降级 = 质量崩塌无感,比失败更坏;改为可重试的 loud(`inference_unavailable`,retriable:true)。rerank 相反——增强信号,降级安全。非对称由测试锁死 |
| **保留检索大锁,多副本对冲** | 大锁 + remote 退避 = 预热期整副本停摆;而且"在错误的单副本上水平复制 = 复制错误、放大故障面"——F 阶段先做 A–E 全量对抗审查(27 confirmed 修完)才上多副本,同一逻辑 |
| **nginx Plus / Traefik(主动健康检查)** | 开源 nginx 无主动健康检查是已知约束,当前用"被动 max_fails + proxy_next_upstream"覆盖 kill 场景;运行期 not-ready 的缺口被审查确认并进 backlog(见 §4 deploy#1)——选型升级要实测,不拍脑袋 |
| **建库也走 remote** | 建库逐 chunk HTTP 纯负担(需 GPU 机 + 独占语义、无多副本需求);定案建库 local、查询 remote |
| **`restart: always` 求全自愈** | 实测 `unless-stopped` 不拉起手动 kill 的副本后,选择诚实订正文档而非盲目换 `always`——kill 无感的承诺主体是 nginx failover,不是容器自愈 |

数据背书集中在三处:等价性(E1 cosine=1.0000000 / bf16 norm 1.002→1.000 / G2 88 题 87/88——注意 88 题口径与历史 72 题不可混,见 [07 评估方法论](07-evaluation.md));可用性(kill 无感 50/50、轮询 6/6/7);容量(单机形态 ~3.2 req/s 天花板,并发 10 时 p50 3.07s 线性排队零错误)。

---

## 4. 实战复盘:对抗审查确认的问题

写这套学习文档前,对部署子系统又做了一轮深读 + 对抗验证(每条疑似问题先派独立验证员试图反驳)。confirmed 的按纪律分流:行为无关的健壮性修复直接落地,需要 compose/GPU/压测环境验证的延期留档。**"确认了但不能马上改"本身是工程判断**——下面五条延期项每一条都是系统设计面试的现成考点。

### 4.1 本轮已修(节选与本篇相关的)

- **remote 握手校验**:服务端换模型后向量空间会静默错位(此前只有 `full_dim` 下界断言,同维不同语义空间抓不住)。修:首查前一次性 GET `/healthz`,比对 `model_dense` 与客户端模型名、`full_dim ≥ dense_dim`,不符 fail-loud;预热中/不可达不阻断(瞬态留给重试链),single-flight:[src/embedder/remote.py:115-144](../../src/embedder/remote.py#L115)。
- **RemoteReranker.score 签名对齐**:基类 `score(query, docs_text, instruction=None)` 被子类收窄成两参——潜伏的 LSP 违反(按基类契约传 `instruction=` 即 TypeError),恰是当年"rerank instruction 双源 footgun"在客户端侧的复活。修:补形参,None 回落 cfg 与旧行为逐字节一致:[src/embedder/remote.py:190-195](../../src/embedder/remote.py#L190)。
- **RequestLog.flush(timeout) 真超时**:形参被接受但忽略,`q.join()` 无界等待——共享卷 IO 卡死时优雅停机 hang 满宽限被 SIGKILL(Exited 137 掩盖真实退出原因)。修:条件变量带 deadline 轮询,超时 warning 后放弃;停机传 `timeout=5.0`:[src/pharos/obs.py:92](../../src/pharos/obs.py#L92)。

验证口径:修复前 `pytest tests -q` 基线 224 passed / 4 skipped,修复后 259 passed / 5 skipped(唯一新增 skip 是 server-gated 测试,临时起真 Qdrant server 实测 264 passed / 0 skipped)。

### 4.2 延期的五条生产化 backlog(deferred deploy#0–3、#6)

**deploy#0 —— 超时预算跨层脱节(medium)**。nginx `proxy_read_timeout 130s`([deploy/nginx.conf:41](../../deploy/nginx.conf#L41))只对齐了客户端单次 read 尝试(120s),漏了 ×3 重试链:pharos 最坏重试链 ≈ 3×120s + 退避 ≈ **361s**。inference 前向 hang 时,客户端 130s 拿 504,但 uvicorn 对 sync 线程池任务无取消传播——被放弃的请求继续烧线程 + 占 inference 16 个 inflight 槽之一,每个浪费 ~231s 无人接收的工作。修法草图:`_post_retry` 加壁钟总 deadline(从 attempts×read+backoff 推导,clamp 到 < nginx read timeout),两端预算写成互相引用的等式。延期原因:需 WSL compose 实测对齐。**面试考点:超时预算必须自上而下单调递减(入口 > 应用重试链总和 > 单次下游),且没有取消传播时"客户端超时"不等于"服务端停止浪费"。**

**deploy#1 —— "readyz 导流量"只在启动序成立(medium)**。运行期 not-ready 的副本仍会被轮询到:Docker DNS 按运行容器返 IP、不按 health 状态过滤;开源 nginx 无主动健康检查;更隐蔽的是 toolcore 把 `inference_unavailable`/`backend_unavailable` 包成 **HTTP 200** 结构化错误——nginx 的 `proxy_next_upstream http_5xx` 对它彻底失明,被动摘除(max_fails)永不触发。副本本地下游连接坏(进程健在)时,它持续吃 1/N 流量并返回"可重试"错误。修法草图:HTTP service 层对下游不可用类错误以 503 返回(body 不变,agent 契约保持),nginx 现有配置即可 failover + 被动摘除;MCP stdio 路径不动(那里没有 LB,结构化 200 仍是正确契约)。延期原因:方案选型(503 适配 vs 换 Traefik)需实测权衡——下游是共享的,全副本同病时重试纯浪费。**面试考点:LB 健康摘除的信号面(HTTP 状态码)与应用错误契约(结构化 200)冲突时,谁迁就谁;主动 vs 被动健康检查的能力边界。**

**deploy#2 —— readyz 深度不足:exists ≠ 非空(medium)**。`/readyz` 只查 `collection_exists`([src/pharos/service.py:242-243](../../src/pharos/service.py#L242));迁移脚本先建 collection 后灌点,在两步之间被 kill 就留下"存在但 0 点/半库"——readyz 全绿,nginx 导真流量查空库,所有查询返回 `status:empty`(HTTP 200,指标还把 empty 计成功),零报警——正是本项目自己定义的"静默数据损坏"类别。修法两档:轻量 count>0 抽查(60s TTL 缓存,免 N 副本探针高频打 qdrant);治本是迁移改写临时 collection + alias 原子切换,同时消除"空库"与"半库"两个可见窗口。延期原因:需迁移中断场景实测。**面试考点:readiness 探"多深"是个梯度(进程活 < 依赖可达 < 数据就绪);迁移的原子性靠 alias 切换而非"跑完才校验"。**

**deploy#3 —— 重试无 jitter(low)**。退避是确定性的 `backoff*2^attempt`(0.5/1.0s):[src/embedder/remote.py:95](../../src/embedder/remote.py#L95)——inference 闪断/打满时 N 副本的客户端在整点同步重发,semaphore 反复被同一波打满,退避的"错峰"目的落空。值得记的是对抗验证**反驳了原报告的放大系数**:nginx 的上游是 pharos 不是 inference,且 inference 故障经 toolcore 转 200 后 nginx 重试分支不触发——量级是 N×3 而非 N×3×3;被拒请求也不占 inflight 槽,服务端近零成本。所以定级 low。修法一行:退避乘 `random.uniform(0.5, 1.5)`,同步改掉锁死确定性序列的测试([tests/engine/test_remote.py:229](../../tests/engine/test_remote.py#L229))。延期原因:收益需压测验证。**面试考点:thundering herd 与 full jitter;以及"审查报告本身也要被审查"——放大系数错一个乘数,优先级就错一档。**

**deploy#6 —— GPU hang 运行期健康盲区(medium)**。`_readiness` 只由 warmup 写入的 `state.ready/err` 决定;运行期 CUDA 假死(进程活着、持 `gpu_lock` 不返回)不改变任何状态——`/readyz` 永绿、pharos 探它也绿、healthcheck 永绿、restart 不触发。请求逐个烧 120s read-timeout,16 个 inflight 槽被 hang 的前向永久占满后,信号量兜底让后续请求快速 503——服务事实死亡而无自动恢复。修法草图:前向记录"持锁时长 + 最近成功时间戳",超阈值让 `_readiness` 转 `stuck` 503;再配 watchdog 线程超更高阈值 `os._exit(1)` 让 restart 拉起——**这才是该栈里唯一真正的自动恢复通道**(readyz 503 只能摘流量,compose unhealthy 不触发重启)。延期原因:需注入故障实测。**面试考点:liveness 的语义应覆盖"进程活着但永远不再产出"(deadlock/hang);"探针绿"与"在干活"是两个命题。**

五条合起来是一张现成的"生产化 gap 清单":超时对齐、摘除信号、就绪深度、重试风暴、hang 检测——面试里被问"你这套系统上生产还差什么"时,能逐条报出根因、触发条件、修法和为什么还没修,比"都做完了"可信一个量级。

---

## 5. 面试怎么讲

### 30 秒版(电梯稿)

> 我把一个单进程 RAG 服务改造成三层可扩缩架构:唯一吃 GPU 的模型前向拆成独立推理服务,应用层脱 torch 做成 250MB 无状态镜像可以 ×N,嵌入式向量库迁到 server 模式——多副本的真开关其实是后者的单进程文件锁,不是 GPU。等价性靠"服务端全维返回 + 客户端截维"做成结构保证,local 建库 remote 查询向量数学一致。nginx 动态 DNS 轮询加跨副本重试,实测 docker kill 一个副本 50 个持续请求零失败。最后有个诚实结论:多副本扩的是非 GPU 并发、崩溃隔离和滚动升级,吞吐上界仍被单卡 GPU 串行钉死在 ~3.2 req/s——突破它的路径是 vLLM continuous batching,我做了等价性探针,触发条件是队列深度而不是技术时髦。

### 3 分钟版(结构化展开)

1. **问题定义**(20s):单体三个绑死——GPU 模型进程内加载、嵌入式 Qdrant 文件锁独占、应用逻辑陪绑。要多副本、滚动升级、崩溃隔离,三个都得拆。
2. **瓶颈定位**(30s):反直觉的一点——多副本的硬互斥是嵌入式向量库的**单进程文件锁**,拆 GPU 只是必要非充分。所以六阶段计划的骨架是:先修失败模式(重试/退避),再锁等价性(go/no-go gate),再迁 Qdrant server(真开关),最后容器化 + nginx。顺序刚性:在错误的单副本上水平复制 = 复制错误。
3. **两个核心设计**(60s):①边界画在"纯 GPU 前向 vs 业务逻辑"——推理服务端点无任何业务概念,纯到将来能被 vLLM 无痛替换,这个红利后来在 vLLM Plan B 直接兑现(换后端应用层零改动)。②等价性做成结构而非实测:服务端全维返回,客户端 numpy 截维 + renorm,两端走同一份前向代码。等价性 gate 还逼出一个真 bug——bf16 上 normalize 的 norm≈1.002,COSINE 下方向等价所以肉眼永远看不出,修完 E1 cosine=1.0000000。
4. **可用性链条**(40s):docker kill 无感是一条链——nginx 动态 DNS(10s 内感知 scale)、`proxy_next_upstream non_idempotent`(在途 POST 也重试)、客户端 `TransportError` 全谱重试(审查抓出漏了 kill 最典型的 `RemoteProtocolError`/`ReadError` 两个断连形态)、优雅停机 25s<30s 宽限。实测 3 副本 kill 一个,50/50 全 200。对偶设计:客户端有重试,服务端就要有背压(BoundedSemaphore 满则 503),否则重试把过载放大成雪崩。
5. **诚实边界**(30s):吞吐天花板 = 单卡前向串行 ~3.2 req/s,`--scale` 不改变它;扩的是非 GPU 并发 + 崩溃隔离。突破靠 vLLM continuous batching,我用四道 go/no-go 门探过:G1 cosine 0.99956 边缘时不拍板,交 G2 用 88 题真库对拍 top-k(87/88 一致)——判据设在业务影响层而非中间指标。生产化还有五条已确认的 backlog(超时对齐/摘除盲区/就绪深度/重试 jitter/GPU hang 检测),每条有根因和修法草图。

---

## 6. 追问预演

**Q1:为什么不一开始就用 vLLM/TEI,还要自建推理服务?**
要点:等价性归属。自建 = 两端同一份官方前向代码,等价性是结构性质;vLLM/TEI 的 pooling 自实现,等价性变成要实测的赌注(TEI 还没有 vision-language)。且单卡 4090 场景 continuous batching 无收益。关键词:结构性等价 vs 实测等价、迁移触发条件(队列深度持续 >1)、契约无业务概念所以未来可换。数据:G1 实测 vLLM min cosine 0.99956——连官方两个 transformers 版本之间都有微漂,佐证"pooling 等价不是免费的"。

**Q2:多副本为什么不提升 QPS?那扩它干嘛?**
要点:三重上限——GPU 前向串行是最硬上限(单卡 ~3.2 req/s,gpu_lock + 信号量),副本再多前向也排同一条队;副本间无共享可变检索状态,所以扩的是非 GPU 段并发(ACL/RRF/组装/IO)+ 崩溃隔离 + 滚动升级。反直觉但实测:1 副本和 3 副本吞吐几乎相同。别忘了说清跨副本有意降级的状态(session 去重 ~1/N、stats 单副本)和唯一必须一致的状态(共享 sidecar 卷,查询期只读访问)。

**Q3:怎么保证 local 建的库 remote 查不错位?**
要点:全维返回 + 客户端截维 + 同一份前向;`_mrl_np` 下界断言防配置错位;首查握手校验模型身份(服务端换模型 = 同维不同语义空间,靠 `model_dense` 指纹抓);bf16→fp32 normalize 的坑(norm 1.002)说明"数学等价"要落到 dtype 级;E1/E3 实测 + CPU 守护测试。进阶:逐元素等价不蕴含 top-k 不变(HNSW+RRF 放大微差),所以混建混查要端到端对拍。

**Q4:kill 副本无感,具体哪些环节可能断?**
要点:按链讲——DNS 固化(nginx `resolve` + valid=10s 解决)、POST 默认不跨副本重试(non_idempotent,代价是 ask 可能重复计费,显式取舍)、connect 60s 默认太慢(降 2s)、客户端异常谱漏断连形态(RemoteProtocolError/ReadError,类型继承树细节决定整体承诺)、停机 drain(25s<30s)。加分:`restart: unless-stopped` 不拉起手动 kill 的副本——自愈与 failover 是两回事,实测过才知道。

**Q5:你的健康检查设计,有什么坑?**
要点:三个实测案例——探针与业务共享线程池导致"负载越高就绪信号越假"(async 化 + 专用 limiter)、探活超时预算 9s > healthcheck 5s 自误判、nginx 容器 localhost→::1 假 unhealthy。原则:liveness≠readiness(配置错不该 crashloop);err 优先于 loading;探针资源路径/超时/解析都要与业务隔离。再主动补刀:已知盲区是 GPU hang 后 readyz 永绿(deferred),修法是持锁时长阈值 + watchdog 自杀。

**Q6:服务端 503 和客户端重试一起,怎么不雪崩?**
要点:对偶关系——重试必须配准入控制。BoundedSemaphore(16) 把在途钉死,满则立即 503(非阻塞 acquire),深队列的三宗罪(超时后仍烧 GPU、重试再入队、拥塞正反馈)。已知残缺:退避无 jitter,同步重试波(deferred,量级 N×3,一行 full jitter 可修)。

**Q7:迁移嵌入式→server,除了搬数据还要做什么?**
要点:四件事——store 三分支、配置透传全部生产出口(三出口全漏过,守护测试锁死)、数据迁移防中间态(锁前置/源非空校验/with_vectors/count 校验/幂等)、**ACL 语义重测**(嵌入式 fusion 丢顶层 should 的坑,server 语义不同必须新测,重跑旧测试 = 假绿)。安全等价性与向量等价性同级。

**Q8:这套东西离真生产还差什么?**
要点:直接背 §4.2 五条 backlog + 诚实边界:超时预算对齐(361s vs 130s)、LB 摘除对 200 错误失明、readyz 不查非空、重试无 jitter、GPU hang 盲区;再加 K8s(当前 nginx+compose 是学习载体)、inference 鉴权(现靠内网隔离)、跨形态容量数字未回填。能报出"为什么还没修"(需 compose/压测/故障注入环境实测)比"都修了"更可信。

---

## 7. 动手实验

### 实验 1(CPU 可跑):remote 失败模式全谱走一遍

前置:WSL `conda activate pharos && cd ~/projects/pharos && pip install -e '.[dev]'`(未装齐依赖时 pytest 是 collection error,不是真失败)。

```bash
pytest tests/engine/test_remote.py -v          # 27 个测试,纯 CPU-mock,不碰 GPU
pytest tests/engine/test_concurrency.py -v     # 锁下沉(M1)的 9 个守护测试
```

预期看到:503×2 后第 3 次成功、502/504/500 都重试而 4xx 零重试、`ConnectTimeout/ReadError/RemoteProtocolError` 走 TransportError 谱、退避序列恰为 [0.5, 1.0] 指数、负 retries clamp、bf16 进 `_mrl` 后 norm=1.0(fp32 守护 + 反向守卫)、rerank 抛 `InferenceUnavailable` 被降级而 dense loud(非对称锁死)、慢 encode 退避不阻塞其他查询(nullcontext 锁)。这一个文件就是 P0-1/P0-2/M1/bf16 修复史的活文档——建议对照 [src/embedder/remote.py:67-96](../../src/embedder/remote.py#L67) 逐条读。

再用 10 行代码手玩 docker-kill 的断连形态(体会异常谱完备性):

```bash
python - <<'EOF'
import httpx
from embedder.config import EmbedConfig
from embedder import remote
calls = {"n": 0}
class C:  # 模拟 docker kill 后 keep-alive 死连接:每次 POST 都断连
    def post(self, path, json=None):
        calls["n"] += 1
        raise httpx.RemoteProtocolError("Server disconnected")
cfg = EmbedConfig(inference_url="http://x", inference_retries=2, inference_backoff=0.01)
try:
    remote._post_retry(C(), cfg, "/embed", {})
except Exception as e:
    print(type(e).__name__, "attempts=", calls["n"], "marker=", getattr(e, "inference_unavailable", False))
EOF
# 预期输出:InferenceUnavailable attempts= 3 marker= True
# —— kill 的断连被重试吸收,耗尽后抛语义异常,且 toolcore 无需 import embedder 即可识别(duck-typing marker)
```

### 实验 2(需 GPU + WSL + Docker Desktop 开 WSL Integration):kill 无感 50/50 + 吞吐上界亲测

前置:必须**从 WSL 终端内**跑 compose(bind mount 是 WSL ext4 原生路径),`.env.compose` 配好 GPU_UUID/路径/keys(见 [../SCALE_OUT.md](../SCALE_OUT.md) §5-F 环境前置)。

```bash
docker compose --env-file .env.compose up -d --build --scale pharos=3   # 3 副本 + nginx,入口 :8080
# 终端 A:持续 50 请求
for i in $(seq 50); do curl -s -XPOST localhost:8080/v1/retrieve -H 'X-API-Key: <key>' \
  -H 'Content-Type: application/json' -d '{"query":"Netflix 2015 revenues"}' -o /dev/null -w '%{http_code}\n'; done
# 终端 B:中途杀一个副本
docker kill $(docker compose ps -q pharos | head -1)
docker compose ps        # 观察:被 kill 副本 Exited 137 且 restarts=0(unless-stopped 不自愈手动 kill)
```

预期:50/50 全 200、0 失败;三副本日志目录 `/index/pharos_logs/<hostname>/requests.jsonl` 请求分布近均匀。然后验证吞吐上界:

```bash
python scripts/bench.py --url http://127.0.0.1:8080 --key <key> --clients 10 --n 10
docker compose --env-file .env.compose up -d --scale pharos=1 && # 缩回 1 副本重跑对比
```

预期:3 副本与 1 副本吞吐几乎相同(~3 req/s 级,被 GPU 前向串行钉死),p50 随并发线性上升零错误——把 §2.7 的"三重上限"从文字变成自己跑出来的数,也是"vLLM 触发条件为什么是队列深度"的实证。

---

## 8. 诚实边界

面试中主动承认,比被追问出来强得多:

1. **吞吐没有被扩展**。"多副本"听起来像加吞吐,实际 QPS 天花板 = 单卡前向串行(~3.2 req/s,单机嵌入式形态实测;compose 三容器形态的容量数字未回填,跨形态外推有口径风险)。话术:"我能精确说出这套架构扩了什么、没扩什么,以及突破上界的触发条件和验证计划。"
2. **E2 混建混查(local 建 / remote 查的 top-k 序一致性)未做专门端到端对拍**。等价性有结构保证 + E1/E3 逐元素实测 + 全容器链路功能实测,但 SCALE_OUT 风险表仍把"E1 不蕴含 E2"列为最高风险;vLLM 侧反而先做了同方法的 G2(88 题对拍)。话术:"结构保证让我敢延后它,但上生产前它是必补项,方法和脚本都是现成的。"
3. **迁移校验只固化了 count 一层**。向量真在/sidecar 覆盖/ACL fail-closed 三层是会话内手验,未进脚本——脚本退出语自己承认这一点。半库中间态 + readyz 不查非空的组合是已确认的静默损坏窗口(deferred deploy#2)。
4. **五条生产化 backlog 是"确认了但没修"**(§4.2):超时预算脱节、LB 对 200 错误失明、readyz 深度、重试无 jitter、GPU hang 盲区。每条都有根因、触发条件、修法草图和延期理由(需 compose/压测/故障注入环境)——延期是纪律不是懒惰:改动需要真环境验证,盲改就是本仓工程纪律里的"盲目修补"。
5. **单机学习载体的天花板**:nginx+compose 不是 K8s(无主动健康检查、无 autoheal、无 HPA);inference 无鉴权(靠内网 + 不发布端口);SessionRegistry/stats 的跨副本降级是显式接受的;`encode_image` 跨机是 base64 TODO(只建库用、建库同机,影响面小)。
6. **vLLM 只过了等价门,没过收益门**。G4 吞吐对比未跑,所以它还是 Plan B;G1 微漂的根因(transformers 版本差)是"头号嫌疑"而非定论。话术:"我不会拿'基本通过'当'通过'——G2 87/88 里那 1 个 top-1 翻转的样本我知道它长什么样(尾部近重复换序)。"

---

*本篇锚点均已按 2026-07-07 工作区代码逐一核验;引用数据的口径:88 题 gold 基线(勿与历史 72 题混)、单机嵌入式容量表(2026-07-04)、F 阶段多副本实测(2026-07-07)。*
