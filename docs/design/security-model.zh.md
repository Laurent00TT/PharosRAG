<div align="right">

**中文** | [English](security-model.md)

</div>

# 安全 & 运维模型

> 多数小团队 RAG 项目止步的一层。NaviKB 从这里开始。

## 为什么有这份文档

知识库不只是搜索引擎,它装着真实用户在意的文档。谁能读什么、谁在什么时候
删了什么、怎么拍一致快照备份、怎么从一次失败的 ingest 里恢复 —— 这些是
**运营**关切,不是"有就更好"。本文说明 NaviKB 怎么处理它们。

读者是 (1) 评估能否把文档托付给 NaviKB 的 operator;(2) 思考加新功能时
触及读 / 写 / 删路径的贡献者。

## 威胁模型(当前)

NaviKB 为一个**刻意限制的威胁模型**设计。明确说清楚它防御什么 —— 以及
**不**防御什么 —— 比声称广泛的安全性更重要。

### 在 scope 内

- **可信网络上的不可信 MCP / REST 客户端。** 任何到达 server 的客户端都
  必须出示合法 user token。没有匿名读路径。没有 unauthenticated 的
  legacy fallback。
- **member-vs-admin 权限边界。** Member 只能改自己拥有的文档。Admin-only
  操作(maintenance toggle、reassign、force-purge、include_deprecated
  搜索)被门禁住,以 HTTP 403 干净拒绝。
- **跨进程一致性。** Worker 进程和 server 进程都能写共享 SQLite + Qdrant;
  协调设计上防止静默过期读(cache-epoch 计数器)和静默半完成写
  (两阶段 audit、claim 上的 BEGIN IMMEDIATE)。
- **destructive 操作的 audit 不可抵赖。** Delete 和 reassign 产生两阶段
  audit 行(.attempted + .completed),即便操作中途崩溃也留下可恢复的
  trail。
- **抗 brute-force auth。** 来自同一 source IP 的认证失败超过阈值后聚合成
  `auth.brute_force` audit 行;doctor 露出这些。
- **带 retention 的软删除。** Doc DELETE 是 flag 翻转,不是物理清除。
  retention 内可 restore。GC 只在 retention 明确过期之后才清。

### 故意 out of scope

- **网络层攻击者。** NaviKB 自己不终结 TLS、不实现 reverse-proxy 风格的
  请求整形、自己没 WAF。把它暴露超出 `127.0.0.1` 的 operator 应该放在
  真 reverse proxy 后面(Caddy / nginx / Cloudflare Tunnel),带 TLS +
  (可选)mTLS 或更高层的 auth gate。
- **被入侵的维护者机器。** 如果跑 NaviKB 的主机被入侵了,一切都无所谓:
  攻击者拿到带每个用户 token hash 的 SQLite 文件、每条 audit 行、每份
  文档的向量表示。我们不假装防御这个。
- **LLM 的侧信道攻击。** 来自用户提交文档内容的 prompt injection 是任何
  RAG 系统的真实风险。NaviKB 分区 evidence vs hint(见下面)但**不**声称
  中和 prompt injection —— resource payload 上的安全 header 是 mitigation,
  不是解决方案。
- **多租户强隔离。** NaviKB 有基于 owner_id 的权限但**没有**实现 per-tenant
  SQLite 命名空间、per-tenant Qdrant collection、per-tenant audit。小团队
  形状是"可信 operator,一个组织单元"。SaaS 多租户部署需要不同设计。

## Auth: 硬切换到 user token

进入系统有两条路径: REST 端点和 MCP sub-app(挂在 `/mcp`)。两者都走同一个
`authenticate_token` 函数。没有第三条路径;没有匿名 fallback。

Token 由 `manage_users.py create <username> --role {admin|member}` 创建,
**只 print 一次**到 stdout。明文 token 被 hash 成 Argon2-class 存到 `users`
表。后续认证是请求 token 的 hash 跟存储 hash 的常量时间比较。

### bootstrap refuse-start 不变量

Server 在 users 表为空时**拒绝启动**(带 FATAL 日志行)。理由: 没有至少一个
admin user,没人能认证,但每个端点还会 401 —— 看起来像通用 auth bug。在
启动时大声失败让 misconfiguration 显而易见。

所以第一次 setup 流程是:

```text
1. 初始化数据库(跑任何调 UsersStore.init 的脚本)
2. python scripts/manage_users.py create <名字> --role admin
3. 现在 server 能启动了
```

这个在 quickstart 里(还没公开)有记录,会是代码 ship 之后"我刚装为啥不
work"最常见的来源。

### Token hash 存储

Token 存为 Argon2id hash,不是明文。Argon2id 参数默认 OWASP-推荐值;参数在
hash 里就有版本号(Argon2 标准的 `$argon2id$v=19$m=...$t=...$p=...` 前缀),
未来参数改变可以在下次成功登录时 re-hash,不破坏老 token。

### Brute-force 保护

`auth.failed` 事件按 source IP 在一个进程内计数器(`BruteForceTracker`)里
聚合。同一 IP 在窗口内 N 次失败后,一行 `auth.brute_force` 写到 audit log
(带 IP、count、window)并重置计数器。攻击爆发会产生一行信号丰富的 audit,
不是 N 行低信号的 `auth.failed`。

Tracker 状态是内存里的。进程重启会丢窗口,这意味着攻击者最多能在下次聚合
触发之前多拿 "threshold − 1" 次尝试。我们认为这是不引入 Redis 依赖的合理
trade-off。

## Audit: 三种语义类

Audit log 是 NaviKB 最安全关键的 state。每个触及文档或用户的操作都走三种
契约之一。

### 类 1: best-effort

Action 类型: `auth.failed`、`auth.brute_force`、`authz.denied`、
`doc.ingest`、`job.cancel`、`doc.restore` 等。

这些直接调 `audit_log.write(action, ...)`。写入到 SQLite audit_log 表。
如果 DB insert 失败(transient lock、磁盘满等),产生一个 error trace,失败
被吞掉 —— 主操作继续。意图是绝不让 audit 失败破坏一个非 destructive 操作。

### 类 2: 单阶段强制

Action 类型: `user.create`、`user.disable`、`user.role_change`、
`user.key_reset`。

这些是用户管理操作,audit 行是操作正确性的一部分,不是旁观。它们用
`make_record()` factory + 调用者自己的事务 session,user 表变更和 audit
行一起 commit(或一起 rollback)。Audit 写不了,user 操作也失败 —— 不存在
"user 创建了但没 audit 行"的可能。

### 类 3: 两阶段

Action 类型: `doc.delete`、`doc.reassign`。

这些是 destructive 文档操作,audit trail 必须**即使操作中途失败也要存活**。
契约:

```text
1. audit_log.write_attempted("doc.delete", target_id=..., ...)
     → 写 doc.delete.attempted 行,best-effort
2. 主操作运行(例如 mark_deleted + cache invalidate)
3. audit_log.write_completed("doc.delete", ...)
     → 写 doc.delete.completed 行,强制
        - 先试 SQLite insert
        - 失败时 append 到 JSONL fallback 文件并 fsync
        - 两个都失败时,raise AuditWriteFailure
          → 调用方的 HTTP handler 映射成 503,带文档化的
            "操作完成但 audit 丢失"的 detail 让 admin 手动 reconcile
```

这意味着:

- 如果 `.attempted` 在磁盘但 `.completed` 不在,操作可能发生也可能没 ——
  admin 调查
- 如果两个都在磁盘,操作成功且有 trail
- 如果都不在磁盘,操作从未开始(或在 best-effort 尝试前就失败)

JSONL fallback 在 SQLite DB 崩溃时存活,每行后 fsync 意味着即使断电,至少
一个终态会留在磁盘。

### 两阶段明确**不**做什么

两阶段**不**让操作原子。`.attempted` 和 `.completed` 之间有窗口,崩溃会留下
"尝试过但状态未知"。契约是诚实的: audit log 会告诉你"我们在时间 T 试图删
D-42" —— admin 然后通过看 D-42 的实际状态来 reconcile。

这是小团队栈的正确 trade。真正原子的实现需要要么 SQLite + Qdrant 两阶段
提交(SQLite 没有支持的干净方式),要么意图操作的 write-ahead log(JSONL
fallback 近似就是这个)。

## 软删除: restore + GC 的核心

NaviKB 的 DELETE 是纯 flag 翻转:

```python
# DELETE /documents/{id} 做什么:
1. require_owner_or_admin(doc, user)            # 调用方不对 → 403
2. write_attempted("doc.delete", ...)           # best-effort phase 1
3. meta_db.mark_deleted(doc_id)                 # status=deleted, deleted_at=now
4. invalidate_all_search_caches()               # 本地 + 跨进程 bump
5. emit("document.soft_deleted", ...)           # trace event
6. write_completed("doc.delete", ...)           # 强制 phase 2
```

**关键: DELETE 不动 Qdrant 或图片 store。** 向量留着。图片留着。active-gate
(status != active → 不被检索、不通过 GET 暴露)让 doc 对读路径不可见;
不擦除底层字节。

### 为什么重要

这是让 restore 有意义的唯一设计。如果 DELETE 立刻清向量,"restore" 就是
metadata-only 翻转,留给 Agent 检索的内容已经没了。retention 内保留内容
完整,让 restore 是真的恢复操作:

```python
# POST /documents/{id}/restore 做什么:
1. require_owner_or_admin(doc, user)
2. meta_db.restore_document(doc_id)             # status=active, deleted_at=NULL
3. invalidate_all_search_caches()
4. audit_log.write("doc.restore", ...)          # best-effort, 单行
```

文档立即重新可搜。向量、图片、nav 条目都从没被删过。

### GC: 最终的硬删除

`scripts/gc_deleted_docs.py` 是真正清理的脚本。它:

1. server 报告 maintenance ON 时拒绝运行(避免跟备份脚本竞争)
2. 扫 `documents` where `status='deleted'`
3. 跳过 `deleted_at IS NULL` 的行(pre-T5 数据 —— 没有真值算 retention),
   除非显式传 `--force-purge`
4. 跳过 `(now - deleted_at).days < retention_days`(默认 30)的行
5. 对每个候选,按顺序:
   1. 按 `doc_id` 删 Qdrant 向量
   2. 删 `image_storage_path/<doc_id>/` 下的图片文件
   3. 删该 doc_id 的 NavIndex 条目
   4. DELETE metadata 行

顺序是"软到硬" —— Qdrant 先,metadata 最后 —— 这样部分失败留下下次 GC
能从中恢复的状态,不会孤立数据。

GC 是 operator 调度的,不是自动的。理由在 [architecture.zh.md](architecture.zh.md#设计预想到的失败模式):
静默自动清理是知识库最危险的默认之一。Operator 通过 cron / systemd timer
/ Task Scheduler 明确调度 GC。

## Maintenance: 跨进程暂停

Maintenance flag 是单行 SQLite 表(`maintenance_state`)。Server 和 worker
都读。`on=1` 时:

- Server 的写端点返回 HTTP 503(`require_no_maintenance` FastAPI 依赖)
- Worker 的 `claim_next_item` 返回 None(worker 在 poll interval 之间空转)
- MCP 写工具拒绝执行(返回 `reason: write_tools_disabled`)
- Admin 端点自己**保持可用**(否则你没法把 maintenance 关掉)
- 读路径(GET 端点、MCP 读工具)保持可用

翻 ON 触发 admin 端点里的 drain loop: 等待最多 5 分钟让 active worker
claim(lease 未过期)完成,然后返回 `{on: true, drained: true|false}`。
5 分钟上限存在是为了 HTTP 请求不挂死;如果 `drained: false`,备份脚本
决定是否继续(默认拒绝;传 `--best-effort` 覆盖)。

### 为什么用 SQLite flag 不用 asyncio.Event

进程内信号不能跨 worker-server 边界。Worker 是独立 OS 进程;server 里的
asyncio.Event 永远到不了它。NaviKB 里每个跨进程 state 都为此存在 SQLite。

性能成本是每个写请求一次 indexed PK SELECT —— 小团队写入速率下亚毫秒级。
我们仔细测过,在延迟 profile 里不可见。

## Cache-epoch: 跨进程失效原语

搜索缓存是进程内的 TTLCache(默认 5 分钟 TTL)。Worker 能 invalidate 自己
的缓存;server 能 invalidate 自己的。两者都不能直接 invalidate 对方。

修法: 单行 SQLite `cache_epoch` 表,两个进程都写。Server 把当前 epoch
值折进它的 cache key。Worker 在每次 ingest / deprecate 之后 bump epoch。
epoch 变了,所有旧 cache key 变 stale,下一个请求 miss 并重算。

具体:

```python
# Server 搜索路径
epoch = await cache_epoch_store.get_epoch()    # ~0.1ms PK 查询
key = cache_key(query, ..., epoch=epoch)
cached = local_cache.get(key)
if cached is not None: return cached
# ... 重算 ...

# Worker ingest 路径(成功 commit 之后)
await cache_epoch_store.bump()                 # UPDATE epoch = epoch + 1
```

读路径多一次 SQLite SELECT。便宜且值得。

## 备份: 一致性原语

Operator 驱动的备份用 `scripts/backup_kb.py`。协议:

```text
1. POST /admin/maintenance_mode {on: true}     # 等最多 5 分钟 drain
2. 每个 KB SQLite DB: sqlite3.Connection.backup()
                                              # online backup API,
                                              # 捕获未 checkpoint 的 WAL
3. shutil.copytree(qdrant_path, dest/qdrant,
                   ignore=*.db, *.db-wal, *.db-shm, *.lock, *.jsonl)
                                              # Qdrant 自己的文件,
                                              # 排除我们的 SQLite + lock
4. shutil.copytree(image_storage_path, dest/images)
5. 打开拷贝的 kb_metadata.db 并 reset maintenance_state.on=0
                                              # 这样恢复备份时
                                              # 不会以 maintenance 状态启动
6. 写 manifest.json(ts、schema_version、每文件 sha256)
7. POST /admin/maintenance_mode {on: false}   # 始终(finally 块)
```

**迭代了好几轮才搞对的关键细节:**

- `shutil.copy` 在 WAL-mode SQLite 文件上会漏掉 `-wal` 侧文件里未
  checkpoint 的页。SQLite 的 `Connection.backup()` API 是唯一不
  checkpoint 就能拿到一致 online 快照的方式。
- qdrant_path 目录同时装着我们的 SQLite 库**和** Qdrant 自己的 state。
  `shutil.copytree(ignore=...)` 排除前者,以防它们被双重备份成损坏。
- 备份包含 maintenance 开着时拍的 SQLite 快照。如果不在拷贝里 reset flag,
  恢复的备份会以锁定状态启动。第 5 步不明显但必要。
- `--strict` 是默认,不是 opt-in。`drained: false` 的备份静默继续等于
  写入中途快照。operator 必须显式传 `--best-effort` 接受那个风险。

## Evidence vs Hint: 引用契约

这是安全模型关切,不只是数据模型关切。LLM 生成的页面描述**不是** evidence
—— 把它当作原始证据引用是幻觉放大器。

NaviKB 在每个 Agent 能看见的响应里都分区:

- `evidence_fields`: 解析的文本、图片 URL、figure caption、figure 索引 ——
  parser 从源文档抽取的内容
- `hint_fields`: `generated_description` —— LLM 写关于这页的内容

MCP `evidence_response_to_mcp_payload` formatter 在文本通道里给每个结果
打标签:

- 有 `text_chunk` → `result_type: evidence`,标 "Text preview (evidence)"
- 只有 `vision_description` → `result_type: hint`,标
  "Generated description (RECALL HINT only — not evidence; fetch the
  page resource to cite)"

Agent prompt 调过,只从 `evidence_fields` 引用。如果 Agent 忽略分区引用了
hint,那是下游失败模式 —— 但契约在 API 表面被强制,失败可见。

## Doctor: 运维的 ground truth

`scripts/doctor.py` 是 NaviKB 暴露给 operator 问 "我系统真的一致吗" 的表面。
它**不写入**(它明确以 `readonly=True` 打开 job store 以避免拿 BEGIN
IMMEDIATE 写锁):

- **单机检查**(默认): config 合法性、storage 路径、Qdrant collection 存在性、
  HTTP 服务可达性
- **深度 reconcile**(`--no-reconcile` 跳过):
  - 每个 active 文档在 Qdrant 里至少有一个 text 向量
  - 每个 Qdrant `doc_id` 都有对应 metadata 行
  - 每个 active 文档有 image 目录
  - 没有软删除文档遗留向量(GC 待运行)
  - 没有 image 目录没对应 metadata 行
- **Team view**(`team` 子命令): 每用户 24h 活动、active worker、按 owner
  的队列深度、近期 destructive 操作、带 retention 倒计时的软删除 inventory、
  孤立 owner、maintenance 状态

发现按 OK / WARN / ERROR 分类,带可操作的消息。ERROR 发现包含具体 doc_id
和建议的修复命令(例如 "运行 python scripts/gc_deleted_docs.py")。

## 这一层**不**解决的事

- **网络暴露。** 用 reverse proxy + TLS。NaviKB 设计是本地可信,不是面向
  互联网。
- **文档级 secret。** 每个文档归且仅归一个用户(或 NULL = admin-only);
  我们没有行级 redaction 或字段级加密。别把国家安全文档放进 NaviKB。
- **内部威胁的 audit 篡改检测。** Audit log 在 SQLite 里;有 shell 访问的
  admin 能 `UPDATE audit_log SET ...`,我们不会检测到。Append-only 区块链式
  audit 在小团队栈的 scope 之外。
- **合规认证。** 没有 SOC 2、HIPAA、ISO 27001。Audit trail 的设计支持你想
  朝那个方向 build,但我们不做任何声明。

## 接下来读什么

- [`architecture.zh.md`](architecture.zh.md) —— 这个安全模型所在的四层架构
- [`navigation-first.zh.md`](navigation-first.zh.md) —— auth / audit /
  evidence-hint 分区都支撑的主访问模式
- [`comparison.zh.md`](comparison.zh.md) —— 这个运维深度跟典型 RAG 框架
  对比起来怎样
