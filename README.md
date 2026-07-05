# Pharos(法罗斯)—— 多格式 agentic RAG 系统

> 亚历山大灯塔守在亚历山大图书馆旁,为航船指路。Pharos 为团队的文档库导航。

把 [chunk-test-repo](../chunk-test-repo) 里逐个打磨过的 RAG 组件(chunker / embedder / generator /
mcp_server 工具层)组装成一个**面向小团队内部知识库的完整服务**——多身份鉴权、可观测、systemd 托管,双出口:

| 出口 | 命令 | 消费方 | 模式 |
|---|---|---|---|
| **HTTP API** | `pharos serve` | curl / 脚本 / 前端 | 闭管道问答(`/v1/ask`:检索→grounding→DeepSeek→带引用答案)+ 6 个检索端点 |
| **MCP** | `pharos mcp` | Claude Code 等 agent | agentic RAG(agent 自己决定何时检索/怎么改写/要不要多跳) |

## 架构(一图)

```
                       ┌──────────────────────────────────────────────┐
 curl / 脚本 ─ HTTP ──►│  pharos serve(FastAPI 守护进程,WSL GPU)     │
   /v1/ask /v1/retrieve│  · 独占嵌入式 Qdrant + 常驻 Qwen3-VL 8B       │──► ~/rag_real
                       │  · generator(DeepSeek)供 /v1/ask            │    (qdrant + sidecar,
 Claude Code ─ stdio ─►│  · ACL 身份启动时绑定(fail-closed)          │     77 篇 / 7652 chunk)
   pharos mcp(薄适配器,│  · per-session 去重(X-Pharos-Session)       │
   零 GPU,毫秒启动)  └──────────────────────────────────────────────┘
                                │ 引擎(path-dep)
                        chunk-test-repo:chunker + embedder + generator + mcp_server/toolcore
```

**为什么是守护进程**:嵌入式 Qdrant 单客户端独占锁 + 8B 模型加载 1-2 分钟。老的 stdio 直连模式每个
agent 会话各开一个进程,既抢锁又重付加载;Pharos 让守护进程独占资源,MCP 变成毫秒级启动的 HTTP
薄适配器,多会话共享同一个热后端。详见 [docs/DESIGN.md](docs/DESIGN.md)。

## 快速开始(WSL `navikb` 环境)

服务由 systemd 托管(开机自启 + 崩溃自愈):`systemctl status pharos` / `journalctl -u pharos -f`。
运维、部署、密钥全流程见 [docs/OPERATIONS.md](docs/OPERATIONS.md)。团队成员日常只需拿到自己的
API key,然后:

```bash
conda activate navikb
# 问答(闭管道,带引用):
python -m pharos ask "库里关于 X 的内容有哪些?"
python -m pharos health

# Claude Code agentic:本仓 .mcp.json 已配置(rag → pharos mcp 薄适配器),
# 各成员在 .mcp.json 里填自己的 PHAROS_API_KEY 即可(前提:守护进程在跑)。
```

管理员建库(语料 = MinerU 解析产物目录):

```bash
sudo systemctl stop pharos                                       # 单客户端锁,先停
python -m pharos index --corpus <parsed_dir> --dest ~/rag_real
sudo systemctl start pharos
```

## 部署模式(身份)

三种身份模式,按配置自动选(设计见 [DESIGN D10](docs/DESIGN.md)):

| 模式 | 触发 | 场景 | 身份 |
|---|---|---|---|
| **keys** | 设 `PHAROS_KEYS_FILE` | **团队部署(默认)** | 每请求 X-API-Key → 独立身份(name+tenant+principals) |
| **legacy** | 只设 `PHAROS_API_KEY` | 单人 / 一道门槛 | 单密钥 + 启动绑定身份 |
| **open** | 都不配 | 本地开发,仅回环 | 启动绑定的单身份 |

团队部署(详见 [docs/OPERATIONS.md](docs/OPERATIONS.md)):

```bash
python -m pharos keys new alice --tenant demo --admin      # 发身份,key 只打印一次
python -m pharos keys new bob   --tenant demo --principals g_eng
# .env 里设 PHAROS_KEYS_FILE=~/pharos.keys.json,要局域网访问再加 PHAROS_HOST=0.0.0.0
sudo systemctl restart pharos
```

## 配置

全走环境变量(`.env`,见 [.env.example](.env.example)):`PHAROS_KEYS_FILE`(团队多身份)/
`PHAROS_INDEX_DIR`(默认 `~/rag_real`)/ `PHAROS_COLLECTION` / `PHAROS_PORT`(8787)/ `PHAROS_HOST`
(非回环强制 keys 模式)/ `PHAROS_LOG_DIR`(请求日志)/ `DEEPSEEK_API_KEY`(`/v1/ask` 用)/
`PHAROS_ENGINE`(引擎仓路径,默认同级 `../chunk-test-repo`)。

## 安全模型(必读)

- **身份 = "谁在问"(Pharos 层),ACL = "能看什么"(引擎层)**:X-API-Key 解析成身份,逐请求把
  该身份的 tenant/principals 传给引擎,引擎 ACL 硬过滤兑现可见性(经 acl_regression 五用户矩阵验证)。
- **fail-closed 处处**:未知/缺失 key 一律 401;keys 文件格式错拒绝启动;**非回环绑定强制 keys 模式**
  (不允许把整库裸奔到局域网);未接权限的文档默认拒绝所有人。
- **会话隔离**:跨调用去重登记按 `身份|会话` 隔离,不同用户互不可见。
- **不泄密**:请求日志记身份名不记 key、query 可截断/关闭;错误响应不泄内部细节与文档存在性;
  检索正文标 `trust: untrusted`(防注入)。
- 默认只绑 `127.0.0.1`;HTTPS/公网明确非目标(要远程走隧道)。

## 测试

```bash
python -m pytest tests -q            # 59 项 CPU 单测(fake retriever + MockLLM,不碰 GPU/网络)
```

GPU 冒烟、并发压测、备份演练、安全评审记录见 [docs/TESTING.md](docs/TESTING.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | 目标/架构/关键决策(含否决的备选)/团队版 D10-D12/风险 |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | 模块地图/请求路径/锁模型/与引擎的接缝 |
| [docs/API.md](docs/API.md) | HTTP API 契约(端点/参数/鉴权/status 状态机) |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | **团队运维手册**:部署/密钥/容量实测/备份恢复/故障排查 |
| [docs/TESTING.md](docs/TESTING.md) | 测试矩阵 + 实测证据(CPU/引擎回归/GPU 冒烟/压测/演练/安全评审) |
| [docs/TODO.md](docs/TODO.md) | 路线图:后续候选 + 明确不做 |
| [docs/COMPONENT_NOTES.md](docs/COMPONENT_NOTES.md) | **对既有组件的异议/修复留痕**(供一起 review) |

引擎组件的设计/评估文档在 chunk-test-repo(入口:[docs/OVERVIEW.md](../chunk-test-repo/docs/OVERVIEW.md))。
