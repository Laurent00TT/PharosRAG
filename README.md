# Pharos(法罗斯)—— 个人多格式 agentic RAG 系统

> 亚历山大灯塔守在亚历山大图书馆旁,为航船指路。Pharos 为你的私人藏书导航。

把 [chunk-test-repo](../chunk-test-repo) 里逐个打磨过的 RAG 组件(chunker / embedder / generator /
mcp_server 工具层)组装成一个**可日常使用的完整产品**,双出口:

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

```bash
cp .env.example .env                 # 默认 PHAROS_TENANT=demo,匹配 ~/rag_real 建库身份
conda activate navikb

# 1) 起守护进程(独占索引;首次 retrieve 时加载 dense 模型,~20s-2min)
#    日常已由 systemd 托管(开机自启+崩溃自愈):systemctl status pharos / journalctl -u pharos -f
#    手动跑(python -m pharos serve)仅调试用 —— 先 sudo systemctl stop pharos(索引单客户端锁)
python -m pharos serve

# 2) 问答(另一个终端;闭管道,带引用)
python -m pharos ask "Netflix 2015 年的营收是多少?"
python -m pharos health

# 3) Claude Code agentic:本仓 .mcp.json 已配置(rag → pharos mcp 薄适配器),
#    在 pharos 目录开 Claude Code 即可;前提是守护进程在跑。
```

建库(语料换成你自己的 MinerU 解析产物目录):

```bash
python -m pharos index --corpus <parsed_dir> --dest ~/rag_real   # 先停 serve(单客户端锁)
```

## 配置

全走环境变量(`.env`,见 [.env.example](.env.example)):`PHAROS_TENANT`(必需,fail-closed)/
`PHAROS_PRINCIPALS` / `PHAROS_INDEX_DIR`(默认 `~/rag_real`)/ `PHAROS_COLLECTION` / `PHAROS_PORT`(8787)/
`PHAROS_API_KEY`(可选接入门槛)/ `DEEPSEEK_API_KEY`(`/v1/ask` 用)/ `PHAROS_ENGINE`(引擎仓路径,
默认同级 `../chunk-test-repo`)。

## 安全模型(必读)

- **ACL 身份启动时绑定**,HTTP 客户端 / agent 不能经参数改;`PHAROS_TENANT` 未设 → 一切检索 fail-closed 返回空。
- **部署即授权**:能连上端口的人 = 能看到该身份可见的内容。默认只绑 `127.0.0.1`;要暴露到局域网,设 `PHAROS_API_KEY`。
- 检索正文一律标 `trust: untrusted`(数据不是指令,防 prompt 注入);错误响应不泄内部细节与文档存在性。

## 测试

```bash
python -m pytest tests -q            # 25 项 CPU 单测(fake retriever + MockLLM,不碰 GPU/网络)
```

GPU 冒烟与实测记录见 [docs/TESTING.md](docs/TESTING.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | 目标/架构/8 项关键决策(含否决的备选)/风险 |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | 模块地图/请求路径/锁模型/与引擎的接缝 |
| [docs/API.md](docs/API.md) | HTTP API 契约(端点/参数/status 状态机) |
| [docs/TESTING.md](docs/TESTING.md) | 测试矩阵 + 实测证据(CPU/引擎回归/GPU 冒烟) |
| [docs/TODO.md](docs/TODO.md) | 版本规划:v0.1 已完成清单 + 后续路线 |
| [docs/COMPONENT_NOTES.md](docs/COMPONENT_NOTES.md) | **对既有组件的异议/修复留痕**(供一起 review) |

引擎组件的设计/评估文档在 chunk-test-repo(入口:[docs/OVERVIEW.md](../chunk-test-repo/docs/OVERVIEW.md))。
