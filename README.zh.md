<div align="right">

**中文** | [English](README.md)

</div>

# NaviKB

**Navigable Knowledge Base** —— 面向个人与小团队的 navigation-first、local-first 知识库。

> 🔬 **状态：研究 / 示例 —— 早期 alpha。** 代码已在此且能跑，但这是一个*参考实现*，
> 不是开箱即用的产品。请预期粗糙之处：干净机器上的安装尚未经第二人独立复现，评测体系
> 仍在完善。诚实的「能用什么、不能用什么、风险在哪」见 [docs/status.zh.md](docs/status.zh.md)。

---

## 是什么

NaviKB 围绕主流 RAG 栈做错的两件事来构建：

1. **导航，而非检索，才是首要访问模式。** 多数 RAG 把文档当成一袋扁平 chunk，靠 dense
   相似度找正确文本。NaviKB 把每篇文档的*结构*（标题、表格、图、页区间）索引成一个
   pointer-only 的 **NavIndex**，让 agent 像人类读者一样在结构上行走。
2. **生成内容不是证据。** LLM 对某页的描述只是一个召回提示。把它当原文引用是幻觉放大器。
   NaviKB 把每个响应切分为 `evidence_fields`（解析文本、图 URL、caption）与 `hint_fields`
   （LLM 生成的描述），并告诉 agent 哪些可以引用。

NaviKB 面向个人与小团队工作的真实形态：几百到几千份 PDF、少数几个操作者、强烈偏好本地运行、
不依赖 SaaS。

## 不是什么

- **不是托管服务。** 一切运行在你的机器或局域网。
- **不是向量搜索的包装。** 检索是四通道之一（text dense / text sparse / vision /
  description），经 RRF 融合后重排 —— 但始终锚定在可导航的结构上。
- **不是知识图谱。** 无实体抽取、无本体归纳、无 LLM 构建的关系边。图就是文档自己的目录。
- **不是企业平台。** 无多租户编排、无 Kubernetes operator、无 Celery / Redis / Kafka。
  SQLite + Qdrant + 一个 worker 进程就是全部技术栈。

## 为什么是这个形态

这个设计是对 2026 年 RAG 两个走偏趋势的回应：

- **走向更重的基础设施** —— 托管向量库、图数据库、agent 平台、可观测栈。对有专职 ops 的
  企业很好，对个人和 2-5 人团队则不友好。
- **远离文档结构** —— 把一切切块、嵌入，指望相似度能找到。丢掉了让真实文档可导航的那些
  affordance（章节、图、表、"见第 3 章"）。

NaviKB 在两点上都取相反立场：小体积、深度尊重结构、经 MCP 原生面向 agent。

详细形态见 [docs/design/architecture.zh.md](docs/design/architecture.zh.md)；把导航设为首要访问
模式的设计理由见 [docs/design/navigation-first.zh.md](docs/design/navigation-first.zh.md)。

## 模型与部署是参考，不是要求

NaviKB 把每个模型都解耦在可配置端点之后（`.env` 里的 `*_SERVER_URL` / `*_MODEL_ID`）。
我们恰好验证过的那套栈 —— Qwen3-VL 做文本+视觉嵌入与重排、MILCO 做学习式稀疏检索、
Qwen3.6-VL 做页面描述、MinerU 做版面解析 —— 只是**一种参考配置**，不是硬依赖。

把端点指向你手头任何兼容服务：本地 llama.cpp / LM Studio、经 SSH 隧道的远程 GPU 主机、
或某个 OpenAI 兼容的托管 API，引擎都不在乎。这个仓库交付的是架构、四通道检索、导航层、
以及操作面（鉴权、审计、软删除、备份、GC）—— 具体模型可替换，部署拓扑是示例而非强制。

## 快速开始

> 这是研究/示例栈。下面的命令是起点，不是硬化的安装路径 —— 复现性见
> [状态说明](docs/status.zh.md)。

```powershell
# 1. 安装（Python 3.11+）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[server,dev]"
.\.venv\Scripts\python.exe -m pip install -e ".[mineru]"   # PDF 版面解析（可选）

# 2. 配置 —— 复制模板，然后把 *_SERVER_URL / *_MODEL_ID 指向你自己的模型端点
#    （见上面「模型与部署是参考」）。
Copy-Item .env.example .env

# 3. 起本地控制面
python scripts/serve.py --port 8000     # FastAPI tool server（REST + MCP）
python scripts/worker.py                 # 后台 ingestion worker

# 或用 docker compose 一并起 main + Qdrant + worker：
# docker compose -f docker/docker-compose.local.yml --env-file .env up -d --build
```

服务默认绑定 `127.0.0.1`。创建管理员账号，然后摄取一份 PDF —— NaviKB 解析 PDF 版面，
所以吃 `.pdf` 文件。本仓库不附带任何第三方 PDF；自带你的，或用内置的**合成**源
（`datasets/pdf_corpus_v1/synthetic_sources/`，见 `scripts/pdf_corpus_*.py`）构建样例 PDF：

```powershell
python scripts/manage_users.py create <username> --role admin
python scripts/ingest.py --path .\path\to\document.pdf
```

所有 API 请求须带 `Authorization: Bearer <user_token>` 头。搜索，或让 agent 基于检索证据回答问题：

```powershell
curl -X POST http://127.0.0.1:8000/search_docs `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:KB_USER_TOKEN" `
  -d "{\"query\":\"审批阈值\",\"top_k\":5}"

curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:KB_USER_TOKEN" `
  -d "{\"question\":\"审批阈值是多少？\"}"
```

本地健康检查：`python scripts/doctor.py`。MCP 工具面挂载在 `/mcp`。

## 可选的工作台

一个 Web 工作台（React + Vite）在独立仓库
**[navikb-workbench](https://github.com/Laurent00TT/navikb-workbench)**。它是 core 的
`/ui/api` 面的薄 HTTP 客户端 —— 不共享代码。先起这个 core，再对着它跑工作台（其 dev server
把 `/ui/api` 代理回 `127.0.0.1:8000`）。没有它，core 也能完全 headless 运行。

## 差异化对比

| 关注点 | NaviKB | 典型 RAG（LangChain / LlamaIndex） | GraphRAG | NaviRAG（研究） |
|---|---|---|---|---|
| 首要访问模式 | 指针导航 + 四通道检索 | 扁平 chunk 上的 dense 检索 | 在 LLM 抽取实体上做图游走 | 层级式 LLM 驱动探索 |
| Local-first | ✅ 默认 | ⚠ 可能但不常见 | ❌ 通常需 Neo4j 级基础设施 | ❌ 研究阶段 |
| MCP 原生 | ✅ 一等工具面 | ⚠ 经 adapter | ❌ | ❌ |
| 引用纪律 | ✅ 强制 evidence/hint 切分 | ❌ 调用方自决 | ❌ | ❌ |
| 操作面（鉴权/审计/备份/GC） | ✅ 端到端 | ❌ "自己造" | ⚠ 仅企业栈 | ❌ |
| 状态 | 研究 / 示例，早期 alpha | 成熟 | 成熟 | 研究原型 |

每一格的详细论证 —— 以及一节诚实的「NaviKB 哪里更弱」—— 见
[docs/design/comparison.zh.md](docs/design/comparison.zh.md)。

## 文档地图

| 文档 | 用途 |
|---|---|
| [README](README.zh.md)（你在这） | 30 秒概览 + 快速开始 |
| [docs/status.zh.md](docs/status.zh.md) | 诚实的成熟度：能用什么、不能用什么、未决风险 |
| [docs/design/architecture.zh.md](docs/design/architecture.zh.md) | 四层架构：ingest → storage → retrieve → serve |
| [docs/design/navigation-first.zh.md](docs/design/navigation-first.zh.md) | 为何 NavIndex；pointer-only 设计；与 chunking 的取舍 |
| [docs/design/security-model.zh.md](docs/design/security-model.zh.md) | 硬切换鉴权、两阶段审计、软删除、维护标志、GC 保留 |
| [docs/design/comparison.zh.md](docs/design/comparison.zh.md) | 与 LangChain / LlamaIndex / GraphRAG / NaviRAG 的深度对比 |
| [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) | 欢迎什么反馈与贡献 |

## 快速链接

- 📋 [状态与诚实成熟度](docs/status.zh.md)
- 🏛 [架构](docs/design/architecture.zh.md)
- 🖥 [工作台（独立仓库）](https://github.com/Laurent00TT/navikb-workbench)
- 💬 [讨论：开 issue](../../issues/new/choose) —— 设计反馈、对比纠错、文档清晰度
- 🤝 [贡献](CONTRIBUTING.zh.md)
- 🔒 [安全策略](SECURITY.md)
- 📜 [License: Apache-2.0](LICENSE)

## 命名

**NaviKB** = **Navi**gable + **K**nowledge **B**ase。

读作 "nah-vee-K-B"（三音节）或 "navy-K-B"（偷懒的两音节）。把项目定位在 2026 年
*navigable knowledge base* 研究方向（见
[NaviRAG](https://arxiv.org/abs/2604.12766)、
["Don't Retrieve, Navigate"](https://arxiv.org/abs/2604.14572)），但聚焦于不同的一层：
那些是检索算法研究原型；NaviKB 是这类设计底下的生产-运维栈。
