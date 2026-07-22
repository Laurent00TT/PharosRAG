<div align="center">

<img src="docs/assets/pharos-banner.svg" alt="Pharos —— 为团队的文档库导航" width="880">

# Pharos

![Python](https://img.shields.io/badge/Python-3.10+-F5B84A?style=flat-square&labelColor=0B1E3A)
![RAG](https://img.shields.io/badge/RAG-agentic%20%2B%20闭管道-6FA8DC?style=flat-square&labelColor=0B1E3A)
![Exits](https://img.shields.io/badge/出口-HTTP%20%2B%20MCP-F5B84A?style=flat-square&labelColor=0B1E3A)
![Docs](https://img.shields.io/badge/学习文档-12%20篇-1D9E75?style=flat-square&labelColor=0B1E3A)

**把 PDF、扫描件、docx、pptx、xlsx 变成一个能问答、带企业级权限、答案可溯源的本地知识库。**
几个 RAG 组件各自打磨到位,再收进同一个仓库,拼成一套能直接交给小团队用的服务:多身份鉴权、可观测、systemd 托管都是现成的。

</div>

---

> 亚历山大灯塔守在亚历山大图书馆旁,为往来的航船指路。**Pharos 做的是同一件事,只不过照亮的是你团队的文档库。**

这不是又一个切块玩具。每个关键决策背后都有实测撑着,也经过好几轮对抗评审来回推敲,如今已经在 77 篇真实文档、7652 个 chunk 上完整跑通。装起来只要一句 `pip install -e '.[dev]'`(src-layout,可编辑安装)。

## 两个出口,同一套语义

同一个知识库,朝两类使用者各开一扇门:

| 出口 | 命令 | 谁来用 | 它做什么 |
|---|---|---|---|
| **HTTP API** | `pharos serve` | curl、脚本、前端 | 闭管道问答:`/v1/ask` 走「检索 → grounding → DeepSeek → 带引用的答案」,外加 6 个检索端点 |
| **MCP** | `pharos mcp` | Claude Code 这类 agent | agentic RAG:何时检索、怎么改写、要不要多跳,全交给 agent 自己判断 |

两扇门背后是同一套语义,靠两条铁律钉死:工具契约只有一个来源(`toolcore`,stdio 和 HTTP 两侧不会漂),身份由服务端说了算(agent 改不动自己的权限)。

## 架构

<div align="center">
<img src="docs/assets/architecture.svg" alt="Pharos 架构:消费方 → 守护进程 → 后端服务" width="100%">
</div>

**为什么做成常驻的守护进程?** 嵌入式 Qdrant 只认一个客户端、独占一把锁,8B 模型光加载就要一两分钟。早先的 stdio 直连模式下,每开一个 agent 会话就起一个进程,既抢锁又要把模型重新加载一遍。Pharos 换了个思路:让守护进程独占这些重资源,MCP 退化成一个毫秒级启动的 HTTP 薄适配器,所有会话共用同一个已经热起来的后端。这背后的取舍见 [docs/DESIGN.md](docs/DESIGN.md)。

<details>
<summary><b>生产多副本形态:从单机走到三层可独立扩缩</b></summary>

<br/>

单节点的问题在于,GPU 模型、嵌入式 Qdrant、应用逻辑全绑在一个进程里,想横向扩就撞天花板。拆开之后是三层,各自能独立扩缩:

- **`inference`**(FastAPI `:8900`)—— 把 GPU 前向单独拎出来,返回全维向量,客户端再截断,两边保持等价;
- **`pharos`** —— 应用层甩掉 torch(配了 `inference_url` 就不再加载模型),于是能 `--scale pharos=N` 起多副本;
- **Qdrant server 模式** —— 多副本共享同一个真源;前面架一层 `nginx` 做负载均衡,`docker kill` 掉某个副本也无感。

```bash
docker compose --env-file .env.compose up -d --scale pharos=3   # 入口 http://127.0.0.1:8080
```

有一点要说清楚:**吞吐的天花板是单卡 inference 的前向速度(GPU 串行),加副本并不会把它抬高**;多副本真正扩的是非 GPU 部分的并发,以及崩溃隔离、滚动升级这些好处。完整过程见 [docs/SCALE_OUT.md](docs/SCALE_OUT.md)。

</details>

## 上手

服务交给 systemd 托管,开机自启、崩了自愈。团队成员日常要做的,就是拿到自己那把 API key,然后:

```bash
conda activate pharos
pip install -e '.[dev]'                        # src-layout,可编辑安装

python -m pharos ask "库里关于 X 的内容有哪些?"   # 闭管道问答,答案带引用
python -m pharos health
```

要在 Claude Code 里用 agentic 模式:把 `.mcp.json.example` 复制成 `.mcp.json`(已 gitignore),填上自己的 `PHAROS_API_KEY` 和本机路径,就能拿到 `rag` 工具(前提是守护进程在跑)。手边没有守护进程时,`pharos mcp --direct` 也行——它 stdio 直连、自己加载 GPU 模型,不依赖常驻服务。

管理员建库(语料就是 MinerU 的解析产物目录):

```bash
sudo systemctl stop pharos                                     # 单客户端锁,得先停
python -m pharos index --corpus <parsed_dir> --dest ~/rag_real
sudo systemctl start pharos
```

## 想弄懂 RAG,而不只是用它?

这个仓库还捎带一套 **[RAG 学习 + 面试文档](docs/learning/)**:把这个被数据反复打脸、又一次次修回来的系统,拆成 **12 篇、4122 行**的教程。从切块、混合检索、权限模型、生成 grounding,一路讲到 agentic、评估方法论、多副本扩展——每篇都配了代码锚点、实测数据、面试话术,还有能自己上手跑的实验。

| 想学什么 | 读哪几篇 |
|---|---|
| RAG 每一层的取舍,以及主流方案都长什么样 | [01 全景](docs/learning/01-rag-overview.md) · [02 切块](docs/learning/02-parsing-chunking.md) · [03 检索](docs/learning/03-retrieval.md) |
| 企业级权限、生成 grounding、agentic | [04 ACL](docs/learning/04-acl-security.md) · [05 生成](docs/learning/05-generation-grounding.md) · [06 Agentic](docs/learning/06-agentic-mcp.md) |
| RAG 到底该怎么评(这套系统最花心思的地方) | [07 评估方法论](docs/learning/07-evaluation.md) |
| 系统设计、单机到多副本、工程方法论 | [08 服务化](docs/learning/08-service-architecture.md) · [09 扩展](docs/learning/09-scale-out.md) · [10 故事集](docs/learning/10-methodology-stories.md) |
| 面试临阵磨枪 | [11 面试题库](docs/learning/11-interview-qa.md) |

## 安全模型

- **身份管「谁在问」,ACL 管「能看到什么」。** X-API-Key 先解析成身份,再逐请求把这个身份的 tenant 和 principals 交给检索层,由 ACL 做硬过滤兑现可见性(这条链路经 `acl_regression` 的五用户矩阵验证过)。
- **处处 fail-closed。** key 认不出或没带,一律 401;keys 文件格式不对,直接拒绝启动;绑到非回环地址就强制走 keys 模式(不许把整个库裸奔到局域网);没挂权限的文档,默认谁都看不到。
- **会话之间互相隔离。** 跨调用去重按「身份 | 会话」分桶,不同人看不见彼此。**也不漏底细**:日志只记身份名、不记 key;报错不吐内部细节;检索回来的正文一律标 `trust: untrusted`,防提示注入。
- 默认只绑 `127.0.0.1`;HTTPS 和公网访问明确不在目标内,真要远程就走隧道。

<details>
<summary><b>三种身份模式,以及配置项</b></summary>

<br/>

| 模式 | 怎么触发 | 用在哪 | 身份怎么来 |
|---|---|---|---|
| **keys** | 设了 `PHAROS_KEYS_FILE` | **团队部署(默认)** | 每个请求凭 X-API-Key 得到独立身份(name + tenant + principals) |
| **legacy** | 只设了 `PHAROS_API_KEY` | 单人用,或只想加道门槛 | 单密钥,启动时绑定身份 |
| **open** | 两个都不设 | 本地开发,只走回环 | 启动时绑定的那一个身份 |

```bash
python -m pharos keys new alice --tenant demo --admin      # 发身份,key 只打印这一次
python -m pharos keys new bob   --tenant demo --principals g_eng
# .env 里设 PHAROS_KEYS_FILE=~/pharos.keys.json;要局域网访问再加 PHAROS_HOST=0.0.0.0
sudo systemctl restart pharos
```

配置都收在仓根那一个 `.env` 里(见 [.env.example](.env.example),统一 `PHAROS_*` 前缀):`PHAROS_KEYS_FILE`、`PHAROS_CORPUS_DIR`、`PHAROS_INDEX_DIR`(默认 `~/rag_real`)、`PHAROS_COLLECTION`、`PHAROS_PORT`(8787)、`PHAROS_HOST`(绑非回环会强制 keys 模式)、`PHAROS_LOG_DIR`、`DEEPSEEK_API_KEY`(`/v1/ask` 要用)。部署、密钥、运维的完整流程见 [docs/OPERATIONS.md](docs/OPERATIONS.md)。

</details>

## 测试

```bash
python -m pytest tests -q                          # 全套 CPU 单测:产品层 + 引擎层,含 ACL 谓词
python -m pytest tests -q --ignore=tests/engine    # 只跑产品层(fake retriever + MockLLM,不碰 GPU、不联网)
```

GPU 端到端的 0-leak 回归(`eval/acl_regression.py`,跑在 WSL + 4090)、并发压测、备份演练、安全评审记录,都在 [docs/TESTING.md](docs/TESTING.md)。

## 文档索引

| 文档 | 讲什么 |
|---|---|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | **系统总览(权威入口)**:全貌、组件地图、这套文档怎么读 |
| [docs/learning/](docs/learning/) | **RAG 学习 + 面试文档(12 篇)**:从切块讲到评估,带代码锚点和面试话术 |
| [docs/DESIGN.md](docs/DESIGN.md) | 目标、架构、关键决策(含被否掉的备选)、多身份与可观测、风险 |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | 模块地图、请求路径、锁模型、检索层接缝 |
| [docs/API.md](docs/API.md) | HTTP API 契约:端点、参数、鉴权、status 状态机 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | **团队运维手册**:部署、密钥、容量实测、备份恢复、故障排查 |
| [docs/SCALE_OUT.md](docs/SCALE_OUT.md) | 单机走到多副本(阶段 A–F):三层拆分、负载均衡、等价性验证 |
| [docs/TESTING.md](docs/TESTING.md) | 测试矩阵加实测证据:单测、ACL 回归、GPU 冒烟、压测、演练、安全 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 往哪走、以及**明确不做**什么(附理由) |
| [docs/COMPONENT_NOTES.md](docs/COMPONENT_NOTES.md) · [docs/PROVENANCE.md](docs/PROVENANCE.md) | 对既有组件的异议留痕、来历与口径断代 |

组件级的设计与评估在 [docs/components/](docs/components/)(chunker / embedder / generator / mcp-server),方法论在 [docs/methodology/](docs/methodology/)。

<div align="center">
<br/>
<sub>Pharos · 自建的多格式 agentic RAG · 本机 4090 / WSL <code>pharos</code></sub>
</div>
