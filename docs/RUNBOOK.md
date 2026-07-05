# RUNBOOK —— 本地启动与运维速查

pharos 跑在 **WSL Ubuntu + conda env `navikb`**(已含 GPU 栈:torch cu128 / transformers / qdrant-client / jieba / mcp)。
下面命令都假设你已 `conda activate navikb` 并 `cd` 到 pharos 仓根。运维细节见 [OPERATIONS.md](OPERATIONS.md)。

## 0. 一次性准备

```bash
conda activate navikb
cd <pharos 仓根>                     # 本机 clone 路径
pip install -e .[dev]                # src-layout 可编辑安装(引擎已折入,import 名不变)
cp .env.example .env                 # 填 DEEPSEEK_API_KEY;示例库用 PHAROS_TENANT=demo
```

> GPU 模型(Qwen3-VL-Embedding-8B / reranker)用 modelscope 下到 `~/models`,首次 `retrieve` 时 lazy 加载(约 1-2 分钟)。
> 非 4090 机器 / 模型放别处:设 `PHAROS_GPU_NAME`(torch device 0 卡名子串,置空=不校验)、`PHAROS_DENSE_MODEL_PATH` / `PHAROS_RERANK_MODEL_PATH`(须含官方 `scripts/`;缺则 `pharos serve` 启动即清晰报错)。

## 1. 启动 / 停止守护进程(独占 GPU + 嵌入式 Qdrant,常驻)

```bash
# 方式 A:systemd(开机自启,推荐;rag MCP 连的就是它)
sudo systemctl start pharos          # 停:stop | 重启:restart | 日志:journalctl -u pharos -f
sudo systemctl status pharos

# 方式 B:手动前台(调试用)
python -m pharos serve               # http://127.0.0.1:8787
```

## 2. 验证

```bash
python -m pharos health              # {status:ok, collection:real, tenant_bound:true, llm_model:..., identity_mode:...}
```

## 3. 日常用

```bash
# 闭管道问答(检索→grounding→DeepSeek→带引用)
python -m pharos ask "库里关于 X 的内容有哪些?"
python -m pharos ask "2021 年净利润多少?" --kind table       # 数值题只在表格块检索
python -m pharos ask "..." --rerank --strategy sparse         # 精排 / 纯关键词选路

# Claude Code agentic:.mcp.json 的 rag 工具 = pharos mcp 薄适配器(秒连热后端)
# 无守护进程时的兜底:自己加载 GPU 模型,首查慢
python -m pharos mcp --direct
```

## 4. 建库(加新文档;需先停守护进程 —— 嵌入式 Qdrant 单客户端锁)

```bash
sudo systemctl stop pharos
# corpus = MinerU 解析产物目录(每篇一个 <doc_type>__<name>/ 含 content_list.json+layout.json)
python -m pharos index --corpus <parsed_dir> --dest ~/rag_real
sudo systemctl start pharos
```

> 从 PDF 生成 parsed/:`pharos parse --manifest sample_manifest.csv --dest <PHAROS_CORPUS_DIR>`
> (MinerU 批量解析,需 `MINERU_TOKEN_*`;清单用 `scripts/select_sample.py` 生成,详见 [../scripts/README.md](../scripts/README.md))。

## 5. 跑 eval(Tier1:DeepSeek 自判,仓内可复现;需停守护进程避 GPU 争用)

```bash
sudo systemctl stop pharos
PHAROS_EVAL_SRC=~/rag_eval_big PHAROS_EVAL_COLLECTION=evalbig \
  python eval/run_eval.py --mode single --judge deepseek --smart-tables --gold eval/gold.jsonl
sudo systemctl start pharos
```

> 旋钮:`--mode single|agentic|decompose|both` `--top-k 6` `--rerank` `--rounds 2` `--limit N`(冒烟)。
> Tier2 权威(双-Claude 异厂裁判)不在仓内可复现,需 Claude Code 多 agent 编排,详见 [../eval/README.md](../eval/README.md)。

## 6. 测试 / 门

```bash
pytest -q                            # CPU 门:179(产品 59 + 引擎 120)
python eval/acl_regression.py        # GPU 发布前门:ACL 0 泄漏(需停守护进程)
```

## 7. 排障速查

| 症状 | 原因 | 处置 |
|---|---|---|
| `ask` 全 `llm_unconfigured` | `.env` 缺 `DEEPSEEK_API_KEY` | 补 key 后 `restart` |
| 检索全空 / `no_identity` | `PHAROS_TENANT` 未设(fail-closed) | 示例库设 `PHAROS_TENANT=demo` |
| `index` 报"已占用" | 守护进程正持锁 | 先 `systemctl stop pharos` |
| Claude Code 里 `rag` 连不上 | 守护进程没起 | `systemctl start pharos`;或 `pharos mcp --direct` |
| eval OOM / 卡 | 守护进程与 eval 抢 GPU | 跑 eval 前先 `stop pharos` |
