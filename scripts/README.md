# scripts/ —— 解析编排 + 工具

把语料 PDF 变成 `pharos index` 能吃的 MinerU 解析产物(`parsed/<doc_id>/`)的工具链,从引擎仓
干净拷贝而来。**这些是离线/批处理工具,不是运行时组件**;需要 MinerU token(放 `.env`,见 `.env.example`)。

## 解析流水线

```
外部数据集            select_sample.py           parse_batch.py                pharos index
knowledge-base/  ──▶  选样 + 均衡 3 账号   ──▶   MinerU 批量解析 + 轮询下载  ──▶  建索引
  datasets/           写 sample_manifest.csv      parsed/<doc_id>/(content_list+layout)
```

- **`select_sample.py`** — 一次性语料构建:从外部数据集(`knowledge-base/datasets`,不在本仓;
  env `PHAROS_KB_ROOT` 覆盖)分层选样,拷进 `corpus/<type>/`,按有效页数在 3 个 MinerU 账号间均衡,
  写 `sample_manifest.csv`。页数缓存 `config/mmdocir_pagecount.json`。
- **`mineru_client.py`** — MinerU 解析 API 客户端(上传 / 轮询 / 下载 zip)。从 `.env` 读 `MINERU_TOKEN_A/B/C`。
- **`parse_batch.py`** — 读 manifest,驱动 MinerU 自动解析,每 15s 轮询,下载到 `parsed/<doc_id>/`。
  可续跑(已解析的跳过),写 `parse_results.csv`。用法:`python scripts/parse_batch.py [manifest] [parsed_dir]`
  (默认 `sample_manifest.csv` + `<repo>/parsed`;生产建议指到 `PHAROS_CORPUS_DIR`)。
- **`parse_office.py`** — docx/pptx/xlsx 走 MinerU 源码后端(env `MINERU_REPO` 指向 MinerU 源仓),
  解析 `corpus_multiformat/` → `parsed_office/`。
- **`bench.py`** — 检索基准小工具。

## 依赖

`pip install -e .[parse]`(pypdf / requests / python-dotenv)。office 解析额外需 MinerU 源仓。
`corpus/ parsed/ parsed_office/` 等大产物均 gitignored(可再生,留仓外)。

## 待办:`pharos parse` CLI(deferred)

当前解析走上面的独立脚本。把 `parse_batch` 包成一等 `pharos parse` 子命令(读 `PHAROS_CORPUS_DIR` /
`MINERU_TOKEN_*`,输出对齐 `pharos index` 的默认语料目录)是后续产品化项 —— 见 `docs/TODO.md`。
