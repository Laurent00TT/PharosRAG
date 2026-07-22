# scripts/ —— 语料构建 + 工具(离线/批处理)

产出 `pharos index` 能吃的 MinerU 解析产物的辅助脚本。**离线工具,不是运行时组件**。

## 解析流水线

```
外部数据集            scripts/select_sample.py        pharos parse                 pharos index
knowledge-base/  ──▶  选样 + 均衡 3 账号        ──▶   MinerU 批量解析 + 下载   ──▶  建索引
  datasets/           写 sample_manifest.csv          parsed/<doc_id>/
```

- **`select_sample.py`** — 一次性语料构建:从外部数据集(env `PHAROS_KB_ROOT`)分层选样,拷进 `corpus/<type>/`,
  按有效页数在 3 个 MinerU 账号间均衡,写 `sample_manifest.csv`。页数缓存 `config/mmdocir_pagecount.json`。
- **`parse_office.py`** — docx/pptx/xlsx 走 MinerU 源码后端(env `MINERU_REPO` 指向 MinerU 源仓),
  解析 `corpus_multiformat/` → `parsed_office/`。
- **`bench.py`** — 检索基准小工具。

> **PDF 批量解析已产品化为一等命令 [`pharos parse`](../src/pharos/parser.py)**(原 `parse_batch.py` + `mineru_client.py` 已并入):
> ```bash
> pharos parse [--manifest sample_manifest.csv] [--dest $PHAROS_CORPUS_DIR] [--corpus-root <仓根>]
> ```
> 读 `MINERU_TOKEN_A/B/C`(pharos/.env),按 (账号, 语言) 分批调 MinerU v4 API、并发上传、轮询、下载解压到
> `<dest>/<doc_id>/`,可续跑(已解析跳过)。默认输出对齐 `PHAROS_CORPUS_DIR`,再默认仓根 `parsed/`。

## 依赖

`pip install -e '.[parse]'`(pypdf / requests / python-dotenv)。office 解析额外需 MinerU 源仓。
`corpus/ parsed/ parsed_office/` 等大产物均 gitignored(可再生,留仓外)。
