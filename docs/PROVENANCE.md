# PROVENANCE —— 引擎折入的溯源

本仓的 RAG 引擎(`src/chunker` / `src/embedder` / `src/generator` / `src/pharos/toolcore.py` /
`src/pharos/mcp_stdio.py`)、评测体系(`eval/`)、组件与方法论文档(`docs/components/`、`docs/methodology/`、
`docs/archive/`)**干净拷贝**自引擎仓 `chunk-test-repo`:

- **来源 commit**:`chunk-test-repo@55bf854`(`docs(OVERVIEW): 去'个人'定位;72→88 评估;补 Pharos 产品化指针`)
- **方式**:干净拷贝(非 git subtree/filter-repo)——引擎的逐文件 git 历史**留在 chunk-test-repo**,本仓不携带其 blame。
- **拷贝后引擎仓归档只读**,不再开发。

## 历史 = 边界解除

历史上 pharos 是引擎仓的**薄产品壳**,靠运行时 path-dep 消费同级 chunk-test-repo,受硬规则
**D12「引擎零改动」** 约束把系统劈成两仓。本次迁移**主动解除 D12**,把引擎完整折入,pharos 成为
自包含单仓完整 RAG 系统。旧的 `bootstrap`/`load_toolcore`(commit `7fbf709` 版本守卫)/`PHAROS_ENGINE`/
`RAG_*` 命名空间均已删除,详见各 commit。

## 承载关键修复的文件(其详细 blame 在引擎仓)

这些文件承载 load-bearing 的正确性/安全修复,若需追溯逐行理由,查 `chunk-test-repo` 的历史:

- `src/embedder/acl.py` —— fail-closed ACL 谓词(`acl_admits`/`acl_split`)。
- `src/embedder/store.py` / `src/embedder/retrieve.py` —— 嵌入式 Qdrant「drops top-level should」fail-open bug 的
  RRF 融合下推修复;资产块不参与 section 去重。
- `src/generator/generate.py` —— 资产命中把 `content_raw` 补回喂 LLM(表格/数值 grounding 修复)。
- `src/generator/signals.py` —— smart-ask 与 eval `--smart-tables` 的单一真源(防漂移)。

## 未迁移的东西

- **大型可再生数据**(~3.2GB:`corpus/ parsed/ chunks*/ …` + 已建索引 `~/rag_real`)——留仓外,可从源 PDF 经 MinerU 重建,配置指向(`PHAROS_CORPUS_DIR` / `PHAROS_INDEX_DIR`)。
- **eval 私有产物**(`gold*.jsonl` / `results_*.json` / `verdicts.json` / `baseline_*.json` / `_judge/` / `_units/`)——含研报节选,gitignored,跑一遍脚本即重建。
- 引擎仓的 `index_real.py` / `index_demo.py`(已被 `src/pharos/indexer.py` 产品化取代)。
