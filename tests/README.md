# tests/ —— 统一测试树

引擎折入 pharos 后,原来「两仓各自 pytest」塌缩成**一套 pytest**(共 179 项)。

## 布局

- `tests/*.py` —— 产品层(service / smart / team / sessions / adapter / review_fixes),纯 CPU(fake retriever + MockLLM,不碰 Qdrant/GPU/网络)。
- `tests/engine/*.py` —— 引擎组件单测,从引擎仓迁入(去掉原 `sys.path.insert`,改用已安装包):
  - **chunker**:`test_core` / `test_table`(读 `tests/engine/fixtures/` 的 MinerU 样例)
  - **embedder**:`test_acl` / `test_sparse` / `test_store` / `test_retrieve`
  - **generator**:`test_generate` / `test_prompt`
  - **MCP**:`test_tools`(经 `pharos.mcp_stdio`,即折入的 stdio 传输)
- `tests/engine/fixtures/` —— chunker 单测的 MinerU 样例(`sample_content_list.json` / `sample_layout.json`)。

## 绿灯门(两档 —— 见不变量 #1)

- **CPU CI 门** = `pytest`(全 179 项;含 embedder `test_acl.py` 的 **ACL 谓词级**断言)。任何 import/store/命名空间改动后必跑。
- **GPU 发布前门** = `python eval/acl_regression.py`(WSL + 4090;44+ 端到端「RRF 融合出口 0 泄漏」断言,**硬依赖 GPU**,不进 CPU CI)。

## 跑

`pip install -e .[dev]` 后 `pytest -q`(navikb 环境已具全部依赖)。契约不漂移由
`tests/test_review_fixes.py::test_transports_contract_no_drift` 把门(HTTP 适配器与 stdio 两传输六工具 docstring 同文 + `_INSTRUCTIONS` 同源自 `toolcore`)。
