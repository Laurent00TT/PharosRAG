# docs/components —— 组件深文档(引擎期原文)

`chunker` / `embedder` / `generator` / MCP 各组件的设计/API/评估文档,随引擎折入本仓时**基本保留原文**(只修跨仓死链与命名空间)。

> ⚠ **引擎期引用未随迁**:这些文档写于组件还是独立包的时期,文中提到的 `examples/run_mineru.py`、
> `scratchpad/verify_dense.py`、`scratchpad/diag_acl.py`、`verify_seal4`、`examples/smoke_deepseek.py`、`e2e*.py`
> 等**验证/演示脚本未随迁入本仓**(引擎期产物)。当前对应的可跑入口:
> - 组件单测 → `tests/engine/`(chunker `test_core`/`test_table`、embedder `test_acl`/`test_sparse`/`test_store`/`test_retrieve`、generator `test_generate`/`test_prompt`、MCP `test_tools`)
> - 建库 → `pharos index`;解析 → `scripts/`(见 `scripts/README.md`);端到端 eval → `eval/`(见 `eval/README.md`)
> - 要复现文中那些具体验证脚本的结果,查引擎仓 `chunk-test-repo`(来源见 [../PROVENANCE.md](../PROVENANCE.md))。

文档索引:`chunker/{README,API,ARCHITECTURE,INTEGRATION}` · `embedder/{README,DESIGN,EVALUATION}` ·
`generator/{README,DESIGN}` · `mcp-server.md`。设计谱系另见 [../methodology/](../methodology/)。
