"""Pharos(法罗斯):多格式 agentic RAG 服务(面向小团队内部知识库)。

亚历山大灯塔守在亚历山大图书馆旁,为航船指路 —— Pharos 为你的私人藏书导航。

双出口,共用同一检索引擎(chunk-test-repo:chunker/embedder/generator/mcp_server.toolcore):
  - `pharos serve`:HTTP 守护进程(独占嵌入式 Qdrant + GPU 模型),/v1/ask 闭管道问答 + 6 个检索端点;
  - `pharos mcp`  :零 GPU 依赖的 MCP 薄适配器(stdio→HTTP),给 Claude Code 等 agent 做 agentic RAG。
"""
__version__ = "0.1.0"
