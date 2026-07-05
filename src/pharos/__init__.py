"""Pharos(法罗斯):自包含多格式 agentic RAG 系统(面向小团队内部知识库)。

亚历山大灯塔守在亚历山大图书馆旁,为航船指路 —— Pharos 为你的私人藏书导航。

单仓完整链路:解析→分块(chunker)→混合检索+rerank+ACL(embedder)→生成+引用(generator),
双出口共用同一检索语义(pharos.toolcore):
  - `pharos serve`:HTTP 守护进程(独占嵌入式 Qdrant + GPU 模型),/v1/ask 闭管道问答 + 6 个检索端点;
  - `pharos mcp`  :零 GPU 依赖的 MCP 薄适配器(stdio→HTTP),给 Claude Code 等 agent 做 agentic RAG
                   (`--direct` 为 no-daemon 兜底,stdio 直连引擎)。
"""
__version__ = "0.3.0"
