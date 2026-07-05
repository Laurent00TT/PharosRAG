"""embedder 配置。组件跑在 WSL `navikb` conda 环境(GPU=4090,按名锁定——见 dense.py)。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# sidecar JSON schema 版本。写侧盖章、读侧校验:不符(含缺失)即抛错要求重新 index_document。
# **改 sidecar 结构(Element/Section 字段、acl_index 编码等)时必须 +1**,否则旧 sidecar 会被静默
# 反序列化成错数据(轻则 small-to-big 取空,重则按位置错位)。把"静默错"变"响亮失败"。
SIDECAR_VERSION = 1


@dataclass
class EmbedConfig:
    # --- dense: Qwen3-VL-Embedding-8B(本地,modelscope 下到 ~/models)---
    dense_model_path: str = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
    dense_dim: int = 1024                  # MRL 截断维度;1024 起步省 4x 存储,按召回实测再调(上限 4096)
    query_instruction: str = "Retrieve relevant documents for the query."
    gpu_name_must_contain: str = "4090"    # FASTEST_FIRST 编号会变 -> 按名断言锁 4090(不是 5070)

    # --- sparse: BM25(jieba 中文分词)---
    stopwords: frozenset[str] = field(default_factory=frozenset)   # 可注入中文停用词

    # --- Qdrant(嵌入式起步;规模大了换 server 模式)---
    qdrant_path: str = os.path.expanduser("~/qdrant_data")
    collection: str = "rag_chunks"

    # --- sidecar:per doc_id 存 elements+sections+acl_index(retrieve 期 assemble_big small-to-big 必需,
    #     Qdrant 只存 chunk 向量,原始 element/section 不进库)---
    sidecar_dir: str = os.path.expanduser("~/rag_sidecar")

    # --- rerank: Qwen3-VL-Reranker-8B(cross-encoder 精排,可选;与 embedding 同源)---
    rerank_model_path: str = os.path.expanduser("~/models/Qwen3-VL-Reranker-8B")
    rerank_instruction: str = "Given a search query, retrieve relevant candidates that answer the query."
    rerank_top_n: int = 50                 # 对 hybrid 召回的前 N 个候选做 cross-encoder 重排

    # --- 检索 ---
    prefetch_limit: int = 50               # 每路(dense/sparse)召回多少进 RRF 融合
    top_k: int = 8
