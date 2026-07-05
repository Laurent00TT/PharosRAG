"""embedder: RAG embed + 检索组件。

Qwen3-VL-Embedding-8B(dense,多模态)+ BM25/jieba(sparse)+ Qdrant(hybrid + ACL 硬过滤)。
接 chunker 的 Chunk[];chunker 的 image_path/acl/source_indices/doc_type 在此兑现。"""
from .acl import acl_admits, acl_split
from .config import EmbedConfig
from .embed import Embedder
from .rerank import Reranker
from .retrieve import Retriever
from .types import Hit, User

__all__ = ["EmbedConfig", "Hit", "User", "Embedder", "Retriever", "Reranker", "acl_admits", "acl_split"]
