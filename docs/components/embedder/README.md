# embedder

RAG 流水线的 **embed + hybrid 检索**组件(接 `chunker` 的 `Chunk[]`)。
`parse → chunk → **embed → retrieve**`。

## 架构

- **dense**:`Qwen/Qwen3-VL-Embedding-8B`(多模态,文本与图像同一 4096 空间)。复用模型自带
  `scripts/qwen3_vl_embedding.py` 的 `Qwen3VLEmbedder`(last-token pooling),不自写。MRL 截到 1024
  (实测几乎无损)。唯一吃 GPU,按名锁定 4090。
- **sparse**:**BM25**(jieba 中文分词 + 正则保精确串如 `gpt-4`/`42000000` + FNV 稳定 hash → uint32,
  打分交 Qdrant `Modifier.IDF`)。零模型、纯 CPU。
- **向量库**:**Qdrant**(嵌入式起步)。named dense+sparse,`query_points` 用 RRF 融合。
- **ACL 硬过滤**(安全边界):fail-closed,租户隔离 +「allow ANY OR public」。**filter 下推到每个
  prefetch**(嵌入式 Qdrant 在 fusion 下会丢顶层 `query_filter` 的 should —— 见 `docs/DESIGN.md` §7#4)。
- **rerank**(可选):`Qwen3-VL-Reranker-8B` cross-encoder 对 hybrid 召回 top-N 重排(实测 MRR 0.566→0.867);`search(rerank=True)` 开启,+16G 显存,默认关。
- **small-to-big**:检索期接 `chunker.assemble_big`,ACL 感知(只取 user 有权看到的原文)。

## 模块

| 模块 | 职责 |
|---|---|
| `config.py` | 配置(模型/维度/Qdrant/sidecar/停用词) |
| `dense.py` | Qwen3-VL 封装(GPU) |
| `sparse.py` | BM25(CPU) |
| `store.py` | Qdrant collection / upsert / hybrid + ACL 硬过滤 |
| `acl.py` | ACL 客户端逻辑(`acl_split` 索引拆字段 / `acl_admits` 检索谓词),与 store 同语义 |
| `embed.py` | `Embedder`:chunk 分流 embed + payload + per-doc sidecar |
| `retrieve.py` | `Retriever`:hybrid 召回 + 可选 rerank + dedup + small-to-big |
| `rerank.py` | `Reranker`:Qwen3-VL-Reranker-8B cross-encoder 精排(可选,GPU) |
| `types.py` | `User` / `Hit` |

## 用法

```python
from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir
from embedder import EmbedConfig, Embedder, Retriever, User

cfg = EmbedConfig()                                   # 默认 ~/models, ~/qdrant_data, ~/rag_sidecar
elements = from_mineru_dir("parsed/<doc>")
result = Chunker().chunk(elements, doc_id="d1", doc_type="academic_paper", lang="en", acl=acl)

emb = Embedder(cfg)
emb.index_document("d1", elements, result, image_root="parsed/<doc>")

# 同进程内复用 store+dense(嵌入式 Qdrant 单 client + 不重 load 8B)
ret = Retriever(cfg, store=emb.store, dense=emb.dense)
for r in ret.search_with_context("...query...", User(tenant="t1", principals=["g_research"])):
    print(r["hit"].text, r["context"].text)
```

## 环境

WSL Ubuntu conda env `navikb`(torch 2.8.0+cu128,GPU=4090)。依赖见 `requirements.txt`。
dense 模型用 modelscope 下到 `~/models/Qwen3-VL-Embedding-8B`(~16GB)。

## 状态

已过 **R1–R5 对抗评审**(R1 ACL 0 发现 / R2 检索修 6 个窗口块状态 / R4 资产 content_raw 计入预算等);`eval/acl_regression.py`
44+ 断言全过(含"禁出口 acl_admits 后 RRF fusion 仍 0 泄漏",证 prefetch 下推本身挡越权)。单测 **34 passed**:
`test_sparse.py`(精确串)、`test_store.py`(ACL 越权不可召回)、`test_acl.py`(ACL 谓词)、`test_retrieve.py`(窗口块状态/出口 ACL/sidecar 版本/去重降级)。
系统级端到端评估见 [OVERVIEW §7](../../OVERVIEW.md)。可选增强(非阻塞):BM25 vs BGE-M3-sparse 权重扫描、est_tokens 接 Qwen3-VL tokenizer。
