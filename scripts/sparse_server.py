# scripts/sparse_server.py
"""MILCO sparse embedding service — FastAPI wrapper.

Model: omai-research/milco-650m (Apache 2.0)
Reference:
  - internal design notes (not published) §3.2
  - MILCO security audit: internal design notes (not published) entry [2026-05-22]

Two endpoints reflect MILCO doc/query pruning asymmetry, which is
encoded into embedding_version per spec §2. Any change to pruning
ratio or source_view setting MUST bump embedding_version and re-ingest
all documents — runtime fallback is forbidden (spec §8.2).

Start (default port 8004 — aligns with Settings.sparse_server_url):
  pip install transformers torch fastapi uvicorn
  python scripts/sparse_server.py --port 8004

Verify:
  curl http://localhost:8004/health
  curl -X POST http://localhost:8004/embed_query \
    -H "Content-Type: application/json" \
    -d '{"query": "test query"}'

Env config (env wins over CLI defaults for container-friendly deploy):
  SPARSE_MODEL                  default omai-research/milco-650m
  SPARSE_MODEL_REVISION         default main (PRODUCTION MUST PIN to a sha)
  SPARSE_DOC_PRUNE_RATIO        default 0.3
  SPARSE_QUERY_PRUNE_RATIO      default 0.0
  SPARSE_USE_SOURCE_VIEW        default true
  SPARSE_DEVICE                 default auto (cuda if available, else cpu).
                                Set to "cpu" to force CPU on resource-constrained
                                hosts. PoC deploys with GPU should leave default —
                                MILCO query embed is 5-10x faster on GPU, batch
                                doc embed 10-15x faster; ~3.6GB VRAM at idle.
  HF_ENDPOINT                   set to https://hf-mirror.com for China users
"""
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    import uvicorn
except ImportError as e:
    raise SystemExit(
        "Install server extras: pip install transformers torch fastapi uvicorn pydantic\n"
        f"{e}"
    )


# Config from env (with sensible defaults)
MODEL_ID = os.environ.get("SPARSE_MODEL", "omai-research/milco-650m")
MODEL_REVISION = os.environ.get("SPARSE_MODEL_REVISION", "main")
DOC_PRUNE_RATIO = float(os.environ.get("SPARSE_DOC_PRUNE_RATIO", "0.3"))
QUERY_PRUNE_RATIO = float(os.environ.get("SPARSE_QUERY_PRUNE_RATIO", "0.0"))
USE_SOURCE_VIEW = os.environ.get("SPARSE_USE_SOURCE_VIEW", "true").lower() == "true"


def _resolve_device() -> str:
    """SPARSE_DEVICE env > auto-detect. Default to GPU when available since
    MILCO is small (~1.3GB BF16) but inference is 5-15x faster on GPU."""
    env_device = os.environ.get("SPARSE_DEVICE", "").lower().strip()
    if env_device in ("cuda", "cpu"):
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _resolve_device()

# Lazy-loaded model
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    from transformers import AutoModel
    logger.info(
        "Loading MILCO: %s @ %s (source_view=%s, doc_prune=%.2f, query_prune=%.2f)",
        MODEL_ID, MODEL_REVISION, USE_SOURCE_VIEW, DOC_PRUNE_RATIO, QUERY_PRUNE_RATIO,
    )
    if MODEL_REVISION == "main":
        logger.warning(
            "MODEL_REVISION='main' — fine for dev, but PRODUCTION should pin "
            "to a specific commit sha (see MILCO security audit recommendation)."
        )
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        revision=MODEL_REVISION,
        dtype="auto",
    )
    _model = _model.to(DEVICE)
    # PyTorch inference mode: train(False) is equivalent to model.eval()
    # (sets dropout/batchnorm to inference behavior; not Python eval)
    _model.train(False)
    logger.info("MILCO loaded on device=%s", DEVICE)
    return _model


def _mass_based_prune(
    indices: list[int], values: list[float], prune_ratio: float,
) -> tuple[list[int], list[float]]:
    """Mass-based pruning: drop tokens whose combined mass <= prune_ratio of total.

    prune_ratio = 0.3 means drop the bottom 30% of mass (keep top 70%);
    prune_ratio = 0.0 means no pruning.

    Returns (indices, values) sorted by weight descending (sparse vectors
    are unordered for Qdrant, but desc-sort makes manual inspection nicer).
    """
    if prune_ratio <= 0.0 or not values:
        return list(indices), list(values)

    paired = sorted(zip(indices, values), key=lambda x: x[1], reverse=True)
    total_mass = sum(v for _, v in paired)
    target_mass = total_mass * (1.0 - prune_ratio)

    kept_indices: list[int] = []
    kept_values: list[float] = []
    accumulated = 0.0
    for idx, val in paired:
        kept_indices.append(int(idx))
        kept_values.append(float(val))
        accumulated += val
        if accumulated >= target_mass:
            break
    return kept_indices, kept_values


def _encode_to_qdrant(
    texts: list[str], prune_ratio: float, source_view: bool,
) -> list[dict]:
    """Encode texts via MILCO, return list of {indices, values} dicts.

    MILCO's encode_text returns a [batch_size, vocab_size] COO sparse
    tensor. We bucket by row, apply pruning, and serialize to Qdrant
    sparse-vector format (indices + values lists).
    """
    model = _load_model()
    # Sub-batch guard (2026-05-30). MILCO encodes the ENTIRE `texts` list in a
    # single forward, so one big document's chunks (often 1000s) spike the
    # activation VRAM to ~15GB — the unbounded-batch peak that pushed the
    # all-online 4-model layout (embed+sparse+rerank+qwen36+mineru) over the
    # 96GB card to OOM. Cap each forward at SPARSE_ENCODE_BATCH (default 48)
    # via recursion → peak ~3GB regardless of doc size, total GPU peak
    # 97.25→89.9GB. Do NOT raise this back to "whole doc" without re-checking
    # the GPU budget. See internal design notes (not published).
    _SUB = int(os.environ.get("SPARSE_ENCODE_BATCH", "48"))
    if len(texts) > _SUB:
        _out: list[dict] = []
        for _s in range(0, len(texts), _SUB):
            _out.extend(_encode_to_qdrant(texts[_s:_s + _SUB], prune_ratio, source_view))
        return _out
    # torch.inference_mode is stricter than no_grad (also disables view
    # tracking + version counter) → less reserved memory per forward pass.
    with torch.inference_mode():
        sparse_tensor = model.encode_text(
            texts, return_dict=False, source_view=source_view,
        ).coalesce()

    batch_size = sparse_tensor.size(0)
    indices_2d = sparse_tensor.indices()  # shape [2, nnz]
    values_1d = sparse_tensor.values()
    rows = indices_2d[0]
    cols = indices_2d[1]

    # Bucket nonzeros by row
    buckets_idx: list[list[int]] = [[] for _ in range(batch_size)]
    buckets_val: list[list[float]] = [[] for _ in range(batch_size)]
    nnz = len(values_1d)
    for i in range(nnz):
        r = rows[i].item()
        buckets_idx[r].append(int(cols[i].item()))
        buckets_val[r].append(float(values_1d[i].item()))

    # Apply mass-based pruning per text
    result: list[dict] = []
    for idx_list, val_list in zip(buckets_idx, buckets_val):
        pruned_idx, pruned_val = _mass_based_prune(idx_list, val_list, prune_ratio)
        result.append({"indices": pruned_idx, "values": pruned_val})

    # Release the encoded sparse tensor + PyTorch caching allocator's
    # peak memory back to the OS. Without this, MILCO's per-batch
    # intermediate tensors (which can spike to several GB for large
    # batch sizes) stay reserved by PyTorch indefinitely, accumulating
    # to 30+ GB over a long-running ingest session. Verified 2026-05-27:
    # sparse process grew from 2.8 GB (idle) to 35.6 GB after a few
    # hours of ingest traffic, starving other GPU services.
    del sparse_tensor, indices_2d, values_1d, rows, cols
    torch.cuda.empty_cache()

    return result


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eager-load on startup so first request doesn't pay the cold-load cost."""
    _load_model()
    yield


app = FastAPI(title="MILCO Sparse Embedding Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if _model is not None else "loading",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_view": USE_SOURCE_VIEW,
        "doc_prune_ratio": DOC_PRUNE_RATIO,
        "query_prune_ratio": QUERY_PRUNE_RATIO,
    }


class DocBatchRequest(BaseModel):
    texts: list[str]


@app.post("/embed_doc")
async def embed_doc(req: DocBatchRequest):
    """Doc-side embedding with sparse_doc_prune_ratio applied."""
    if not req.texts:
        return {"embeddings": []}
    embeddings = _encode_to_qdrant(req.texts, DOC_PRUNE_RATIO, USE_SOURCE_VIEW)
    return {"embeddings": embeddings}


class QueryRequest(BaseModel):
    query: str


@app.post("/embed_query")
async def embed_query(req: QueryRequest):
    """Query-side embedding with sparse_query_prune_ratio applied.

    Returns the single sparse embedding directly (not wrapped in a list).
    """
    if not req.query:
        raise HTTPException(status_code=400, detail="query field required")
    embeddings = _encode_to_qdrant(
        [req.query], QUERY_PRUNE_RATIO, USE_SOURCE_VIEW,
    )
    return embeddings[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
