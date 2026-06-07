#!/usr/bin/env bash
# sparse (Channel 3) — MILCO-650m via navikb scripts/sparse_server.py, port 8004.
# Runs in the `vllm` conda env but FORCED TO CPU (SPARSE_DEVICE=cpu) so it uses
# 0 VRAM — keeps the 48G budget for the three vLLM models.
set -euo pipefail
MC="$HOME/miniconda3"
REPO="/mnt/c/Users/11541/Desktop/projects/navikb"
MODELS="$HOME/navikb-serving/models"

export SPARSE_MODEL="$MODELS/milco-650m"
export SPARSE_DEVICE=cpu
# milco.py hardcodes self.to("cuda") if torch.cuda.is_available() — hide GPUs so
# it stays on CPU (0 VRAM) instead of grabbing the 4090.
export CUDA_VISIBLE_DEVICES=""
# load the two sub-tokenizers offline from local dirs (no HF at request time)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SPARSE_USE_SOURCE_VIEW=true
export SPARSE_DOC_PRUNE_RATIO=0.3
export SPARSE_QUERY_PRUNE_RATIO=0.0
export SPARSE_ENCODE_BATCH=8          # bound the unbounded-batch VRAM/CPU spike (deployment-guide §9)
export HF_ENDPOINT=https://hf-mirror.com

exec "$MC/bin/conda" run -n vllm --no-capture-output \
  python "$REPO/scripts/sparse_server.py" --host 0.0.0.0 --port 8004
