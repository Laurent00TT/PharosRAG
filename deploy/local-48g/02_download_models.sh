#!/usr/bin/env bash
# NaviKB local-48G — Phase 1: download models into ~/navikb-serving/models
# Usage: bash 02_download_models.sh [embed|gen|rerank|milco|mineru|all]
# China mirrors: modelscope (official Qwen/MinerU) + hf-mirror (community embed, MILCO).
set -euo pipefail

MC="$HOME/miniconda3"; CONDA="$MC/bin/conda"
MODELS="$HOME/navikb-serving/models"; mkdir -p "$MODELS"
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/" PIP_TRUSTED_HOST="mirrors.aliyun.com"
export HF_ENDPOINT="https://hf-mirror.com"

run(){ "$CONDA" run -n vllm --no-capture-output "$@"; }

ensure_tools(){
  run python -c "import modelscope" 2>/dev/null || run python -m pip install -q modelscope
  run python -c "import huggingface_hub" 2>/dev/null || run python -m pip install -q -U "huggingface_hub[cli]"
}

dl_ms(){ echo ">> modelscope $1"; run modelscope download --model "$1" --local_dir "$2"; }
dl_hf(){ echo ">> hf $1"; run hf download "$1" --local-dir "$2"; }

# embed/rerank: download official BF16 from modelscope, quantize to FP8 locally
# (hf-mirror file CDN is unreachable via the TUN, so the community FP8 repo is out).
do_embed(){  dl_ms Qwen/Qwen3-VL-Embedding-8B          "$MODELS/qwen3-vl-embed-8b"; }
do_gen(){    dl_ms Qwen/Qwen3-VL-8B-Instruct-FP8        "$MODELS/qwen3-vl-8b-instruct-fp8"; }
do_rerank(){ dl_ms Qwen/Qwen3-VL-Reranker-8B           "$MODELS/qwen3-vl-rerank-8b"; }   # BF16 source for FP8 quantization
do_mineru(){ dl_ms OpenDataLab/MinerU2.5-Pro-2604-1.2B "$MODELS/mineru-vlm"; }
# MILCO is HF-only (not on modelscope) — loop hf download with resume to ride out the flaky TUN.
do_milco(){
  export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
  local i
  for i in $(seq 1 15); do
    echo ">> MILCO attempt $i (hf-mirror, resume)"
    if run hf download omai-research/milco-650m --local-dir "$MODELS/milco-650m"; then
      echo "MILCO ok"; return 0
    fi
    echo "MILCO attempt $i failed; retry in 5s"; sleep 5
  done
  echo "MILCO FAILED after retries"; return 1
}

which="${1:-all}"
ensure_tools
case "$which" in
  embed)  do_embed;;  gen) do_gen;;  rerank) do_rerank;;  milco) do_milco;;  mineru) do_mineru;;
  bulk)   do_embed; do_gen; do_rerank; do_mineru;;   # the modelscope-reliable bulk (no MILCO)
  all)    do_embed; do_gen; do_rerank; do_mineru; do_milco;;
  *) echo "unknown target: $which (embed|gen|rerank|milco|mineru|bulk|all)"; exit 1;;
esac
echo "DOWNLOAD_DONE_$which"
