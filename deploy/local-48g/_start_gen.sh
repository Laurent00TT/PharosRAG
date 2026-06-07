#!/usr/bin/env bash
# generation VLM (Channel 4 description + /ask agent + HyDE query-rewrite, 3 roles, 1 model)
# Qwen3-VL-8B-Instruct-FP8, vLLM generate runner, port 8006, served-model-name qwen3.6-vl
set -euo pipefail
MC="$HOME/miniconda3"; MODELS="$HOME/navikb-serving/models"
DIR="$MODELS/qwen3-vl-8b-instruct-fp8"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1   # pin to the 4090
# generation hits flashinfer JIT (sampler + attention) needing nvcc (absent),
# and vllm_flash_attn isn't installed -> use the pure-PyTorch SDPA backend
# (no compile, no nvcc). Drop fp8 kv-cache (that path forces flashinfer).
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# served-model-name kept as "qwen3.6-vl" so navikb DESCRIPTION_MODEL/AGENT_MODEL need no change.
# ctx 6144: single-page description total tokens ~2300 (2.7x headroom); thinking disabled by client.
exec "$MC/bin/conda" run -n vllm --no-capture-output vllm serve "$DIR" \
  --served-model-name qwen3.6-vl \
  --port 8006 --gpu-memory-utilization 0.28 \
  --max-model-len 6144 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --mm-processor-kwargs '{"max_pixels": 401408, "min_pixels": 50176}' \
  --limit-mm-per-prompt '{"image":1}' \
  --enforce-eager --trust-remote-code
