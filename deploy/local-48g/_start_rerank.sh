#!/usr/bin/env bash
# rerank (search re-ranking) — Qwen3-VL-Reranker-8B FP8, vLLM pooling, port 8005
# THREE must-haves or it silently ranks garbage (vllm #35412):
#   1) --chat-template   2) classifier_from_token ["no","yes"]   3) is_original_qwen3_reranker:true
set -euo pipefail
MC="$HOME/miniconda3"; MODELS="$HOME/navikb-serving/models"
DIR="$MODELS/qwen3-vl-rerank-8b"   # official BF16; vLLM quantizes to FP8 dynamically at load
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1   # pin to the 4090

ARGS=( "$DIR" --served-model-name qwen3-vl-rerank --runner pooling
       --quantization fp8
       --hf_overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
       --port 8005 --gpu-memory-utilization 0.28 --max-model-len 4096 --trust-remote-code )
TPL="$DIR/chat_template.jinja"
[ -f "$TPL" ] && ARGS+=( --chat-template "$TPL" ) || echo "WARN: no chat_template.jinja in $DIR — rerank will rank garbage without it"

exec "$MC/bin/conda" run -n vllm --no-capture-output vllm serve "${ARGS[@]}"
