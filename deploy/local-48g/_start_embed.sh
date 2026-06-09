#!/usr/bin/env bash
# embed (Channel 1 text + Channel 2 vision) — Qwen3-VL-Embedding-8B FP8, vLLM pooling, port 8003
set -euo pipefail
MC="$HOME/miniconda3"; MODELS="$HOME/navikb-serving/models"
DIR="$MODELS/qwen3-vl-embed-8b"   # official BF16; vLLM quantizes to FP8 dynamically at load
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1   # pin to the 4090 (index 1), NOT the 5070

# --quantization fp8: dynamic W8A8 FP8 at load (Ada sm89 supported). Avoids the
# broken offline-quantization path (AutoModel loaded embedding weights as random).
# vision channel needs the chat template (ships in the BF16 repo).
ARGS=( "$DIR" --served-model-name qwen3-vl-embed --runner pooling
       --quantization fp8
       --port 8003 --gpu-memory-utilization 0.26 --max-model-len 4096 --trust-remote-code )
TPL="$DIR/chat_template.jinja"
[ -f "$TPL" ] && ARGS+=( --chat-template "$TPL" ) || echo "WARN: no chat_template.jinja in $DIR — vision channel may 400"

exec "$MC/bin/conda" run -n vllm --no-capture-output vllm serve "${ARGS[@]}"
