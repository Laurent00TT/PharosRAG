#!/usr/bin/env bash
# mineru (parse, ingest only) — navikb scripts/mineru_server.py in the `mineru` env.
# Usage: _start_mineru.sh [port]   (default 8101)
set -euo pipefail
MC="$HOME/miniconda3"; REPO="/mnt/c/Users/11541/Desktop/projects/navikb"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1   # pin to the 4090
export MINERU_MODEL_SOURCE=modelscope
export MINERU_DEVICE_MODE=cuda
export MINERU_PDF_RENDER_THREADS=1          # pypdfium2 thread race -> SIGABRT (#5033)
export MINERU_HYBRID_BATCH_RATIO=8          # cap hybrid batch to avoid VRAM spike
export HF_ENDPOINT=https://hf-mirror.com
PORT="${1:-8101}"
exec "$MC/bin/conda" run -n mineru --no-capture-output \
  python "$REPO/scripts/mineru_server.py" --host 0.0.0.0 --port "$PORT"
