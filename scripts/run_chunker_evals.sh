#!/bin/bash
# Run two evals back-to-back on the default-chunker collection.
set -e
cd "$(dirname "$0")/.."

# Probe for a usable Python in this order:
#   1. .venv/bin/python          — POSIX venv (Linux/macOS/WSL)
#   2. .venv/Scripts/python.exe  — Windows venv (Git Bash, MSYS)
#   3. python3 / python on PATH  — system fallback
# This keeps the script portable across the team's typical shells without
# forcing everyone to symlink one location to another.
if [ -x "./.venv/bin/python" ]; then
    PY="./.venv/bin/python"
elif [ -x "./.venv/Scripts/python.exe" ]; then
    PY="./.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
elif command -v python >/dev/null 2>&1; then
    PY="python"
else
    echo "ERROR: no Python interpreter found (.venv missing and no python on PATH)" >&2
    exit 1
fi
echo "Using interpreter: $PY"

echo "===== Eval 1/2: default chunker + 4 channels ====="
PYTHONIOENCODING=utf-8 "$PY" scripts/eval_search.py \
  --testset datasets/mmdocir/subset_eval.jsonl \
  --top-k 5 \
  --qdrant-path ./qdrant_data_mmdocir_default \
  --snapshot eval_mmdocir_default_baseline.json
echo

echo "===== Eval 2/2: default chunker --no-desc ====="
PYTHONIOENCODING=utf-8 "$PY" scripts/eval_search.py \
  --testset datasets/mmdocir/subset_eval.jsonl \
  --top-k 5 \
  --qdrant-path ./qdrant_data_mmdocir_default \
  --no-desc \
  --snapshot eval_mmdocir_default_no_desc.json
echo

echo "===== Both evals complete ====="
