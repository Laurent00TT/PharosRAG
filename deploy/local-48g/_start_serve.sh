#!/usr/bin/env bash
# NaviKB control plane — FastAPI tool server (REST + /mcp) on :8000.
# CWD = runtime dir on ext4 (SQLite coordination must NOT live on /mnt/c 9p).
set -euo pipefail
MC="$HOME/miniconda3"; REPO="/mnt/c/Users/11541/Desktop/projects/navikb"
RUNTIME="$HOME/navikb-serving/runtime"; mkdir -p "$RUNTIME"; cd "$RUNTIME"
export PYTHONPATH="$REPO/src"
exec "$MC/bin/conda" run -n navikb --no-capture-output \
  python "$REPO/scripts/serve.py" --host 0.0.0.0 --port 8000
