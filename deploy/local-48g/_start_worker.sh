#!/usr/bin/env bash
# NaviKB control plane — background ingestion worker (consumes the SQLite job queue).
# Same runtime CWD as serve so it shares the same SQLite DBs + .env.
set -euo pipefail
MC="$HOME/miniconda3"; REPO="/mnt/c/Users/11541/Desktop/projects/navikb"
RUNTIME="$HOME/navikb-serving/runtime"; mkdir -p "$RUNTIME"; cd "$RUNTIME"
export PYTHONPATH="$REPO/src"
exec "$MC/bin/conda" run -n navikb --no-capture-output \
  python "$REPO/scripts/worker.py"
