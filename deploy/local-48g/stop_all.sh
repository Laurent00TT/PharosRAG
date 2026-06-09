#!/usr/bin/env bash
# Stop all NaviKB local-48G model services (by pidfile process-group, plus a sweep).
RUN="$HOME/navikb-serving/run"
if [ -d "$RUN" ]; then
  for f in "$RUN"/*.pid; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .pid); pid=$(cat "$f" 2>/dev/null || echo)
    [ -n "$pid" ] || continue
    echo "stopping $n (pgid $pid)"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    rm -f "$f"
  done
fi
# sweep any stragglers (full stack)
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "sparse_server.py" 2>/dev/null || true
pkill -f "mineru_server.py" 2>/dev/null || true
pkill -f "scripts/serve.py" 2>/dev/null || true
pkill -f "scripts/worker.py" 2>/dev/null || true
pkill -f "navikb-serving/qdrant/qdrant" 2>/dev/null || true
echo "STOPPED"
