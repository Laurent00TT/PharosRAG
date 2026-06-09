#!/usr/bin/env bash
# NaviKB local-48G — full-stack bring-up, detached so it survives the launching
# shell (run once from a WSL terminal:  bash start_everything.sh).
# Order matters (don't profile rerank+gen concurrently); each waits for health.
set -uo pipefail
DEPLOY=/mnt/c/Users/11541/Desktop/projects/navikb/deploy/local-48g
LOGS=$HOME/navikb-serving/logs; RUN=$HOME/navikb-serving/run
mkdir -p "$LOGS" "$RUN"

up(){ local port=$1 path=${2:-health}; curl -m2 -s -o /dev/null "http://127.0.0.1:$port/$path" 2>/dev/null; }
launch(){ # name script port [healthpath] [arg]
  local name=$1 script=$2 port=$3 path=${4:-health} arg=${5:-}
  if [ "$port" != "0" ] && up "$port" "$path"; then echo "[$name] already up :$port"; return 0; fi
  echo "[$name] launching ..."
  setsid bash "$DEPLOY/$script" $arg > "$LOGS/$name.log" 2>&1 < /dev/null &
  echo $! > "$RUN/$name.pid"
}
wait_h(){ local name=$1 port=$2 path=${3:-health} t=${4:-300} i=0
  while [ "$i" -lt "$t" ]; do up "$port" "$path" && { echo "[$name] HEALTHY :$port (${i}s)"; return 0; }; sleep 4; i=$((i+4)); done
  echo "[$name] !! not healthy in ${t}s — tail $LOGS/$name.log"; tail -15 "$LOGS/$name.log"; return 1; }

vram(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1 2>/dev/null; }

launch qdrant _start_qdrant.sh 6333 healthz   && wait_h qdrant 6333 healthz 60
launch embed  _start_embed.sh  8003 health    && wait_h embed  8003 health 360; echo "VRAM: $(vram)"
launch sparse _start_sparse.sh 8004 health    && wait_h sparse 8004 health 240
launch mineru _start_mineru.sh 8101 health    && wait_h mineru 8101 health 120
launch rerank _start_rerank.sh 8005 health    && wait_h rerank 8005 health 360; echo "VRAM: $(vram)"
launch gen    _start_gen.sh    8006 health    && wait_h gen    8006 health 360; echo "VRAM: $(vram)"
launch serve  _start_serve.sh  8000 health    && wait_h serve  8000 health 120
launch worker _start_worker.sh 0              # background ingest worker (no health port)
echo "=== NaviKB all-online up ===  final VRAM: $(vram)"
bash "$DEPLOY/_status.sh"
