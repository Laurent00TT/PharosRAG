#!/usr/bin/env bash
# NaviKB local-48G — race-safe ordered bring-up of the full all-online model stack
# on the single 4090. Order matters: do NOT profile rerank + gen concurrently
# (they fight for VRAM during startup -> negative KV). See deployment-guide §9.
#
#   embed(8003) -> sparse(8004,CPU) -> mineru(8101) -> rerank(8005) -> gen(8006)
#
# Each service is launched detached (setsid) with logs in ~/navikb-serving/logs/
# and a pidfile in ~/navikb-serving/run/. Re-running skips already-healthy ports.
set -uo pipefail
DEPLOY="/mnt/c/Users/11541/Desktop/projects/navikb/deploy/local-48g"
LOGS="$HOME/navikb-serving/logs"; mkdir -p "$LOGS"
RUN="$HOME/navikb-serving/run";  mkdir -p "$RUN"

vram(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1 2>/dev/null; }

start_svc(){  # name script port [arg]
  local name="$1" script="$2" port="$3" arg="${4:-}"
  if curl -m2 -s -o /dev/null "http://127.0.0.1:${port}/health" 2>/dev/null; then
    echo "[$name] already healthy on $port — skip"; return 0
  fi
  echo "[$name] launching ($script $arg) ..."
  local tmp="/tmp/svc_$(basename "$script")"
  sed 's/\r$//' "$DEPLOY/$script" > "$tmp"
  setsid bash "$tmp" $arg > "$LOGS/$name.log" 2>&1 < /dev/null &
  echo $! > "$RUN/$name.pid"
}

wait_health(){  # name port timeout_s
  local name="$1" port="$2" t="${3:-300}" i=0
  while [ "$i" -lt "$t" ]; do
    if curl -m2 -s -o /dev/null "http://127.0.0.1:${port}/health" 2>/dev/null; then
      echo "[$name] HEALTHY on $port (after ${i}s)"; return 0
    fi
    sleep 3; i=$((i+3))
  done
  echo "[$name] !!! NOT healthy within ${t}s — tail $LOGS/$name.log:"; tail -25 "$LOGS/$name.log"
  return 1
}

echo "=== bring-up start ($(date)) ===; baseline VRAM (4090): $(vram)"
start_svc embed  _start_embed.sh  8003     && wait_health embed  8003 360 || exit 1; echo "VRAM: $(vram)"
start_svc sparse _start_sparse.sh 8004     && wait_health sparse 8004 240 || exit 1; echo "VRAM: $(vram)"
start_svc mineru _start_mineru.sh 8101     && wait_health mineru 8101 180 || exit 1; echo "VRAM: $(vram)"
start_svc rerank _start_rerank.sh 8005     && wait_health rerank 8005 360 || exit 1; echo "VRAM: $(vram)"
start_svc gen    _start_gen.sh    8006     && wait_health gen    8006 360 || exit 1; echo "VRAM: $(vram)"
echo "=== ALL_ONLINE_UP ($(date)) ==="
echo "final VRAM (4090): $(vram)"
for p in 8003 8004 8005 8006 8101; do
  printf "  :%s health=" "$p"; curl -m3 -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$p/health" 2>/dev/null || echo "ERR"
done
