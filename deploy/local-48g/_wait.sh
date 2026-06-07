#!/usr/bin/env bash
# _wait.sh <port> [timeout_s] — poll /health, report VRAM on the 4090.
PORT="$1"; T="${2:-200}"; i=0
while [ "$i" -lt "$T" ]; do
  c=$(curl -m2 -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  if [ "$c" = "200" ]; then
    echo "PORT $PORT HEALTHY after ${i}s"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1
    exit 0
  fi
  sleep 4; i=$((i+4))
done
echo "PORT $PORT NOT healthy within ${T}s"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1
exit 1
