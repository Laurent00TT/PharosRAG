#!/usr/bin/env bash
echo "=== service health ==="
for hp in "embed 8003" "sparse 8004" "rerank 8005" "gen 8006" "mineru 8101" "serve 8000"; do
  set -- $hp
  printf "  %-8s :%s  " "$1" "$2"
  curl -m3 -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$2/health" 2>/dev/null || echo down
done
printf "  %-8s :%s  " "qdrant" "6333"; curl -m3 -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:6333/healthz" 2>/dev/null || echo down
echo "=== 4090 VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1
echo "STATUS_DONE"
