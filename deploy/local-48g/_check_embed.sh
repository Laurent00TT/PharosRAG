#!/usr/bin/env bash
for i in $(seq 1 45); do
  code=$(curl -m2 -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8003/health 2>/dev/null)
  if [ "$code" = "200" ]; then echo "EMBED HEALTHY (after ~$((i*4))s)"; break; fi
  sleep 4
done
echo "=== /health ==="; curl -m3 -s -o /dev/null -w 'code=%{http_code}\n' http://127.0.0.1:8003/health
echo "=== 4090 VRAM ==="; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1
echo "=== /v1/models ==="; curl -m3 -s http://127.0.0.1:8003/v1/models; echo
echo "=== embed log tail ==="; tail -18 /home/tiantian/navikb-serving/logs/embed.log
echo "CHECK_EMBED_DONE"
