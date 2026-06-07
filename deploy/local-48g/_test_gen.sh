#!/usr/bin/env bash
echo "=== health (embed/rerank/gen) ==="
for p in 8003 8005 8006; do printf "  :%s " "$p"; curl -m3 -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$p/health"; done
echo "=== 4090 VRAM ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1
echo "=== gen chat completion test ==="
curl -m40 -s http://127.0.0.1:8006/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-vl","messages":[{"role":"user","content":"Reply with exactly the word: OK"}],"max_tokens":16,"temperature":0}' > /tmp/gen.json 2>&1
python3 -c "import json; d=json.load(open('/tmp/gen.json')); print('gen reply =', repr(d['choices'][0]['message']['content']))"
echo "DONE"
