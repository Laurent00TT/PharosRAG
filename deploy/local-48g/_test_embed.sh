#!/usr/bin/env bash
echo "=== 4090 VRAM (embed FP8 resident) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1
echo "=== /v1/models ==="
curl -m3 -s http://127.0.0.1:8003/v1/models
echo
echo "=== embedding test ==="
curl -m 20 -s http://127.0.0.1:8003/v1/embeddings -H "Content-Type: application/json" -d '{"model":"qwen3-vl-embed","input":"approval threshold"}' > /tmp/emb.json 2>&1
python3 -c "import json; d=json.load(open('/tmp/emb.json')); v=d['data'][0]['embedding']; print('embedding dim =', len(v), '| first3 =', [round(x,4) for x in v[:3]])"
echo "TEST_EMBED_DONE"
