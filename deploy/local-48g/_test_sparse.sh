#!/usr/bin/env bash
echo "=== health ==="; curl -m5 -s http://127.0.0.1:8004/health; echo
echo "=== embed_query test ==="
curl -m40 -s http://127.0.0.1:8004/embed_query -H "Content-Type: application/json" \
  -d '{"query":"approval threshold escalation"}' > /tmp/sp.json 2>&1
python3 -c "import json; d=json.load(open('/tmp/sp.json')); print('sparse nnz =', len(d.get('indices',[])), '| sample idx', d.get('indices',[])[:5], '| sample val', [round(x,3) for x in d.get('values',[])[:5]])"
echo "DONE"
