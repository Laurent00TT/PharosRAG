#!/usr/bin/env bash
TOKEN="kb_admin_dCKnAsswHGzxdnvcv0qrkE-fLtFdBDEBXaeYtb0DMcM"
Q="${1:-what is the main contribution of this paper}"
echo "=== search_docs: $Q ==="
curl -m 150 -s http://127.0.0.1:8000/search_docs \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d "{\"query\":\"$Q\",\"top_k\":3}" > /tmp/search.json 2>&1
echo "http bytes: $(wc -c < /tmp/search.json)"
python3 /mnt/c/Users/11541/Desktop/projects/navikb/deploy/local-48g/_search.py
echo "SEARCH_DONE"
