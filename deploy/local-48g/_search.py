import json, sys
d = json.load(open("/tmp/search.json"))
if not isinstance(d, dict):
    print("resp type", type(d), str(d)[:400]); sys.exit()
print("response keys:", list(d.keys()))
res = d.get("results") or d.get("hits") or d.get("documents") or d.get("matches") or []
print("num results:", len(res))
for i, r in enumerate(res[:3]):
    if isinstance(r, dict):
        txt = r.get("text") or r.get("content") or r.get("snippet") or r.get("chunk") or ""
        sc = r.get("score") or r.get("rrf_score") or r.get("rerank_score")
        print(f"  [{i}] score={sc} page={r.get('page')} doc={r.get('doc_id') or r.get('document_id')}")
        print("       " + str(txt)[:180].replace(chr(10), " "))
