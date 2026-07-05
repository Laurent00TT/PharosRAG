"""Full parse driver: submit all sampled docs to MinerU across 3 accounts, poll, download.

Groups docs by (api_key, language) -> one batch each (<=50 files). Uploads concurrently,
lets MinerU auto-parse, polls every 15s, downloads each zip into parsed/<doc_id>/.
Resumable: docs already parsed (content_list present) are skipped. Writes parse_results.csv.
"""
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mineru_client as mc

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / (sys.argv[1] if len(sys.argv) > 1 else "sample_manifest.csv")
PARSED = REPO / (sys.argv[2] if len(sys.argv) > 2 else "parsed")
POLL_SECS = 15
MAX_POLL_MIN = 60


def already_done(doc_id):
    d = PARSED / doc_id
    return d.exists() and any(d.rglob("*content_list.json"))


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    todo = [r for r in rows if not already_done(r["doc_id"])]
    print(f"{len(rows)} docs total, {len(rows)-len(todo)} already parsed, {len(todo)} to do", flush=True)

    # group into batches by (key, language)
    groups = {}
    for r in todo:
        groups.setdefault((r["api_key"], r["language"]), []).append(r)

    batches = []  # {key, batch_id, name2doc:{name:doc_id}}
    for (key, lang), grp in groups.items():
        files = []
        for r in grp:
            f = {"name": r["doc_id"] + ".pdf", "data_id": r["doc_id"]}
            if r["page_ranges"]:
                f["page_ranges"] = r["page_ranges"]
            files.append(f)
        print(f"submit batch key={key} lang={lang} files={len(files)}", flush=True)
        batch_id, urls = mc.create_batch(key, files, language=lang)
        # upload concurrently
        def up(args):
            r, url = args
            return mc.upload(url, REPO / r["corpus_path"])
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(up, zip(grp, urls)))
        batches.append({
            "key": key, "batch_id": batch_id,
            "name2doc": {r["doc_id"] + ".pdf": r["doc_id"] for r in grp},
        })
        print(f"  uploaded {len(grp)} files, batch_id={batch_id}", flush=True)

    (REPO / "batches.json").write_text(json.dumps(batches, indent=2))

    # poll all batches until done
    results = {}   # doc_id -> {state, zip_url, err}
    pending = {b["batch_id"]: b for b in batches}
    deadline = time.time() + MAX_POLL_MIN * 60
    while pending and time.time() < deadline:
        time.sleep(POLL_SECS)
        for batch_id, b in list(pending.items()):
            try:
                res = mc.poll_batch(b["key"], batch_id)
            except Exception as e:
                print(f"  poll err {batch_id}: {e}", flush=True); continue
            done_here = 0
            for st in res:
                doc_id = b["name2doc"].get(st["file_name"])
                if not doc_id or doc_id in results:
                    if st["state"] in mc.DONE_STATES:
                        done_here += 1
                    continue
                if st["state"] == "done":
                    results[doc_id] = {"state": "done", "zip_url": st["full_zip_url"], "err": ""}
                    done_here += 1
                elif st["state"] == "failed":
                    results[doc_id] = {"state": "failed", "zip_url": "", "err": st.get("err_msg", "")}
                    done_here += 1
            n_total = len(b["name2doc"])
            print(f"  [{b['key']}] {batch_id[:8]} {done_here}/{n_total} done", flush=True)
            if done_here >= n_total:
                del pending[batch_id]

    # download all done zips concurrently
    done_docs = [(d, r["zip_url"]) for d, r in results.items() if r["state"] == "done"]
    print(f"\ndownloading {len(done_docs)} zips...", flush=True)
    def dl(args):
        doc_id, url = args
        names = mc.download_and_extract(url, PARSED / doc_id)
        return doc_id, names is not None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, ok in ex.map(dl, done_docs):
            print(f"  {'OK ' if ok else 'ERR'} {doc_id}", flush=True)

    # write results csv
    with open(REPO / "parse_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "state", "err"])
        for r in rows:
            res = results.get(r["doc_id"])
            if already_done(r["doc_id"]) and not res:
                w.writerow([r["doc_id"], "done", "(pre-existing)"])
            elif res:
                w.writerow([r["doc_id"], res["state"], res["err"]])
            else:
                w.writerow([r["doc_id"], "timeout", ""])

    ok = sum(1 for r in results.values() if r["state"] == "done")
    fail = sum(1 for r in results.values() if r["state"] == "failed")
    print(f"\nDONE: {ok} ok, {fail} failed, {len(todo)-len(results)} timed-out", flush=True)


if __name__ == "__main__":
    main()
