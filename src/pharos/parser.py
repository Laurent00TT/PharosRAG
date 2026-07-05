"""`pharos parse`:把语料 PDF 经 MinerU 批量解析成 `parsed/<doc_id>/`(供 `pharos index` 建库)。

产品化自引擎期 scripts/{parse_batch,mineru_client}.py。读 manifest CSV(scripts/select_sample.py 生成),
按 (api_key, language) 分批调 MinerU v4 API,并发上传、轮询、下载 zip 解压到 dest/<doc_id>/。可续跑
(已解析的跳过)。MinerU token 从 pharos/.env 读(MINERU_TOKEN_A/B/C,与其余配置同一 .env)。

流程(https://mineru.net/apiManage/docs):POST file-urls/batch -> PUT 各 presigned url(裸二进制,无 Content-Type,
自动开始解析)-> GET extract-results/batch/{id} 轮询 -> 下载 full_zip_url 解压。
"""
from __future__ import annotations

import csv
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

BASE = "https://mineru.net"
DONE_STATES = {"done", "failed"}


def _tokens() -> dict:
    config.load_env()   # 载入 pharos/.env -> MINERU_TOKEN_*
    return {k: os.environ.get(f"MINERU_TOKEN_{k}") for k in ("A", "B", "C")}


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _create_batch(token, files, language=None, model_version="vlm", enable_formula=True, enable_table=True):
    """files: [{"name","data_id", optional "page_ranges"}]。返回 (batch_id, file_urls)(顺序对齐 files)。"""
    import requests
    body = {"files": files, "model_version": model_version,
            "enable_formula": enable_formula, "enable_table": enable_table}
    if language:
        body["language"] = language
    r = requests.post(f"{BASE}/api/v4/file-urls/batch", headers=_headers(token), json=body, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"create_batch failed: {j}")
    return j["data"]["batch_id"], j["data"]["file_urls"]


def _upload(file_url, path, retries=3) -> bool:
    """PUT 文件到 presigned URL(务必不带 Content-Type)。"""
    import requests
    for attempt in range(retries):
        try:
            with open(path, "rb") as f:
                r = requests.put(file_url, data=f, timeout=600)
            r.raise_for_status()
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! upload failed {Path(path).name}: {e}", flush=True)
                return False
            time.sleep(2 * (attempt + 1))
    return False


def _poll_batch(token, batch_id) -> list:
    import requests
    r = requests.get(f"{BASE}/api/v4/extract-results/batch/{batch_id}", headers=_headers(token), timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"poll failed: {j}")
    return j["data"].get("extract_result", [])


def _download_and_extract(zip_url, dest_dir, retries=3):
    import requests
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.get(zip_url, timeout=600)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(dest_dir)
            return sorted(p.name for p in dest_dir.iterdir())
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! download failed {zip_url}: {e}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))
    return None


def _already_done(dest: Path, doc_id: str) -> bool:
    d = dest / doc_id
    return d.exists() and any(d.rglob("*content_list.json"))


def run_parse(manifest: str, dest: str, corpus_root: str,
              poll_secs: int = 15, max_poll_min: int = 60) -> int:
    """按 manifest 批量解析 -> dest/<doc_id>/。corpus_root 解析 manifest 里相对的 corpus_path。返回成功篇数。"""
    manifest_p, dest_p, corpus_p = Path(manifest), Path(dest), Path(corpus_root)
    if not manifest_p.exists():
        raise SystemExit(f"manifest 不存在:{manifest_p}(用 scripts/select_sample.py 生成 sample_manifest.csv)")
    tokens = _tokens()
    rows = list(csv.DictReader(open(manifest_p, encoding="utf-8")))
    todo = [r for r in rows if not _already_done(dest_p, r["doc_id"])]
    print(f"{len(rows)} docs total, {len(rows) - len(todo)} already parsed, {len(todo)} to do", flush=True)
    if not todo:
        print(f"全部已解析 -> {dest_p}", flush=True)
        return 0
    missing = sorted({r["api_key"] for r in todo} - {k for k, v in tokens.items() if v})
    if missing:
        raise SystemExit(f"缺 MINERU_TOKEN_{' / MINERU_TOKEN_'.join(missing)}(放 pharos/.env)。")

    # 分批(按 key + 语言),建 batch + 并发上传
    groups: dict = {}
    for r in todo:
        groups.setdefault((r["api_key"], r["language"]), []).append(r)
    batches = []
    for (key, lang), grp in groups.items():
        files = []
        for r in grp:
            f = {"name": r["doc_id"] + ".pdf", "data_id": r["doc_id"]}
            if r.get("page_ranges"):
                f["page_ranges"] = r["page_ranges"]
            files.append(f)
        print(f"submit batch key={key} lang={lang} files={len(files)}", flush=True)
        batch_id, urls = _create_batch(tokens[key], files, language=lang)

        def _up(pair):
            r, url = pair
            # corpus_path 可能是 Windows 反斜杠写法 -> 规范化,兼容 WSL/Linux
            return _upload(url, corpus_p / r["corpus_path"].replace("\\", "/"))
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_up, zip(grp, urls)))
        batches.append({"key": key, "batch_id": batch_id,
                        "name2doc": {r["doc_id"] + ".pdf": r["doc_id"] for r in grp}})
        print(f"  uploaded {len(grp)} files, batch_id={batch_id}", flush=True)

    # 轮询到完成
    results: dict = {}
    pending = {b["batch_id"]: b for b in batches}
    deadline = time.time() + max_poll_min * 60
    while pending and time.time() < deadline:
        time.sleep(poll_secs)
        for batch_id, b in list(pending.items()):
            try:
                res = _poll_batch(tokens[b["key"]], batch_id)
            except Exception as e:
                print(f"  poll err {batch_id}: {e}", flush=True)
                continue
            done_here = 0
            for st in res:
                doc_id = b["name2doc"].get(st["file_name"])
                if not doc_id or doc_id in results:
                    if st["state"] in DONE_STATES:
                        done_here += 1
                    continue
                if st["state"] == "done":
                    results[doc_id] = {"state": "done", "zip_url": st["full_zip_url"]}
                    done_here += 1
                elif st["state"] == "failed":
                    results[doc_id] = {"state": "failed", "zip_url": "", "err": st.get("err_msg", "")}
                    done_here += 1
            print(f"  [{b['key']}] {batch_id[:8]} {done_here}/{len(b['name2doc'])} done", flush=True)
            if done_here >= len(b["name2doc"]):
                del pending[batch_id]

    # 并发下载解压
    done_docs = [(d, r["zip_url"]) for d, r in results.items() if r["state"] == "done"]
    print(f"\ndownloading {len(done_docs)} zips -> {dest_p} ...", flush=True)

    def _dl(pair):
        doc_id, url = pair
        return doc_id, _download_and_extract(url, dest_p / doc_id) is not None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for doc_id, ok in ex.map(_dl, done_docs):
            print(f"  {'OK ' if ok else 'ERR'} {doc_id}", flush=True)

    ok = sum(1 for r in results.values() if r["state"] == "done")
    fail = sum(1 for r in results.values() if r["state"] == "failed")
    print(f"\nDONE -> {dest_p}  {ok} ok, {fail} failed, {len(todo) - len(results)} timed-out", flush=True)
    return ok
