"""Minimal MinerU v4 API client: batch upload -> auto-parse -> poll -> download zip.

Flow (per https://mineru.net/apiManage/docs):
  POST /api/v4/file-urls/batch  -> {data:{batch_id, file_urls:[...]}}  (urls match files order)
  PUT  <file_url>  (binary body, NO Content-Type header)               -> parsing auto-starts
  GET  /api/v4/extract-results/batch/{batch_id}  -> {data:{extract_result:[{file_name,state,full_zip_url,...}]}}
"""
import io
import os
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE = "https://mineru.net"
REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

TOKENS = {k: os.getenv(f"MINERU_TOKEN_{k}") for k in ("A", "B", "C")}
DONE_STATES = {"done", "failed"}


def _headers(key):
    return {"Authorization": f"Bearer {TOKENS[key]}", "Content-Type": "application/json"}


def create_batch(key, files, model_version="vlm", language=None,
                 enable_formula=True, enable_table=True):
    """files: list of {"name","data_id", optional "page_ranges","is_ocr"}. Returns (batch_id, file_urls)."""
    body = {"files": files, "model_version": model_version,
            "enable_formula": enable_formula, "enable_table": enable_table}
    if language:
        body["language"] = language
    r = requests.post(f"{BASE}/api/v4/file-urls/batch", headers=_headers(key),
                      json=body, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"create_batch failed: {j}")
    return j["data"]["batch_id"], j["data"]["file_urls"]


def upload(file_url, path, retries=3):
    """PUT the file to the presigned URL. Must NOT send Content-Type."""
    for attempt in range(retries):
        try:
            with open(path, "rb") as f:
                r = requests.put(file_url, data=f, timeout=600)
            r.raise_for_status()
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! upload failed {Path(path).name}: {e}")
                return False
            time.sleep(2 * (attempt + 1))


def poll_batch(key, batch_id):
    r = requests.get(f"{BASE}/api/v4/extract-results/batch/{batch_id}",
                     headers=_headers(key), timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"poll failed: {j}")
    return j["data"].get("extract_result", [])


def download_and_extract(zip_url, dest_dir, retries=3):
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
                print(f"  ! download failed {zip_url}: {e}")
                return None
            time.sleep(3 * (attempt + 1))
