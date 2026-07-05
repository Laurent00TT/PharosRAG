"""Stratified sampling for the chunking experiment.

Reads the manifests of the three usable datasets, normalizes doc_type across them,
picks a breadth-first sample spanning every type x page-size bucket, copies the
selected PDFs into corpus/<type>/, balances the load across 3 MinerU accounts by
effective page count, and writes sample_manifest.csv.

Deterministic: no RNG. Selection spreads evenly across the page-size range of each
type so we always capture small/medium/large variants of every document type.
"""
import csv
import json
import os
import re
import shutil
from pathlib import Path

# 语料构建脚本(一次性):从外部数据集(knowledge-base/datasets)选样并写 sample_manifest.csv。
# KB_ROOT = 外部数据集根(不在本仓;env PHAROS_KB_ROOT 覆盖);REPO = 本仓根(__file__ 相对)。
KB_ROOT = Path(os.environ.get("PHAROS_KB_ROOT") or os.path.expanduser("~/knowledge-base"))
DS = KB_ROOT / "datasets"
REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "corpus"
CACHE = REPO / "config" / "mmdocir_pagecount.json"

PAGE_CAP = 50          # cap pages sent per doc to bound API cost
KEYS = ["A", "B", "C"]
PER_KEY_PAGE_LIMIT = 1000

# normalized_type -> target number of docs (breadth-first, all types covered)
TARGETS = {
    "financial_research_zh": 12,
    "academic_paper":        10,
    "law":                   10,
    "financial_report_en":    8,
    "government":             6,
    "research_report":        5,
    "form":                   5,
    "guidebook":              4,
    "slides_tutorial":        4,
    "brochure":               4,
    "policy":                 4,
    "admin_industry":         2,
    "tech_report":            2,
    "news":                   1,
}

MMDOCIR_DOMAIN_MAP = {
    "Academic paper": "academic_paper",
    "Financial report": "financial_report_en",
    "Laws": "law",
    "Government": "government",
    "Research report / Introduction": "research_report",
    "Guidebook": "guidebook",
    "Tutorial/Workshop": "slides_tutorial",
    "Brochure": "brochure",
    "Administration/Industry file": "admin_industry",
    "News": "news",
}

PDFCORPUS_TYPE_MAP = {
    "bill_or_public_law": "law",
    "synthetic_internal_policy": "policy",
    "irs_form_or_publication": "form",
    "academic_paper": "academic_paper",
    "nasa_technical_report": "tech_report",
}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pdf_page_count(path):
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path), strict=False).pages)
    except Exception as e:
        print(f"  ! page count failed for {path.name}: {e}")
        return None


def build_candidates():
    cands = []

    # 1) financial — Chinese sell-side / industry research
    for r in read_jsonl(DS / "financial_semiconductor_reports_v1" / "manifest.jsonl"):
        src = KB_ROOT / r["local_path"]
        if not src.exists():
            continue
        cands.append({
            "doc_type": "financial_research_zh",
            "dataset": "financial",
            "domain": r.get("domain", ""),
            "language": "ch",
            "page_count": r.get("page_count") or 0,
            "layout_tags": ";".join(r.get("layout_tags", [])),
            "src": src,
            "stem": src.stem,
        })

    # 2) pdf_corpus native types. benchmark_pdf (which duplicates mmdocir) is implicitly
    #    skipped: PDFCORPUS_TYPE_MAP has no entry for it, so the `if not t: continue` drops it.
    for r in read_jsonl(DS / "pdf_corpus_v1" / "manifest.jsonl"):
        t = PDFCORPUS_TYPE_MAP.get(r.get("doc_type"))
        if not t:
            continue
        src = KB_ROOT / r["local_path"]
        if not src.exists():
            continue
        cands.append({
            "doc_type": t,
            "dataset": "pdf_corpus",
            "domain": r.get("domain", ""),
            "language": r.get("language", "en"),
            "page_count": r.get("page_count") or 0,
            "layout_tags": ";".join(r.get("layout_tags", [])),
            "src": src,
            "stem": src.stem,
        })

    # 3) mmdocir — diverse domains, page counts via pypdf (cached)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    docdir = DS / "mmdocir" / "extracted" / "doc_pdfs"
    mm = list(read_jsonl(DS / "mmdocir" / "MMDocIR_annotations.jsonl"))
    print(f"counting pages for {len(mm)} mmdocir docs (cached={len(cache)})...")
    for r in mm:
        t = MMDOCIR_DOMAIN_MAP.get(r.get("domain"))
        if not t:
            continue
        name = r["doc_name"]
        src = docdir / name
        if not src.exists():
            continue
        if name not in cache:
            cache[name] = pdf_page_count(src)
        pc = cache[name]
        if not pc:
            continue
        cands.append({
            "doc_type": t,
            "dataset": "mmdocir",
            "domain": r.get("domain", ""),
            "language": "en",
            "page_count": pc,
            "layout_tags": "",
            "src": src,
            "stem": src.stem,
        })
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return cands


def spread_pick(items, n):
    """Pick n items spread evenly across the page-size range (sorted), deterministic."""
    items = sorted(items, key=lambda x: (x["page_count"], x["stem"]))
    # drop pathological 0-page / >200-page (API hard limit)
    items = [x for x in items if 1 <= x["page_count"] <= 200]
    if len(items) <= n:
        return items
    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    seen, out = set(), []
    for i in idx:
        while i in seen and i < len(items) - 1:
            i += 1
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def assign_keys(selected):
    """Greedy load-balance by effective pages across the 3 accounts."""
    load = {k: 0 for k in KEYS}
    for d in sorted(selected, key=lambda x: -x["eff_pages"]):
        k = min(KEYS, key=lambda kk: load[kk])
        d["api_key"] = k
        load[k] += d["eff_pages"]
    return load


def main():
    cands = build_candidates()
    by_type = {}
    for c in cands:
        by_type.setdefault(c["doc_type"], []).append(c)
    print("\n== candidate pool ==")
    for t in TARGETS:
        print(f"  {t:24s} available={len(by_type.get(t, [])):4d}  target={TARGETS[t]}")

    selected = []
    for t, n in TARGETS.items():
        picked = spread_pick(by_type.get(t, []), n)
        for c in picked:
            c["eff_pages"] = min(c["page_count"], PAGE_CAP)
            c["page_ranges"] = "" if c["page_count"] <= PAGE_CAP else f"1-{PAGE_CAP}"
            c["doc_id"] = f"{t}__{re.sub(r'[^A-Za-z0-9._-]', '_', c['stem'])}"[:120]
            selected.append(c)

    load = assign_keys(selected)

    # copy into corpus/<type>/ and write manifest
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    rows = []
    for d in selected:
        dest = CORPUS / d["doc_type"] / (d["doc_id"] + ".pdf")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d["src"], dest)
        rows.append({
            "doc_id": d["doc_id"], "doc_type": d["doc_type"], "dataset": d["dataset"],
            "domain": d["domain"], "language": d["language"],
            "page_count": d["page_count"], "eff_pages": d["eff_pages"],
            "page_ranges": d["page_ranges"], "layout_tags": d["layout_tags"],
            "api_key": d["api_key"], "corpus_path": str(dest.relative_to(REPO)),
            "src_path": str(d["src"]),
        })

    rows.sort(key=lambda r: (r["doc_type"], r["doc_id"]))
    with open(REPO / "sample_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_pages = sum(r["eff_pages"] for r in rows)
    print(f"\n== selected {len(rows)} docs, {total_pages} effective pages ==")
    tc = {}
    for r in rows:
        tc[r["doc_type"]] = tc.get(r["doc_type"], 0) + 1
    for t in TARGETS:
        print(f"  {t:24s} {tc.get(t,0)}")
    print(f"\n== per-key load (pages) ==  {load}")
    kc = {k: sum(1 for r in rows if r['api_key'] == k) for k in KEYS}
    print(f"== per-key files ==  {kc}")
    print(f"\nmanifest -> {REPO / 'sample_manifest.csv'}")


if __name__ == "__main__":
    main()
