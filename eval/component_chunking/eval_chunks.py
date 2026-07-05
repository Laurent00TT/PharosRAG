"""Ground-truth evaluation of chunk quality against MMDocIR question evidence.

For each labeled question on a doc we parsed, localize the evidence region(s)
(page + bbox + page_size) to content_list element(s), then use the chunk
`source_indices` bridge to find which chunk(s) hold that evidence, and score whether
the chunking KEPT THE EVIDENCE TOGETHER (the thing chunking actually controls).

Verdicts (denominator separates measurement limits from chunking quality):
  out_of_range  evidence page beyond our PAGE_CAP=50          (measurement limit; excluded)
  unlocalized   in-range but no content element overlaps it    (measurement limit; excluded)
  missing       localized to element(s) kept in NO chunk       (chunking dropped it: noise/heading)
  single        evidence elements all land in ONE chunk        (good)
  split         evidence elements span >1 chunk                (fragmentation risk)

chunking-quality denominator = missing + single + split (localized, in-range).
Plus channel-kind match (Table/Chart/Figure -> table/chart/image chunk) and soft answer recall.

NOTE on coordinates (validated empirically, see PROCESS_LOG):
- MMDocIR bbox uses its own per-region page_size; MinerU uses layout.json page_size. Both
  normalized to [0,1]. Page index aligns directly (offset 0) on clean docs; we do NOT
  per-doc hack offsets. `type` is a STRINGIFIED list ("['Figure']") -> ast-parsed.
"""
import ast
import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

KB = Path(os.environ.get("PHAROS_KB_ROOT") or os.path.expanduser("~/knowledge-base"))  # 外部数据集根(BYO;env PHAROS_KB_ROOT 覆盖)
REPO = Path(__file__).resolve().parent.parent
PARSED = REPO / "parsed"
CHUNKS = REPO / "chunks"
ANN_FILE = KB / "datasets" / "mmdocir" / "MMDocIR_annotations.jsonl"

COV_EVID = 0.3     # element counts if it covers >=30% of the evidence area, OR
COV_ELEM = 0.5     # the element is >=50% inside the evidence region
CHANNEL_KIND = {"Table": "table", "Chart": "chart", "Figure": "image",
                "Pure-text (Plain-text)": "text", "Generalized-text (Layout)": "text"}


def parse_listish(v):
    """MMDocIR stores type/answer as either a plain str or a stringified list."""
    if isinstance(v, list):
        return [str(x) for x in v]
    s = str(v)
    if s.startswith("[") and s.endswith("]"):
        try:
            x = ast.literal_eval(s)
            return [str(i) for i in x] if isinstance(x, list) else [s]
        except Exception:
            return [s]
    return [s]


def _key(s):
    return re.sub(r"\s+", "", (s or "")).lower()[:24]


def doc_scale(cl, layout):
    """Median (sx, sy) mapping layout/PDF-point space -> content_list render space, derived
    by text-matching elements. content_list bbox = layout bbox * scale about origin (0,0),
    scale is per-doc constant but x!=y and varies across docs (adversarial-review F6/F1).
    Returns (sx, sy) or (None, None) if too few matches."""
    lay = {}
    for pg in layout.get("pdf_info", []):
        p = pg.get("page_idx", 0)
        for b in pg.get("para_blocks", []) or []:
            txt = "".join(s.get("content", "") for ln in b.get("lines", []) or []
                          for s in ln.get("spans", []) or [])
            lay.setdefault(p, []).append((_key(txt), b.get("bbox")))
    sxs, sys = [], []
    for e in cl:
        eb = e.get("bbox")
        k = _key(e.get("text", ""))
        if not eb or len(k) < 8:
            continue
        for tk, lb in lay.get(e.get("page_idx"), []):
            if not lb or not tk:
                continue
            if tk.startswith(k[:12]) or k.startswith(tk[:12]):
                if lb[0] > 2 and lb[2] > 2:
                    sxs += [eb[0] / lb[0], eb[2] / lb[2]]
                if lb[1] > 2 and lb[3] > 2:
                    sys += [eb[1] / lb[1], eb[3] / lb[3]]
                break
    if len(sxs) < 4 or len(sys) < 4:
        return None, None
    return statistics.median(sxs), statistics.median(sys)


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter_area(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def channels(qtype):
    return [t for t in parse_listish(qtype) if t in CHANNEL_KIND]


def load_doc(doc_id):
    dd = PARSED / doc_id
    cl = json.loads([p for p in dd.rglob("*content_list.json") if "_v2" not in p.name][0]
                    .read_text(encoding="utf-8"))
    lay = json.loads(list(dd.rglob("layout.json"))[0].read_text(encoding="utf-8"))
    pages = {pg["page_idx"] for pg in lay["pdf_info"]}
    scale = doc_scale(cl, lay)
    chunks = [json.loads(l) for l in open(CHUNKS / f"{doc_id}.jsonl", encoding="utf-8")]
    idx2chunk = {}
    for ch in chunks:
        for i in ch["source_indices"]:
            idx2chunk[i] = ch
    return cl, pages, scale, idx2chunk


def region_elements(region, cl, pages, scale, cov_evid=COV_EVID, cov_elem=COV_ELEM):
    """GT bbox (PDF points) scaled INTO content_list render space (same origin) so the two
    coordinate systems match. Returns None if page beyond parsed range; else
    {hits:set, best:idx|None} where best = max-overlap element even if below threshold."""
    p = region["page"]
    if p not in pages:
        return None
    sx, sy = scale
    if sx is None:
        return {"hits": set(), "best": None}
    gb = region["bbox"]
    g = [gb[0] * sx, gb[1] * sy, gb[2] * sx, gb[3] * sy]
    ga = area(g)
    hits, best = set(), (0.0, None)
    for i, e in enumerate(cl):
        if e.get("page_idx") != p or not e.get("bbox"):
            continue
        eb = e["bbox"]
        ia = inter_area(g, eb)
        if ia <= 0:
            continue
        cev = ia / ga if ga else 0
        cel = ia / area(eb) if area(eb) else 0
        if max(cev, cel) > best[0]:
            best = (max(cev, cel), i)
        if cev >= cov_evid or cel >= cov_elem:
            hits.add(i)
    return {"hits": hits, "best": best[1]}


def eval_doc(doc_id, doc_type, ann, cov_evid=COV_EVID, cov_elem=COV_ELEM):
    cl, pages, scale, idx2chunk = load_doc(doc_id)
    out = []
    for q in ann["questions"]:
        regions = q.get("layout_mapping", [])
        if not regions:
            continue
        any_in_range = False
        ev_idxs, best_idxs = set(), []
        for r in regions:
            res = region_elements(r, cl, pages, scale, cov_evid, cov_elem)
            if res is None:
                continue
            any_in_range = True
            ev_idxs |= res["hits"]
            if res["best"] is not None:
                best_idxs.append(res["best"])
        if not any_in_range:
            verdict = "out_of_range"
        elif not ev_idxs:
            # distinguish "overlap existed but below threshold" from "zero overlap (parse gap)"
            if best_idxs:
                verdict = "unlocalized_threshold"
            else:
                verdict = "unlocalized_zero"
        else:
            hit_chunks = {idx2chunk[i]["chunk_id"] for i in ev_idxs if i in idx2chunk}
            verdict = "missing" if not hit_chunks else ("single" if len(hit_chunks) == 1 else "split")

        chs = channels(q.get("type"))
        asset_chs = [c for c in chs if CHANNEL_KIND[c] in ("table", "chart", "image")]
        kind_match = kind_match_strict = None
        if verdict in ("single", "split") and chs:
            hit_kinds = {idx2chunk[i]["kind"] for i in ev_idxs if i in idx2chunk}
            kind_match = bool(hit_kinds & {CHANNEL_KIND[c] for c in chs})
            if asset_chs:   # strict: a TRUE-asset question must hit an asset chunk, not just text
                kind_match_strict = bool(hit_kinds & {CHANNEL_KIND[c] for c in asset_chs})
        ans_hit = None
        if verdict in ("single", "split"):
            blob = norm_text(" ".join(
                (idx2chunk[i]["text"] + " " + (idx2chunk[i].get("content_raw") or ""))
                for i in ev_idxs if i in idx2chunk))
            ans_hit = any(norm_text(a) in blob for a in parse_listish(q.get("A")) if len(norm_text(a)) >= 2)
        out.append({"doc_id": doc_id, "doc_type": doc_type, "verdict": verdict,
                    "channel": "+".join(parse_listish(q.get("type"))), "is_asset": bool(chs),
                    "kind_match": kind_match, "kind_match_strict": kind_match_strict, "ans_hit": ans_hit})
    return out


def run(stem2, ann, cov_evid, cov_elem):
    allr = []
    for stem, (doc_id, dt) in stem2.items():
        if stem + ".pdf" not in ann or not (CHUNKS / f"{doc_id}.jsonl").exists():
            continue
        allr.extend(eval_doc(doc_id, dt, ann[stem + ".pdf"], cov_evid, cov_elem))
    return allr


def main():
    rows = list(csv.DictReader(open(REPO / "sample_manifest.csv", encoding="utf-8")))
    stem2 = {Path(r["src_path"]).stem: (r["doc_id"], r["doc_type"]) for r in rows}
    ann = {a["doc_name"]: a for a in (json.loads(l) for l in open(ANN_FILE, encoding="utf-8"))}

    def pct(x, d): return round(100 * x / d, 1) if d else 0.0

    allr = run(stem2, ann, COV_EVID, COV_ELEM)
    docs = len({r["doc_id"] for r in allr})
    vc = Counter(r["verdict"] for r in allr)
    localized = [r for r in allr if r["verdict"] in ("missing", "single", "split")]
    nl = len(localized)

    # threshold sensitivity (partial-F1): single% under stricter coverage
    strict = run(stem2, ann, 0.5, 0.7)
    sl = [r for r in strict if r["verdict"] in ("missing", "single", "split")]
    single_strict = pct(sum(1 for r in sl if r["verdict"] == "single"), len(sl))

    asset = [r for r in localized if r["kind_match"] is not None]
    asset_strict = [r for r in localized if r["kind_match_strict"] is not None]   # true-asset only
    ansq = [r for r in localized if r["ans_hit"] is not None]
    report = {
        "docs_evaluated": docs, "questions_total": len(allr), "verdicts": dict(vc),
        "excluded_out_of_range": vc.get("out_of_range", 0),
        "excluded_unlocalized_threshold": vc.get("unlocalized_threshold", 0),
        "excluded_unlocalized_zero_parse_gap": vc.get("unlocalized_zero", 0),
        "localized_questions": nl,
        "evidence_single_chunk_pct": pct(vc.get("single", 0), nl),
        "evidence_split_pct": pct(vc.get("split", 0), nl),
        "evidence_missing_pct": pct(vc.get("missing", 0), nl),
        "evidence_single_pct_STRICT_thresh": single_strict,
        "asset_channel_kind_match_pct": pct(sum(1 for r in asset if r["kind_match"]), len(asset)),
        "asset_channel_kind_match_pct_TRUE_ASSET": pct(sum(1 for r in asset_strict if r["kind_match_strict"]), len(asset_strict)),
        "asset_questions": len(asset), "true_asset_questions": len(asset_strict),
        "answer_string_recall_pct_SOFT_advisory": pct(sum(1 for r in ansq if r["ans_hit"]), len(ansq)),
    }
    (REPO / "analysis" / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # by doc_type with DISTINCT DOC COUNT (F5: don't let one doc's questions masquerade as a type)
    bydt = defaultdict(Counter)
    docset = defaultdict(set)
    for r in localized:
        bydt[r["doc_type"]][r["verdict"]] += 1
        docset[r["doc_type"]].add(r["doc_id"])
    with open(REPO / "analysis" / "eval_by_doctype.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_type", "distinct_docs", "n_localized", "single%", "split%", "missing%", "note"])
        for dt, c in sorted(bydt.items(), key=lambda x: -sum(x[1].values())):
            t = sum(c.values())
            nd = len(docset[dt])
            note = "single-doc: NOT generalizable" if nd <= 1 else ("n<=2 docs" if nd <= 2 else "")
            w.writerow([dt, nd, t, pct(c["single"], t), pct(c["split"], t), pct(c["missing"], t), note])

    print("== chunk-quality eval vs MMDocIR ground truth (coords-fixed) ==")
    print(f"distinct docs={docs} questions={len(allr)}  | excluded: out_of_range={vc.get('out_of_range',0)} "
          f"unlocalized_threshold={vc.get('unlocalized_threshold',0)} unlocalized_zero={vc.get('unlocalized_zero',0)}  | localized={nl}\n")
    print(f"  evidence SINGLE chunk     : {report['evidence_single_chunk_pct']}%  (strict thresh: {single_strict}%)")
    print(f"  evidence SPLIT            : {report['evidence_split_pct']}%")
    print(f"  evidence MISSING (dropped): {report['evidence_missing_pct']}%")
    print(f"  asset kind match (any)    : {report['asset_channel_kind_match_pct']}%  (n={len(asset)})")
    print(f"  asset kind match (TRUE)   : {report['asset_channel_kind_match_pct_TRUE_ASSET']}%  (n={len(asset_strict)})")
    print(f"  answer recall (soft/advis): {report['answer_string_recall_pct_SOFT_advisory']}%  (n={len(ansq)})\n")
    print(f"{'doc_type':22s} {'docs':>4} {'n':>4} {'single%':>8} {'split%':>7} {'missing%':>9}")
    for dt, c in sorted(bydt.items(), key=lambda x: -sum(x[1].values())):
        t = sum(c.values())
        print(f"{dt:22s} {len(docset[dt]):>4} {t:>4} {pct(c['single'],t):>8} {pct(c['split'],t):>7} {pct(c['missing'],t):>9}")
    print("\nwrote analysis/eval_report.json + analysis/eval_by_doctype.csv")


if __name__ == "__main__":
    main()
