"""Apples-to-apples comparison of 3 chunking strategies on the SAME ground-truth docs,
using the SAME evidence-preservation eval (see eval_chunks.py / EVALUATION.md):

  A) ours      — JSON-native chunker (chunks/<doc>.jsonl, mapped via source_indices)
  B) chonkie   — RecursiveChunker(recipe=markdown) over MinerU's full.md
  C) docling   — Docling HybridChunker over (MinerU full.md -> DoclingDocument)

Unification: evidence (page+bbox) -> content_list element idx (shared, via doc_scale) -> chunk.
For ours: idx->chunk via source_indices (exact). For text chunkers (B/C): idx->chunk via
substring match of the element's text inside a chunk's text. The localized-question
denominator is identical across strategies, so single/split/missing% are directly comparable.

NOTE: chonkie LateChunker changes embeddings, not boundaries -> same structural verdicts as
recursive, so recursive represents chonkie here (embedding quality is out of scope for a
structural evidence-preservation metric).
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import eval_chunks as E   # reuse doc_scale, region_elements, parse_listish, ANN_FILE, COV_*

REPO = Path(__file__).resolve().parent.parent
PARSED = REPO / "parsed"
CHUNKS = REPO / "chunks"

CHONKIE_CHUNK_SIZE = 2048   # chars (character tokenizer) ~= 512 tokens for English


# ---------- per-doc inputs ----------
def load_cl_layout(doc_id):
    dd = PARSED / doc_id
    cl = json.loads([p for p in dd.rglob("*content_list.json") if "_v2" not in p.name][0]
                    .read_text(encoding="utf-8"))
    lay = json.loads(list(dd.rglob("layout.json"))[0].read_text(encoding="utf-8"))
    pages = {pg["page_idx"] for pg in lay["pdf_info"]}
    scale = E.doc_scale(cl, lay)
    md_files = [p for p in dd.rglob("*.md")]
    md = md_files[0].read_text(encoding="utf-8") if md_files else ""
    md_path = str(md_files[0]) if md_files else None
    return cl, pages, scale, md, md_path


def element_text(el):
    t = el.get("text")
    if t:
        return t
    for k in ("table_caption", "image_caption", "chart_caption"):
        v = el.get(k)
        if v:
            return " ".join(v) if isinstance(v, list) else str(v)
    return ""


# ---------- element -> chunk maps (one per strategy) ----------
def map_ours(doc_id):
    """idx -> {chunk_id} from our chunker's source_indices (exact)."""
    f = CHUNKS / f"{doc_id}.jsonl"
    m = defaultdict(set)
    if not f.exists():
        return m
    for ch in (json.loads(l) for l in open(f, encoding="utf-8")):
        for i in ch["source_indices"]:
            m[i].add(ch["chunk_id"])
    return m


def map_text(cl, chunk_texts):
    """idx -> {chunk_index} via substring match of element text inside chunk text."""
    norm_chunks = [re.sub(r"\s+", "", (t or "").lower()) for t in chunk_texts]
    m = defaultdict(set)
    for i, el in enumerate(cl):
        et = re.sub(r"\s+", "", element_text(el).lower())
        if len(et) < 20:        # too short to match reliably -> skip (counts as unmapped)
            continue
        key = et[:60]
        for ci, nc in enumerate(norm_chunks):
            if key in nc:
                m[i].add(ci)
    return m


# ---------- chunkers ----------
def chonkie_chunks(md):
    from chonkie import RecursiveChunker
    c = RecursiveChunker.from_recipe("markdown", lang="en", chunk_size=CHONKIE_CHUNK_SIZE)
    return [ch.text for ch in c(md)]


_DOCLING = {}
def docling_chunks(md_path):
    if "chunker" not in _DOCLING:
        from docling.document_converter import DocumentConverter
        from docling.chunking import HybridChunker
        _DOCLING["conv"] = DocumentConverter()
        _DOCLING["chunker"] = HybridChunker()
    doc = _DOCLING["conv"].convert(md_path).document
    return [c.text for c in _DOCLING["chunker"].chunk(dl_doc=doc)]


# ---------- shared evidence localization (strategy-independent) ----------
def question_evidence(q, cl, pages, scale):
    """Return (in_range, ev_idxs set). Identical across strategies."""
    in_range = False
    ev = set()
    for r in q.get("layout_mapping", []):
        res = E.region_elements(r, cl, pages, scale)
        if res is None:
            continue
        in_range = True
        ev |= res["hits"]
    return in_range, ev


def verdict_localized(ev_idxs, emap):
    chunks = set()
    for i in ev_idxs:
        chunks |= emap.get(i, set())
    if not chunks:
        return "missing"
    return "single" if len(chunks) == 1 else "split"


def ours_chunk_texts(doc_id):
    f = CHUNKS / f"{doc_id}.jsonl"
    return [ch["text"] for ch in (json.loads(l) for l in open(f, encoding="utf-8"))] if f.exists() else []


def is_asset_question(q):
    return any(E.CHANNEL_KIND[c] in ("table", "chart", "image") for c in E.channels(q.get("type")))


def main():
    rows = list(csv.DictReader(open(REPO / "sample_manifest.csv", encoding="utf-8")))
    stem2 = {Path(r["src_path"]).stem: (r["doc_id"], r["doc_type"]) for r in rows}
    ann = {a["doc_name"]: a for a in (json.loads(l) for l in open(E.ANN_FILE, encoding="utf-8"))}

    have_docling = True
    try:
        import docling  # noqa
    except Exception as e:
        have_docling = False
        print(f"[docling unavailable: {repr(e)[:80]}]\n")

    # ours_exact = source_indices bridge (its real-world value); the rest + ours_fair use the
    # SAME text-substring bridge so boundary/asset differences are measured symmetrically.
    strategies = ["ours_exact", "ours_fair", "chonkie"] + (["docling"] if have_docling else [])
    tally = {sc: {s: Counter() for s in strategies} for sc in ("all", "text_only")}
    base = {"all": Counter(), "text_only": Counter()}
    docs = 0
    errs = Counter()

    for stem, (doc_id, dt) in stem2.items():
        if stem + ".pdf" not in ann or not (CHUNKS / f"{doc_id}.jsonl").exists():
            continue
        cl, pages, scale, md, md_path = load_cl_layout(doc_id)
        maps = {"ours_exact": map_ours(doc_id), "ours_fair": map_text(cl, ours_chunk_texts(doc_id))}
        try:
            maps["chonkie"] = map_text(cl, chonkie_chunks(md))
        except Exception:
            errs["chonkie"] += 1; maps["chonkie"] = {}
        if have_docling:
            try:
                maps["docling"] = map_text(cl, docling_chunks(md_path))
            except Exception as e:
                errs["docling"] += 1; maps["docling"] = {}
                if errs["docling"] <= 2:
                    print(f"  docling err on {doc_id}: {repr(e)[:120]}")
        docs += 1
        for q in ann[stem + ".pdf"]["questions"]:
            if not q.get("layout_mapping"):
                continue
            in_range, ev = question_evidence(q, cl, pages, scale)
            scopes = ["all"] + ([] if is_asset_question(q) else ["text_only"])
            for sc in scopes:
                if not in_range:
                    base[sc]["out_of_range"] += 1; continue
                if not ev:
                    base[sc]["unlocalized"] += 1; continue
                base[sc]["localized"] += 1
                for s in strategies:
                    tally[sc][s][verdict_localized(ev, maps[s])] += 1

    def report(scope, title):
        nl = base[scope]["localized"]
        def pct(x): return round(100 * x / nl, 1) if nl else 0.0
        print(f"\n===== {title}  (localized={nl}, excluded oor={base[scope]['out_of_range']} "
              f"unloc={base[scope]['unlocalized']}) =====")
        print(f"{'strategy':12s} {'single%':>8} {'split%':>7} {'missing%':>9}")
        for s in strategies:
            t = tally[scope][s]
            print(f"{s:12s} {pct(t['single']):>8} {pct(t['split']):>7} {pct(t['missing']):>9}")
        return {s: {k: tally[scope][s][k] for k in ("single", "split", "missing")} for s in strategies}

    print(f"docs={docs}" + (f"  chunker errors={dict(errs)}" if errs else ""))
    print("\nNOTE: ours_exact uses source_indices (exact); ours_fair/chonkie/docling all use the SAME "
          "text-substring bridge -> compare those three for boundary/asset差异, ours_exact 是 ours 的真实上限。")
    out = {"docs": docs,
           "all": report("all", "ALL evidence (incl. tables/figures/charts)"),
           "text_only": report("text_only", "TEXT-channel evidence only (无资产不对称，最公平)")}
    (REPO / "analysis" / "compare_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote analysis/compare_report.json")


if __name__ == "__main__":
    main()
