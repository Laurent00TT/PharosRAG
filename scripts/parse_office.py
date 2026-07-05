"""COMMITTED parse path for docx/pptx: MinerU's NATIVE office backend (local, no API/quota,
**zero VLM/ML models** — verified pure-rule OOXML parsing) -> content_list.json, the SAME schema
as PDF/scanned. The existing from_mineru adapter + chunker then handle them on the identical path.

This is the project's chosen office parser (over the hand-rolled from_docx/from_pptx, which stay
as a zero-dependency fallback only). Rationale: fairly measured, MinerU office ties the hand-rolled
adapter AND Tika on docx text coverage (~95%), but its content_list schema is purpose-built for
chunking (text_level / list_items / table_body / image_caption / chart / OMML->LaTeX) and unifies
docx/pptx with the PDF+scanned pipeline — no JVM (unlike Tika), no heavy ML deps.

xlsx is NOT here: MinerU emits a whole sheet as one table element (useless for a 200k-row grid);
spreadsheets stay on the dedicated table_chunker (row-group + header context).

Setup: needs the MinerU source repo (set $MINERU_REPO or edit DEFAULT_MINERU) + the office-subset
deps only (pip install loguru mammoth lxml pydantic boto3 python-docx python-pptx) — NOT MinerU's
full torch/vlm stack. Run: python scripts/parse_office.py [name-filter]
"""
import csv
import html
import json
import os
import re
import sys
import zipfile
from pathlib import Path

DEFAULT_MINERU = os.path.expanduser("~/MinerU")   # MinerU 源仓;env MINERU_REPO 覆盖
sys.path.insert(0, os.environ.get("MINERU_REPO", DEFAULT_MINERU))

from loguru import logger                                        # noqa: E402
logger.remove()                                                  # silence MinerU debug spam

from mineru.backend.office.docx_analyze import office_docx_analyze       # noqa: E402
from mineru.backend.office.pptx_analyze import office_pptx_analyze       # noqa: E402
from mineru.backend.office.office_middle_json_mkcontent import union_make  # noqa: E402
from mineru.utils.enum_class import MakeMode                     # noqa: E402
from mineru.data.data_reader_writer import FileBasedDataWriter   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "parsed_office"
ANALYZE = {".docx": office_docx_analyze, ".pptx": office_pptx_analyze}


def _junk_alt(s):
    """Reject alt-text that is NOT an authored description but an auto-filled filename / path / id.
    Word & PowerPoint default a picture's descr/title to its filename when the author wrote none."""
    s = (s or "").strip()
    if len(s) < 3:
        return True
    if re.search(r"\.(png|jpe?g|gif|emf|wmf|bmp|tiff?|svg|webp)\b", s, re.I):  # contains an image ext
        return True
    if "\\" in s or re.search(r"[A-Za-z]:/", s) or s.count("/") >= 2:          # a file path
        return True
    if "_" in s and " " not in s:                  # identifier-like single token (allah_anim, IMG_001)
        return True
    if " " not in s and re.search(r"\d", s):       # single token with digits -> likely a filename/id
        return True
    return not re.search(r"[A-Za-z一-鿿]{3,}", s)   # must contain a real word


_BOILER = re.compile(r"\s*(Description automatically generated|AI-generated content may be incorrect\.?"
                     r"|生成的(说明|描述))\s*", re.I)
# Office/Azure "AI alt text" suggestions — hedged auto-captions, never an authored description (review)
_AUTOCAP = re.compile(r"(?i)(\bwith (?:low|medium|high) confidence\s*$|^a picture containing\b"
                      r"|^a close-?up of\b|^a (?:blue|red|green|black|white|gray|grey)\b.*\blogo\b"
                      r"|^image result for\b|^screen ?clip\b|^graphical user interface\b)")


def _alt_of(m):
    if not m:
        return ""
    a = (re.search(r'descr="([^"]*)"', m.group(0)) or re.search(r'title="([^"]*)"', m.group(0)))
    if not a:
        return ""
    alt = re.sub(r"\s+", " ", _BOILER.sub(" ", html.unescape(a.group(1)))).strip()  # strip PPT boilerplate
    if alt.lower() in ("text", "picture", "image", "logo", "a picture", "diagram", "graphical user interface"):
        return ""                                # generic auto-caption with no real content
    if _AUTOCAP.search(alt):                      # Office AI-suggested hedge caption, not authored
        return ""
    return "" if _junk_alt(alt) else alt


def pic_alt_texts(path: Path):
    """Alt-text (descr/title) for each PICTURE drawing in reading order ("" if junk/absent).
    docx: <wp:inline|anchor> with a:blip -> wp:docPr; pptx: <p:pic> -> p:cNvPr (slides in order)."""
    with zipfile.ZipFile(path) as z:
        if path.suffix.lower() == ".docx":
            x = z.read("word/document.xml").decode("utf-8", "ignore")
            blocks = [b for b in re.findall(r"<wp:(?:inline|anchor).*?</wp:(?:inline|anchor)>", x, re.S)
                      if "a:blip" in b]
            return [_alt_of(re.search(r"<wp:docPr[^>]*>", b)) for b in blocks]
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n)),
                        key=lambda n: int(re.search(r"slide(\d+)", n).group(1)))
        alts = []
        for n in slides:
            x = z.read(n).decode("utf-8", "ignore")
            for pic in re.findall(r"<p:pic>.*?</p:pic>", x, re.S):
                alts.append(_alt_of(re.search(r"<p:cNvPr[^>]*>", pic)))
        return alts


def parse_one(f: Path):
    d = OUT / f.stem
    (d / "images").mkdir(parents=True, exist_ok=True)
    # B②: pass an image_writer so MinerU crops+stores each picture and fills content_list img_path
    # (verbatim images/<hash>.jpg). MinerU does the image<->element association INTERNALLY — the naive
    # "align by OOXML media order" is unreliable (media holds master/decorative/duplicate images; e.g.
    # acl_gov has 20 media files but only 3 content images, which MinerU crops exactly).
    image_writer = FileBasedDataWriter(str(d / "images"))
    middle_json, _ = ANALYZE[f.suffix.lower()](f.read_bytes(), image_writer=image_writer)
    content_list = union_make(middle_json["pdf_info"], MakeMode.CONTENT_LIST, "images")
    # Tier-0 alt-text enrichment: fill empty image captions, but ONLY when the picture count aligns
    # with MinerU's image elements (MinerU drops/merges some images -> order match unreliable; skip
    # rather than attach a wrong neighbour's description). Conservative by design.
    imgs = [e for e in content_list if e.get("type") == "image"]
    alts = pic_alt_texts(f)
    applied, aligned = 0, abs(len(alts) - len(imgs)) <= 1 and bool(alts)
    if aligned:
        for e, a in zip(imgs, alts):
            if a and not (e.get("image_caption") or []):
                e["image_caption"] = [a]
                applied += 1
    d = OUT / f.stem
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{f.stem}_content_list.json").write_text(
        json.dumps(content_list, ensure_ascii=False), encoding="utf-8")
    return len(content_list), len(imgs), applied, aligned


def main():
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ok, err, alt_total, alt_skip = [], 0, 0, 0, 0
    for fmt in ("docx", "pptx"):
        for f in sorted((ROOT / "corpus_multiformat" / fmt).glob(f"*.{fmt}")):
            if flt and flt not in f.name:
                continue
            try:
                n, nimg, applied, aligned = parse_one(f)
                ok += 1; alt_total += applied
                if nimg and not aligned:
                    alt_skip += 1
                rows.append({"doc_id": f.stem, "format": fmt, "elements": n,
                             "images": nimg, "alt_applied": applied, "alt_aligned": int(aligned)})
                print(f"  OK  {f.name[:46]:46} -> {n} elems, {nimg} img, alt+{applied}"
                      f"{'' if aligned or not nimg else ' (skip:count-mismatch)'}", flush=True)
            except Exception as e:
                err += 1
                print(f"  ERR {f.name[:46]:46} {type(e).__name__}: {e}", flush=True)
    with (OUT / "_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "format", "elements", "images",
                                           "alt_applied", "alt_aligned"])
        w.writeheader(); w.writerows(rows)
    print(f"\nDONE: {ok} ok, {err} err -> {OUT}")
    print(f"  alt-text: filled {alt_total} image captions; {alt_skip} docs skipped (count mismatch)")


if __name__ == "__main__":
    main()
