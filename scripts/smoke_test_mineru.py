"""End-to-end smoke test for mineru_server: send a PDF and validate the
middle_json structure.

Goals:
  1. Verify the model loads (the first hit triggers ModelSingleton load into VRAM)
  2. Confirm middle_json contains the fields our parser expects
     (pdf_info[i].para_blocks, BlockType, ...)
  3. Confirm page_image_b64 is injected per page
  4. Report parse latency and a structure overview

No business-code dependencies — uses only stdlib + httpx + fpdf2 + pypdfium2.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _make_test_pdf(path: Path) -> None:
    """Generate a small PDF with a heading, body text, and a table for smoke testing."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Knowledge Base Smoke Test", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Section 1: Approval Process", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7,
        "Approvals must be completed within 3 working days. Amounts over 1M "
        "require CEO signature. Lower tiers go through manager approval."
    )
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Section 1.1: Tiers", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, "Tier 1: under 100K - Manager", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "Tier 2: 100K-1M  - VP", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, "Tier 3: over 1M  - CEO", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Section 2: Reference Table", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=11)
    # Simple "table" — fpdf2 cell grid that MinerU's layout detector can pick up
    headers = ["Tier", "Amount", "Approver"]
    rows = [["1", "<100K", "Manager"], ["2", "100K-1M", "VP"], ["3", ">1M", "CEO"]]
    for h in headers:
        pdf.cell(45, 8, h, border=1)
    pdf.cell(0, 8, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for row in rows:
        for c in row:
            pdf.cell(45, 8, c, border=1)
        pdf.cell(0, 8, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8001")
    parser.add_argument("--mode", default="hybrid",
                        choices=["hybrid", "vlm", "pipeline"])
    parser.add_argument("--pdf", type=Path, default=None,
                        help="custom PDF path (omit to generate the bundled smoke test PDF)")
    parser.add_argument("--save-json", type=Path, default=None,
                        help="dump the full middle_json to a file for inspection")
    args = parser.parse_args()

    # Prepare the PDF
    if args.pdf is None:
        tmp_pdf = Path("smoke_test.pdf")
        _make_test_pdf(tmp_pdf)
        print(f"Generated test PDF: {tmp_pdf} ({tmp_pdf.stat().st_size} bytes)")
        pdf_path = tmp_pdf
    else:
        pdf_path = args.pdf
        print(f"Using user PDF: {pdf_path}")

    # Health check
    import httpx
    print(f"\n[1/3] Health check {args.server}/health ...")
    h = httpx.get(f"{args.server}/health", timeout=10)
    h.raise_for_status()
    print(f"  -> {h.json()}")

    # /parse call
    print(f"\n[2/3] POST /parse mode={args.mode} (first call loads models, may take 30s-2min)...")
    t0 = time.monotonic()
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(
            f"{args.server}/parse",
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
            data={
                "mode": args.mode,
                "lang": "ch",
                "formula_enable": "true",
                "table_enable": "true",
                "include_page_images": "true",
            },
        )
    elapsed = time.monotonic() - t0
    print(f"  -> HTTP {resp.status_code} ({elapsed:.1f}s)")
    if resp.status_code != 200:
        print(f"  body: {resp.text[:500]}")
        sys.exit(1)
    middle_json = resp.json()

    # Structure validation
    print(f"\n[3/3] Validating middle_json structure...")
    print(f"  top-level keys: {sorted(middle_json.keys())}")

    pdf_info = middle_json.get("pdf_info", [])
    print(f"  pdf_info pages: {len(pdf_info)}")

    if not pdf_info:
        print("  [FAIL] pdf_info is empty — middle_json schema may have changed")
        sys.exit(1)

    for i, page_info in enumerate(pdf_info):
        keys = sorted(page_info.keys())
        para = page_info.get("para_blocks", [])
        preproc = page_info.get("preproc_blocks", [])
        has_image = "page_image_b64" in page_info
        print(f"\n  Page {i}:")
        print(f"    keys: {keys}")
        print(f"    para_blocks: {len(para)} | preproc_blocks: {len(preproc)} | "
              f"page_image_b64: {has_image}")
        # List block types (first 20)
        chosen = para if para else preproc
        types_seen = []
        for b in chosen[:20]:
            t = b.get("type", "?")
            types_seen.append(t)
        print(f"    block types (first 20): {types_seen}")

    # Key check: the parser assumes para_blocks exists (preferred) or falls back to preproc_blocks
    has_any = all(("para_blocks" in p) or ("preproc_blocks" in p) for p in pdf_info)
    print(f"\n  [{'PASS' if has_any else 'FAIL'}] every page has para_blocks or preproc_blocks")

    # Verify page_image_b64 injection
    has_imgs = sum(1 for p in pdf_info if p.get("page_image_b64"))
    print(f"  [{'PASS' if has_imgs == len(pdf_info) else 'FAIL'}] "
          f"page_image_b64 injection: {has_imgs}/{len(pdf_info)}")

    # Optional dump
    if args.save_json:
        # Don't save page_image_b64 (too big); keep everything else verbatim
        slim = json.loads(json.dumps(middle_json))
        for p in slim.get("pdf_info", []):
            if "page_image_b64" in p:
                p["page_image_b64"] = f"<{len(p['page_image_b64'])} chars elided>"
        args.save_json.write_text(
            json.dumps(slim, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  full middle_json (sans b64) -> {args.save_json}")

    print(f"\n[DONE] Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
