# PDF Corpus v1 Source Notes

This directory tracks Corpus v1 metadata, not large PDF assets.

## Rules

- Keep public PDFs under `raw/` as local artifacts unless a later review explicitly approves redistribution.
- Record source URLs, download URLs, and license notes in `manifest.jsonl`.
- Mark uncertain licensing or access terms with `collection_policy: "metadata_only"` or `collection_policy: "do_not_redistribute"`.
- Keep run outputs under `derived/`; commit only `.gitkeep` scaffolding unless a specific snapshot is intentionally reviewed and approved.

## Source Families

- `mmdocir`: local benchmark PDFs and annotations.
- `govinfo`: official public government PDFs; also includes official IRS PDF forms/publications because Corpus v1 v0.2 uses the existing controlled value rather than adding an `irs` source family.
- `sec`: company filings and annual-report style documents. SEC metadata-only seeds are intentionally excluded from the main downloaded PDF candidate manifest until HTML-to-PDF conversion or annual-report PDF sourcing is implemented.
- `manual`: product manuals, technical guides, and whitepapers.
- `academic`: papers, reports, course notes, NASA NTRS public PDFs, and arXiv PDFs. arXiv API expansion is optional because the API can rate-limit large requests.
- `docvqa_like`: scanned or form-heavy documents used for OCR/layout stress.
- `omnidocbench`: parser/layout benchmark documents.
- `synthetic`: generated internal-style business documents with deterministic answer keys.
- `other`: reviewed one-off sources.
