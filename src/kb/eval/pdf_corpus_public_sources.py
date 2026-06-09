"""Official/public PDF seed records for Corpus v1.

These seeds are deliberately small and conservative. They provide a reusable
starting pool without requiring API keys; the script can optionally download
the PDFs into ignored local storage.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pdf_corpus import make_manifest_record


@dataclass(frozen=True)
class PublicPdfSeed:
    doc_key: str
    source_family: str
    source_url: str
    download_url: str
    local_path: str
    domain: str
    doc_type: str
    layout_tags: list[str]
    license_note: str
    notes: str


IRS_SEEDS = [
    ("f1040", "Form 1040 U.S. Individual Income Tax Return"),
    ("i1040gi", "Instructions for Form 1040"),
    ("fw4", "Form W-4 Employee Withholding Certificate"),
    ("fw9", "Form W-9 Request for Taxpayer Identification Number"),
    ("iw9", "Instructions for Form W-9"),
    ("f941", "Form 941 Employer Quarterly Federal Tax Return"),
    ("i941", "Instructions for Form 941"),
    ("p15", "Publication 15 Employer Tax Guide"),
    ("p17", "Publication 17 Federal Income Tax"),
    ("p334", "Publication 334 Tax Guide for Small Business"),
    ("p463", "Publication 463 Travel Gift and Car Expenses"),
    ("p505", "Publication 505 Tax Withholding and Estimated Tax"),
    ("p583", "Publication 583 Starting a Business and Keeping Records"),
    ("p946", "Publication 946 How To Depreciate Property"),
    ("p970", "Publication 970 Tax Benefits for Education"),
    ("f1120", "Form 1120 U.S. Corporation Income Tax Return"),
    ("i1120", "Instructions for Form 1120"),
    ("f1065", "Form 1065 U.S. Return of Partnership Income"),
]

GOVINFO_FIXED_SEEDS = [
    ("BILLS-118hr1ih", "House Bill 118 HR 1 Introduced"),
    ("BILLS-118s1is", "Senate Bill 118 S 1 Introduced"),
]

NASA_SEEDS = [
    ("20040121077", "The NASA Technical Report Server", "20040121077.pdf"),
    ("20160003607", "NASA Technical Report Example 20160003607", "20160003607.pdf"),
    ("20030015752", "NASA Scientific and Technical Information", "20030015752.pdf"),
    (
        "20210025399",
        "Developing the Foundations of an Exploration Class Medical System",
        "Dev_Found_%20Expl_Class_Med_Syst_Bridg_Gap_bw_LEO_Mars_FINAL_REV2_SP%2012062021.docx.pdf",
    ),
]

ARXIV_SEEDS = [
    ("1706.03762", "Attention Is All You Need"),
    ("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    ("2104.09864", "LayoutLMv2"),
    ("2112.10752", "LayoutLMv3"),
    ("2310.06825", "Mistral 7B"),
    ("2308.12950", "Qwen Technical Report"),
    ("2407.10671", "Qwen2 Technical Report"),
    ("2210.03629", "ReAct"),
]

def build_public_seed_records(
    *,
    root: Path,
    download: bool,
    limit: int | None = None,
    timeout_s: int = 30,
    max_mb: float = 15.0,
) -> list[dict[str, Any]]:
    seeds = _all_seeds()
    if limit is not None:
        seeds = seeds[:limit]

    records = []
    for seed in seeds:
        pdf_path = root / seed.local_path
        local_path = seed.local_path
        notes = seed.notes
        if download and seed.download_url.lower().endswith(".pdf"):
            ok, msg = _download_pdf(seed.download_url, pdf_path, timeout_s=timeout_s, max_mb=max_mb)
            notes = f"{notes}; download={msg}"
            if not ok:
                local_path = seed.local_path

        records.append(make_manifest_record(
            doc_key=seed.doc_key,
            source_family=seed.source_family,
            source_url=seed.source_url,
            download_url=seed.download_url,
            local_path=local_path,
            language="en",
            domain=seed.domain,
            doc_type=seed.doc_type,
            layout_tags=seed.layout_tags,
            quality_tier="candidate",
            eval_role="background",
            gold_priority=0,
            license_note=seed.license_note,
            collection_policy="download_ok" if seed.download_url.lower().endswith(".pdf") else "metadata_only",
            notes=notes,
            root=root,
            ingest_status="downloaded" if (root / local_path).exists() else "pending",
            parse_status="unknown",
        ))
    return records


def _all_seeds() -> list[PublicPdfSeed]:
    seeds: list[PublicPdfSeed] = []

    for form_id, title in IRS_SEEDS:
        seeds.append(PublicPdfSeed(
            doc_key=f"irs-{form_id}",
            source_family="govinfo",
            source_url="https://www.irs.gov/forms-pubs",
            download_url=f"https://www.irs.gov/pub/irs-pdf/{form_id}.pdf",
            local_path=f"datasets/pdf_corpus_v1/raw/govinfo/irs-{form_id}.pdf",
            domain="government_tax",
            doc_type="irs_form_or_publication",
            layout_tags=["forms", "tables", "text"],
            license_note="official IRS public PDF",
            notes=title,
        ))

    govinfo_packages = list(GOVINFO_FIXED_SEEDS)
    govinfo_packages.extend(
        (f"PLAW-118publ{number}", f"Public Law 118-{number}")
        for number in range(1, 161)
    )
    seen_packages = set()
    for package_id, title in govinfo_packages:
        if package_id in seen_packages:
            continue
        seen_packages.add(package_id)
        seeds.append(PublicPdfSeed(
            doc_key=f"govinfo-{package_id.lower()}",
            source_family="govinfo",
            source_url=f"https://www.govinfo.gov/app/details/{package_id}",
            download_url=f"https://www.govinfo.gov/content/pkg/{package_id}/pdf/{package_id}.pdf",
            local_path=f"datasets/pdf_corpus_v1/raw/govinfo/{package_id}.pdf",
            domain="government_legislation",
            doc_type="bill_or_public_law",
            layout_tags=["text"],
            license_note="official GovInfo public PDF",
            notes=title,
        ))

    for citation_id, title, download_name in NASA_SEEDS:
        seeds.append(PublicPdfSeed(
            doc_key=f"nasa-ntrs-{citation_id}",
            source_family="academic",
            source_url=f"https://ntrs.nasa.gov/citations/{citation_id}",
            download_url=f"https://ntrs.nasa.gov/api/citations/{citation_id}/downloads/{download_name}",
            local_path=f"datasets/pdf_corpus_v1/raw/academic/nasa-ntrs-{citation_id}.pdf",
            domain="technical_report",
            doc_type="nasa_technical_report",
            layout_tags=["figures", "text"],
            license_note="NASA NTRS public content; verify individual record distribution limits before redistribution",
            notes=title,
        ))

    for arxiv_id, title in ARXIV_SEEDS:
        seeds.append(PublicPdfSeed(
            doc_key=f"arxiv-{arxiv_id.replace('.', '-')}",
            source_family="academic",
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            download_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            local_path=f"datasets/pdf_corpus_v1/raw/academic/arxiv-{arxiv_id.replace('.', '-')}.pdf",
            domain="research_paper",
            doc_type="academic_paper",
            layout_tags=["figures", "tables", "multi_column", "text"],
            license_note="arXiv public paper; follow paper license before redistribution",
            notes=title,
        ))

    return seeds


def _download_pdf(url: str, path: Path, *, timeout_s: int, max_mb: float) -> tuple[bool, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return True, f"exists_bytes={path.stat().st_size}"
    request = urllib.request.Request(url, headers={"User-Agent": "NaviKB corpus builder"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(int(max_mb * 1024 * 1024) + 1)
        if len(data) > max_mb * 1024 * 1024:
            return False, f"skipped_over_{max_mb:.1f}mb"
        if "pdf" not in content_type.lower() and not data.startswith(b"%PDF"):
            return False, f"not_pdf_content_type={content_type}"
        path.write_bytes(data)
        return True, f"ok_bytes={len(data)}"
    except Exception as exc:
        return False, f"failed_{type(exc).__name__}:{exc}"
