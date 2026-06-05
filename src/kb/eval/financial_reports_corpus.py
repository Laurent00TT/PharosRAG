"""Bilingual semiconductor financial-report corpus helpers.

The collector is intentionally conservative:
  - use public Eastmoney research metadata and public PDF URLs;
  - keep English sources as curated public/redacted PDFs;
  - store provenance, access notes, and hashes in a manifest;
  - do not touch logged-in, paid, or CAPTCHA-gated sources.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .pdf_corpus import make_manifest_record


FINANCIAL_CORPUS_DIR = Path("datasets/financial_semiconductor_reports_v1")
EASTMONEY_API_URL = "https://reportapi.eastmoney.com/report/list"
EASTMONEY_DETAIL_URL = "https://data.eastmoney.com/report/zw_industry.jshtml"
EASTMONEY_REFERER = "https://data.eastmoney.com/report/industry.jshtml"

DEFAULT_ZH_KEYWORDS = [
    "半导体",
    "集成电路",
    "芯片",
    "晶圆",
    "封测",
    "存储",
    "功率半导体",
    "半导体设备",
    "半导体材料",
    "先进封装",
    "AI芯片",
    "EDA",
    "SiC",
    "碳化硅",
]

DEFAULT_EN_KEYWORDS = [
    "semiconductor",
    "semiconductors",
    "semiconductor equipment",
    "foundry",
    "wafer",
    "memory",
    "HBM",
    "advanced packaging",
    "EUV",
    "SiC",
    "power semiconductor",
    "AI chips",
    "semiconductor supply chain",
]


@dataclass(frozen=True)
class EnglishReportSeed:
    doc_key: str
    source_family: str
    institution: str
    title: str
    source_url: str
    download_url: str
    local_name: str
    doc_type: str
    license_note: str
    notes: str


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    message: str
    bytes_written: int = 0


ENGLISH_PUBLIC_REPORT_SEEDS = [
    EnglishReportSeed(
        doc_key="msci-semiconductors-equipment-2025-redacted",
        source_family="msci",
        institution="MSCI ESG Research",
        title="Industry Report | Semiconductors & Semiconductor Equipment | 2025 January",
        source_url="https://www.msci.com/",
        download_url=(
            "https://www.msci.com/documents/1296102/55971585/"
            "MSCI_Semiconductors_Redacted_Industry_Report_Final.pdf"
        ),
        local_name="msci-semiconductors-equipment-2025-redacted.pdf",
        doc_type="industry_report_redacted",
        license_note="public redacted MSCI PDF; internal eval only; do not redistribute",
        notes="English public/redacted industry report seed.",
    ),
    EnglishReportSeed(
        doc_key="spglobal-electronic-chemicals-semiconductors-abstract-2025",
        source_family="spglobal",
        institution="S&P Global",
        title="Electronic Chemicals - Semiconductors, Silicon and IC Process Chemicals Abstract",
        source_url="https://www.spglobal.com/",
        download_url=(
            "https://www.spglobal.com/content/dam/spglobal/ci/en/documents/"
            "products/pdf/CI_0425_SCUP_Electronic_Chemicals-Semiconductors_"
            "Silicon_IC_Process_Chemicals_Abstract.pdf"
        ),
        local_name="spglobal-electronic-chemicals-semiconductors-abstract-2025.pdf",
        doc_type="industry_report_sample",
        license_note="public S&P Global sample/abstract PDF; internal eval only; do not redistribute",
        notes="English sample PDF with semiconductor materials coverage.",
    ),
    EnglishReportSeed(
        doc_key="sia-state-of-us-semiconductor-industry-2023",
        source_family="sia",
        institution="Semiconductor Industry Association",
        title="2023 State of the U.S. Semiconductor Industry",
        source_url="https://www.semiconductors.org/",
        download_url=(
            "https://www.semiconductors.org/wp-content/uploads/2023/08/"
            "SIA_State-of-Industry-Report_2023_Final_080323.pdf"
        ),
        local_name="sia-state-of-us-semiconductor-industry-2023.pdf",
        doc_type="industry_association_report",
        license_note="public SIA PDF; internal eval only; verify terms before redistribution",
        notes="English industry association report seed.",
    ),
    EnglishReportSeed(
        doc_key="sia-state-of-us-semiconductor-industry-2021",
        source_family="sia",
        institution="Semiconductor Industry Association",
        title="2021 State of the U.S. Semiconductor Industry",
        source_url="https://www.semiconductors.org/",
        download_url=(
            "https://www.semiconductors.org/wp-content/uploads/2021/09/"
            "2021-SIA-State-of-the-Industry-Report.pdf"
        ),
        local_name="sia-state-of-us-semiconductor-industry-2021.pdf",
        doc_type="industry_association_report",
        license_note="public SIA PDF; internal eval only; verify terms before redistribution",
        notes="English industry association report seed.",
    ),
    EnglishReportSeed(
        doc_key="sia-green-book-v18-sample-2025",
        source_family="sia",
        institution="Semiconductor Industry Association",
        title="SIA Green Book v18 Sample",
        source_url="https://www.semiconductors.org/",
        download_url="https://www.semiconductors.org/wp-content/uploads/2025/01/Green-Book-v18_Sample.pdf",
        local_name="sia-green-book-v18-sample-2025.pdf",
        doc_type="industry_statistics_sample",
        license_note="public SIA sample PDF; internal eval only; do not redistribute",
        notes="English semiconductor statistics sample seed.",
    ),
    EnglishReportSeed(
        doc_key="goldman-sachs-green-technology-cycle-sic-2022",
        source_family="goldman_sachs",
        institution="Goldman Sachs Global Investment Research",
        title="The Green Technology Cycle: SiC",
        source_url="https://www.goldmansachs.com/",
        download_url=(
            "https://www.goldmansachs.com/pdfs/insights/pages/gs-research/"
            "the-green-technology-cycle-sic/report.pdf"
        ),
        local_name="goldman-sachs-green-technology-cycle-sic-2022.pdf",
        doc_type="investment_research_redacted",
        license_note="public redacted Goldman Sachs research PDF; internal eval only; do not redistribute",
        notes="English investment research seed focused on SiC supply chain.",
    ),
    EnglishReportSeed(
        doc_key="goldman-sachs-inflation-here-today-gone-tomorrow-2021",
        source_family="goldman_sachs",
        institution="Goldman Sachs Global Investment Research",
        title="Inflation: Here Today, Gone Tomorrow?",
        source_url="https://www.goldmansachs.com/",
        download_url=(
            "https://www.goldmansachs.com/pdfs/insights/pages/gs-research/"
            "inflation-here-today-gone-tomorrow/inflation-here-today-gone-tomorrow.pdf"
        ),
        local_name="goldman-sachs-inflation-semiconductor-supply-chain-2021.pdf",
        doc_type="investment_research_redacted",
        license_note="public redacted Goldman Sachs research PDF; internal eval only; do not redistribute",
        notes="English macro research seed with semiconductor supply-chain discussion.",
    ),
    EnglishReportSeed(
        doc_key="jpmorgan-working-capital-index-2023",
        source_family="jpmorgan",
        institution="J.P. Morgan",
        title="Strategies for Resiliency: Working Capital Index Report 2023",
        source_url="https://www.jpmorgan.com/",
        download_url=(
            "https://www.jpmorgan.com/content/dam/jpm/treasury-services/documents/"
            "strategies-for-resiliency-working-capital-index-report-2023-ada.pdf"
        ),
        local_name="jpmorgan-working-capital-index-2023.pdf",
        doc_type="financial_industry_report",
        license_note="public J.P. Morgan PDF; internal eval only; do not redistribute",
        notes="English financial report seed with semiconductor inventory/working-capital coverage.",
    ),
    EnglishReportSeed(
        doc_key="msci-semiconductors-arent-green-transcript",
        source_family="msci",
        institution="MSCI",
        title="Semiconductors aren't green - Transcript",
        source_url="https://www.msci.com/",
        download_url=(
            "https://www.msci.com/documents/1296102/25589897/"
            "Semiconductors%2Baren_t%2Bgreen%2B-%2BTranscript.pdf"
        ),
        local_name="msci-semiconductors-arent-green-transcript.pdf",
        doc_type="research_transcript",
        license_note="public MSCI transcript PDF; internal eval only; do not redistribute",
        notes="English transcript seed; useful for non-traditional financial-report layout.",
    ),
    EnglishReportSeed(
        doc_key="gpec-industry-series-semiconductors-2025",
        source_family="other_financial",
        institution="Greater Phoenix Economic Council",
        title="Industry Series Reports: Semiconductors",
        source_url="https://www.gpec.org/",
        download_url="https://www.gpec.org/wp-content/uploads/Industry-Series-Reports-Semiconductors.pdf",
        local_name="gpec-industry-series-semiconductors-2025.pdf",
        doc_type="industry_report",
        license_note="public GPEC PDF; internal eval only; verify terms before redistribution",
        notes="English regional industry report seed.",
    ),
]


def parse_jsonp(text: str) -> dict[str, Any]:
    stripped = text.strip()
    match = re.match(r"^[\w$.]+\((.*)\)\s*;?$", stripped, flags=re.S)
    if match:
        stripped = match.group(1)
    return json.loads(stripped)


def fetch_eastmoney_page(
    *,
    page_no: int,
    page_size: int,
    begin_date: str,
    end_date: str,
    timeout_s: int = 20,
) -> dict[str, Any]:
    params = {
        "industryCode": "*",
        "pageSize": str(page_size),
        "industry": "*",
        "rating": "*",
        "ratingchange": "*",
        "beginTime": begin_date,
        "endTime": end_date,
        "pageNo": str(page_no),
        "fields": "",
        "qType": "1",
        "orgCode": "",
        "code": "*",
        "rcode": "",
        "cb": "callback",
    }
    url = f"{EASTMONEY_API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers=_browser_headers(referer=EASTMONEY_REFERER),
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        text = response.read().decode("utf-8", "replace")
    return parse_jsonp(text)


def fetch_eastmoney_page_with_retries(
    *,
    page_no: int,
    page_size: int,
    begin_date: str,
    end_date: str,
    timeout_s: int = 20,
    attempts: int = 3,
    retry_delay_s: float = 1.5,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fetch_eastmoney_page(
                page_no=page_no,
                page_size=page_size,
                begin_date=begin_date,
                end_date=end_date,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < attempts and retry_delay_s > 0:
                time.sleep(retry_delay_s * attempt)
    if last_exc is not None:
        raise last_exc
    return {}


def eastmoney_record_matches(
    item: dict[str, Any],
    *,
    keywords: Iterable[str] = DEFAULT_ZH_KEYWORDS,
) -> bool:
    haystack = " ".join(
        str(item.get(field) or "")
        for field in ("title", "industryName", "researcher", "orgSName")
    ).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def eastmoney_pdf_url(info_code: str) -> str:
    return f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf"


def eastmoney_detail_url(info_code: str) -> str:
    return f"{EASTMONEY_DETAIL_URL}?infocode={urllib.parse.quote(info_code)}"


def eastmoney_to_manifest_record(
    item: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    info_code = str(item["infoCode"])
    local_path = FINANCIAL_CORPUS_DIR / "raw" / "zh" / "eastmoney" / f"{info_code}.pdf"
    pdf_exists = (root / local_path).exists()
    download_message = str(item.get("_download_message") or "")
    download_ok = item.get("_download_ok")
    notes = (
        f"title={item.get('title', '')}; institution={item.get('orgSName', '')}; "
        f"industry={item.get('industryName', '')}; publish_date={item.get('publishDate', '')}"
    )
    if download_message:
        notes = f"{notes}; download={download_message}"
    ingest_status = "downloaded" if pdf_exists else "pending"
    if download_ok is False and not pdf_exists:
        ingest_status = "failed"
    record = make_manifest_record(
        doc_key=f"eastmoney-{info_code.lower()}",
        source_family="eastmoney",
        source_url=eastmoney_detail_url(info_code),
        download_url=eastmoney_pdf_url(info_code),
        local_path=local_path.as_posix(),
        language="zh",
        domain="finance_semiconductor",
        doc_type="sell_side_or_industry_research",
        layout_tags=["text", "tables", "charts"],
        quality_tier="candidate",
        eval_role="background",
        gold_priority=0,
        license_note="public Eastmoney research PDF; internal eval only; do not redistribute",
        collection_policy="do_not_redistribute",
        notes=notes,
        root=root,
        ingest_status=ingest_status,
    )
    record["meta"] = {
        "institution": item.get("orgSName") or item.get("orgName") or "",
        "institution_full": item.get("orgName") or "",
        "researcher": item.get("researcher") or "",
        "publish_date": item.get("publishDate") or "",
        "title": item.get("title") or "",
        "industry_name": item.get("industryName") or "",
        "industry_code": item.get("industryCode") or "",
        "attach_size_kb": item.get("attachSize"),
        "attach_pages": item.get("attachPages"),
        "rating": item.get("sRatingName") or item.get("emRatingName") or "",
    }
    return record


def build_english_seed_records(
    *,
    root: Path,
    download: bool,
    limit: int | None = None,
    timeout_s: int = 30,
    max_mb: float = 80.0,
    delay_s: float = 0.5,
) -> list[dict[str, Any]]:
    seeds = ENGLISH_PUBLIC_REPORT_SEEDS[:limit] if limit is not None else ENGLISH_PUBLIC_REPORT_SEEDS
    records: list[dict[str, Any]] = []
    for seed in seeds:
        local_path = FINANCIAL_CORPUS_DIR / "raw" / "en" / seed.source_family / seed.local_name
        notes = seed.notes
        if download:
            result = download_pdf(seed.download_url, root / local_path, timeout_s=timeout_s, max_mb=max_mb)
            notes = f"{notes}; download={result.message}"
            if delay_s > 0:
                time.sleep(delay_s)

        record = make_manifest_record(
            doc_key=seed.doc_key,
            source_family=seed.source_family,
            source_url=seed.source_url,
            download_url=seed.download_url,
            local_path=local_path.as_posix(),
            language="en",
            domain="finance_semiconductor",
            doc_type=seed.doc_type,
            layout_tags=["text", "tables", "charts"],
            quality_tier="candidate",
            eval_role="background",
            gold_priority=0,
            license_note=seed.license_note,
            collection_policy="do_not_redistribute",
            notes=notes,
            root=root,
            ingest_status="downloaded" if (root / local_path).exists() else "pending",
        )
        record["meta"] = {
            "institution": seed.institution,
            "title": seed.title,
            "access_type": "public_redacted_or_public_pdf",
        }
        records.append(record)
    return records


def collect_eastmoney_records(
    *,
    root: Path,
    target: int,
    max_pages: int,
    page_size: int,
    begin_date: str,
    end_date: str,
    download: bool,
    timeout_s: int = 30,
    max_mb: float = 80.0,
    delay_s: float = 0.8,
    keywords: Iterable[str] = DEFAULT_ZH_KEYWORDS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page_no in range(1, max_pages + 1):
        try:
            data = fetch_eastmoney_page_with_retries(
                page_no=page_no,
                page_size=page_size,
                begin_date=begin_date,
                end_date=end_date,
                timeout_s=timeout_s,
            )
        except Exception:
            continue
        rows = data.get("data") or []
        if not rows:
            break
        for item in rows:
            info_code = str(item.get("infoCode") or "")
            if not info_code or info_code in seen:
                continue
            if not eastmoney_record_matches(item, keywords=keywords):
                continue
            seen.add(info_code)
            local_path = root / FINANCIAL_CORPUS_DIR / "raw" / "zh" / "eastmoney" / f"{info_code}.pdf"
            item_for_manifest = dict(item)
            if download:
                result = download_pdf(
                    eastmoney_pdf_url(info_code),
                    local_path,
                    timeout_s=timeout_s,
                    max_mb=max_mb,
                    referer=eastmoney_detail_url(info_code),
                )
                item_for_manifest["_download_ok"] = result.ok
                item_for_manifest["_download_message"] = result.message
                if delay_s > 0:
                    time.sleep(delay_s)
            records.append(eastmoney_to_manifest_record(item_for_manifest, root=root))
            if len(records) >= target:
                return records
        if delay_s > 0:
            time.sleep(delay_s)
    return records


def default_date_range() -> tuple[str, str]:
    return "2023-01-01", date.today().isoformat()


def download_pdf(
    url: str,
    path: Path,
    *,
    timeout_s: int,
    max_mb: float,
    referer: str | None = None,
) -> DownloadResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return DownloadResult(True, f"exists_bytes={path.stat().st_size}", path.stat().st_size)

    headers = _browser_headers(referer=referer)
    ok, data, content_type, message = _fetch_bytes(url, headers=headers, timeout_s=timeout_s, max_mb=max_mb)

    if ok and data.startswith(b"%PDF"):
        path.write_bytes(data)
        return DownloadResult(True, f"ok_bytes={len(data)}", len(data))

    if "pdf.dfcfw.com" in urllib.parse.urlparse(url).netloc and data:
        cookie = extract_dfcfw_cookie_header(data.decode("utf-8", "ignore"))
        if cookie:
            headers["Cookie"] = cookie
            ok, data, content_type, message = _fetch_bytes(
                url, headers=headers, timeout_s=timeout_s, max_mb=max_mb
            )
            if ok and data.startswith(b"%PDF"):
                path.write_bytes(data)
                return DownloadResult(True, f"ok_after_cookie_bytes={len(data)}", len(data))

    if ok:
        message = f"not_pdf_content_type={content_type}; prefix={data[:40]!r}"
    return DownloadResult(False, message, 0)


def extract_dfcfw_cookie_header(challenge_html: str) -> str | None:
    status_match = re.search(
        r"WTKkN:(\d+),bOYDu:(\d+),dtzqS:function\(a,n\).*?wyeCN:(\d+)",
        challenge_html,
        flags=re.S,
    )
    ssid_match = re.search(r"case\"3\":t=.*?\(t,(\d+)\)", challenge_html, flags=re.S)
    if not status_match or not ssid_match:
        return None
    status = sum(int(part) for part in status_match.groups())
    ssid = int(ssid_match.group(1))
    return f"__tst_status={status}#; EO_Bot_Ssid={ssid}"


def _fetch_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout_s: int,
    max_mb: float,
) -> tuple[bool, bytes, str, str]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(int(max_mb * 1024 * 1024) + 1)
    except urllib.error.HTTPError as exc:
        return False, b"", "", f"http_{exc.code}:{exc.reason}"
    except Exception as exc:
        return False, b"", "", f"failed_{type(exc).__name__}:{exc}"
    if len(data) > max_mb * 1024 * 1024:
        return False, data, content_type, f"skipped_over_{max_mb:.1f}mb"
    return True, data, content_type, f"ok_bytes={len(data)}"


def _browser_headers(*, referer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _user_agent() -> str:
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36 "
        "NaviKB-financial-report-corpus"
    )
