# src/kb/parser/document_classifier.py
"""Document classifier — quickly classify a document at ingest entry to skip
crash-prone paths.

Ported from MinerU's pdf_classify.py decision tree. Zero extra dependencies
beyond pypdfium2.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# Thresholds (ported from MinerU pdf_classify.py)
CHARS_THRESHOLD = 50
HIGH_IMAGE_COVERAGE_THRESHOLD = 0.8
TEXT_QUALITY_BAD_THRESHOLD = 0.03      # abnormal-char ratio > 3% → corrupted
TEXT_QUALITY_MIN_CHARS = 300
MAX_SAMPLE_PAGES = 10
MAX_PAGE_ASPECT_RATIO = 10.0


Classification = Literal["native_text", "scanned", "encrypted", "corrupted", "invalid"]


@dataclass
class DocClassification:
    classification: Classification
    is_processable: bool
    page_count: int
    signals: dict = field(default_factory=dict)
    error: str | None = None


# Only PDFs go through pypdfium2 classification. Other formats (docx / doc) are
# left for MinerU to handle directly.
_PDF_SUFFIXES = {".pdf"}


def classify_document(path: Path) -> DocClassification:
    """Lightweight classification on a PDF. No side effects, no model calls.

    Non-PDF formats (docx / doc / etc.) are passed through as native_text with
    page_count=-1 (unknown), to be sorted out by MinerU. Downstream loggers
    should render page_count<0 as 'unknown'.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in _PDF_SUFFIXES:
        # Non-PDF: -1 is the "unknown page count" sentinel (distinct from 0=invalid empty)
        return DocClassification(
            classification="native_text",
            is_processable=True,
            page_count=-1,
            signals={"reason": "non_pdf_passthrough", "suffix": path.suffix.lower()},
        )

    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise SystemExit(
            "Need pypdfium2 for document classification: pip install pypdfium2"
        )

    pdf_bytes = path.read_bytes()

    # 1. Encryption check (the most common reason pypdfium2 fails to open)
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
    except pdfium.PdfiumError as e:
        msg = str(e).lower()
        if "password" in msg or "encrypt" in msg:
            return DocClassification(
                classification="encrypted",
                is_processable=False,
                page_count=0,
                error=str(e),
            )
        return DocClassification(
            classification="invalid",
            is_processable=False,
            page_count=0,
            error=str(e),
        )

    try:
        page_count = len(pdf)
        if page_count == 0:
            return DocClassification(
                classification="invalid",
                is_processable=False,
                page_count=0,
                error="empty document",
            )

        # 2. Sample pages
        sample_indices = _sample_indices(page_count, MAX_SAMPLE_PAGES)

        # 3. Per-page char counts
        avg_chars = _avg_chars(pdf, sample_indices)

        # 4. Extreme aspect ratio check
        for idx in sample_indices:
            page = pdf[idx]
            try:
                width, height = page.get_size()
                if max(width, height) / max(min(width, height), 1) > MAX_PAGE_ASPECT_RATIO:
                    return DocClassification(
                        classification="scanned",
                        is_processable=True,
                        page_count=page_count,
                        signals={"reason": "extreme_aspect_ratio",
                                 "avg_chars_per_page": avg_chars},
                    )
            finally:
                page.close()

        # 5. Decision
        if avg_chars < CHARS_THRESHOLD:
            return DocClassification(
                classification="scanned",
                is_processable=True,         # scans are processable (OCR path)
                page_count=page_count,
                signals={"reason": "low_text_density",
                         "avg_chars_per_page": avg_chars},
            )

        # 6. Text quality (abnormal-char ratio)
        abnormal_ratio = _abnormal_char_ratio(pdf, sample_indices)
        if (
            abnormal_ratio >= TEXT_QUALITY_BAD_THRESHOLD
            and avg_chars * len(sample_indices) >= TEXT_QUALITY_MIN_CHARS
        ):
            return DocClassification(
                classification="corrupted",
                is_processable=False,
                page_count=page_count,
                signals={"reason": "encoding_corruption",
                         "abnormal_char_ratio": abnormal_ratio,
                         "avg_chars_per_page": avg_chars},
            )

        return DocClassification(
            classification="native_text",
            is_processable=True,
            page_count=page_count,
            signals={
                "avg_chars_per_page": avg_chars,
                "abnormal_char_ratio": abnormal_ratio,
            },
        )
    finally:
        pdf.close()


def _sample_indices(total: int, max_samples: int) -> list[int]:
    """Evenly-spaced page indices."""
    if total <= max_samples:
        return list(range(total))
    step = total / max_samples
    return [int(i * step) for i in range(max_samples)]


def _avg_chars(pdf, indices: list[int]) -> float:
    """Average non-blank chars per sampled page."""
    total = 0
    counted = 0
    for idx in indices:
        page = pdf[idx]
        try:
            text = page.get_textpage().get_text_range() or ""
            cleaned = text.strip()
            if cleaned:
                total += len(cleaned)
            counted += 1
        except Exception:
            pass
        finally:
            page.close()
    if counted == 0:
        return 0
    return total / counted


def _abnormal_char_ratio(pdf, indices: list[int]) -> float:
    """Ratio of abnormal characters (U+FFFD, control chars, etc.)."""
    abnormal_chars = {chr(0xFFFD), chr(0)}
    total = 0
    abnormal = 0
    for idx in indices:
        page = pdf[idx]
        try:
            text = page.get_textpage().get_text_range() or ""
            for c in text:
                total += 1
                if c in abnormal_chars or (ord(c) < 32 and c not in "\t\n\r"):
                    abnormal += 1
        except Exception:
            pass
        finally:
            page.close()
    if total == 0:
        return 0
    return abnormal / total
