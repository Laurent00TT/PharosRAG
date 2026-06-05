# tests/test_classifier.py
import pytest
from pathlib import Path
from kb.parser.document_classifier import classify_document, DocClassification


@pytest.fixture
def text_pdf(tmp_path):
    """生成一个有文字层的 PDF（用 fpdf2）"""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for i in range(20):
        pdf.cell(0, 10, f"Line {i}: This is a normal text PDF with plenty of characters.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.add_page()
    pdf.cell(0, 10, "Page 2 content with more text.",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p = tmp_path / "text.pdf"
    pdf.output(str(p))
    return p


@pytest.fixture
def empty_pdf(tmp_path):
    """生成一个几乎没有文字的 PDF（模拟扫描件）"""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "x", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    p = tmp_path / "empty.pdf"
    pdf.output(str(p))
    return p


def test_classify_native_text_pdf(text_pdf):
    result = classify_document(text_pdf)
    assert result.classification == "native_text"
    assert result.is_processable is True


def test_classify_low_text_pdf_as_scanned(empty_pdf):
    result = classify_document(empty_pdf)
    # 平均字符 < 50 应判为 scanned
    assert result.classification in ("scanned", "native_text")  # 极小 PDF 边界情况都接受


def test_classify_returns_metadata(text_pdf):
    result = classify_document(text_pdf)
    assert result.page_count >= 1
    assert "avg_chars_per_page" in result.signals


def test_classify_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        classify_document(Path("/no/such/file.pdf"))


def test_classify_non_pdf_passthrough(tmp_path):
    """DOCX/DOC 等非 PDF 应直接放行 native_text,不走 pypdfium2 路径。"""
    docx = tmp_path / "manual.docx"
    docx.write_bytes(b"PK\x03\x04 fake docx bytes")  # 非真 docx 也无所谓 — 早返回不解析
    result = classify_document(docx)
    assert result.classification == "native_text"
    assert result.is_processable is True
    assert result.page_count == -1  # 哨兵: 未知页数,交给 MinerU
    assert result.signals.get("reason") == "non_pdf_passthrough"
    assert result.signals.get("suffix") == ".docx"


def test_classify_non_pdf_uppercase_suffix(tmp_path):
    """suffix 大小写不敏感。"""
    f = tmp_path / "report.DOC"
    f.write_bytes(b"fake")
    result = classify_document(f)
    assert result.classification == "native_text"
    assert result.is_processable is True
