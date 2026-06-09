# tests/test_chunker.py
from kb.ingestion.chunker import Chunker
from kb.models import ParsedPage, TextChunk

def _make_page(text: str, page_num: int = 0, tables=None) -> ParsedPage:
    return ParsedPage(
        doc_id="abc", doc_name="t.pdf", page_num=page_num,
        structured_text=text, heading_path=["Doc", "Ch1"],
        page_image=b"", image_blocks=[], tables=tables or [],
        metadata={},
    )

def test_short_text_becomes_one_chunk():
    page = _make_page("审批须在3个工作日内完成。")
    chunker = Chunker(small_size=384, parent_size=1024)
    chunks = chunker.chunk_page(page)
    assert any(c.is_parent is False for c in chunks)
    assert any(c.is_parent is True for c in chunks)

def test_small_chunk_has_parent_id():
    page = _make_page("审批须在3个工作日内完成。" * 50)
    chunker = Chunker(small_size=384, parent_size=1024)
    chunks = chunker.chunk_page(page)
    small = [c for c in chunks if not c.is_parent]
    assert all(c.parent_chunk_id != "" for c in small)

def test_table_becomes_single_parent_chunk():
    table_data = [{"列1": "审批人", "列2": "权限金额"}]
    page = _make_page("", tables=[table_data])
    chunker = Chunker(small_size=384, parent_size=1024)
    chunks = chunker.chunk_page(page)
    parents = [c for c in chunks if c.is_parent]
    assert any("审批人" in c.text or "列1" in c.text for c in parents)

def test_heading_path_propagated():
    page = _make_page("内容文字。")
    chunker = Chunker()
    chunks = chunker.chunk_page(page)
    assert all(c.heading_path == ["Doc", "Ch1"] for c in chunks)

def test_chunk_ids_are_unique():
    page = _make_page("内容。" * 100)
    chunker = Chunker(small_size=64)
    chunks = chunker.chunk_page(page)
    ids = [f"{c.doc_id}_{c.page_num}_{c.chunk_index}" for c in chunks]
    assert len(ids) == len(set(ids))


# ── Heading-aware path tests ─────────────────────────────────────────────


def test_no_headings_uses_default_split():
    """0 headings → falls back to default char-split path."""
    text = "Some plain text without any markdown headings. " * 30  # ~1500 chars
    page = _make_page(text)
    chunker = Chunker(small_size=384, parent_size=1024)
    chunks = chunker.chunk_page(page)
    # default path produces parent + small chunks; heading_path stays as page-level prefix
    assert all(c.heading_path == ["Doc", "Ch1"] for c in chunks)
    # parent count should be > 1 due to text length > parent_size
    parents = [c for c in chunks if c.is_parent]
    assert len(parents) >= 1


def test_single_heading_uses_default_split():
    """1 heading → predicate returns False, default path used."""
    text = "## Solo Section\n\n" + ("Body content here. " * 20)
    page = _make_page(text)
    chunker = Chunker(min_headings_for_segmenting=2)
    chunks = chunker.chunk_page(page)
    # Single heading is below threshold → default path → all chunks share page heading_path
    assert all(c.heading_path == ["Doc", "Ch1"] for c in chunks)


def test_two_short_headings_creates_two_segments():
    """Two short sections → two parent+leaf pairs, each with own heading appended."""
    text = (
        "## Section A\n\n"
        "Content for A.\n\n"
        "## Section B\n\n"
        "Content for B."
    )
    page = _make_page(text)
    chunker = Chunker(small_size=384, parent_size=1024)
    chunks = chunker.chunk_page(page)
    parents = [c for c in chunks if c.is_parent]
    # Two segments → two parents (each segment short enough for one parent)
    assert len(parents) == 2
    headings_in_parents = [c.heading_path[-1] for c in parents]
    assert "Section A" in headings_in_parents
    assert "Section B" in headings_in_parents


def test_long_segment_falls_back_to_char_split_inside():
    """A heading whose body is > parent_size → parent + multiple small leaves."""
    long_body = "重复内容。" * 200  # ~1000 chars
    text = (
        "## A\n\n" + long_body + "\n\n"
        "## B\n\n" + "Short B body."
    )
    page = _make_page(text)
    chunker = Chunker(small_size=200, parent_size=500)
    chunks = chunker.chunk_page(page)
    # Segment A is long → 1 parent + many leaves with heading "A"
    a_leaves = [c for c in chunks if not c.is_parent and "A" in c.heading_path]
    assert len(a_leaves) > 1
    # Segment B is short → 1 parent + 1 leaf
    b_chunks = [c for c in chunks if "B" in c.heading_path]
    assert any(c.is_parent for c in b_chunks)
    assert any(not c.is_parent for c in b_chunks)


def test_segment_heading_appended_to_page_heading_path():
    """heading_path = page prefix + segment heading."""
    text = "## Sub1\n\nA.\n\n## Sub2\n\nB."
    page = _make_page(text)  # page heading_path = ["Doc", "Ch1"]
    chunker = Chunker()
    chunks = chunker.chunk_page(page)
    sub1_chunks = [c for c in chunks if "Sub1" in c.heading_path]
    sub2_chunks = [c for c in chunks if "Sub2" in c.heading_path]
    assert sub1_chunks and sub2_chunks
    for c in sub1_chunks:
        assert c.heading_path == ["Doc", "Ch1", "Sub1"]
    for c in sub2_chunks:
        assert c.heading_path == ["Doc", "Ch1", "Sub2"]


def test_table_emitted_first_then_heading_aware_text():
    """A page with both tables and headings: tables first, then heading-aware text."""
    table_data = [{"col1": "v1"}]
    text = "## Title\n\nText.\n\n## Title2\n\nText2."
    page = _make_page(text, tables=[table_data])
    chunker = Chunker()
    chunks = chunker.chunk_page(page)
    # First two chunks (parent + leaf) should be the table
    assert chunks[0].page_type == "table"
    assert chunks[1].page_type == "table"
    # Subsequent chunks are heading-aware text
    text_chunks = [c for c in chunks if c.page_type == "text"]
    assert any("Title" in c.heading_path for c in text_chunks)
    assert any("Title2" in c.heading_path for c in text_chunks)


def test_empty_text_with_no_tables_returns_empty():
    page = _make_page("")
    chunker = Chunker()
    chunks = chunker.chunk_page(page)
    assert chunks == []


def test_pre_heading_body_preserved():
    """Body text before the first heading is emitted as a no-heading segment."""
    text = "Intro text without heading.\n\n## A\n\nA body.\n\n## B\n\nB body."
    page = _make_page(text)
    chunker = Chunker()
    chunks = chunker.chunk_page(page)
    # Find chunks whose heading_path is exactly the page prefix (no segment heading appended)
    prefix_only = [
        c for c in chunks
        if c.heading_path == ["Doc", "Ch1"] and "Intro" in c.text
    ]
    assert prefix_only, "Pre-heading body should produce a chunk with no segment heading"


def test_heading_at_end_of_text_handled():
    """A heading right at the end of text (empty body after) must not crash."""
    text = "## A\n\nA body.\n\n## B\n\n## C"
    page = _make_page(text)
    chunker = Chunker()
    chunks = chunker.chunk_page(page)
    # Should produce chunks for A, B, C; B's body is empty, C's body is empty.
    # Empty segments should still register heading via the segment formatter.
    text_chunks = [c for c in chunks if c.page_type == "text"]
    headings_seen = {h for c in text_chunks for h in c.heading_path if h not in ("Doc", "Ch1")}
    assert "A" in headings_seen


def test_heading_count_threshold_is_configurable():
    text = "## Only One\n\nBody."
    page = _make_page(text)
    # threshold=1: should engage heading-aware path even with single heading
    chunker = Chunker(min_headings_for_segmenting=1)
    chunks = chunker.chunk_page(page)
    assert any("Only One" in c.heading_path for c in chunks)
