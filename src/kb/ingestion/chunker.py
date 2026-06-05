# src/kb/ingestion/chunker.py
"""Heading-aware chunker with safe fallback to default char-split.

See internal design notes (not published) for the
design discussion. Summary:
- Tables: same as before (parent + leaf, full table preserved)
- Plain text: split at markdown-heading boundaries IF the page has >=2 headings,
  else fall back to the original char-based parent/small split
- heading_path: page-level prefix + per-segment local heading
"""
import json
import re

from kb.models import ParsedPage, TextChunk


# Match a markdown heading line (## title / ### sub) at the start of a line.
# MinerU emits these into structured_text from doc_title / title / paragraph_title blocks.
_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class Chunker:
    def __init__(
        self,
        small_size: int = 384,
        parent_size: int = 1024,
        overlap: int = 64,
        min_headings_for_segmenting: int = 2,
    ) -> None:
        self.small_size = small_size
        self.parent_size = parent_size
        self.overlap = overlap
        self.min_headings_for_segmenting = min_headings_for_segmenting

    def chunk_page(self, page: ParsedPage) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        idx = 0

        # 1. Tables: keep whole as a parent chunk (don't split).
        # Also emit a leaf chunk (is_parent=False) so the table can be retrieved
        # via text search — search_text_dense/sparse force is_parent=False filtering.
        for table in page.tables:
            table_text = json.dumps(table, ensure_ascii=False)
            parent_id = f"{page.doc_id}_p{page.page_num}_t{idx}"
            chunks.append(self._make_chunk(
                page, idx, table_text, page.heading_path,
                page_type="table", parent_chunk_id=parent_id, is_parent=True,
            ))
            idx += 1
            chunks.append(self._make_chunk(
                page, idx, table_text, page.heading_path,
                page_type="table", parent_chunk_id=parent_id, is_parent=False,
            ))
            idx += 1

        # 2. Plain text. Heading-aware path if structure exists, else fallback.
        text = page.structured_text.strip()
        if not text:
            return chunks

        if self._has_meaningful_headings(text):
            chunks.extend(self._chunk_by_headings(page, text, idx_start=idx))
        else:
            chunks.extend(self._chunk_default(page, text, idx_start=idx))
        return chunks

    # ── Heading-aware path ───────────────────────────────────────────────

    def _has_meaningful_headings(self, text: str) -> bool:
        """True if text contains >= min_headings_for_segmenting heading lines.
        With <2 headings, segmenting buys nothing over the default split."""
        return len(_HEADING_LINE_RE.findall(text)) >= self.min_headings_for_segmenting

    def _chunk_by_headings(
        self, page: ParsedPage, text: str, idx_start: int
    ) -> list[TextChunk]:
        """Split text at heading boundaries. Each segment = heading + body
        until the next heading. Long segments fall back to char-split inside
        the segment, with the segment's heading appended to heading_path."""
        out: list[TextChunk] = []
        idx = idx_start
        segments = self._split_into_segments(text)
        for seg_heading, seg_body, _seg_level in segments:
            # The full segment text = heading line + body. Used for parent/short-segment text.
            seg_text = self._format_segment(seg_heading, seg_body)
            local_heading_path = (
                list(page.heading_path) + [seg_heading] if seg_heading
                else list(page.heading_path)
            )
            parent_id = f"{page.doc_id}_p{page.page_num}_s{idx}"

            if len(seg_text) <= self.parent_size:
                # Short segment: emit a parent + identical-text leaf so retrieval
                # hits the leaf and enrich can return the parent.
                out.append(self._make_chunk(
                    page, idx, seg_text, local_heading_path,
                    page_type="text", parent_chunk_id=parent_id, is_parent=True,
                ))
                idx += 1
                out.append(self._make_chunk(
                    page, idx, seg_text, local_heading_path,
                    page_type="text", parent_chunk_id=parent_id, is_parent=False,
                ))
                idx += 1
            else:
                # Long segment: parent = full segment, leaves = char-split of body
                # (heading itself stays only in the parent text — no point indexing
                # heading-only fragments).
                out.append(self._make_chunk(
                    page, idx, seg_text, local_heading_path,
                    page_type="text", parent_chunk_id=parent_id, is_parent=True,
                ))
                idx += 1
                # Char-split the *whole segment text* (incl. heading) so the heading
                # is also reachable via small-chunk retrieval.
                for small_text in self._split_chars(seg_text, self.small_size, self.overlap):
                    out.append(self._make_chunk(
                        page, idx, small_text, local_heading_path,
                        page_type="text", parent_chunk_id=parent_id, is_parent=False,
                    ))
                    idx += 1
        return out

    @staticmethod
    def _split_into_segments(text: str) -> list[tuple[str, str, int]]:
        """Walk the text, split at heading lines.
        Returns list of (heading_text, body_text, heading_level).

        Pre-heading body (text before first heading) is emitted as a segment
        with heading="" so we don't drop content.
        """
        matches = list(_HEADING_LINE_RE.finditer(text))
        if not matches:
            return [("", text, 0)]

        segments: list[tuple[str, str, int]] = []
        # Body before first heading (if any)
        first_start = matches[0].start()
        if first_start > 0:
            pre = text[:first_start].strip()
            if pre:
                segments.append(("", pre, 0))

        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            segments.append((heading, body, level))
        return segments

    @staticmethod
    def _format_segment(heading: str, body: str) -> str:
        """Render segment as a single string for chunk text."""
        if heading and body:
            return f"## {heading}\n\n{body}"
        if heading:
            return f"## {heading}"
        return body

    # ── Default char-split path (kept for back-compat / fallback) ────────

    def _chunk_default(
        self, page: ParsedPage, text: str, idx_start: int
    ) -> list[TextChunk]:
        out: list[TextChunk] = []
        idx = idx_start
        for parent_text in self._split_chars(text, self.parent_size, self.overlap):
            parent_id = f"{page.doc_id}_p{page.page_num}_c{idx}"
            out.append(self._make_chunk(
                page, idx, parent_text, page.heading_path,
                page_type="text", parent_chunk_id=parent_id, is_parent=True,
            ))
            idx += 1
            for small_text in self._split_chars(parent_text, self.small_size, self.overlap):
                out.append(self._make_chunk(
                    page, idx, small_text, page.heading_path,
                    page_type="text", parent_chunk_id=parent_id, is_parent=False,
                ))
                idx += 1
        return out

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_chunk(
        self,
        page: ParsedPage,
        idx: int,
        text: str,
        heading_path: list[str],
        *,
        page_type: str,
        parent_chunk_id: str,
        is_parent: bool,
    ) -> TextChunk:
        return TextChunk(
            doc_id=page.doc_id, doc_name=page.doc_name,
            page_num=page.page_num, chunk_index=idx,
            text=text, heading_path=heading_path,
            has_visual_on_page=bool(page.image_blocks),
            page_type=page_type, parent_chunk_id=parent_chunk_id,
            is_parent=is_parent,
        )

    @staticmethod
    def _split_chars(text: str, size: int, overlap: int) -> list[str]:
        """Split by character count — simple version. Production can swap in
        token-aware splitting later."""
        if size <= 0 or overlap < 0:
            raise ValueError("size must be > 0, overlap must be >= 0")
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            step = max(1, size - overlap)  # at least 1 to avoid infinite loop
            start += step
        return chunks

    # Public alias for backward-compat
    @staticmethod
    def _split(text: str, size: int, overlap: int) -> list[str]:
        return Chunker._split_chars(text, size, overlap)
