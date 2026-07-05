"""Stable data contracts for the chunker component (parse -> [chunk] -> embed)."""
from __future__ import annotations

from dataclasses import dataclass, field


def _acl_openness(acl: dict) -> tuple:
    """A coarse 'how many people can see this' score; LARGER == more open. Only used to break the
    rare idx-in-two-chunks tie in ChunkResult.acl_index (pick the stricter). Conservative: an unset
    or empty-allow policy scores as closed; a `public` visibility scores as most open."""
    acl = acl or {}
    return (1 if acl.get("visibility") == "public" else 0,
            0 if acl.get("unset") else 1,
            len(acl.get("allow") or []))


def _stricter(a: dict, b: dict) -> dict:
    """Return whichever acl is at least as restrictive (the safer choice on conflict)."""
    return a if _acl_openness(a) <= _acl_openness(b) else b


@dataclass
class Element:
    """Normalized parse-stage output unit. A parser adapter (e.g. from_mineru) produces a
    list[Element]; the chunker consumes ONLY this — so the parse stage stays swappable."""
    idx: int                                  # reading order (0-based, dense)
    kind: str                                 # text|table|image|chart|list|header|footer|page_number
    text: str | None = None
    text_level: int | None = None             # heading-level hint from parser (if any)
    page: int = 0
    bbox: list | None = None
    list_items: list[str] | None = None
    caption: str | None = None                # normalized table/image/chart caption
    footnote: str | None = None
    table_body: str | None = None             # HTML (tables) — generation payload
    asset_content: str | None = None          # VLM markdown/mermaid (image/chart) — low trust
    sub_type: str | None = None
    merge_prev: bool = False                  # continues the previous block (cross-page)
    image_path: str | None = None             # MinerU cropped-asset path (images/*.jpg), relative to the
    #   MinerU output root; carried VERBATIM (core does no IO). Lets the embed stage VL-embed the raw image.


@dataclass
class Section:
    sec_id: str
    doc_id: str
    level: int
    title: str
    breadcrumb: list[str]                     # ancestor titles incl. self
    start_idx: int
    end_idx: int                              # exclusive
    parent_sec_id: str | None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    kind: str                                 # text|table|image|chart
    text: str                                 # what the embedding stage embeds
    content_raw: str | None                   # generation payload for assets (HTML / VLM content)
    breadcrumb: list[str]
    section_path: str
    section_id: str | None
    section_anchor: list[int] | None          # [start_idx, end_idx] of enclosing section
    page_start: int
    page_end: int
    source_indices: list[int]                 # Element idxs this chunk came from (traceability)
    n_tokens: int
    trust: str                                # high|low
    flags: list[str] = field(default_factory=list)
    lang: str = "en"
    doc_meta: dict = field(default_factory=dict)   # document-level metadata (title/source/date/doc_type/...);
    #                                                goes into the vector-store payload for FILTER + citation.
    acl: dict = field(default_factory=dict)        # access-control policy stamped from ingest. SECURITY field:
    #   retrieval MUST hard pre-filter on it; FAIL-CLOSED (an unset/empty-allow chunk is denied by default).
    image_path: str | None = None                  # raw cropped-image path for image/chart chunks (relative to
    #   MinerU output root), so the embed stage can VL-embed the picture. None for text/table (table -> HTML in
    #   content_raw; equation -> LaTeX text). Carried verbatim; downstream resolves the path. See MULTIFORMAT §12.
    doc_type: str | None = None                    # stamped at chunk time; assemble_big reads it to pick the
    #   per-doc_type token budget at QUERY time (seal review: was absent on Chunk -> always DEFAULT_BUDGET).


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    sections: list[Section]
    banners: frozenset[str] = frozenset()     # doc-level running banners (computed once at chunk time)

    def sections_by_id(self) -> dict[str, Section]:
        return {s.sec_id: s for s in self.sections}

    def acl_index(self) -> dict[int, dict]:
        """element idx -> the acl of the chunk that produced it. assemble_big needs this to gather
        small-to-big context WITHOUT crossing an ACL boundary (a public hit must NOT pull a tightened
        sibling's raw text). On the rare idx-in-two-chunks conflict, the STRICTER acl wins (fail-closed:
        an `unset`/empty-allow policy is treated as at least as strict as a populated one)."""
        out: dict[int, dict] = {}
        for ch in self.chunks:
            for i in ch.source_indices or []:
                prev = out.get(i)
                if prev is None or _stricter(ch.acl, prev) is ch.acl:
                    out[i] = ch.acl
        return out


@dataclass
class BigBlock:
    text: str
    resolved_section: str | None
    breadcrumb: list[str]
    n_tokens: int
    climbed: int
    anchor: list[int] | None
    note: str = ""
    acl: dict | None = None     # SECURITY: the effective acl of the assembled text (anchored at the
    #   hit chunk; only same-ACL-as-hit content is gathered when an acl_index is supplied). None means
    #   the block was assembled WITHOUT acl awareness (legacy single-ACL-doc mode) — re-verify before use.
    windowed: bool = False      # True == produced by _window_within (a token-bounded WINDOW / truncated slice,
    #   NOT a complete section). Retrieval layer maps this to context_status="section_window" so the agent knows
    #   it may be partial and can `expand`. climbed>0 vs full_section both imply a whole section; windowed does NOT.
