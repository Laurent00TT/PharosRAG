from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


NavEntryType = Literal[
    "document",
    "section",
    "page",
    "table",
    "figure",
    "flowchart",
    "topic",
    "appendix",
]
NavEntrySource = Literal[
    "parser_heading",
    "page_label",
    "table_caption",
    "figure_caption",
    "page_type_rule",
    "keyword_rule",
    "manual",
]
NavEdgeType = Literal[
    "parent_child",
    "next",
    "previous",
    "related_by_heading",
    "related_by_page_window",
    "related_by_keyword",
    "same_page",
    "same_document",
]


@dataclass
class NavEntry:
    entry_id: str
    label: str
    normalized_label: str
    entry_type: NavEntryType
    source: NavEntrySource
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    order_index: int
    parent_entry_id: str | None
    resource_uris: list[str]
    source_ref_ids: list[str]
    aliases: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def contains_generated_content(self) -> bool:
        return False


@dataclass
class NavEdge:
    from_entry_id: str
    to_entry_id: str
    edge_type: NavEdgeType
    weight: float
    source: str


@dataclass(frozen=True)
class NavSourceRef:
    ref_id: str
    source_kind: Literal["heading", "page", "table", "figure", "chunk"]
    doc_id: str
    doc_name: str
    page_num: int
    chunk_index: int | None
    exact_text: str
    text_hash: str
    resource_uri: str


@dataclass(frozen=True)
class FetchTarget:
    reason: str
    resource_uri: str
    doc_id: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class EvidenceHit:
    """Compact projection of SearchResult for HybridNavigatorResponse.

    Intentionally does NOT carry text_chunk / vision_description / image_url.
    Agent must call `kb://documents/{doc_id}/pages/{page_num}` resource to
    read original content. This keeps the hybrid response pointer-only,
    consistent with the navigation-only invariant.

    Carries lightweight ranking signals (``heading_path``, ``match_channels``,
    ``page_type``) so the UI can show "why" a hit ranked where it did
    without making a second fetch. These come straight from the underlying
    SearchResult and are NOT regenerated content.
    """

    doc_id: str
    doc_name: str
    page_num: int
    resource_uri: str
    score: float
    heading_path: list[str] = field(default_factory=list)
    match_channels: list[str] = field(default_factory=list)
    page_type: str = ""


@dataclass
class NavHit:
    entry_id: str
    label: str
    entry_type: NavEntryType
    score: float
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    resource_uris: list[str]
    source_refs: list[NavSourceRef]
    neighbor_entry_ids: list[str]
    match_channels: list[str]
    contains_generated_content: bool = False


@dataclass
class HybridNavigatorResponse:
    query: str
    evidence_hits: list[EvidenceHit]
    nav_hits: list[NavHit]
    suggested_fetches: list[FetchTarget]
    routing_reason: str
