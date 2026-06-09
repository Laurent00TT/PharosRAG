# src/kb/models.py
from dataclasses import dataclass, field
from typing import Optional
from datetime import date

@dataclass
class ParsedPage:
    doc_id: str
    doc_name: str
    page_num: int
    structured_text: str
    heading_path: list[str]
    page_image: bytes
    image_blocks: list[dict]
    tables: list[dict]
    metadata: dict

@dataclass
class TextChunk:
    doc_id: str
    doc_name: str
    page_num: int
    chunk_index: int
    text: str
    heading_path: list[str]
    has_visual_on_page: bool
    page_type: str
    parent_chunk_id: str
    is_parent: bool = False

@dataclass
class VisionPage:
    doc_id: str
    doc_name: str
    page_num: int
    image_url: str
    heading_path: list[str]
    vision_tier: str
    page_type: str = "text"

@dataclass
class DescPage:
    doc_id: str
    doc_name: str
    page_num: int
    page_type: str
    figure_index: Optional[str]
    figure_caption: Optional[str]
    position_hint: Optional[str]
    description: str
    key_entities: list[str]
    image_url: str
    heading_path: list[str]

@dataclass
class SearchResult:
    """A single ranked search hit returned to callers.

    score: Always the cross-encoder rerank score (0-1, sigmoid-normalized).
        Internal pipeline stages briefly use RRF scores, but they are
        replaced by the reranker before SearchEngine.search() returns.
        External consumers can rely on score being on the [0, 1] scale.
    match_channels: Which retrieval channels (dense/sparse/vision/desc)
        contributed this hit during RRF fusion.
    """
    doc_id: str
    doc_name: str
    page_num: int
    heading_path: list[str]
    text_chunk: str
    vision_description: Optional[str]
    figure_index: Optional[str]
    figure_caption: Optional[str]
    image_url: str
    page_type: str
    doc_version: Optional[str]
    effective_date: Optional[date]
    score: float
    match_channels: list[str]

@dataclass
class SearchResponse:
    results: list[SearchResult]
    total: int
    suggested_next_queries: list[str] = field(default_factory=list)
    low_confidence: bool = False
    confidence_reason: str = ""
    # Cold-start UX fields. Both populated together by SearchEngine when total == 0
    # AND list_active_doc_ids() is empty. Otherwise both remain at their defaults.
    message: str = ""
    suggested_upload_topics: list[str] = field(default_factory=list)

@dataclass
class DocMeta:
    doc_id: str
    doc_name: str
    version: Optional[str]
    status: str
    effective_date: Optional[date]
    expiry_date: Optional[date]

@dataclass
class DocumentSummary(DocMeta):
    page_count: int = 0
    has_vision: bool = False
