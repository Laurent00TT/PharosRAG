"""chunker — a pluggable chunking component for a RAG indexing pipeline.

Pipeline position:   parse  ->  [ chunker ]  ->  embed
The chunker consumes normalized Element[] (produced by a parser adapter) and emits
Chunk[] (embed `chunk.text`, store the rest as metadata) + Section[] (the heading tree).
At query time, `assemble_big` does small-to-big retrieval (climb / merge siblings).

Quick start:
    from chunker import Chunker
    from chunker.adapters.mineru import from_mineru

    elements = from_mineru(content_list_json, layout_json)     # parse-stage adapter
    result = Chunker().chunk(elements, doc_id="d1", doc_type="academic_paper", lang="en")
    # -> embed [c.text for c in result.chunks]; persist result.sections
    big = Chunker().assemble_big(hit_chunk, result, elements)  # query-time small-to-big
"""
from .core import RESTRICTED_ACL, Chunker
from .meta import extract_doc_meta
from .retrieve import assemble_big
from .types import BigBlock, Chunk, ChunkResult, Element, Section

__all__ = ["Chunker", "assemble_big", "extract_doc_meta", "RESTRICTED_ACL",
           "Element", "Chunk", "Section", "ChunkResult", "BigBlock"]
