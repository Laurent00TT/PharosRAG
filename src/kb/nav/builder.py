from __future__ import annotations

from dataclasses import dataclass

from kb.models import ParsedPage
from kb.nav.models import NavEdge, NavEntry
from kb.nav.normalize import nav_entry_id, normalize_label
from kb.nav.resources import page_range_uri, page_uri

# TODO: keyword_rule/key_entities aliases (per design §6.2) require
# literal-string presence verification in source text before being
# added to entry.aliases. Out of scope for Task 8 (heading + page only).


@dataclass
class NavBuildResult:
    entries: list[NavEntry]
    edges: list[NavEdge]


class NavIndexBuilder:
    def build_from_pages(self, pages: list[ParsedPage]) -> NavBuildResult:
        entries: list[NavEntry] = []
        edges: list[NavEdge] = []
        if not pages:
            return NavBuildResult(entries=[], edges=[])

        first = pages[0]
        # Doc root spans the actual page_num range from pages, NOT 1..len(pages).
        # Parser emits page_num via mineru's page_idx which is 0-based, so the
        # earlier 1-based root both lost page 0 and pointed at a non-existent
        # last page. Using min/max keeps the whole nav layer in one consistent
        # numbering scheme (0-based, matching qdrant payload).
        doc_page_start = min(p.page_num for p in pages)
        doc_page_end = max(p.page_num for p in pages)
        doc_entry_id = nav_entry_id(
            first.doc_id, "document", first.doc_name, doc_page_start, doc_page_end,
        )
        entries.append(NavEntry(
            entry_id=doc_entry_id,
            label=first.doc_name,
            normalized_label=normalize_label(first.doc_name),
            entry_type="document",
            source="page_label",
            doc_id=first.doc_id,
            doc_name=first.doc_name,
            page_start=doc_page_start,
            page_end=doc_page_end,
            order_index=0,
            parent_entry_id=None,
            resource_uris=[page_range_uri(first.doc_id, doc_page_start, doc_page_end)],
            source_ref_ids=[],
        ))

        previous_page_entry_id: str | None = None
        order = 1
        for page in pages:
            heading = " > ".join(page.heading_path) if page.heading_path else f"Page {page.page_num}"
            entry_type = "flowchart" if page.metadata.get("page_type") == "flowchart" else "page"
            entry_id = nav_entry_id(page.doc_id, entry_type, heading, page.page_num, page.page_num)
            entries.append(NavEntry(
                entry_id=entry_id,
                label=heading,
                normalized_label=normalize_label(heading),
                entry_type=entry_type,
                source="parser_heading" if page.heading_path else "page_label",
                doc_id=page.doc_id,
                doc_name=page.doc_name,
                page_start=page.page_num,
                page_end=page.page_num,
                order_index=order,
                parent_entry_id=doc_entry_id,
                resource_uris=[page_uri(page.doc_id, page.page_num)],
                source_ref_ids=[],
            ))
            edges.append(NavEdge(doc_entry_id, entry_id, "parent_child", 1.0, "builder"))
            if previous_page_entry_id:
                edges.append(NavEdge(previous_page_entry_id, entry_id, "next", 1.0, "builder"))
                edges.append(NavEdge(entry_id, previous_page_entry_id, "previous", 1.0, "builder"))
            previous_page_entry_id = entry_id
            order += 1
        return NavBuildResult(entries=entries, edges=edges)
