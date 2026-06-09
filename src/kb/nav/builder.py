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
        # Duplicate page_num collides page entry_ids (id = hash(.., "Page N",
        # page_num, page_num)) → store's INSERT OR REPLACE silently drops one and
        # orphans its next/prev edges. Fail loud; auto_build swallows this and
        # skips nav for the malformed doc instead of building a corrupt tree.
        page_nums = [p.page_num for p in pages]
        if len(set(page_nums)) != len(page_nums):
            from collections import Counter
            dups = sorted(n for n, c in Counter(page_nums).items() if c > 1)
            raise ValueError(
                f"build_from_pages: duplicate page_num(s) {dups} for doc "
                f"{first.doc_id!r}; nav requires one page per page_num"
            )
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
        doc_entry = NavEntry(
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
        )
        entries.append(doc_entry)

        # ── Multi-level section tree from heading_path ──────────────────────
        # heading_path is the parser's heading_stack snapshot per page (see
        # parser/_middle_to_pages.py): the full ancestor chain root→…→current,
        # e.g. ["5 Security", "5.3 Escalation"]. We rebuild the section tree by
        # MIRRORING that stack: `active` holds the currently-open section chain
        # as (normalized-label, entry) pairs (normalized so OCR/whitespace/case
        # jitter does not fragment a section). For each page we keep common prefix with
        # the previous page, pop what no longer continues, then push the newly
        # entered levels as fresh `section` entries.
        #
        # Why a stack and NOT a global {prefix: entry_id} dict: two DISTINCT
        # sections that happen to share a label (e.g. two un-numbered top-level
        # "Overview" H1s separated by a "Methods" H1) must not collapse into one
        # entry. The stack treats them as push→pop→push — two independent
        # sections with different page_start → different entry_id → no
        # cross-spanning range. A global dict would merge them, and the
        # coverage-based page_end below would then swallow everything between.
        active: list[tuple[str, NavEntry]] = []
        order = 1
        previous_page_entry_id: str | None = None

        for page in pages:
            heading_path = page.heading_path or []
            # Compare on the NORMALIZED label so OCR/whitespace/case jitter between
            # consecutive pages (e.g. "5 Security" vs "5 Security ") does not split
            # one section into two fragments. Raw labels stay on the entries.
            norm_path = [normalize_label(h) for h in heading_path]

            # 1) common-prefix length between the open chain and this page
            common = 0
            while (
                common < len(active)
                and common < len(heading_path)
                and active[common][0] == norm_path[common]
            ):
                common += 1
            # 2) close sections that no longer continue (their page_end was
            #    finalized by the pages visited while they were open)
            del active[common:]
            # 3) open the newly-entered levels as section entries
            for depth in range(common, len(heading_path)):
                label = heading_path[depth]
                parent_entry = active[-1][1] if active else doc_entry
                # full path → id distinguishes same-named sections under
                # different ancestors; page_start distinguishes same-named
                # INDEPENDENT sections (the two-"Overview" case above). page_end
                # is held at page_start so the id stays stable while page_end
                # grows during the loop. CONTRACT: any future "find-or-create
                # section" lookup MUST also pass page_start as BOTH range args —
                # the stored id encodes page_start..page_start, NOT ..page_end.
                full_label = " > ".join(heading_path[: depth + 1])
                section_id = nav_entry_id(
                    page.doc_id, "section", full_label, page.page_num, page.page_num,
                )
                section = NavEntry(
                    entry_id=section_id,
                    label=label,
                    normalized_label=normalize_label(label),
                    entry_type="section",
                    source="parser_heading",
                    doc_id=page.doc_id,
                    doc_name=page.doc_name,
                    page_start=page.page_num,
                    page_end=page.page_num,  # extended below as pages are visited
                    order_index=order,
                    parent_entry_id=parent_entry.entry_id,
                    resource_uris=[],  # backfilled after the loop (page_end final)
                    source_ref_ids=[],
                )
                entries.append(section)
                edges.append(NavEdge(parent_entry.entry_id, section_id, "parent_child", 1.0, "builder"))
                active.append((norm_path[depth], section))
                order += 1
            # 4) this page belongs to every open ancestor → extend their
            #    page_end (coverage-based: the real max page they contain).
            #    A section pushed THIS iteration has page_end == page.page_num, so
            #    the strict `>` is False for it (no self-extension) — do NOT relax
            #    to `>=` or every section would over-count by one page.
            for _, section in active:
                if page.page_num > section.page_end:
                    section.page_end = page.page_num

            # 5) page entry parents to the deepest open section, or the doc root
            #    when heading_path is empty (degraded page-as-entry fallback).
            #    Its label is the page anchor ("Page N"), NOT the heading path:
            #    the heading levels are already owned by the section ancestors
            #    (reachable via parent_entry_id). Carrying the full heading_path
            #    on the page too would make every page under "5.3 Escalation"
            #    also match a nav search for "escalation", duplicating the
            #    section hit. Pages are land-on anchors; sections own titles.
            page_parent = active[-1][1] if active else doc_entry
            page_label = f"Page {page.page_num}"
            entry_type = "flowchart" if page.metadata.get("page_type") == "flowchart" else "page"
            entry_id = nav_entry_id(page.doc_id, entry_type, page_label, page.page_num, page.page_num)
            entries.append(NavEntry(
                entry_id=entry_id,
                label=page_label,
                normalized_label=normalize_label(page_label),
                entry_type=entry_type,
                source="page_label",
                doc_id=page.doc_id,
                doc_name=page.doc_name,
                page_start=page.page_num,
                page_end=page.page_num,
                order_index=order,
                parent_entry_id=page_parent.entry_id,
                resource_uris=[page_uri(page.doc_id, page.page_num)],
                source_ref_ids=[],
            ))
            edges.append(NavEdge(page_parent.entry_id, entry_id, "parent_child", 1.0, "builder"))
            if previous_page_entry_id:
                edges.append(NavEdge(previous_page_entry_id, entry_id, "next", 1.0, "builder"))
                edges.append(NavEdge(entry_id, previous_page_entry_id, "previous", 1.0, "builder"))
            previous_page_entry_id = entry_id
            order += 1

        # 6) backfill section resource_uris now that page_end is final
        for entry in entries:
            if entry.entry_type == "section":
                entry.resource_uris = [page_range_uri(entry.doc_id, entry.page_start, entry.page_end)]

        return NavBuildResult(entries=entries, edges=edges)
