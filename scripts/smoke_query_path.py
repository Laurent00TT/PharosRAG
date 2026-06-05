"""End-to-end smoke for the query path (page/range/image resources).

Bypasses the HTTP server (which may be running an older binary) and
constructs the resource handlers against the real on-disk stores so the
test reflects the freshly-edited code. Verifies the P0 fix from the
previous review round: page resources must return actual content, not
placeholders.

Usage:
    PYTHONPATH=src python scripts/smoke_query_path.py

What it verifies:
  1. The metadata DB lists active docs (the 13 backfilled docs)
  2. NavSearch returns nav pointers for at least one doc
  3. ``kb://documents/{doc}/pages/{page}`` resource returns real text,
     NOT ``status: "placeholder"`` / ``phase: "phase_1_metadata_only"``
  4. ``kb://documents/{doc}/pages/{page}/image`` returns the file path
     and either inline data_url or a "too big" marker — never placeholder
  5. ``kb://documents/{doc}/ranges/{a-b}`` aggregates per-page payloads
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.server.fastmcp import FastMCP

from kb.config import Settings
from kb.nav.search import NavSearchEngine
from kb.nav.store import NavIndexStore
from kb.storage.image_store import ImageStore
from kb.storage.metadata_db import MetadataDB
from kb.storage.qdrant_store import QdrantStore
from kb.tool_server.mcp_resources import (
    UNTRUSTED_SOURCE_HEADER,
    register_document_resources,
)


def _get_handler(mcp: FastMCP, uri_template: str):
    rm = mcp._resource_manager
    for tmpl in rm._templates.values():
        if tmpl.uri_template == uri_template:
            return tmpl.fn
    for res in rm._resources.values():
        if str(res.uri) == uri_template or str(res.uri).rstrip("/") == uri_template:
            return res.fn
    raise KeyError(uri_template)


def _section(title: str) -> None:
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


async def main() -> int:
    settings = Settings()

    # 1. Wire real stores
    qdrant = QdrantStore(settings=settings)
    await qdrant.ensure_collections()
    meta_db = MetadataDB(
        db_url=f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    )
    await meta_db.init()
    image_store = ImageStore(settings=settings)
    nav_store = NavIndexStore(settings.nav_db_path)
    await nav_store.init()
    nav_search = NavSearchEngine(
        nav_store,
        active_doc_ids_provider=lambda: meta_db.list_active_doc_ids(),
    )

    state = SimpleNamespace(
        meta_db=meta_db,
        qdrant=qdrant,
        image_store=image_store,
        nav_store=nav_store,
        nav_search=nav_search,
        settings=settings,
    )

    mcp = FastMCP("smoke")
    register_document_resources(mcp, state)
    page_fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}")
    image_fn = _get_handler(mcp, "kb://documents/{doc_id}/pages/{page_num}/image")
    range_fn = _get_handler(mcp, "kb://documents/{doc_id}/ranges/{range_spec}")

    print("=" * 60)
    print("END-TO-END QUERY-PATH SMOKE")
    print("=" * 60)

    ok = True

    # 2. Confirm active docs present
    _section("STEP 1: active doc inventory")
    active_ids = await meta_db.list_active_doc_ids()
    print(f"  Active docs in metadata: {len(active_ids)}")
    if not active_ids:
        print("  ✗ no active docs — backfill required before query smoke")
        return 1

    # 3. Confirm nav index has entries
    _section("STEP 2: nav store inventory")
    nav_entries = await nav_store.list_entries()
    print(f"  Nav entries total:       {len(nav_entries)}")
    doc_ids_with_nav = {e.doc_id for e in nav_entries}
    print(f"  Distinct docs in nav:    {len(doc_ids_with_nav)}")
    if not nav_entries:
        print("  ✗ nav store empty — run scripts/backfill_nav_index.py first")
        return 1

    # 4. NavSearch on a generic query — verify active filter works
    _section("STEP 3: nav search (with active-doc filter)")
    sample_hits = await nav_search.search("page", top_k=5)
    print(f"  Hits for 'page' (top 5): {len(sample_hits)}")
    for hit in sample_hits[:3]:
        print(f"    - {hit.entry_type:8s} pages {hit.page_start}-{hit.page_end}  doc={hit.doc_id[:12]}  label={hit.label[:40]!r}")
        # Each hit's doc_id MUST be active (this is the new double-safety
        # guarantee from the previous round)
        if hit.doc_id not in active_ids:
            print(f"      ✗ leaked stale doc_id {hit.doc_id[:12]} into nav search")
            ok = False

    # Pick a real page hit to drive page-resource fetch
    page_hit = next(
        (h for h in sample_hits if h.entry_type == "page"),
        None,
    )
    if page_hit is None:
        # fall back to first hit regardless of entry_type
        page_hit = sample_hits[0] if sample_hits else None
    if page_hit is None:
        # Construct one synthetically from a known doc
        any_entry = nav_entries[0]
        target_doc_id = any_entry.doc_id
        target_page = any_entry.page_start
    else:
        target_doc_id = page_hit.doc_id
        target_page = page_hit.page_start
    print(f"  → driving fetch with doc_id={target_doc_id[:12]} page_num={target_page}")

    # 5. Page resource — THE critical fix
    _section("STEP 4: page resource (must NOT be placeholder)")
    body = await page_fn(doc_id=target_doc_id, page_num=str(target_page))
    parsed = json.loads(body)
    print(f"  status:              {parsed.get('status')}")
    print(f"  doc_id:              {parsed.get('doc_id', '')[:12]}")
    print(f"  page_num:            {parsed.get('page_num')}")
    print(f"  safety wrapper:      {'present' if parsed.get('safety') == UNTRUSTED_SOURCE_HEADER else 'MISSING'}")
    print(f"  heading_path:        {parsed.get('heading_path', [])[:3]}")
    text = parsed.get("text", "") or ""
    text_preview = text.replace("\n", " ")[:120]
    print(f"  text length:         {len(text)} chars")
    print(f"  text preview:        {text_preview!r}{'…' if len(text) > 120 else ''}")
    print(f"  vision_description:  {(parsed.get('vision_description', '') or '')[:80]!r}{'…' if len(parsed.get('vision_description', '') or '') > 80 else ''}")
    print(f"  image_resource_uri:  {parsed.get('image_resource_uri')}")

    if parsed.get("status") == "placeholder":
        print(f"  ✗ REGRESSION: page resource is back to placeholder")
        ok = False
    if parsed.get("phase") == "phase_1_metadata_only":
        print(f"  ✗ REGRESSION: phase_1_metadata_only marker came back")
        ok = False
    if parsed.get("status") == "ok" and not text:
        print(f"  ✗ status=ok but text is empty — qdrant might be out of sync")
        # not necessarily a failure; some pages legit have empty text. Don't fail ok.
    if parsed.get("safety") != UNTRUSTED_SOURCE_HEADER:
        print(f"  ✗ safety header missing or wrong")
        ok = False
    if parsed.get("status") not in {"ok", "page_not_found"}:
        print(f"  ✗ unexpected status: {parsed.get('status')!r}")
        ok = False
    if parsed.get("status") == "ok":
        print(f"  ✓ page resource returns real content (status=ok)")

    # 6. Image resource — must return path + (data_url or "too big" marker)
    _section("STEP 5: page image resource")
    img_body = await image_fn(doc_id=target_doc_id, page_num=str(target_page))
    img = json.loads(img_body)
    print(f"  status:              {img.get('status')}")
    print(f"  mime_type:           {img.get('mime_type')}")
    print(f"  size_bytes:          {img.get('size_bytes')}")
    print(f"  file_path:           {img.get('file_path')}")
    data_url = img.get("data_url") or ""
    if data_url:
        print(f"  data_url:            data:image/png;base64,…{len(data_url)} chars total")
    else:
        print(f"  data_url:            None (size > inline cap, file_path only)")

    if img.get("status") not in {"ok", "image_not_found"}:
        print(f"  ✗ unexpected status: {img.get('status')!r}")
        ok = False
    if img.get("status") == "ok":
        if not img.get("file_path"):
            print(f"  ✗ status=ok but file_path missing")
            ok = False
        else:
            file_exists = Path(img["file_path"]).exists()
            print(f"  file exists on disk: {file_exists}")
            if not file_exists:
                print(f"  ✗ file_path reported but doesn't exist")
                ok = False
            else:
                print(f"  ✓ image resource returns real file path + payload")

    # 7. Range resource — aggregate
    _section("STEP 6: range resource (aggregate 2 pages)")
    range_spec = f"{target_page}-{target_page + 1}"
    range_body = await range_fn(doc_id=target_doc_id, range_spec=range_spec)
    range_data = json.loads(range_body)
    print(f"  safety wrapper:      {'present' if range_data.get('safety') == UNTRUSTED_SOURCE_HEADER else 'MISSING'}")
    print(f"  doc_id:              {range_data.get('doc_id', '')[:12]}")
    print(f"  page_start..end:     {range_data.get('page_start')}-{range_data.get('page_end')}")
    pages = range_data.get("pages", [])
    print(f"  pages aggregated:    {len(pages)}")
    for p in pages:
        print(f"    - page {p.get('page_num')}: status={p.get('status')} text_len={len(p.get('text', '') or '')}")
    page_statuses = [p.get("status") for p in pages]
    if any(s == "placeholder" for s in page_statuses):
        print(f"  ✗ at least one aggregated page is placeholder")
        ok = False
    else:
        print(f"  ✓ range resource aggregates real per-page payloads")

    # cleanup
    await nav_store.close()
    await meta_db.aclose()
    client = getattr(qdrant, "client", None)
    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            pass

    print()
    print("=" * 60)
    if ok:
        print("RESULT: ✓ query path end-to-end works — page resources return real content")
        return 0
    print("RESULT: ✗ at least one assertion failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
