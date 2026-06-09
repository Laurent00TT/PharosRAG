"""End-to-end MCP server smoke test via the streamable-HTTP transport.

Speaks raw JSON-RPC at /mcp/ (no MCP client SDK) so we test the wire format
exactly as an external agent would see it.

Covers:
  1. initialize handshake (protocol version, capabilities, session id)
  2. tools/list (schema for kb_search and kb_list + nav tools)
  3. tools/call kb_list
  4. tools/call kb_search (English query)
  5. tools/call kb_search (Chinese query)
  6. tools/call kb_search with top_k=99 (server clamps to 20)
  7. tools/call kb_search with no-match query (graceful empty)
  8. tools/call unknown_tool (proper JSON-RPC error)
  9. Two concurrent sessions (isolation)
 10. resources/list (kb://documents and friends)
 11. resources/templates/list (page + nav templates)
 12. resources/read kb://documents (JSON body parses)
 13. kb_search_nav (returns structuredContent.nav_hits)
 14. kb_hybrid_search (evidence_hits + nav_hits + suggested_fetches)
 15. kb_get_status (capability snapshot)
 16. kb_list_documents (per-doc nav_indexed bool)
 17. kb_get_toc (probe doc_id from Test 16)
 18. kb_expand_nav_entry (gentle 404 on nonexistent)
 19. kb_add_nav_alias (write gate off → saved=False)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx

URL = "http://localhost:8000/mcp/"


def _resolve_token() -> str:
    """Resolve the user token from CLI --token or KB_USER_TOKEN env.

    T1b hard cutover: every /mcp request must carry a valid user token
    (Authorization: Bearer ... or X-API-Key: ...). The smoke script
    no longer works against an unauthenticated server.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        help="User token (kb_...). Falls back to KB_USER_TOKEN env var.",
    )
    args = parser.parse_args()
    token = args.token or os.environ.get("KB_USER_TOKEN")
    if not token:
        print(
            "ERROR: no user token provided.\n"
            "  Either:\n"
            "    $env:KB_USER_TOKEN = '<your token>'  (PowerShell)\n"
            "    export KB_USER_TOKEN='<your token>'   (bash)\n"
            "  Or pass --token <token>.\n"
            "\n"
            "To create one:\n"
            "    python scripts/manage_users.py create <username> --role admin",
            file=sys.stderr,
        )
        sys.exit(2)
    return token


HEADERS_BASE = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    # "Authorization": "Bearer <token>" is injected at __main__ time
    # so module import (e.g. from tests) doesn't trigger argparse/sys.exit.
}


def _parse_sse(text: str) -> dict:
    """Pick out the first JSON-RPC payload from an SSE response body."""
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line.removeprefix("data:").strip())
    raise RuntimeError(f"No SSE data in: {text[:200]!r}")


async def open_session(client: httpx.AsyncClient) -> str:
    """initialize + notifications/initialized, return session_id."""
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "smoke-test", "version": "1.0"},
        },
    }
    r = await client.post(URL, headers=HEADERS_BASE, json=req)
    r.raise_for_status()
    sid = r.headers.get("mcp-session-id")
    assert sid, "server didn't return mcp-session-id header"
    init = _parse_sse(r.text)
    assert init["result"]["serverInfo"]["name"] == "kb"
    # follow-up "initialized" notification
    await client.post(
        URL,
        headers={**HEADERS_BASE, "mcp-session-id": sid},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    return sid


async def rpc(client: httpx.AsyncClient, sid: str, rpc_id: int, method: str, params: dict | None = None) -> dict:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    r = await client.post(URL, headers={**HEADERS_BASE, "mcp-session-id": sid}, json=body)
    r.raise_for_status()
    return _parse_sse(r.text)


async def call_tool(client: httpx.AsyncClient, sid: str, rpc_id: int, name: str, args: dict) -> dict:
    return await rpc(client, sid, rpc_id, "tools/call", {"name": name, "arguments": args})


async def main() -> int:
    failures: list[str] = []
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # ── Test 1: initialize ────────────────────────────────────────────
        print("=== Test 1: initialize ===")
        sid = await open_session(client)
        print(f"  session: {sid}")

        # ── Test 2: tools/list ────────────────────────────────────────────
        print("\n=== Test 2: tools/list ===")
        resp = await rpc(client, sid, 2, "tools/list")
        tools = resp["result"]["tools"]
        names = [t["name"] for t in tools]
        print(f"  tools: {names}")
        # After Phase 5+Dispatch 2: 2 core + 6 nav-read + 3 gated writes = 11 tools.
        # Use issubset so the assertion stays tolerant of future additions.
        expected_min_tools = {
            "kb_search", "kb_list",
            "kb_search_nav", "kb_hybrid_search",
            "kb_expand_nav_entry", "kb_list_documents", "kb_get_toc", "kb_get_status",
            "kb_add_nav_alias", "kb_hide_nav_entry", "kb_rebuild_nav_index",
        }
        if not expected_min_tools.issubset(set(names)):
            missing = expected_min_tools - set(names)
            failures.append(f"tools/list missing expected tools {missing}; got {names}")
        for t in tools:
            schema = t.get("inputSchema", {})
            print(f"    - {t['name']}: required={schema.get('required',[])} props={list(schema.get('properties',{}).keys())}")

        # ── Test 3: kb_list ──────────────────────────────────────────────
        print("\n=== Test 3: kb_list ===")
        resp = await call_tool(client, sid, 3, "kb_list", {})
        text = resp["result"]["content"][0]["text"]
        first_line = text.split("\n", 1)[0]
        print(f"  {first_line}")
        if "Total:" not in text or "documents" not in text:
            failures.append(f"kb_list missing Total/documents: {text[:100]!r}")

        # ── Test 4: kb_search English ────────────────────────────────────
        print("\n=== Test 4: kb_search (English) ===")
        t0 = time.monotonic()
        resp = await call_tool(client, sid, 4, "kb_search", {"query": "carbon emissions of NLP training", "top_k": 2})
        elapsed = time.monotonic() - t0
        text = resp["result"]["content"][0]["text"]
        n_hits = text.count("[#")
        print(f"  hits: {n_hits} | elapsed: {elapsed:.1f}s")
        print(f"  first 200 chars: {text[:200]!r}")
        if n_hits == 0:
            failures.append("kb_search English: 0 hits on a query that should match Strubell paper")
        if "Cite as:" not in text:
            failures.append("kb_search: 'Cite as:' marker missing in output")

        # ── Test 5: kb_search Chinese ────────────────────────────────────
        print("\n=== Test 5: kb_search (Chinese) ===")
        resp = await call_tool(client, sid, 5, "kb_search", {"query": "现在完成时", "top_k": 3})
        text = resp["result"]["content"][0]["text"]
        n_hits = text.count("[#")
        print(f"  hits: {n_hits}")
        if n_hits == 0:
            failures.append("kb_search Chinese: 0 hits on query that should match grammar book")
        # Verify Chinese came through UTF-8 (no mojibake)
        if "Doc:" in text and not any(ord(c) > 127 for c in text):
            failures.append("kb_search Chinese: no non-ASCII in output — possible encoding loss")

        # ── Test 6: top_k clamp ──────────────────────────────────────────
        print("\n=== Test 6: kb_search top_k=99 (should clamp to 20) ===")
        resp = await call_tool(client, sid, 6, "kb_search", {"query": "language", "top_k": 99})
        text = resp["result"]["content"][0]["text"]
        n_hits = text.count("[#")
        print(f"  hits: {n_hits}")
        if n_hits > 20:
            failures.append(f"kb_search top_k=99 returned {n_hits} hits — not clamped")

        # ── Test 7: empty-result query ───────────────────────────────────
        print("\n=== Test 7: kb_search (no-match nonsense) ===")
        resp = await call_tool(client, sid, 7, "kb_search", {"query": "xyzqwertyzzz blorftnaglefooz", "top_k": 5})
        text = resp["result"]["content"][0]["text"]
        print(f"  output: {text[:120]!r}")
        # Either real "no match" string or low-relevance hits — both are acceptable
        # as long as no exception bubbled up
        if "Error:" in text:
            failures.append(f"kb_search nonsense query raised internal error: {text[:100]!r}")

        # ── Test 8: unknown tool ─────────────────────────────────────────
        print("\n=== Test 8: tools/call unknown_tool ===")
        resp = await call_tool(client, sid, 8, "this_tool_does_not_exist", {})
        # MCP returns either JSON-RPC error or result with isError=true
        if "error" in resp:
            print(f"  ✓ got JSON-RPC error: {resp['error'].get('message','?')[:80]}")
        elif resp.get("result", {}).get("isError"):
            print(f"  ✓ got isError=True content")
        else:
            failures.append(f"unknown_tool did not error: {resp!r}")

        # ── Test 9: concurrent sessions ──────────────────────────────────
        print("\n=== Test 9: 2 concurrent sessions ===")
        async with httpx.AsyncClient(timeout=timeout) as client_b:
            sid_b = await open_session(client_b)
            print(f"  session A: {sid}")
            print(f"  session B: {sid_b}")
            if sid == sid_b:
                failures.append("two sessions got the same ID — isolation broken")
            # both call simultaneously
            r_a, r_b = await asyncio.gather(
                call_tool(client, sid, 9, "kb_search", {"query": "by", "top_k": 1}),
                call_tool(client_b, sid_b, 1, "kb_search", {"query": "since", "top_k": 1}),
            )
            text_a = r_a["result"]["content"][0]["text"]
            text_b = r_b["result"]["content"][0]["text"]
            print(f"  A got: {text_a.split(chr(10),1)[0]}")
            print(f"  B got: {text_b.split(chr(10),1)[0]}")
            if not text_a or not text_b:
                failures.append("one of the concurrent calls returned empty")

        # ── Test 10: resources/list returns the 3 KB resources ───────────
        print("\n=== Test 10: resources/list ===")
        resp = await rpc(client, sid, 10, "resources/list")
        resources = resp["result"]["resources"]
        uris = {r["uri"] for r in resources}
        print(f"  found {len(resources)} resources: {sorted(uris)}")
        # Note: MCP returns CONCRETE resources from @mcp.resource decorator; URI
        # templates with {placeholders} show up in resources/templates/list, not here.
        if "kb://documents" not in uris:
            failures.append(f"resources/list missing kb://documents; got {sorted(uris)}")

        # ── Test 11: resources/templates/list returns the parameterized URIs ──
        print("\n=== Test 11: resources/templates/list ===")
        resp = await rpc(client, sid, 11, "resources/templates/list")
        templates = resp["result"].get("resourceTemplates", [])
        template_uris = {t["uriTemplate"] for t in templates}
        print(f"  found {len(templates)} templates: {sorted(template_uris)}")
        if not any("/pages/" in uri for uri in template_uris):
            failures.append(f"templates/list should include page template; got {sorted(template_uris)}")
        # Dispatch 2 added the nav entry + per-doc TOC templates.
        expected_nav_templates = {
            "kb://nav/entries/{entry_id}",
            "kb://documents/{doc_id}/toc",
        }
        if not expected_nav_templates.issubset(template_uris):
            missing = expected_nav_templates - template_uris
            failures.append(f"templates/list missing nav templates {missing}; got {sorted(template_uris)}")

        # ── Test 12: resources/read kb://documents ───────────────────────
        print("\n=== Test 12: resources/read kb://documents ===")
        resp = await rpc(client, sid, 12, "resources/read", {"uri": "kb://documents"})
        contents = resp["result"].get("contents", [])
        if not contents:
            failures.append("resources/read kb://documents returned empty contents")
        else:
            body = contents[0].get("text", "")
            try:
                data = json.loads(body)
                doc_count = len(data.get("documents", []))
                print(f"  read OK, body parses to {doc_count} documents")
            except json.JSONDecodeError:
                failures.append(f"resources/read body is not valid JSON: {body[:120]}")

        # ── Test 13: kb_search_nav ───────────────────────────────────────
        print("\n=== Test 13: kb_search_nav ===")
        resp = await call_tool(client, sid, 13, "kb_search_nav", {"query": "审批流程", "top_k": 3})
        result = resp.get("result", {})
        content = result.get("content", [])
        structured = result.get("structuredContent", {}) or {}
        if not content:
            failures.append("kb_search_nav returned empty content list")
        if "nav_hits" not in structured:
            failures.append(f"kb_search_nav missing structuredContent.nav_hits; got keys {list(structured.keys())}")
        else:
            hits = structured.get("nav_hits", [])
            print(f"  nav_hits: {len(hits)} (empty is OK if nav index not populated)")

        # ── Test 14: kb_hybrid_search ────────────────────────────────────
        print("\n=== Test 14: kb_hybrid_search ===")
        resp = await call_tool(client, sid, 14, "kb_hybrid_search", {"query": "carbon emissions", "top_k": 3})
        result = resp.get("result", {})
        structured = result.get("structuredContent", {}) or {}
        # Tolerate either hybrid-disabled stub ({}) or populated response.
        if structured:
            for k in ("evidence_hits", "nav_hits", "suggested_fetches"):
                if k not in structured:
                    failures.append(f"kb_hybrid_search missing structuredContent.{k}; got keys {list(structured.keys())}")
            print(
                f"  evidence_hits={len(structured.get('evidence_hits', []))} "
                f"nav_hits={len(structured.get('nav_hits', []))} "
                f"suggested_fetches={len(structured.get('suggested_fetches', []))}"
            )
        else:
            print("  hybrid navigator not initialized — empty structuredContent (acceptable)")

        # ── Test 15: kb_get_status ───────────────────────────────────────
        print("\n=== Test 15: kb_get_status ===")
        resp = await call_tool(client, sid, 15, "kb_get_status", {})
        result = resp.get("result", {})
        structured = result.get("structuredContent", {}) or {}
        required_status_keys = {"evidence_ready", "nav_ready", "hybrid_ready", "document_count"}
        missing = required_status_keys - set(structured.keys())
        if missing:
            failures.append(f"kb_get_status missing keys {missing}; got {list(structured.keys())}")
        print(
            f"  evidence_ready={structured.get('evidence_ready')} "
            f"nav_ready={structured.get('nav_ready')} "
            f"hybrid_ready={structured.get('hybrid_ready')} "
            f"document_count={structured.get('document_count')}"
        )

        # ── Test 16: kb_list_documents ───────────────────────────────────
        print("\n=== Test 16: kb_list_documents ===")
        resp = await call_tool(client, sid, 16, "kb_list_documents", {})
        result = resp.get("result", {})
        structured = result.get("structuredContent", {}) or {}
        docs = structured.get("documents", [])
        print(f"  total documents: {len(docs)}")
        if "documents" not in structured:
            failures.append(f"kb_list_documents missing structuredContent.documents; got {list(structured.keys())}")
        for d in docs:
            if "nav_indexed" not in d or not isinstance(d["nav_indexed"], bool):
                failures.append(f"kb_list_documents entry missing bool nav_indexed: {d}")
                break
        # Capture a probe doc_id for the kb_get_toc test below.
        probe_doc_id = docs[0]["doc_id"] if docs else "doc"
        print(f"  probe_doc_id for next test: {probe_doc_id!r}")

        # ── Test 17: kb_get_toc ──────────────────────────────────────────
        print("\n=== Test 17: kb_get_toc ===")
        resp = await call_tool(client, sid, 17, "kb_get_toc", {"doc_id": probe_doc_id})
        result = resp.get("result", {})
        structured = result.get("structuredContent", {}) or {}
        if "entries" not in structured:
            failures.append(f"kb_get_toc missing structuredContent.entries; got {list(structured.keys())}")
        entries = structured.get("entries", [])
        print(f"  TOC entries: {len(entries)} (0 is OK if nav not built for this doc)")

        # ── Test 18: kb_expand_nav_entry (nonexistent → gentle 404) ──────
        print("\n=== Test 18: kb_expand_nav_entry (nonexistent) ===")
        resp = await call_tool(client, sid, 18, "kb_expand_nav_entry", {"entry_id": "nonexistent"})
        result = resp.get("result", {})
        content_list = result.get("content", [])
        text = content_list[0].get("text", "") if content_list else ""
        structured = result.get("structuredContent", {}) or {}
        if "not" not in text.lower() and "not initialized" not in text.lower():
            failures.append(f"kb_expand_nav_entry should gently report missing; got {text[:120]!r}")
        neighbors = structured.get("neighbors", [])
        print(f"  neighbors: {len(neighbors)} (empty expected for nonexistent id)")

        # ── Test 19: kb_add_nav_alias (write gate) ───────────────────────
        print("\n=== Test 19: kb_add_nav_alias (default write gate = disabled) ===")
        resp = await call_tool(client, sid, 19, "kb_add_nav_alias", {"entry_id": "x", "alias": "y"})
        result = resp.get("result", {})
        structured = result.get("structuredContent", {}) or {}
        if structured.get("saved") is not False:
            failures.append(f"kb_add_nav_alias should return saved=False when write gate off; got {structured}")
        if structured.get("reason") != "write_tools_disabled":
            failures.append(f"kb_add_nav_alias reason should be 'write_tools_disabled'; got {structured.get('reason')!r}")
        print(f"  saved={structured.get('saved')} reason={structured.get('reason')!r}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"✗ {len(failures)} FAILURE(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("✓ All MCP smoke tests passed.")
    return 0


if __name__ == "__main__":
    HEADERS_BASE["Authorization"] = f"Bearer {_resolve_token()}"
    raise SystemExit(asyncio.run(main()))
