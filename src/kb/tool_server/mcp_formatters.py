from __future__ import annotations

from typing import Any


def truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + " [truncated]"


def evidence_hit_to_structured(hit: Any) -> dict:
    # Security review P0 #2: surface text-chunk presence at structured-
    # content level so machine consumers (downstream Agents, eval
    # pipelines) can partition evidence from hint without re-parsing
    # the text channel. text_chunk and image_resource_uri are
    # citable evidence; generated description is a hint only.
    has_text_evidence = bool((hit.text_chunk or "").strip())
    has_generated_desc = bool((hit.vision_description or "").strip())
    return {
        "result_type": "evidence" if has_text_evidence else "hint",
        "doc_id": hit.doc_id,
        "doc_name": hit.doc_name,
        "page_num": hit.page_num,
        "page_type": hit.page_type,
        "score": round(float(hit.score), 4),
        "match_channels": list(hit.match_channels or []),
        "resource_uri": f"kb://documents/{hit.doc_id}/pages/{hit.page_num}",
        "image_resource_uri": f"kb://documents/{hit.doc_id}/pages/{hit.page_num}/image" if hit.image_url else "",
        "cite_as": f"[{hit.doc_id}:{hit.page_num}]",
        "has_text_evidence": has_text_evidence,
        "has_generated_description": has_generated_desc,
    }


def evidence_hit_to_text(hit: Any, index: int, text_budget: int = 800) -> str:
    """Render one hit for the MCP text channel.

    Security review P0 #2: text_chunk and vision_description are DIFFERENT
    classes of content. text_chunk is parsed-from-the-page raw text (real
    evidence the Agent can cite); vision_description is LLM-generated
    description of the rendered page image (a recall hint, NOT evidence —
    citing it would amplify model hallucination). Render each with a
    distinct label so the Agent doesn't conflate them.
    """
    text_chunk = (hit.text_chunk or "").strip()
    desc_hint = (hit.vision_description or "").strip()
    if text_chunk:
        preview = truncate_text(text_chunk, text_budget)
        body = f"Text preview (evidence):\n{preview}"
    elif desc_hint:
        preview = truncate_text(desc_hint, text_budget)
        body = (
            f"Generated description (RECALL HINT only — not evidence; "
            f"fetch the page resource to cite):\n{preview}"
        )
    else:
        body = "(no text or description — fetch page resource for content)"
    return (
        f"[#{index} score={hit.score:.2f}]\n"
        f"Doc: {hit.doc_name} ({hit.doc_id})\n"
        f"Page: {hit.page_num}\n"
        f"Resource: kb://documents/{hit.doc_id}/pages/{hit.page_num}\n"
        f"{body}\n"
        f"Cite as: [{hit.doc_id}:{hit.page_num}]"
    )


def evidence_response_to_mcp_payload(response: Any, text_budget: int = 800) -> dict:
    return {
        "content": "\n\n".join(
            evidence_hit_to_text(hit, index, text_budget=text_budget)
            for index, hit in enumerate(response.results, 1)
        ) or "No matching documents found.",
        "structuredContent": {
            "results": [evidence_hit_to_structured(hit) for hit in response.results],
            "total": response.total,
        },
    }
