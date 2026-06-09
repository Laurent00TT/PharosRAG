from __future__ import annotations

from kb.nav.resources import page_range_uri, page_uri


def test_page_uri():
    assert page_uri("doc", 3) == "kb://documents/doc/pages/3"


def test_page_range_uri():
    assert page_range_uri("doc", 3, 5) == "kb://documents/doc/ranges/3-5"


def test_nav_entry_has_no_generated_content_field():
    from kb.nav.models import NavEntry
    fields = NavEntry.__dataclass_fields__
    assert "content_markdown" not in fields
    assert "summary" not in fields


def test_nav_entry_contains_generated_content_is_false():
    from kb.nav.models import NavEntry
    entry = NavEntry(
        entry_id="e1",
        label="Approval",
        normalized_label="approval",
        entry_type="section",
        source="parser_heading",
        doc_id="doc",
        doc_name="manual.pdf",
        page_start=1,
        page_end=3,
        order_index=1,
        parent_entry_id=None,
        resource_uris=["kb://documents/doc/ranges/1-3"],
        source_ref_ids=["ref-1"],
    )
    assert entry.contains_generated_content is False


def test_evidence_hit_carries_no_text_body():
    from kb.nav.models import EvidenceHit
    fields = EvidenceHit.__dataclass_fields__
    for forbidden in ("text_chunk", "vision_description", "image_url", "content_markdown"):
        assert forbidden not in fields, f"EvidenceHit must stay pointer-only; saw {forbidden}"
