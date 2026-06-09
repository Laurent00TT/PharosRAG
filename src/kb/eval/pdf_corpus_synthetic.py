"""Deterministic synthetic internal PDFs for Corpus v1 gold eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pdf_corpus import (
    compute_md5,
    make_manifest_record,
)


SYNTHETIC_TOPICS = [
    ("expense_policy", "Expense Reimbursement Policy", "Finance Operations", "CFO", "2 business days", "RED-EXPENSE"),
    ("approval_matrix", "Approval Authority Matrix", "Governance", "CEO", "1 business day", "RED-APPROVAL"),
    ("incident_response", "Security Incident Response SOP", "Security", "CISO", "30 minutes", "RED-INCIDENT"),
    ("vendor_onboarding", "Vendor Onboarding Checklist", "Procurement", "VP Procurement", "3 business days", "RED-VENDOR"),
    ("data_retention", "Data Retention Policy", "Compliance", "General Counsel", "5 business days", "RED-RETENTION"),
    ("meeting_actions", "Meeting Action Item Protocol", "Operations", "Chief of Staff", "1 business day", "RED-ACTION"),
    ("release_notes", "Product Release Notes Procedure", "Product", "VP Product", "2 business days", "RED-RELEASE"),
    ("access_review", "Quarterly Access Review SOP", "IT Governance", "CIO", "4 business days", "RED-ACCESS"),
    ("travel_policy", "Business Travel Policy", "People Operations", "COO", "2 business days", "RED-TRAVEL"),
    ("procurement_policy", "Procurement Exception Policy", "Procurement", "CFO", "3 business days", "RED-PROCURE"),
    ("privacy_request", "Privacy Request Handling SOP", "Privacy", "DPO", "24 hours", "RED-PRIVACY"),
    ("change_management", "Change Management Policy", "Engineering", "CTO", "1 business day", "RED-CHANGE"),
    ("customer_escalation", "Customer Escalation Playbook", "Customer Success", "VP Customer Success", "4 hours", "RED-CUSTOMER"),
    ("backup_restore", "Backup and Restore Runbook", "Platform", "Head of Infrastructure", "1 hour", "RED-BACKUP"),
    ("asset_disposal", "Asset Disposal Policy", "IT Asset Management", "Controller", "5 business days", "RED-ASSET"),
    ("training_policy", "Mandatory Training Policy", "People Operations", "Chief People Officer", "7 business days", "RED-TRAINING"),
    ("contract_review", "Contract Review SOP", "Legal", "General Counsel", "2 business days", "RED-CONTRACT"),
    ("risk_register", "Operational Risk Register Procedure", "Risk", "Chief Risk Officer", "3 business days", "RED-RISK"),
    ("support_handoff", "Support Handoff Procedure", "Support", "Director of Support", "8 hours", "RED-HANDOFF"),
    ("model_eval", "Model Evaluation Governance Policy", "AI Governance", "AI Governance Chair", "2 business days", "RED-MODEL"),
]


def generate_synthetic_corpus(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir = root / "datasets" / "pdf_corpus_v1" / "raw" / "synthetic"
    source_dir = root / "datasets" / "pdf_corpus_v1" / "synthetic_sources"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for idx, (slug, title, domain, level3_approver, ack_sla, exception_keyword) in enumerate(
        SYNTHETIC_TOPICS,
        start=1,
    ):
        doc_key = f"synthetic-{slug}-v1"
        pdf_path = raw_dir / f"{slug}.pdf"
        source_path = source_dir / f"{slug}.md"
        pages = _build_pages(
            title=title,
            domain=domain,
            level3_approver=level3_approver,
            ack_sla=ack_sla,
            exception_keyword=exception_keyword,
        )
        _write_source_markdown(source_path, title, pages)
        _render_pdf(pdf_path, title, pages)

        local_path = _relative_posix(pdf_path, root)
        records.append(make_manifest_record(
            doc_key=doc_key,
            source_family="synthetic",
            source_url=f"local://synthetic/{slug}",
            download_url="",
            local_path=local_path,
            language="en",
            domain=_domain_slug(domain),
            doc_type="synthetic_internal_policy",
            layout_tags=["tables", "text"],
            quality_tier="gold",
            eval_role="gold",
            gold_priority=100 - idx,
            license_note="generated local synthetic fixture",
            collection_policy="download_ok",
            notes=f"Generated from synthetic_sources/{slug}.md",
            root=root,
            parse_status="text_pdf",
        ))

        doc_id = compute_md5(pdf_path)
        queries.extend(_build_queries(
            doc_key=doc_key,
            doc_name=pdf_path.name,
            doc_id=doc_id,
            title=title,
            domain=_domain_slug(domain),
            level3_approver=level3_approver,
            ack_sla=ack_sla,
            exception_keyword=exception_keyword,
        ))

    return records, queries


def _build_pages(
    *,
    title: str,
    domain: str,
    level3_approver: str,
    ack_sla: str,
    exception_keyword: str,
) -> list[list[str]]:
    return [
        [
            f"{title}",
            f"Domain: {domain}",
            f"Acknowledgement SLA: {ack_sla}",
            "Scope: This policy applies to all full-time employees, contractors, and temporary operators.",
            "Owner: Corpus v1 Synthetic Governance Office.",
            "The acknowledgement clock starts when a request is submitted in the approved system of record.",
        ],
        [
            "Approval Matrix",
            "Level 1: Team Manager approval for low-risk or low-cost requests.",
            "Level 2: Department VP approval for medium-risk or cross-team requests.",
            f"Level 3: {level3_approver} approval for high-risk, regulated, or executive-impact requests.",
            "Evidence Required: requester name, business justification, expected completion date, and audit ticket.",
        ],
        [
            "Exceptions and Escalation",
            f"Immediate escalation keyword: {exception_keyword}",
            "Any request containing the immediate escalation keyword must be routed before normal queue ordering.",
            "Emergency exceptions expire after 14 calendar days unless renewed by the Level 3 approver.",
            "Audit evidence must include the original request, approval trail, and final disposition.",
        ],
    ]


def _build_queries(
    *,
    doc_key: str,
    doc_name: str,
    doc_id: str,
    title: str,
    domain: str,
    level3_approver: str,
    ack_sla: str,
    exception_keyword: str,
) -> list[dict[str, Any]]:
    base = {
        "doc_key": doc_key,
        "doc_name": doc_name,
        "domain": domain,
    }
    return [
        {
            "query": f"How quickly must requests be acknowledged under the {title}?",
            "relevant": [[doc_id, 0]],
            "metadata": {
                **base,
                "query_type": "fact_lookup",
                "expected_channel": "text",
                "difficulty": "easy",
                "answer": ack_sla,
                "evidence_note": "Acknowledgement SLA on the overview page.",
            },
        },
        {
            "query": f"Who approves Level 3 requests in the {title}?",
            "relevant": [[doc_id, 1]],
            "metadata": {
                **base,
                "query_type": "table_lookup",
                "expected_channel": "mixed",
                "difficulty": "medium",
                "answer": level3_approver,
                "evidence_note": "Approval Matrix Level 3 row.",
            },
        },
        {
            "query": f"Which keyword triggers immediate escalation in the {title}?",
            "relevant": [[doc_id, 2]],
            "metadata": {
                **base,
                "query_type": "exception_rule",
                "expected_channel": "sparse",
                "difficulty": "medium",
                "answer": exception_keyword,
                "evidence_note": "Exceptions and Escalation page.",
            },
        },
    ]


def _write_source_markdown(path: Path, title: str, pages: list[list[str]]) -> None:
    parts = [f"# {title}", ""]
    for idx, page in enumerate(pages, start=1):
        parts.extend([f"## Page {idx}", ""])
        parts.extend(page)
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def _render_pdf(path: Path, title: str, pages: list[list[str]]) -> None:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title(title)
    pdf.set_author("NaviKB synthetic corpus generator")
    for page_lines in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 10, page_lines[0], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        for line in page_lines[1:]:
            pdf.multi_cell(0, 7, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(str(path))


def _domain_slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
