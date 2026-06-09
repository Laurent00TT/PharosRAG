from __future__ import annotations

from dataclasses import dataclass

from kb.nav.models import NavEntry


@dataclass(frozen=True)
class NavLintIssue:
    code: str
    entry_id: str
    message: str


def lint_nav_entry(entry: NavEntry) -> list[NavLintIssue]:
    issues: list[NavLintIssue] = []
    if entry.page_start > entry.page_end:
        issues.append(NavLintIssue("invalid_page_range", entry.entry_id, "page_start must be <= page_end"))
    if not entry.resource_uris:
        issues.append(NavLintIssue("missing_resource_uri", entry.entry_id, "navigation entry must point to resources"))
    if entry.contains_generated_content:
        issues.append(NavLintIssue("generated_content_forbidden", entry.entry_id, "navigation entries must not contain generated content"))
    return issues


def lint_nav_entries(entries: list[NavEntry]) -> list[NavLintIssue]:
    """Batch lint convenience for doctor."""
    out: list[NavLintIssue] = []
    for entry in entries:
        out.extend(lint_nav_entry(entry))
    return out
