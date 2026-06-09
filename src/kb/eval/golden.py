from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# "wiki" intentionally removed in the navigation-only refactor — nav
# entries are pointers, not generated content. The runtime guard below
# (line "if expected_mode not in VALID_MODES") rejects any jsonl line
# that still carries expected_mode="wiki" so the dead semantic can't
# silently sneak back in via a hand-edited golden file.
VALID_EXPECTED_MODES: frozenset[str] = frozenset({"evidence", "hybrid", "navigation"})


@dataclass(frozen=True)
class ExpectedSource:
    doc_id: str
    page_num: int


@dataclass(frozen=True)
class GoldenCase:
    id: str
    query: str
    intent: str
    expected_mode: Literal["evidence", "hybrid", "navigation"]
    expected_sources: list[ExpectedSource]


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    p = Path(path)
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            expected_mode = data["expected_mode"]
            if expected_mode not in VALID_EXPECTED_MODES:
                raise ValueError(
                    f"expected_mode={expected_mode!r} not in {sorted(VALID_EXPECTED_MODES)} — "
                    f"'wiki' was removed during the navigation-only refactor"
                )
            sources = [
                ExpectedSource(doc_id=str(item["doc_id"]), page_num=int(item["page_num"]))
                for item in data["expected_sources"]
            ]
            cases.append(
                GoldenCase(
                    id=str(data["id"]),
                    query=str(data["query"]),
                    intent=str(data["intent"]),
                    expected_mode=expected_mode,
                    expected_sources=sources,
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{p}:{line_no}: malformed golden record — {exc}"
            ) from exc
    return cases
