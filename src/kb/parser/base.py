# src/kb/parser/base.py
"""Parser protocol — implementations must turn a Path into a list of ParsedPage.

Two implementations live alongside each other and are independent:
- kb.parser.mineru_local.MinerULocalParser  → HTTP to local mineru_server
                                              (CPU or per-PDF GPU subprocess)
- kb.parser.mineru_api.MinerUAPIParser       → mineru.net SaaS API

Selection happens at startup via Settings.parser_backend, resolved by
kb.parser.make_parser(). There is intentionally NO automatic fallback
between the two.
"""
from pathlib import Path
from typing import Protocol

from kb.models import ParsedPage


class Parser(Protocol):
    """Document → list[ParsedPage]. Stateful (HTTP client, settings)."""

    async def parse_async(self, path: Path) -> list[ParsedPage]:
        ...


class ParseError(Exception):
    """Raised when parsing fails irrecoverably (failed task, malformed
    response, missing expected file in result zip, etc.).

    The ingestion pipeline catches this and emits an ingest.error event,
    skipping the offending document but continuing with the next."""
