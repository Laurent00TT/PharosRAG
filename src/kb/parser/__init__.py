# src/kb/parser/__init__.py
"""Parser backend factory.

`make_parser(settings)` resolves Settings.parser_backend at startup and
returns a single Parser instance for the IngestionPipeline. There is
intentionally no automatic fallback between local and API backends —
one is chosen at startup, and that choice is final for the process lifetime.
"""
from kb.config import Settings
from kb.parser.base import Parser, ParseError
from kb.parser.mineru_local import MinerULocalParser

__all__ = ["make_parser", "Parser", "ParseError", "MinerULocalParser"]


def make_parser(settings: Settings) -> Parser:
    """Construct a Parser based on settings.parser_backend.

    Raises ValueError on misconfiguration (invalid backend value, missing
    required token for the cloud backend) so problems surface at startup
    rather than at first /ingest.
    """
    backend = settings.parser_backend
    if backend == "mineru_local":
        return MinerULocalParser(settings=settings)
    if backend == "mineru_api":
        # At least one of mineru_api_tokens (multi) or mineru_api_token (single)
        # must be set. mineru_api_tokens wins when both are present.
        if not (settings.mineru_api_tokens or settings.mineru_api_token):
            raise ValueError(
                "mineru_api_token or mineru_api_tokens is required when "
                "parser_backend='mineru_api'. Set MINERU_API_TOKEN(S) in .env, "
                "or use parser_backend='mineru_local'."
            )
        # Lazy import keeps local-only environments from needing the API client.
        from kb.parser.mineru_api import MinerUAPIParser
        return MinerUAPIParser(settings=settings)
    if backend == "mineru_openai_local":
        # PoC v1.1 (P-10A): cloud-deployed mineru-openai-server with
        # vlm-vllm-async-engine backend. HTTP protocol still being verified
        # against a live deployment — parse_async raises NotImplementedError
        # until internal design notes (not published) [P-10A] is resolved.
        # See an internal smoke script for the protocol discovery flow.
        from kb.parser.mineru_openai_local import MinerUOpenAILocalParser
        return MinerUOpenAILocalParser(settings=settings)
    raise ValueError(f"Unknown parser_backend: {backend!r}")
