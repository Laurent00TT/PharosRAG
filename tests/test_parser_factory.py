# tests/test_parser_factory.py
"""Factory tests for make_parser.

Verifies the startup-time selection of MinerULocalParser vs MinerUAPIParser
based on Settings.parser_backend, including the validation that an empty
mineru_api_token blocks the cloud backend.
"""
import pytest

from kb.config import Settings
from kb.parser import make_parser
from kb.parser.mineru_local import MinerULocalParser


def _settings(tmp_path, **overrides) -> Settings:
    base = {
        "qdrant_path": str(tmp_path / "q"),
        "image_storage_path": str(tmp_path / "i"),
        "ingest_use_gpu_mineru": False,
    }
    base.update(overrides)
    return Settings(**base)


def test_factory_returns_local_parser_when_configured(tmp_path):
    s = _settings(tmp_path, parser_backend="mineru_local")
    parser = make_parser(s)
    assert isinstance(parser, MinerULocalParser)


def test_factory_returns_api_parser_when_configured(tmp_path):
    pytest.importorskip("kb.parser.mineru_api",
                        reason="MinerUAPIParser implemented in Stage 2 (Task 6)")
    s = _settings(
        tmp_path,
        parser_backend="mineru_api",
        mineru_api_token="t0k3n_value",
    )
    parser = make_parser(s)
    # Avoid hard import at top so mineru_local-only environments don't
    # need the API parser. Verify by class name to keep the test loose.
    assert type(parser).__name__ == "MinerUAPIParser"


def test_factory_raises_without_token_in_api_mode(tmp_path):
    # Need to clear BOTH multi-token and single-token to trigger the guard.
    s = _settings(
        tmp_path,
        parser_backend="mineru_api",
        mineru_api_token="",
        mineru_api_tokens="",
    )
    with pytest.raises(ValueError, match="mineru_api_token"):
        make_parser(s)


def test_factory_raises_unknown_backend(tmp_path):
    # Settings(Literal[...]) blocks construction with an invalid value, so we
    # build a valid Settings then override the field at runtime to exercise
    # the make_parser error path.
    s = _settings(tmp_path, parser_backend="mineru_local")
    object.__setattr__(s, "parser_backend", "wibble")
    with pytest.raises(ValueError, match="Unknown parser_backend"):
        make_parser(s)
