# tests/conftest.py
#
# ── Env workaround: torch/torchvision version mismatch on this venv ────
# `import torchvision` raises `RuntimeError: operator torchvision::nms
# does not exist` because the installed torchvision is incompatible
# with the installed torch. transformers probes for torchvision via
# importlib.util.find_spec at import time; by patching find_spec to
# return None for torchvision BEFORE the first transformers import in
# the session, transformers caches _torchvision_available=False and
# cleanly degrades (the KB project does not actually use torchvision —
# `grep -r torchvision src/kb/` returns zero matches).
#
# This patch MUST run at conftest module-import time, NOT inside a
# pytest fixture. pytest fixtures (even session-scoped, autouse=True)
# execute AFTER collection — and several test files (test_channels.py,
# test_search.py, etc.) top-level-import FlagEmbedding -> transformers
# at collection time. If the patch is in a fixture, transformers
# imports first, caches _torchvision_available=True, then the failing
# torchvision::nms op is hit later and collection crashes. Doing the
# patch at module top runs it before pytest collects any test file.
import importlib.util as _ilu
_orig_find_spec = _ilu.find_spec
def _patched_find_spec(name, *args, **kwargs):
    if name == "torchvision":
        return None
    return _orig_find_spec(name, *args, **kwargs)
_ilu.find_spec = _patched_find_spec

# Early-load 顺序: torch/pyarrow/pandas 在 FlagEmbedding 之前 import,
# 避免 Windows 下 .pyd 加载顺序导致的 access violation
# (FlagEmbedding 内部某个 native 扩展会跟 pyarrow 抢 DLL,先各自加载再交互就稳定)。
import torch  # noqa: F401  E402 — must follow find_spec patch
import pyarrow  # noqa: F401  E402
import pandas  # noqa: F401  E402

import pytest  # noqa: E402
from pathlib import Path  # noqa: E402
from fpdf import FPDF  # noqa: E402
from fpdf.enums import XPos, YPos  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import io  # noqa: E402
from kb.config import Settings  # noqa: E402

@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    """2页英文 PDF，第1页纯文字，第2页含简单表格描述"""
    tmp = tmp_path_factory.mktemp("fixtures")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Chapter 1: Approval Process", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, "Approval must be completed within 3 working days.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, "Amounts over 1M require CEO signature.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.add_page()
    pdf.cell(0, 10, "Chapter 2: Amount Tiers", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, "Tier 1: under 100K - Manager approval", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, "Tier 2: 100K-1M - VP approval", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, "Tier 3: over 1M - CEO approval", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    path = tmp / "sample.pdf"
    pdf.output(str(path))
    return path

@pytest.fixture(scope="session")
def sample_image() -> bytes:
    """简单的测试页面图片（白底黑字）"""
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Approval Flow Chart", fill="black")
    draw.rectangle([100, 150, 700, 200], outline="black")
    draw.text((110, 160), "Step 1: Submit Request", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.fixture
def test_settings(tmp_path) -> Settings:
    # Force the local-mineru path with GPU off so tests don't need a real
    # mineru_api_token or a running GPU worker. Anyone needing different
    # values should construct Settings(...) explicitly inside their test.
    return Settings(
        qdrant_path=str(tmp_path / "qdrant_data"),
        image_storage_path=str(tmp_path / "images"),
        parser_backend="mineru_local",
        ingest_use_gpu_mineru=False,
    )


@pytest.fixture(autouse=True)
def _isolate_env_file(monkeypatch):
    """Disable .env loading during tests.

    Several config tests assert the in-class default values (e.g.
    description_adapter == "openai_compat"); when the developer's local .env
    overrides those (with anthropic / a local proxy / etc.), those tests would
    otherwise fail with assertion errors that have nothing to do with the
    code change being tested.

    By patching SettingsConfigDict.env_file to None for the duration of each
    test, Settings() falls back to class defaults unless a test explicitly
    passes overrides via constructor kwargs (which is the recommended way
    to vary config in a test).
    """
    new_config = {**Settings.model_config, "env_file": None}
    monkeypatch.setattr(Settings, "model_config", new_config)


@pytest.fixture
async def auth_setup(tmp_path_factory):
    """Provide a UsersStore prepopulated with an admin and a member user.
    Returns (admin_plaintext, member_plaintext, store).

    Tests that need REST/MCP auth in T1b should depend on this fixture
    plus inject ``store`` into ``app.state.users_store`` before the
    AuthMiddleware runs (i.e. before any request)."""
    from kb.auth.users import UsersStore

    db_path = tmp_path_factory.mktemp("auth") / "users.db"
    store = UsersStore(db_url=f"sqlite+aiosqlite:///{db_path}")
    await store.init()
    admin_pt, _ = await store.create_user(username="admin0", role="admin")
    member_pt, _ = await store.create_user(username="member0", role="member")
    yield admin_pt, member_pt, store
    await store.aclose()


@pytest.fixture
async def audit_setup(tmp_path):
    """Standalone AuditLog (no users) for audit-only tests."""
    from kb.audit.log import AuditLog
    from kb.storage.metadata_db import MetadataDB  # noqa: F401 — future tests

    db_url = f"sqlite+aiosqlite:///{tmp_path}/kb_metadata.db"
    audit = AuditLog(
        db_url=db_url,
        fallback_path=tmp_path / "audit_fallback.jsonl",
    )
    await audit.init()
    yield audit
    await audit.aclose()


@pytest.fixture
async def test_app_with_users(tmp_path, monkeypatch):
    """Full FastAPI TestClient: real lifespan, seeded admin + alice + bob,
    one document doc-alice-1 owned by alice. Returns dict with keys:
        client, admin_token, alice_token, bob_token,
        admin_user_id, alice_user_id, bob_user_id, audit_log.

    Tests unpack: ``setup = test_app_with_users; client = setup["client"]; ...``
    """
    import os

    # Note: the torchvision find_spec patch lives in the session-scoped
    # _patch_find_spec_torchvision autouse fixture at the top of conftest.py,
    # so transformers is already correctly degraded by the time this fixture
    # imports kb.tool_server.app.

    from fastapi.testclient import TestClient

    from kb.auth.users import UsersStore
    from kb.storage.metadata_db import MetadataDB
    from kb.tool_server import app as app_module

    # Lifespan derives users_db_url + meta_db url from settings.qdrant_path
    # (see src/kb/tool_server/app.py lifespan around lines 97-102):
    #   f"sqlite+aiosqlite:///{settings.qdrant_path}/kb_metadata.db"
    # We mirror that derivation so pre-seeded users and the documents
    # row land in the same SQLite file the lifespan will reopen.
    monkeypatch.setattr(app_module.settings, "qdrant_path", str(tmp_path / "qd"))
    monkeypatch.setattr(
        app_module.settings, "audit_fallback_path",
        str(tmp_path / "audit_fallback.jsonl"),
    )
    os.makedirs(f"{tmp_path}/qd", exist_ok=True)

    db_url = f"sqlite+aiosqlite:///{tmp_path}/qd/kb_metadata.db"

    meta = MetadataDB(db_url=db_url)
    await meta.init()
    store = UsersStore(db_url=db_url)
    await store.init()
    admin_plain, admin_user = await store.create_user(
        username="admin", role="admin",
    )
    alice_plain, alice = await store.create_user(
        username="alice", role="member",
    )
    bob_plain, bob = await store.create_user(
        username="bob", role="member",
    )
    # Seed one doc and set owner_id to alice (T3 will populate owner_id via
    # ingest). The UPDATE is parameterless string-interp; alice.user_id is
    # a controlled UUID value, not user input — acceptable for a TEST
    # fixture, deliberately not over-engineered.
    await meta.upsert_document(
        doc_id="doc-alice-1",
        doc_name="alice-manual.pdf",
        version=None,
        effective_date=None,
        expiry_date=None,
        supersedes=None,
    )
    async with meta._engine.begin() as conn:
        await conn.exec_driver_sql(
            f"UPDATE documents SET owner_id='{alice.user_id}' "
            f"WHERE doc_id='doc-alice-1'"
        )
    await store.aclose()
    await meta.aclose()

    client = TestClient(app_module.app)
    with client as c:
        yield {
            "client":        c,
            "admin_token":   admin_plain,
            "alice_token":   alice_plain,
            "bob_token":     bob_plain,
            "admin_user_id": admin_user.user_id,
            "alice_user_id": alice.user_id,
            "bob_user_id":   bob.user_id,
            "audit_log":     c.app.state.audit_log,
        }
