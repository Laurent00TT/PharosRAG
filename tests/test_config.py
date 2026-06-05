# tests/test_config.py
import pytest
from kb.config import Settings


def test_default_service_urls(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    assert s.mineru_server_url == "http://localhost:8001"


def test_mineru_model_source_default(tmp_path):
    """国内默认 ModelScope (HF 直连慢)。"""
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    assert s.mineru_model_source == "modelscope"


def test_default_embedding_version(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    # v1.1 栈:Qwen3-VL-Embedding-8B (4096) + MILCO sparse。
    # 任何 model/dim/sparse 变化必须 bump 此版本以隔离 collection(spec §2)。
    assert s.embedding_version == "qwen3vl8b_dim4096_milco650_sourceview_docprune30_qprune0_v01"


def test_description_adapter_config(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    assert s.description_adapter == "openai_compat"
    assert s.description_url == "http://localhost:1235/v1"
    assert s.description_supports_vision is True
    assert s.description_model == "qwen3.6-27b"


def test_agent_adapter_config(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    assert s.agent_adapter == "openai_compat"
    assert s.agent_url == "http://localhost:1235/v1"
    assert s.agent_model == "qwen3.6-27b"
    # Qwen3.6-27B 原生多模态 → 默认 True 启用真多模态消息
    assert s.agent_supports_vision is True


def test_query_rewrite_settings_default_to_agent_defaults(test_settings):
    assert test_settings.query_rewrite_adapter == "openai_compat"
    assert test_settings.query_rewrite_url == "http://localhost:1235/v1"
    assert test_settings.query_rewrite_api_key == "lm-studio"
    assert test_settings.query_rewrite_model == "qwen3.6-27b"
    assert test_settings.query_rewrite_supports_vision is False


def test_job_worker_defaults(test_settings):
    assert test_settings.job_worker_poll_interval_s == 2.0
    assert test_settings.job_worker_lease_seconds == 60
    assert test_settings.job_worker_heartbeat_interval_s == 0.0


def test_agent_image_budget_defaults(test_settings):
    assert test_settings.agent_max_images_per_turn == 3
    assert test_settings.agent_max_image_bytes_per_turn == 8_000_000


def test_new_runtime_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MAX_IMAGES_PER_TURN", "1")
    monkeypatch.setenv("AGENT_MAX_IMAGE_BYTES_PER_TURN", "1024")
    monkeypatch.setenv("JOB_WORKER_POLL_INTERVAL_S", "0.5")
    monkeypatch.setenv("JOB_WORKER_LEASE_SECONDS", "10")

    s = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
    )

    assert s.agent_max_images_per_turn == 1
    assert s.agent_max_image_bytes_per_turn == 1024
    assert s.job_worker_poll_interval_s == 0.5
    assert s.job_worker_lease_seconds == 10


def test_search_defaults(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    assert s.hyde_enabled is True
    assert s.multi_query_enabled is True
    assert s.multi_query_n == 3
    assert s.kb_profile == "balanced"


def test_fast_profile_effective_search_config(tmp_path):
    s = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
        kb_profile="fast",
    )

    cfg = s.effective_search_config()

    assert cfg["hyde_enabled"] is False
    assert cfg["multi_query_enabled"] is False
    assert cfg["multi_query_n"] == 1


def test_quality_profile_effective_search_config(tmp_path):
    """quality with no explicit overrides yields the preset (HyDE+MQ on, n=3)."""
    s = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
        kb_profile="quality",
    )

    cfg = s.effective_search_config()

    assert cfg["hyde_enabled"] is True
    assert cfg["multi_query_enabled"] is True
    assert cfg["multi_query_n"] == 3


def test_explicit_env_override_beats_profile(tmp_path):
    """6.6 invariant: any field the user explicitly sets (via env or
    kwarg) MUST win over the profile preset. Profile presets are only
    defaults for fields the user didn't touch."""
    # fast profile says hyde=False; user explicitly says hyde=True → user wins.
    s = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
        kb_profile="fast",
        hyde_enabled=True,
    )
    cfg = s.effective_search_config()
    assert cfg["hyde_enabled"] is True   # explicit override beats profile
    # multi_query_* not explicitly set → profile preset applies
    assert cfg["multi_query_enabled"] is False
    assert cfg["multi_query_n"] == 1

    # quality profile says n=3; user explicitly says n=1 → user wins.
    # (Old buggy behavior would have returned max(1, 3) = 3.)
    s2 = Settings(
        qdrant_path=str(tmp_path / "q2"),
        image_storage_path=str(tmp_path / "i2"),
        kb_profile="quality",
        multi_query_n=1,
    )
    cfg2 = s2.effective_search_config()
    assert cfg2["multi_query_n"] == 1   # explicit override wins, not max()
    # hyde / multi_query_enabled not touched → quality preset applies
    assert cfg2["hyde_enabled"] is True
    assert cfg2["multi_query_enabled"] is True



def test_reranker_default(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    # v1.1:HTTP 客户端连 Qwen3-VL-Reranker server(进程内 bge-reranker-v2-m3 已删)
    assert s.reranker_server_url == "http://localhost:8005"
    assert s.reranker_model_id == "qwen3-vl-rerank"


def test_observability_defaults(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    assert s.log_dir == "./logs"
    assert s.log_max_mb == 50
    assert s.log_backup_count == 5


def test_no_legacy_fields(tmp_path):
    s = Settings(qdrant_path=str(tmp_path), image_storage_path=str(tmp_path))
    # 确保旧字段彻底删除
    assert not hasattr(s, "vision_embedding_tier")
    assert not hasattr(s, "vision_remote_url")
    assert not hasattr(s, "copilot_api_key")
    assert not hasattr(s, "copilot_endpoint")
    assert not hasattr(s, "copilot_deployment")
    assert not hasattr(s, "blob_storage_url")
    assert not hasattr(s, "blob_container")
    assert not hasattr(s, "blob_sas_expiry_minutes")
    assert not hasattr(s, "use_local_storage")
    assert not hasattr(s, "qdrant_is_local")
    # PoC v1.0 → v1.1: legacy colqwen vision server + in-process bge-reranker
    # fields removed (modules + scripts/colqwen_server.py deleted too)
    assert not hasattr(s, "colqwen_server_url")
    assert not hasattr(s, "reranker_model")
    assert not hasattr(s, "reranker_device")
    # T1b hard cutover (I-7): legacy single-key auth fields must be gone
    assert not hasattr(s, "tool_server_api_key")
    assert not hasattr(s, "mcp_api_key")
    assert not hasattr(s, "mcp_require_api_key")


def test_no_module_level_settings_instance():
    """Module-level singleton was removed in Plan 4. Settings should be DI'd, not imported."""
    import kb.config as cfg
    assert not hasattr(cfg, "settings"), (
        "module-level `settings` instance must remain removed — "
        "Settings should be dependency-injected, not imported as a singleton"
    )


def test_navigation_mcp_settings_defaults():
    settings = Settings()
    assert settings.nav_enabled is True
    assert settings.nav_db_path.endswith("nav_index.db")
    assert settings.nav_schema_version == "nav-v1"
    assert settings.nav_auto_build_on_ingest is True
    assert settings.mcp_enable_write_tools is False
    assert settings.hybrid_nav_enabled is True


def test_old_wiki_settings_are_removed():
    """Cleanup invariant: Phase -1 + Step 0.5 must remove all wiki_* fields.
    If any field below survives, content-wiki dead code can sneak back in."""
    settings = Settings()
    forbidden = [
        "wiki_enabled",
        "wiki_db_path",
        "wiki_export_path",
        "wiki_schema_version",
        "wiki_prompt_version",
        "wiki_auto_build_on_ingest",
        "wiki_default_status",
        "wiki_min_source_refs",
        "wiki_max_evidence_chars_per_prompt",
        "wiki_max_pages_per_build",
        "wiki_compile_backend",
        "wiki_compile_max_llm_calls_per_doc",
        "wiki_compile_max_wall_seconds",
        "wiki_compile_page_batch_size",
        "wiki_compile_max_input_tokens_per_doc",
        "wiki_compile_over_budget_strategy",
        "wiki_compile_external_api_key",
    ]
    leaked = [name for name in forbidden if hasattr(settings, name)]
    assert not leaked, f"Old wiki settings leaked through cleanup: {leaked}"


def test_deleted_doc_retention_default_is_30():
    s = Settings()
    assert s.deleted_doc_retention_days == 30
