# src/kb/config.py
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow third-party env vars (e.g. HF_ENDPOINT, HF_HOME, TRANSFORMERS_CACHE)
        # to live in .env without breaking Settings construction. Declared fields
        # are still strictly type-validated.
        extra="ignore",
    )

    # ── Storage ──────────────────────────────────────────────────────────────
    qdrant_path: str = "./qdrant_data"
    qdrant_url: str = ""                                # empty = local file mode
    image_storage_path: str = "./storage/images"
    # Original-document store: keep the source PDF at ingest so the workbench
    # (and MCP agents) can navigate back to the original, and pages can later
    # be re-rendered at higher DPI on demand. STORE_SOURCE_DOCUMENTS=false
    # skips it (saves disk ≈ the corpus PDF size).
    source_storage_path: str = "./storage/source_docs"
    store_source_documents: bool = True

    # ── Embedding version (determines Qdrant collection name) ──────────────
    # Bumping the version forces a fresh collection; old collections stay
    # readable for comparison.
    #
    # 命名规范(spec §2):必须显式编码所有影响 schema 的因素:
    #   <embedding_model>_<dim>_<sparse_model>_<sourceview?>_<doc_prune>_<query_prune>_<iter>
    # 任何下列维度变化都触发新 collection,不复用旧 collection:
    #   - dense 模型 / 维度
    #   - sparse 模型 / 是否 source_view / pruning 比例(doc/query 不对称必须分开标)
    embedding_version: str = "qwen3vl8b_dim4096_milco650_sourceview_docprune30_qprune0_v01"

    # ── Service URLs ───────────────────────────────────────────────────────
    mineru_server_url: str = "http://localhost:8001"
    # Optional: comma-separated list of additional mineru server URLs for
    # multi-instance scaling. When set, MinerULocalParser round-robins
    # parse requests across all listed URLs (each instance loads its own
    # ~4GB model in GPU; throughput scales ~linearly with instance count).
    # When empty, fallback to mineru_server_url single-instance behavior.
    # Example: "http://localhost:8101,http://localhost:8102,http://localhost:8103,http://localhost:8104"
    mineru_server_urls: str = ""

    # ── Ingestion API path guard ───────────────────────────────────────────
    # Comma-separated list of directory roots that the /ingestion/jobs
    # endpoint is allowed to ingest from. Empty list = the guard is OFF
    # (any local file path is accepted — historical behavior, less safe).
    # When set, each submitted path must resolve to a descendant of at
    # least one configured root. Prevents prompt-injection / LLM-driven
    # arbitrary-file ingestion when the tool server is exposed to an agent.
    # Roots may be absolute or relative to the process working directory.
    ingestion_allowed_roots: str = ""

    # ── Parser backend selection (启动时一次决定，不做自动切换) ────────────
    # "mineru_api"  : OpenDataLab MinerU SaaS at mineru.net (requires token).
    #                 Default — measured 14 min for 11 DK PDFs, 6.1x faster
    #                 than the CPU local baseline and 2.6x faster than the
    #                 per-PDF-GPU local mode (see
    #                 internal design notes (not published)).
    # "mineru_local": HTTP to local mineru_server. Backup path; useful when
    #                 offline, behind firewall, or for sensitive documents.
    parser_backend: Literal["mineru_local", "mineru_api", "mineru_openai_local"] = "mineru_api"

    # ── MinerU model source (used by mineru_server / local backend) ────────
    # "modelscope": China CDN (recommended)
    # "huggingface": overseas / VPN
    # "local": fully offline, requires ~/mineru.json with models-dir
    mineru_model_source: str = "modelscope"

    # ── MinerU SaaS API (used when parser_backend=mineru_api) ──────────────
    mineru_api_url: str = "https://mineru.net"
    # Single-token form (back-compat). When mineru_api_tokens is set it
    # supersedes this field.
    mineru_api_token: str = ""
    # Comma-separated list of tokens for round-robin across accounts.
    # Each parse_async call picks the next token, spreading load evenly
    # so a single account's daily quota (~1000 pages) doesn't bottleneck.
    mineru_api_tokens: str = ""
    mineru_api_lang: str = "ch"             # ch / en / japan / korean / ...
    mineru_api_poll_interval_s: float = 5.0
    mineru_api_max_wait_s: float = 600.0
    mineru_api_enable_ocr: bool = True
    mineru_api_enable_formula: bool = True
    mineru_api_enable_table: bool = True

    # ── Local-mineru ingest sub-mode (used when parser_backend=mineru_local) ──
    # When True (default), the local pipeline runs each PDF through a fresh
    # mineru subprocess in cuda mode (per-PDF worker), which /exits after
    # parse so the OS reclaims VRAM (MinerU has no unload API). ~2.4x faster
    # end-to-end on the 5070 + 11 PDF benchmark vs the long-lived CPU
    # mineru_server fallback.
    ingest_use_gpu_mineru: bool = True
    # How long to wait for the mineru worker to come up before /parse.
    ingest_mineru_worker_health_timeout_s: float = 180.0
    # Mineru backend mode for /parse calls.
    #   "hybrid"   — default, layout + OCR + (optional) formula + VLM fusion;
    #                deadlocks under multiprocessing.spawn in some
    #                containerised cloud envs (see deployment notes).
    #   "pipeline" — streaming-API backend without subprocess spawn;
    #                use this for cloud deploys that hit the spawn deadlock.
    #   "vlm"      — pure VLM (Qwen2-VL-based) inference, slower but no
    #                layout/OCR sub-models needed.
    # Read by kb.parser.mineru_local.MinerULocalParser.
    mineru_parse_mode: Literal["hybrid", "pipeline", "vlm"] = "hybrid"

    # ── Chunker config (settings-driven so A/B experiments don't need code edits) ──
    chunker_small_size: int = 384
    chunker_parent_size: int = 1024
    chunker_overlap: int = 64
    # >=2: split text at markdown headings when present.
    # 99 (or any large number): always fall back to char-split — useful as
    # an A/B baseline against the default heading-aware behaviour.
    chunker_min_headings_for_segmenting: int = 2

    # ── Description Channel (independent vision pass) ──────────────────────
    # Defaults point to Qwen3.6-27B (:1235), shared with the agent endpoint.
    # To swap in a smaller vision model (e.g. qwen2-vl-7b) only change this group.
    # Set description_enabled=False to skip the channel entirely (no adapter
    # built, no /describe calls made). Useful when no vision LM is available
    # (e.g. running KB main inside docker without host.docker.internal access
    # to a host-side LM Studio). Indexing still works — pages just won't have
    # the desc/visual_description field populated.
    description_enabled: bool = True
    description_adapter: str = "openai_compat"          # "openai_compat" | "ollama"
    description_url: str = "http://localhost:1235/v1"
    description_api_key: str = "lm-studio"
    description_model: str = "qwen3.6-27b"
    description_supports_vision: bool = True

    # ── Agent LLM (shared by KBAgent / HyDE / Multi-Query) ─────────────────
    # Qwen3.6-27B is natively multimodal, so agent_supports_vision=True enables
    # real multipart messages with image attachments.
    agent_adapter: str = "openai_compat"
    agent_url: str = "http://localhost:1235/v1"
    agent_api_key: str = "lm-studio"
    agent_model: str = "qwen3.6-27b"
    agent_supports_vision: bool = True
    agent_max_images_per_turn: int = 3
    agent_max_image_bytes_per_turn: int = 8_000_000

    # ── Query Rewrite LLM (HyDE / Multi-Query) ─────────────────────────────
    query_rewrite_adapter: str = "openai_compat"
    query_rewrite_url: str = "http://localhost:1235/v1"
    query_rewrite_api_key: str = "lm-studio"
    query_rewrite_model: str = "qwen3.6-27b"
    query_rewrite_supports_vision: bool = False

    # ── 模型升级新栈 PoC (v1.1) ─────────────────────────────────────────────
    # spec: internal design notes (not published)
    # 当前生产栈。PoC v1.0 的 legacy 字段(colqwen_server_url / reranker_model /
    # reranker_device)及其实现模块已在 v1.1 清理中删除,见 the project README。

    # Multimodal Embedding Server — Qwen3-VL-Embedding-8B(text/image/desc 共用同一模型)
    multimodal_embedding_server_url: str = "http://localhost:8003"
    multimodal_embedding_model_id: str = "qwen3-vl-embed"   # vllm --served-model-name
    # Qwen3-VL-Embedding 官方推荐 query 端附加 instruction prefix(检索任务通用)
    # document 端不加。见模型卡 / spec §3.1
    embedding_query_instruction: str = (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: "
    )
    # MRL 维度截断,0 = 用模型原生(4096);spec §3.5 决定走原生
    embedding_output_dim: int = 0
    # 截断 dense 文本输入到 embed 服务 --max-model-len 以内。中文/混合 chunk 经
    # tokenize 后偶尔超 4096(token 膨胀),vllm 默认 400 整篇文档(error_type
    # HTTPStatusError, retryable=false → 文档判失败)。设此值后 TextChannel 传
    # truncate_prompt_tokens,让 vllm 左截断而非报错;embedding 对尾部 token 不敏感。
    # ⚠️ 必须 ≤ 云端 embed 的 --max-model-len(当前 4096),留 8 token 给特殊符号。
    # 治本方案(chunker 按 token 精确切到 <4096)归 v2,需 re-ingest。2026-05-30 诊断。
    embedding_max_input_tokens: int = 4088

    # Reranker Server — Qwen3-VL-Reranker-8B(多模态,vllm --task score)
    # 启动命令需要 --hf_overrides 强制 task,见 internal setup notes (not published) §3.5
    reranker_server_url: str = "http://localhost:8005"
    reranker_model_id: str = "qwen3-vl-rerank"
    # Phase C-2 finding (2026-05-27): Qwen3-VL-Reranker is net-negative
    # on our v1.1 corpus, even after fixing the classifier_from_token
    # order ([yes,no] not [no,yes]). Token fix improved en MRR 0.65→0.73
    # but zh MRR stayed at 0.28 (vs 0.69 with reranker off) and overall
    # recall@10 dropped to 0.82 (vs 1.00 with reranker off). Default
    # True: skip reranker and use raw RRF order. Set False to re-enable
    # reranker for A/B testing or once we swap to a different model.
    # See internal design notes (not published) for full saga.
    #
    # 2026-05-28 flipped True→False: with set-preserving rerank pool (multiplier
    # 1.0, below) the reranker no longer evicts RRF top-k docs, so recall@10 is
    # unchanged from RRF while page-1 precision improves 4x (phase-c §5.4). The
    # earlier -13% recall regression was a pool=2 eviction artifact, not the
    # model. Engine falls back to RRF order if the rerank server is unavailable,
    # so ON-by-default carries no hard availability dependency. Flip back to True
    # for pure-RRF serving.
    search_disable_reranker: bool = False

    # Sparse Server — MILCO 650m(CPU 独立 service)
    sparse_server_url: str = "http://localhost:8004"
    sparse_model: str = "omai-research/milco-650m"
    # **必须 pin commit sha** — MILCO 安全审查报告建议;
    # 默认 main 仅用于本地开发,部署必须显式覆盖为审过的 sha
    sparse_model_revision: str = "main"
    # Mass-based pruning(doc/query 不对称会改变检索行为,见 spec §2 embedding_version 命名规范;
    # 任何修改必须 bump embedding_version 并重 ingest)
    sparse_doc_prune_ratio: float = 0.3
    sparse_query_prune_ratio: float = 0.0
    # 启用 MILCO LexEcho/source_view(影响 sparse 词表语义,会出现在 embedding_version 名)
    sparse_use_source_view: bool = True

    # ── Search defaults ─────────────────────────────────────────────────────
    kb_profile: str = "balanced"
    hyde_enabled: bool = True
    multi_query_enabled: bool = True
    multi_query_n: int = 3
    search_planner_enabled: bool = True
    search_profile: str = "balanced"
    search_low_confidence_threshold: float = 0.2

    # ── Adaptive result cutoff (serving-side; eval always uses fixed top_k) ──
    # When enabled, the served result list may be SHORTER than top_k by cutting
    # where relevance drops off, instead of always padding to top_k with noise.
    # top_k stays the hard maximum. Default OFF — preserves fixed-top_k behaviour
    # until validated. See internal design notes (not published) discussion.
    search_adaptive_cutoff_enabled: bool = False
    # RRF mode (uncalibrated scores): keep results scoring >= ratio × top_score.
    # Relative because RRF magnitude is not comparable across queries.
    search_adaptive_cutoff_ratio: float = 0.5
    # Reranker mode (calibrated sigmoid scores in [0,1]): absolute relevance gate.
    search_adaptive_cutoff_min_score: float = 0.3
    # Always return at least this many (when available), even if scores are low.
    search_adaptive_cutoff_min_results: int = 1

    # Rerank input pool size, as a multiple of top_k. The reranker rescores this
    # many RRF candidates then returns the best top_k.
    #   2.0 = current: rerank top-2k → reranker CAN evict an RRF top-k doc by
    #         scoring it low → recall@k may drop (observed -13% in phase-c §5.3).
    #   1.0 = set-preserving: rerank only the RRF top-k, so recall@k == RRF
    #         recall@k by construction — reranker only reorders, never evicts.
    #         Isolates "does reranker improve top precision?" from recall damage.
    # Capped by the enriched candidate count (enrich pool is top_k*2).
    # 2026-05-28 default flipped 2.0→1.0 (set-preserving) — see phase-c §5.4.
    search_rerank_pool_multiplier: float = 1.0

    # Per-channel RRF weights (multiplier on each channel's 1/(k+rank) term).
    # All 1.0 = classic unweighted RRF. Tunable to up-weight e.g. sparse for
    # keyword-heavy CJK queries or dense for semantic EN. Phase B channel
    # attribution: dense+sparse carry nearly all top-1 hits, vision ~40%, desc
    # negligible (Channel 4 off). Defaults preserve current behaviour.
    search_rrf_weight_dense: float = 1.0
    search_rrf_weight_sparse: float = 1.0
    search_rrf_weight_vision: float = 1.0
    search_rrf_weight_desc: float = 1.0

    # ── Background ingestion jobs ──────────────────────────────────────────
    job_worker_poll_interval_s: float = 2.0
    job_worker_lease_seconds: int = 60
    # 0 lets the runner derive a safe heartbeat interval from the lease.
    job_worker_heartbeat_interval_s: float = 0.0

    # ── Observability ───────────────────────────────────────────────────────
    log_dir: str = "./logs"
    log_max_mb: int = 50
    log_backup_count: int = 5

    # ── Navigation index ───────────────────────────────────────────────────
    nav_enabled: bool = True
    nav_db_path: str = "./qdrant_data/nav_index.db"
    nav_schema_version: str = "nav-v1"
    nav_auto_build_on_ingest: bool = True
    nav_max_aliases_per_entry: int = 20
    nav_max_entries_per_doc: int = 5000
    # EXPERIMENTAL / UNWIRED: nav vector search (semantic-similarity nav
    # ranking on top of the current lexical substring match) is designed
    # but no code path consumes this flag yet. Setting it to false has no
    # effect today; setting it to true does not enable anything. Keeping
    # the flag so existing .env files don't break, but treat its value as
    # "future state intent" until a real vector NavSearch backend lands.
    nav_vector_enabled: bool = True
    # EXPERIMENTAL / UNWIRED: graph-walk expansion (follow parent_child /
    # next_prev edges N hops from a hit) is not implemented; same caveat.
    nav_graph_expand_depth: int = 1
    # WIRED: consumed by HybridNavigator.search via nav_hit_to_fetch_target
    # — widens each suggested fetch range symmetrically around the hit.
    # ``1`` means "exactly the hit's pages"; ``3`` means "hit ± 1 page".
    nav_fetch_window_pages: int = 1

    # ── MCP ────────────────────────────────────────────────────────────────
    mcp_enable_resources: bool = True
    mcp_enable_prompts: bool = True
    mcp_enable_write_tools: bool = False
    mcp_allowed_origins: str = "http://localhost,http://127.0.0.1"
    mcp_tool_text_budget: int = 800
    mcp_tool_result_limit: int = 20

    # ── Hybrid navigator ───────────────────────────────────────────────────
    # WIRED: consumed by HybridNavigator runtime gate via the kb_hybrid_search
    # tool. When false the tool returns reason="hybrid_nav_disabled" rather
    # than running a search (tool stays registered so agent discovery is stable).
    hybrid_nav_enabled: bool = True
    # WIRED: consumed by HybridNavigator.search — when true, every call emits
    # an "hybrid.search" trace event (latency + hit counts + window). Drop
    # the noise by setting to false in production.
    hybrid_trace_enabled: bool = True
    # EXPERIMENTAL / UNWIRED: supervised routing (train a router from
    # accumulated trace samples) is designed but not implemented. Keeping
    # the flag so a future training pipeline knows what threshold to apply.
    hybrid_router_min_samples_for_training: int = 200

    # ── Small-team stage ──────────────────────────────────────────────────────
    # WIRED in T5: gc_deleted_docs.py removes documents with status='deleted'
    # older than this many days. Default 30 mirrors Gmail/Dropbox trash retention.
    # Admins can --force-purge to bypass for sensitive data leaks.
    deleted_doc_retention_days: int = 30

    # ── T2 Audit ───────────────────────────────────────────────────────────
    # WIRED: AuditLog._append_fallback_jsonl_synced — when DB insert fails
    # in the two-phase mandatory completed path, append here with fsync to
    # survive a process crash. (single-phase mandatory uses caller-session
    # ORM and does NOT go through this fallback — see A-1.)
    audit_fallback_path: str = "./logs/audit_fallback.jsonl"
    # WIRED: GET /audit query default + cap (member abuse guard).
    audit_default_limit: int = 50
    audit_max_limit: int = 500

    # ── T2 Brute force ─────────────────────────────────────────────────────
    # WIRED: BruteForceTracker — N failures within window_s from same
    # source_ip → one auth.brute_force audit row, clear counter. Memory
    # only; process restart loses the window (acceptable trade — attacker
    # only gets threshold-1 extra attempts before the next window's audit).
    brute_force_window_s: int = 300
    brute_force_threshold: int = 10

    # Profile defaults for the search knobs. ``balanced`` / ``offline``
    # intentionally fall through to the explicit Settings defaults so users
    # who never set KB_PROFILE see the same behavior as pre-6.6.
    # ``fast`` and ``quality`` are opinionated presets — but they must NOT
    # override an env-explicit value (spec §6.6: "Profile 不应覆盖用户显式
    # 配置"). The override check below uses Pydantic's model_fields_set,
    # which contains exactly the field names the user passed to __init__
    # (either via env or kwarg). Fields whose value came from the class
    # default are absent.
    _PROFILE_DEFAULTS = {
        "fast":    {"hyde_enabled": False, "multi_query_enabled": False, "multi_query_n": 1},
        "quality": {"hyde_enabled": True,  "multi_query_enabled": True,  "multi_query_n": 3},
    }

    def effective_search_config(self) -> dict:
        """Resolve search knobs from (profile default → env explicit) precedence.

        Returns a dict with `hyde_enabled`, `multi_query_enabled`,
        `multi_query_n`. Order of precedence (last wins):
          1. ``balanced`` / ``offline``: defer to Settings field values
             (which themselves came from env or class default).
          2. ``fast`` / ``quality``: apply profile preset.
          3. Any field the user EXPLICITLY set (via env or kwarg) wins
             over the profile preset — model_fields_set reports it.
        """
        profile = self.kb_profile if self.kb_profile in self._PROFILE_DEFAULTS else None
        explicitly_set = self.model_fields_set

        def _resolve(name: str, profile_value):
            # User explicitly set this field → honor it.
            if name in explicitly_set:
                return getattr(self, name)
            return profile_value

        if profile is None:
            # balanced / offline / anything unknown: defer to settings values.
            return {
                "hyde_enabled": self.hyde_enabled,
                "multi_query_enabled": self.multi_query_enabled,
                "multi_query_n": self.multi_query_n,
            }

        preset = self._PROFILE_DEFAULTS[profile]
        return {
            "hyde_enabled": _resolve("hyde_enabled", preset["hyde_enabled"]),
            "multi_query_enabled": _resolve("multi_query_enabled", preset["multi_query_enabled"]),
            # For multi_query_n in 'quality' the legacy behavior was
            # ``max(self.multi_query_n, 3)`` — i.e. quality could only INCREASE
            # the count. Under explicit-override semantics, a user who sets
            # multi_query_n=5 keeps 5 (already wins via _resolve). A user who
            # sets multi_query_n=1 keeps 1 (explicit override wins). A user
            # who never set it falls through to the preset default of 3.
            "multi_query_n": _resolve("multi_query_n", preset["multi_query_n"]),
        }
