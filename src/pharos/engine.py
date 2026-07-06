"""引擎装配:把 RAG 引擎组件(embedder/generator)接进 Pharos 服务层。

引擎三包(chunker/embedder/generator)已折入本仓 `src/`,与 pharos 同为可安装包 —— 直接 import,
不再有跨仓 sys.path 注入 / 文件路径加载 toolcore 那套 path-dep(历史 D12 边界已解除,见 docs/PROVENANCE.md)。

并发(M1 锁下沉,docs/SCALE_OUT.md 阶段B):**不再用一把 LockedRetriever 大锁串行所有检索** —— 那把大锁会让
remote 后端的 dense HTTP 重试退避(预热/滚动重启期)持锁阻塞整副本查询(整副本雪崩)。改为**锁下沉到资源类**:
  - Qdrant 单 client 段 -> `Store._lock`(local+remote 都要;嵌入式单 client 非线程安全);
  - GPU 模型前向 -> `Dense`/`Reranker._fwd_lock`(仅 local;remote override 成 nullcontext,HTTP 天然并发);
  - query LRU / lazy load -> `Dense._cache_lock` / `_load_lock`。
remote encode 的 HTTP + 退避因此落在所有锁外,不阻塞其他查询。LLM 网络调用不在任何锁内(见 service.py)。
换 Qdrant server 模式后 `Store._lock` 可整体删除(阶段 F/Q3),那时 remote `Retriever` 真正无锁、可多副本。
"""
from __future__ import annotations

import os

from embedder import EmbedConfig, Retriever, User, acl_admits
from generator import Generator, OpenAICompatibleLLM


def build_retriever(cfg) -> Retriever:
    """建真检索器(打开嵌入式 Qdrant;dense 模型首查 lazy 加载)。锁下沉到各资源类,不再包 LockedRetriever。"""
    # 仅 **local 后端** 需要本地模型:dense/rerank 的官方 scripts/ 是运行时刚需(dense.py/rerank.py sys.path
    # 注入 qwen3_vl_*),缺则早报清晰错、而非首查深处 ModuleNotFoundError。**remote 后端(inference_url 非空)
    # 本进程不加载模型、不需要模型文件在本地** —— 跳过检查,这正是"应用层脱 GPU/脱模型"的体现。
    if not cfg.inference_url:
        scripts = os.path.join(cfg.dense_model_path, "scripts")
        if not os.path.isdir(scripts):
            raise SystemExit(
                f"dense 模型 scripts 目录缺失:{scripts}\n"
                f"模型(含官方 scripts/)需在 PHAROS_DENSE_MODEL_PATH(现 {cfg.dense_model_path});"
                f"用 modelscope 下 Qwen3-VL-Embedding-8B 会带 scripts/。或配 PHAROS_INFERENCE_URL 用远程推理服务。")
    ecfg = EmbedConfig(qdrant_path=cfg.qdrant_path, sidecar_dir=cfg.sidecar_dir,
                       collection=cfg.collection, dense_dim=cfg.dense_dim,
                       dense_model_path=cfg.dense_model_path, rerank_model_path=cfg.rerank_model_path,
                       gpu_name_must_contain=cfg.gpu_name, inference_url=cfg.inference_url)
    return Retriever(ecfg)


def build_user(cfg):
    """启动时绑定的 ACL 身份(客户端/agent 不可经参数改)。tenant 空 -> 一切 fail-closed 返回空。"""
    return User(tenant=cfg.tenant, principals=cfg.principals)


def build_generator(retriever, cfg):
    """闭管道生成器:generator.Generator + DeepSeek(OpenAI-compat)。
    注入 acl_admits 做出口防御纵深(每个进 prompt 的命中块二次 fail-closed 校验)。
    缺 API key 时 OpenAICompatibleLLM 抛 ValueError —— 调用方转成结构化 llm_unconfigured。"""
    llm = OpenAICompatibleLLM(model=cfg.llm_model, base_url=cfg.llm_base_url,
                              api_key_env=cfg.llm_api_key_env, thinking=False,
                              max_tokens=cfg.llm_max_tokens)
    return Generator(retriever, llm, acl_check=acl_admits)
