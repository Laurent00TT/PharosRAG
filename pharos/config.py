"""Pharos 配置:全走环境变量(PHAROS_*),启动时读取一次成不可变 dataclass。

设计:不引入 toml/yaml —— 与引擎组件(RAG_* 环境变量)同一风格,个人部署一个 .env 就够;
.env 读取用极简解析(复用 eval/_common.py 的做法,免 python-dotenv 依赖),已存在的环境变量优先。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# 仓库根(pharos/pharos/config.py 的上上级)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env_file(path: str) -> None:
    """极简 .env 加载;setdefault 语义(真环境变量 > .env 文件)。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def _default_engine() -> str:
    """默认引擎仓:与 pharos 同级的 chunk-test-repo。"""
    return os.path.join(os.path.dirname(REPO_ROOT), "chunk-test-repo")


def load_env() -> str:
    """加载 pharos/.env,再用引擎仓 .env 兜底(DEEPSEEK_API_KEY 历史上放引擎仓)。返回引擎根路径。"""
    _load_env_file(os.path.join(REPO_ROOT, ".env"))
    engine = os.environ.get("PHAROS_ENGINE") or _default_engine()
    _load_env_file(os.path.join(engine, ".env"))
    return engine


@dataclass(frozen=True)
class PharosConfig:
    # 引擎(chunk-test-repo)
    engine: str = ""
    # 身份(ACL):fail-closed —— tenant 为空则一切检索/列举返回空
    tenant: str = ""
    principals: list[str] = field(default_factory=list)
    # 索引
    index_dir: str = ""
    qdrant_path: str = ""
    sidecar_dir: str = ""
    collection: str = "real"
    dense_dim: int = 1024
    # 服务
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str = ""            # 可选:设了则 /v1/* 需 X-API-Key
    max_context_tokens: int = 12000
    # LLM(闭管道 /v1/ask)
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_max_tokens: int = 2000


def from_env() -> PharosConfig:
    engine = load_env()
    index_dir = os.path.expanduser(os.environ.get("PHAROS_INDEX_DIR", "~/rag_real"))
    principals = [p.strip() for p in os.environ.get("PHAROS_PRINCIPALS", "").split(",") if p.strip()]
    return PharosConfig(
        engine=engine,
        tenant=os.environ.get("PHAROS_TENANT", "").strip(),
        principals=principals,
        index_dir=index_dir,
        qdrant_path=os.environ.get("PHAROS_QDRANT_PATH") or os.path.join(index_dir, "qdrant"),
        sidecar_dir=os.environ.get("PHAROS_SIDECAR_DIR") or os.path.join(index_dir, "sidecar"),
        collection=os.environ.get("PHAROS_COLLECTION", "real"),
        dense_dim=int(os.environ.get("PHAROS_DENSE_DIM", "1024")),
        host=os.environ.get("PHAROS_HOST", "127.0.0.1"),
        port=int(os.environ.get("PHAROS_PORT", "8787")),
        api_key=os.environ.get("PHAROS_API_KEY", "").strip(),
        max_context_tokens=int(os.environ.get("PHAROS_MAX_CONTEXT_TOKENS", "12000")),
        llm_base_url=os.environ.get("PHAROS_LLM_BASE_URL", "https://api.deepseek.com"),
        llm_model=os.environ.get("PHAROS_LLM_MODEL", "deepseek-v4-flash"),
        llm_api_key_env=os.environ.get("PHAROS_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        llm_max_tokens=int(os.environ.get("PHAROS_LLM_MAX_TOKENS", "2000")),
    )


def adapter_base_url() -> str:
    """MCP 薄适配器要连的守护进程地址(适配器进程只需要这个 + 可选 API key)。"""
    load_env()
    return os.environ.get("PHAROS_URL", f"http://127.0.0.1:{os.environ.get('PHAROS_PORT', '8787')}")
