"""Pharos 配置:全走环境变量(PHAROS_*),启动时读取一次成不可变 dataclass。

设计:不引入 toml/yaml —— 单实例部署一个 .env 就够;.env 读取用极简解析(免 python-dotenv 依赖),
已存在的环境变量优先。全部配置统一 PHAROS_* 命名空间(引擎折入本仓后不再有 RAG_* 第二命名空间)。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# 仓库根(pharos/src/pharos/config.py 的上上上级 —— 用于定位 .env)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_INLINE_COMMENT = re.compile(r"\s+#.*$")


def _parse_env_value(v: str) -> str:
    """评审修:.env 常见写法容错——成对引号剥掉(引号内原样保留,含 #);
    未引号值剥行内注释(`8787  # 服务端口` -> `8787`,否则 int() 崩启动)。"""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return _INLINE_COMMENT.sub("", v).strip()


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
            v = _parse_env_value(v)
            if v:
                os.environ.setdefault(k.strip(), v)


def _int_env(name: str, default: int) -> int:
    """评审修:整数配置错值给出指名报错,不裸 ValueError 崩启动。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"环境变量 {name} 必须是整数(收到 {raw!r}),请检查 .env。")


def load_env() -> None:
    """加载 pharos/.env(引擎已折入本仓,不再有第二个 .env 兜底)。"""
    _load_env_file(os.path.join(REPO_ROOT, ".env"))


@dataclass(frozen=True)
class PharosConfig:
    # 身份(ACL):fail-closed —— tenant 为空则一切检索/列举返回空
    tenant: str = ""
    principals: list[str] = field(default_factory=list)
    # 索引
    index_dir: str = ""
    qdrant_path: str = ""
    sidecar_dir: str = ""
    collection: str = "real"
    dense_dim: int = 1024
    corpus_dir: str = ""         # 入库语料目录(MinerU 解析产物;留仓外,PHAROS_CORPUS_DIR)
    # 服务
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str = ""            # legacy 单密钥模式(单人门槛)
    keys_file: str = ""          # keys 模式(团队,D10):JSON 文件路径,配了即启用多身份
    log_dir: str = ""            # 请求日志目录(D11);空=关
    log_queries: bool = True     # 日志是否含 query 文本(截断;内网默认开)
    max_context_tokens: int = 12000
    # smart-ask(/v1/ask 的产品层智能:数值题表格补检 + 拒答 hints;off=纯净模式)
    smart_ask: bool = True
    # LLM(闭管道 /v1/ask)
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_max_tokens: int = 2000


def from_env() -> PharosConfig:
    load_env()
    index_dir = os.path.expanduser(os.environ.get("PHAROS_INDEX_DIR", "~/rag_real"))
    principals = [p.strip() for p in os.environ.get("PHAROS_PRINCIPALS", "").split(",") if p.strip()]
    # 评审修:显式覆盖项同样 expanduser(否则写 ~/x 会在字面 "./~" 下开出空库)
    qdrant = os.environ.get("PHAROS_QDRANT_PATH", "").strip()
    sidecar = os.environ.get("PHAROS_SIDECAR_DIR", "").strip()
    return PharosConfig(
        tenant=os.environ.get("PHAROS_TENANT", "").strip(),
        principals=principals,
        index_dir=index_dir,
        qdrant_path=os.path.expanduser(qdrant) if qdrant else os.path.join(index_dir, "qdrant"),
        sidecar_dir=os.path.expanduser(sidecar) if sidecar else os.path.join(index_dir, "sidecar"),
        collection=os.environ.get("PHAROS_COLLECTION", "real"),
        dense_dim=_int_env("PHAROS_DENSE_DIM", 1024),
        corpus_dir=os.path.expanduser(os.environ.get("PHAROS_CORPUS_DIR", "").strip()),
        host=os.environ.get("PHAROS_HOST", "127.0.0.1"),
        port=_int_env("PHAROS_PORT", 8787),
        api_key=os.environ.get("PHAROS_API_KEY", "").strip(),
        keys_file=os.path.expanduser(os.environ.get("PHAROS_KEYS_FILE", "").strip()),
        log_dir=os.path.expanduser(os.environ.get("PHAROS_LOG_DIR", "~/pharos_logs").strip()),
        log_queries=os.environ.get("PHAROS_LOG_QUERIES", "on").strip().lower() not in ("off", "0", "false"),
        smart_ask=os.environ.get("PHAROS_SMART_ASK", "on").strip().lower() not in ("off", "0", "false"),
        max_context_tokens=_int_env("PHAROS_MAX_CONTEXT_TOKENS", 12000),
        llm_base_url=os.environ.get("PHAROS_LLM_BASE_URL", "https://api.deepseek.com"),
        llm_model=os.environ.get("PHAROS_LLM_MODEL", "deepseek-v4-flash"),
        llm_api_key_env=os.environ.get("PHAROS_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        llm_max_tokens=_int_env("PHAROS_LLM_MAX_TOKENS", 2000),
    )


def adapter_base_url() -> str:
    """MCP 薄适配器要连的守护进程地址(适配器进程只需要这个 + 可选 API key)。"""
    load_env()
    return os.environ.get("PHAROS_URL", f"http://127.0.0.1:{os.environ.get('PHAROS_PORT', '8787')}")
