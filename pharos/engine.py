"""引擎装配:把 chunk-test-repo 的组件接进 Pharos。

依赖方式 = **path-dep**(sys.path 插入引擎仓的 {chunker,embedder,generator}/src),与引擎仓自身
的用法(server.py / eval/_common.py)完全一致 —— 单机个人部署,不引入发布/安装环节(决策见 docs/DESIGN.md §4)。

toolcore(mcp_server/toolcore.py,纯 stdlib 工具语义层)按**文件路径**加载,不经 sys.path ——
避免顶层模块名(toolcore)在别的路径下被撞名。

并发:嵌入式 Qdrant 单客户端 + GPU 模型前向非线程安全,FastAPI 会用线程池并发跑 sync 端点,
故所有 retriever 调用经 `LockedRetriever` 串行化;LLM 网络调用不在锁内(见 service.py)。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading

_TOOLCORE_CACHE: dict = {}


def bootstrap(engine_root: str) -> None:
    """把引擎仓三个组件包的 src 目录插入 sys.path(幂等)。引擎缺失时给出可操作的报错。"""
    if not os.path.isdir(engine_root):
        raise FileNotFoundError(
            f"引擎仓不存在:{engine_root}。设 PHAROS_ENGINE 指向 chunk-test-repo,或把两仓放同级目录。")
    for pkg in ("chunker", "embedder", "generator"):
        p = os.path.join(engine_root, pkg, "src")
        if not os.path.isdir(p):
            raise FileNotFoundError(f"引擎组件缺失:{p}(chunk-test-repo 不完整?)")
        if p not in sys.path:
            sys.path.insert(0, p)


def load_toolcore(engine_root: str):
    """按文件路径加载引擎的工具语义层(mcp_server/toolcore.py,纯 stdlib、无 GPU/FastMCP 依赖)。
    HTTP 端点与 MCP 薄适配器共用它,保证与引擎 stdio server 的契约(状态机/错误映射/_INSTRUCTIONS)不漂移。"""
    path = os.path.join(engine_root, "mcp_server", "toolcore.py")
    if path in _TOOLCORE_CACHE:
        return _TOOLCORE_CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(f"toolcore 不存在:{path}(引擎仓需要 >= toolcore 拆分版本,commit 7fbf709)")
    spec = importlib.util.spec_from_file_location("pharos_engine_toolcore", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TOOLCORE_CACHE[path] = mod
    return mod


class _LockedStore:
    """retriever.store 的加锁代理(toolcore._list_impl 走 store.list_documents)。"""

    def __init__(self, store, lock: threading.Lock):
        self._store, self._lock = store, lock

    def list_documents(self, user):
        with self._lock:
            return self._store.list_documents(user)


class LockedRetriever:
    """检索器加锁代理:嵌入式 Qdrant/GPU 前向按调用串行化。只代理 toolcore + Generator 用到的方法面。"""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()
        self.store = _LockedStore(inner.store, self._lock)

    def search_with_context(self, *a, **kw):
        with self._lock:
            return self._inner.search_with_context(*a, **kw)

    def get_document(self, *a, **kw):
        with self._lock:
            return self._inner.get_document(*a, **kw)

    def get_outline(self, *a, **kw):
        with self._lock:
            return self._inner.get_outline(*a, **kw)

    def expand(self, *a, **kw):
        with self._lock:
            return self._inner.expand(*a, **kw)

    def search_grouped(self, *a, **kw):
        with self._lock:
            return self._inner.search_grouped(*a, **kw)


def build_retriever(cfg) -> LockedRetriever:
    """建真检索器(打开嵌入式 Qdrant = 取得单客户端锁;dense 模型首查 lazy 加载)。"""
    bootstrap(cfg.engine)
    from embedder import EmbedConfig, Retriever
    ecfg = EmbedConfig(qdrant_path=cfg.qdrant_path, sidecar_dir=cfg.sidecar_dir,
                       collection=cfg.collection, dense_dim=cfg.dense_dim)
    return LockedRetriever(Retriever(ecfg))


def build_user(cfg):
    """启动时绑定的 ACL 身份(客户端/agent 不可经参数改)。tenant 空 -> 一切 fail-closed 返回空。"""
    bootstrap(cfg.engine)
    from embedder import User
    return User(tenant=cfg.tenant, principals=cfg.principals)


def build_generator(retriever, cfg):
    """闭管道生成器:generator.Generator + DeepSeek(OpenAI-compat)。
    注入 acl_admits 做出口防御纵深(每个进 prompt 的命中块二次 fail-closed 校验)。
    缺 API key 时 OpenAICompatibleLLM 抛 ValueError —— 调用方转成结构化 llm_unconfigured。"""
    bootstrap(cfg.engine)
    from embedder import acl_admits
    from generator import Generator, OpenAICompatibleLLM
    llm = OpenAICompatibleLLM(model=cfg.llm_model, base_url=cfg.llm_base_url,
                              api_key_env=cfg.llm_api_key_env, thinking=False,
                              max_tokens=cfg.llm_max_tokens)
    return Generator(retriever, llm, acl_check=acl_admits)
