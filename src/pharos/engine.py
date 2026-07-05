"""引擎装配:把 RAG 引擎组件(embedder/generator)接进 Pharos 服务层。

引擎三包(chunker/embedder/generator)已折入本仓 `src/`,与 pharos 同为可安装包 —— 直接 import,
不再有跨仓 sys.path 注入 / 文件路径加载 toolcore 那套 path-dep(历史 D12 边界已解除,见 docs/PROVENANCE.md)。

并发:嵌入式 Qdrant 单客户端 + GPU 模型前向非线程安全,FastAPI 会用线程池并发跑 sync 端点,
故所有 retriever 调用经 `LockedRetriever` 串行化;LLM 网络调用不在锁内(见 service.py)。
"""
from __future__ import annotations

import threading

from embedder import EmbedConfig, Retriever, User, acl_admits
from generator import Generator, OpenAICompatibleLLM


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
    ecfg = EmbedConfig(qdrant_path=cfg.qdrant_path, sidecar_dir=cfg.sidecar_dir,
                       collection=cfg.collection, dense_dim=cfg.dense_dim)
    return LockedRetriever(Retriever(ecfg))


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
