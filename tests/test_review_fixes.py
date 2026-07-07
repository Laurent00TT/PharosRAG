"""对抗评审 P1(2026-07-02)修复的回归测试:每条 confirmed / 自核实属实的发现钉一个测试。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from _fakes import FakeRetriever, make_app, make_cfg, make_user
from pharos import config as pconfig
from pharos import mcp_adapter as A
from pharos.indexer import run_index


# ---------- C2:no_identity hint 必须指向 PHAROS_TENANT(不是引擎的 RAG_TENANT)----------
def test_no_identity_hint_names_pharos_vars():
    app = make_app(user=make_user(tenant=""), cfg=make_cfg(tenant=""))
    with TestClient(app) as c:
        r1 = c.post("/v1/retrieve", json={"query": "q"}).json()
        r2 = c.post("/v1/ask", json={"query": "q"}).json()
    for r in (r1, r2):
        assert r["status"] == "no_identity"
        assert "PHAROS_TENANT" in r["hint"] and "RAG_TENANT" not in r["hint"]


# ---------- C1:indexer 拒绝 restricted + 空 allow(静默不可见陷阱)----------
def test_indexer_rejects_restricted_without_allow():
    with pytest.raises(SystemExit, match="allow"):
        run_index(make_cfg(), visibility="restricted", allow="")


def test_indexer_restricted_with_allow_passes_guard():
    # 有 principals 时守卫放行(随后因语料目录不存在退出 —— 证明卡的是 ACL 守卫而非别的)
    with pytest.raises(SystemExit, match="语料目录不存在"):
        run_index(make_cfg(), corpus="Z:/definitely/not/exist", visibility="restricted", allow="g_hr")


# ---------- 模型路径上浮 config:缺 scripts/ 目录时 build_retriever 启动即清晰报错 ----------
def test_build_retriever_asserts_model_scripts(tmp_path):
    from pharos.engine import build_retriever
    with pytest.raises(SystemExit, match="scripts 目录缺失"):
        build_retriever(make_cfg(dense_model_path=str(tmp_path / "no-such-model")))


# ---------- 阶段D 审查 M1:PHAROS_QDRANT_URL 必须真传到 Store(QdrantClient(url=)),否则 server 链不通(我曾宣称透传却漏改 engine)----------
def test_qdrant_url_reaches_store_via_engine():
    """守护 engine.build_retriever 透传 qdrant_url。mock QdrantClient 避免真连,纯 CPU;删 engine 的 qdrant_url= 即红。"""
    from unittest import mock

    from pharos.engine import build_retriever
    cfg = make_cfg(qdrant_url="http://fake:6333", inference_url="http://fake:8900")   # inference_url 非空 -> 跳 scripts 检查
    with mock.patch("embedder.store.QdrantClient") as MockQC:
        build_retriever(cfg)
    MockQC.assert_called_once()
    assert MockQC.call_args.kwargs.get("url") == "http://fake:6333", \
        f"engine 未透传 qdrant_url 到 Store(QdrantClient 调用={MockQC.call_args})"


# ---------- pharos parse:manifest 缺失时清晰报错(不静默/不裸抛)----------
def test_pharos_parse_missing_manifest(tmp_path):
    from pharos.parser import run_parse
    with pytest.raises(SystemExit, match="manifest 不存在"):
        run_parse(str(tmp_path / "nope.csv"), str(tmp_path / "out"), str(tmp_path))


# ---------- ask:工厂非 ValueError 异常不再裸抛 500 ----------
def test_ask_factory_runtime_error_degrades():
    def boom(retriever, cfg):
        raise ModuleNotFoundError("openai")
    with TestClient(make_app(generator_factory=boom)) as c:
        r = c.post("/v1/ask", json={"query": "q"})
    assert r.status_code == 200 and r.json()["status"] == "ask_failed"


# ---------- 适配器:doc_id URL 编码 / 空 doc_id 本地拒 / 3xx 结构化 ----------
class _StubClient:
    def __init__(self, responder):
        self.responder = responder
        self.base_url = "http://stub:8787"
        self.calls: list = []

    def request(self, method, path, json=None, params=None):
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        return self.responder(method, path, json, params)


def test_adapter_quotes_doc_id(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(200, json={"status": "ok"}))
    monkeypatch.setattr(A, "_client", sc)
    A.get_document("Q1#report/v2")
    assert sc.calls[0]["path"] == "/v1/documents/Q1%23report%2Fv2"   # #、/ 不再截断/改路由
    A.get_outline("研报 2026")
    assert "%" in sc.calls[1]["path"] and sc.calls[1]["path"].endswith("/outline")


def test_adapter_empty_doc_id_local_bad_arg(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(200, json={"status": "ok"}))
    monkeypatch.setattr(A, "_client", sc)
    assert A.get_document("")["status"] == "bad_arg"
    assert A.get_outline("")["status"] == "bad_arg"
    assert sc.calls == []                    # 未发 HTTP(否则空 doc_id 打到列表/307)


def test_adapter_3xx_structured(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(307, headers={"location": "/v1/documents"}))
    monkeypatch.setattr(A, "_client", sc)
    out = A.get_document("d1")
    assert out["status"] == "backend_unavailable" and out["retriable"] is True


# ---------- config:.env 行内注释/引号、int 指名报错、覆盖路径 expanduser ----------
def test_parse_env_value():
    assert pconfig._parse_env_value("8787  # 服务端口") == "8787"
    assert pconfig._parse_env_value('"sk-abc # not comment"') == "sk-abc # not comment"
    assert pconfig._parse_env_value("'  spaced  '") == "  spaced  "
    assert pconfig._parse_env_value("plain") == "plain"


def test_int_env_named_error(monkeypatch):
    monkeypatch.setenv("PHAROS_PORT", "8787  x")
    with pytest.raises(SystemExit, match="PHAROS_PORT"):
        pconfig._int_env("PHAROS_PORT", 8787)


def test_override_paths_expanduser(monkeypatch):
    monkeypatch.setenv("PHAROS_QDRANT_PATH", "~/qtest")
    monkeypatch.setenv("PHAROS_SIDECAR_DIR", "~/stest")
    cfg = pconfig.from_env()
    assert cfg.qdrant_path == os.path.expanduser("~/qtest")     # 不再落在字面 "./~" 目录
    assert cfg.sidecar_dir == os.path.expanduser("~/stest")


# ---------- 契约不漂移:HTTP 适配器与 stdio-direct 两个传输的六工具 docstring 逐一相同,
#            且 _INSTRUCTIONS 单一真源来自 in-repo toolcore(不再 exec 外部引擎仓) ----------
def test_transports_contract_no_drift():
    from pharos import mcp_stdio as S
    from pharos import toolcore
    for name in ["retrieve", "list_documents", "get_document", "get_outline", "expand", "retrieve_grouped"]:
        adp_doc = (getattr(A, name).__doc__ or "").strip()
        std_doc = (getattr(S, name).__doc__ or "").strip()
        assert adp_doc == std_doc, f"{name} docstring 两传输不同文"
    assert A._tc is toolcore                          # HTTP 适配器的工具语义真源 = in-repo toolcore
    assert S._INSTRUCTIONS is toolcore._INSTRUCTIONS  # stdio 的 instructions 同源(单一真源结构强制)


# ---------- N3:/v1/ask 检索过滤透传 ----------
from generator import Generator, MockLLM
from _fakes import make_res, make_hit


def test_ask_forwards_filters_to_retriever():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="表数据 6,779,511")])
    app = make_app(retriever=ret, generator_factory=lambda r, c: Generator(r, MockLLM()))
    with TestClient(app) as c:
        r = c.post("/v1/ask", json={"query": "营收?", "kind": "table",
                                    "doc_ids": ["d1"], "strategy": "sparse"}).json()
    assert r["status"] == "ok"
    call = ret.calls[0]
    assert call["kind"] == "table" and call["doc_ids"] == ["d1"] and call["strategy"] == "sparse"


def test_ask_bad_strategy_structured():
    app = make_app(generator_factory=lambda r, c: Generator(r, MockLLM()))
    with TestClient(app) as c:
        r = c.post("/v1/ask", json={"query": "q", "strategy": "weird"}).json()
    assert r["status"] == "bad_arg" and "strategy" in r["hint"]


# ---------- 阶段E:/readyz 探下游 + 豁免鉴权(F 的 nginx 健康导流前提;唯一代码改动)----------
def _ret_with_qdrant(collection_exists):
    """给 FakeRetriever 补一个 store.client.collection_exists(默认 fake 无 .client)。"""
    ret = FakeRetriever()
    ret.store = SimpleNamespace(client=SimpleNamespace(collection_exists=collection_exists))
    return ret


def test_readyz_ready_when_downstream_ok():
    """local 模式(inference_url 空):store.client.collection_exists 可调 → ready 200。"""
    with TestClient(make_app(retriever=_ret_with_qdrant(lambda name: True))) as c:
        r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_readyz_503_when_qdrant_down():
    """collection_exists 抛错 → 结构化 503 qdrant_unavailable(不裸崩;nginx 据此摘流量,healthz 仍 200)。"""
    def boom(name):
        raise RuntimeError("connection refused to http://qdrant:6333")
    with TestClient(make_app(retriever=_ret_with_qdrant(boom))) as c:
        r = c.get("/readyz")
        assert c.get("/healthz").status_code == 200      # liveness 不受下游影响(不 crashloop)
    assert r.status_code == 503 and r.json()["status"] == "qdrant_unavailable"
    # 评审 sec-2:503 体【不得】回 str(e)——否则未鉴权探针读到内网 host:port。删 log.warning 里的 exc_info 不影响,但删源码里的 deter 收紧即红
    assert "qdrant:6333" not in r.text and "connection refused" not in r.text


def test_readyz_collection_missing_503():
    """评审 compose-B:collection 不存在(新起 server 未迁移)→ 503 collection_missing,不绿着查空库。
    删 service.py 里 `if not exists` 那支即红(退回"qdrant 可达就 ready"的假绿)。"""
    with TestClient(make_app(retriever=_ret_with_qdrant(lambda name: False))) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "collection_missing"


def test_readyz_exempt_from_auth():
    """legacy 鉴权模式无 API key:/readyz 不 401(探针无 key 可达);/v1/ 无 key 仍 401(证鉴权真在)。
    删 auth 白名单里的 /readyz 即红 —— healthcheck 在 keys/legacy 模式永远 unhealthy。"""
    app = make_app(retriever=_ret_with_qdrant(lambda name: True), cfg=make_cfg(api_key="secret"))
    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 200                              # 无 key 也 200(豁免)
        assert c.post("/v1/retrieve", json={"query": "q"}).status_code == 401   # 无 key → 401(鉴权真在)


# ---------- 阶段F 审查:/readyz 的 inference 探活分支(remote 模式;nginx 导流量正依赖它,原零测试覆盖)----------
class _FakeAsyncClient:
    """假 httpx.AsyncClient:async with 上下文 + get(status 或抛异常)。仿 service.readyz 的 AsyncClient 用法。"""
    def __init__(self, status=None, exc=None):
        self._status, self._exc = status, exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, timeout=None):
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(status_code=self._status)


def test_readyz_inference_ready(monkeypatch):
    """remote 模式:qdrant 有集合 + inference /readyz 200 → 整体 ready 200。"""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(status=200))
    app = make_app(retriever=_ret_with_qdrant(lambda name: True),
                   cfg=make_cfg(inference_url="http://inf:8900"))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_readyz_inference_not_ready(monkeypatch):
    """inference 还在预热(/readyz 503)→ pharos /readyz 503 inference_not_ready(nginx 据此不导流量到本副本)。"""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(status=503))
    app = make_app(retriever=_ret_with_qdrant(lambda name: True),
                   cfg=make_cfg(inference_url="http://inf:8900"))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "inference_not_ready"


def test_readyz_inference_unavailable_no_leak(monkeypatch):
    """inference 连不上(探活抛异常)→ 503 inference_unavailable,且响应体不泄内网 host:port(sec-2 同纪律)。"""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **k: _FakeAsyncClient(exc=httpx.ConnectError("refused to http://inf:8900")))
    app = make_app(retriever=_ret_with_qdrant(lambda name: True),
                   cfg=make_cfg(inference_url="http://inf:8900"))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "inference_unavailable"
    assert "inf:8900" not in r.text and "refused" not in r.text


# ---------- 阶段F 审查:mcp_stdio 出口透传 inference_url(D 阶段"漏改出口"同款坑在 remote 开关上复发)----------
def test_mcp_stdio_config_passes_inference_url(monkeypatch):
    """守护 mcp_stdio._config 透传 inference_url + 模型路径/gpu_name。删任一透传即红 —— agentic 出口静默退回 local。"""
    from pharos import mcp_stdio as S
    monkeypatch.setenv("PHAROS_INFERENCE_URL", "http://inf:8900")
    monkeypatch.setenv("PHAROS_TENANT", "demo")
    monkeypatch.setenv("PHAROS_DENSE_MODEL_PATH", "/custom/emb")
    monkeypatch.setenv("PHAROS_GPU_NAME", "5090")
    ec = S._config()
    assert ec.inference_url == "http://inf:8900"        # 核心:remote 开关到位
    assert ec.dense_model_path == "/custom/emb"          # 模型路径同源
    assert ec.gpu_name_must_contain == "5090"            # gpu 名同源


# ---------- 阶段F 审查:失败模式 4 字段 PHAROS_* → PharosConfig → engine 透传到 EmbedConfig ----------
def test_inference_failure_fields_env_to_embedconfig(monkeypatch):
    """PHAROS_INFERENCE_RETRIES/TIMEOUT/... 经 from_env 进 PharosConfig,再经 engine.build_retriever 透传到
    EmbedConfig(否则重试/超时钉死默认值,阶段F 调参必须改码重建镜像)。"""
    from unittest import mock

    from pharos.engine import build_retriever
    monkeypatch.setenv("PHAROS_INFERENCE_RETRIES", "5")
    monkeypatch.setenv("PHAROS_INFERENCE_BACKOFF", "1.5")
    monkeypatch.setenv("PHAROS_INFERENCE_TIMEOUT", "90")
    cfg = pconfig.from_env()
    assert cfg.inference_retries == 5 and cfg.inference_backoff == 1.5 and cfg.inference_timeout == 90.0
    # 透传到 EmbedConfig:mock Retriever 截获 ecfg(避免真建 Store/Dense)
    cfg2 = make_cfg(inference_url="http://inf:8900", inference_retries=5, inference_backoff=1.5)
    with mock.patch("pharos.engine.Retriever") as MockRet:
        build_retriever(cfg2)
    ecfg = MockRet.call_args.args[0]
    assert ecfg.inference_retries == 5 and ecfg.inference_backoff == 1.5


# ---------- 修复:PHAROS_ASK_MAX_CONTEXT_TOKENS 经 config → engine 透传到 Generator(闭管道 context 预算)----------
def test_ask_max_context_tokens_env_to_generator(monkeypatch):
    """守护透传链:env → PharosConfig.ask_max_context_tokens → build_generator(max_context_tokens=)。
    删 engine 的 max_context_tokens= 透传即红 —— 配了 env 却静默不生效(D 阶段"漏改出口"同款坑)。"""
    from unittest import mock

    from pharos.engine import build_generator
    monkeypatch.setenv("PHAROS_ASK_MAX_CONTEXT_TOKENS", "6000")
    cfg = pconfig.from_env()
    assert cfg.ask_max_context_tokens == 6000
    with mock.patch("pharos.engine.OpenAICompatibleLLM"), \
         mock.patch("pharos.engine.Generator") as MockGen:
        build_generator(None, make_cfg(ask_max_context_tokens=6000))
        build_generator(None, make_cfg())                      # 默认 0 -> None(行为完全不变)
    assert MockGen.call_args_list[0].kwargs.get("max_context_tokens") == 6000
    assert MockGen.call_args_list[1].kwargs.get("max_context_tokens") is None


# ---------- 阶段F 审查:P2-3 服务端 /rerank 消费客户端 instruction(收敛"客户端唯一真相")----------
def test_reranker_score_consumes_client_instruction():
    """Reranker.score(instruction=) 非 None 时用调用方值、None 时回落服务端 cfg。inference_server /rerank 据此
    把 payload 的 q.instruction 送进模型(修 P2-3 双源 footgun)。删 rerank.py 的 instruction 形参即红。"""
    import contextlib

    from embedder.config import EmbedConfig
    from embedder.rerank import Reranker
    rr = Reranker.__new__(Reranker)
    rr.cfg = EmbedConfig(rerank_instruction="SERVER_DEFAULT")
    rr._load = lambda: None
    rr._fwd_lock = contextlib.nullcontext()
    captured = {}
    rr._model = SimpleNamespace(process=lambda inp: (captured.update(inp), [0.5, 0.5])[1])
    rr.score("q", ["a", "b"], instruction="CLIENT_WINS")
    assert captured["instruction"] == "CLIENT_WINS"       # 客户端值优先(收敛唯一真相)
    rr.score("q", ["a", "b"], instruction=None)
    assert captured["instruction"] == "SERVER_DEFAULT"     # None 回落服务端默认(向后兼容)


# ========== embedder 簇对抗验证修复(2026-07-07)==========
# 修复1:index_document 删旧时机重排(编码/sidecar tmp 为纯准备 -> delete->upsert->replace 毫秒级收尾)
from dataclasses import dataclass, field


@dataclass
class _Chunk:
    """最小 chunk 替身:覆盖 embed._payload 读取的全部字段。"""
    chunk_id: str
    doc_id: str
    text: str
    kind: str = "text"
    content_raw: str = ""
    breadcrumb: list = field(default_factory=list)
    section_path: str = ""
    section_id: str = "s1"
    section_anchor: str | None = None
    page_start: int = 0
    page_end: int = 0
    source_indices: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    lang: str = "en"
    doc_type: str = "unknown"
    image_path: str | None = None
    doc_meta: dict = field(default_factory=dict)
    acl: dict = field(default_factory=lambda: {"tenant": "t1", "allow": [], "visibility": "public", "unset": False})


@dataclass
class _El:
    idx: int
    text: str


class _ChunkResult:
    def __init__(self, chunks):
        self.chunks = chunks
        self.sections = []
        self.banners = frozenset()

    def acl_index(self):
        return {0: {"tenant": "t1", "allow": [], "visibility": "public", "unset": False}}


class _FakeDense:
    """假 dense:fail_at=N 表示第 N 次 encode_text 起抛 exc(模拟 remote 重试耗尽 / GPU 断言快失败)。"""
    def __init__(self, dim=8, fail_at=None, exc=None):
        self.dim, self.fail_at, self.exc = dim, fail_at, exc
        self.n = 0

    def encode_text(self, texts, instruction=None):
        import numpy as np
        self.n += 1
        if self.fail_at is not None and self.n >= self.fail_at:
            raise self.exc or RuntimeError("boom")
        return np.array([[0.1] * self.dim] * len(texts), dtype="float32")


def _mk_embedder(tmp_path, dense):
    from embedder import EmbedConfig, Embedder
    cfg = EmbedConfig(qdrant_path=":memory:", dense_dim=8, collection="t", sidecar_dir=str(tmp_path))
    return Embedder(cfg, dense=dense)


def test_index_document_encode_failure_keeps_old_index(tmp_path):
    """修复1核心回归:重索引在编码期失败(旧实现已先 delete_by_doc)-> 旧 point/旧 sidecar 必须原样可用。
    修复前:旧向量全灭、该 doc 从库中持久消失直到人工重跑;修复后:编码是纯准备,失败零副作用。"""
    from embedder.errors import InferenceUnavailable
    emb = _mk_embedder(tmp_path, _FakeDense())
    chunks = [_Chunk("doc#0", "doc", "第一段"), _Chunk("doc#1", "doc", "第二段")]
    emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")
    assert emb.store.client.count("t").count == 2
    sidecar_v1 = open(os.path.join(str(tmp_path), "doc.json"), encoding="utf-8").read()
    emb.dense = _FakeDense(fail_at=2, exc=InferenceUnavailable("重试耗尽"))   # 第 2 个 chunk 编码时炸
    with pytest.raises(InferenceUnavailable):
        emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")
    assert emb.store.client.count("t").count == 2, "编码失败是纯准备阶段,旧向量必须仍可检索(修复前=0)"
    assert open(os.path.join(str(tmp_path), "doc.json"), encoding="utf-8").read() == sidecar_v1, "旧 sidecar 不得被动"
    assert not os.path.exists(os.path.join(str(tmp_path), "doc.json.tmp")), "无 tmp 残留"


def test_index_document_post_delete_failure_loud(tmp_path, capsys):
    """修复1:delete 之后失败(upsert 炸)-> 该 doc 真脱库,必须响亮告警"已脱库需重跑"再 raise,
    且 sidecar 不得半更新(replace 未执行,tmp 清理)。"""
    emb = _mk_embedder(tmp_path, _FakeDense())
    chunks = [_Chunk("doc#0", "doc", "第一段")]
    emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")

    def boom(points):
        raise RuntimeError("qdrant write failed")
    emb.store.upsert = boom
    with pytest.raises(RuntimeError, match="qdrant write failed"):
        emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")
    err = capsys.readouterr().err
    assert "已脱库" in err and "doc" in err, "post-delete 失败必须响亮告警,不许静默 raise 被上游'跳过'吞掉"
    assert os.path.exists(os.path.join(str(tmp_path), "doc.json")), "旧 sidecar 保留(与新向量未落一致)"
    assert not os.path.exists(os.path.join(str(tmp_path), "doc.json.tmp")), "tmp 清理"


def test_run_index_collects_failures_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """修复1配套:indexer 不再静默"跳过"失败 doc —— 收集清单、DONE 行汇总、SystemExit 非零退出。"""
    import pharos.indexer as IX
    corpus = tmp_path / "corpus"
    (corpus / "typeA__doc_bad").mkdir(parents=True)
    (corpus / "typeA__doc_ok").mkdir()
    monkeypatch.setattr(IX, "from_mineru_dir", lambda d: [SimpleNamespace(text="正文")])
    monkeypatch.setattr(IX, "Chunker", lambda: SimpleNamespace(chunk=lambda els, **kw: SimpleNamespace(chunks=[1, 2])))

    class _FakeEmb:
        def __init__(self, cfg):
            pass

        def index_document(self, d, els, res, image_root):
            if "bad" in d:
                raise RuntimeError("推理服务不可用")
            return {}
    monkeypatch.setattr(IX, "Embedder", _FakeEmb)
    with pytest.raises(SystemExit) as ei:
        IX.run_index(make_cfg(), corpus=str(corpus), dest=str(tmp_path / "idx"))
    out = capsys.readouterr().out
    assert "失败 typeA__doc_bad" in out                       # 逐条失败行
    assert "失败 1 篇" in out                                  # DONE 汇总
    assert "typeA__doc_bad" in str(ei.value), "退出消息须带失败清单供重跑"


# 修复6(收窄版):search_with_context 同节折叠计数外露(SearchResults.section_folded_n)
def test_search_with_context_counts_section_folds():
    from embedder.retrieve import Retriever, SearchResults
    from embedder.types import Hit, User

    pub = {"tenant": "t", "visibility": "public", "allow": [], "unset": False}

    def hit(cid, sid):
        return Hit(chunk_id=cid, doc_id="d", kind="text", text="t", score=0.9, payload={"section_id": sid})

    def mk(hits):
        r = Retriever.__new__(Retriever)                     # 不走 __init__(不建 Store/Dense)
        r.search = lambda q, u, top_k=None, **kw: hits
        r._assemble = lambda h, u, cache=None: SimpleNamespace(acl=pub, text="ctx", anchor=None)
        return r
    out = mk([hit("c1", "s1"), hit("c2", "s1"), hit("c3", "s1"), hit("c4", "s2")]).search_with_context(
        "q", User("t", []))
    assert isinstance(out, SearchResults) and [o["hit"].chunk_id for o in out] == ["c1", "c4"]
    assert out.section_folded_n == 2, "c2/c3 同 s1 折叠不进返回,计数必须外露(区分'库存尽'与'被折叠')"
    assert mk([hit("c1", "s1")]).search_with_context("q", User("t", [])).section_folded_n == 0
