"""Retriever.search_with_context 的契约/安全单测(纯 CPU,mock search+_assemble,不依赖 GPU/Qdrant)。
覆盖:dedup None 不折叠(seal#6)、出口 ACL 二次校验(seal#2/铁律5)、sidecar 缺失降级(seal#10)、
同 anchor big-block 去重(lazy-tree review)、sidecar 版本绑定(写侧盖章 + 读侧不符响亮失败)。"""
import json
import os
import tempfile


from embedder.config import SIDECAR_VERSION, EmbedConfig
from embedder.retrieve import Retriever
from embedder.types import Hit, User


class _Ctx:
    def __init__(self, acl=None, anchor=None):
        self.acl = acl
        self.text = "ctx"
        self.anchor = anchor


def _hit(cid, doc, sid):
    return Hit(chunk_id=cid, doc_id=doc, kind="text", text="t", score=0.9, payload={"section_id": sid})


def _ret(hits, assemble):
    r = Retriever.__new__(Retriever)               # 不走 __init__(不建 Store/Dense)
    r.search = lambda q, u, top_k=None, rerank=False, doc_ids=None, doc_type=None, kind=None, strategy="hybrid", rerank_top_n=None: hits
    r._assemble = assemble
    return r


def test_none_section_not_folded():
    # seal#6:section_id=None 是"无小节"不是"同一小节",不能折叠(否则同 doc 无小节命中只剩首个)
    hits = [_hit("c1", "d", None), _hit("c2", "d", None), _hit("c3", "d", "s1"), _hit("c4", "d", "s1")]
    out = _ret(hits, lambda h, u, cache=None: _Ctx()).search_with_context("q", User(tenant="t", principals=[]))
    ids = [o["hit"].chunk_id for o in out]
    assert ids == ["c1", "c2", "c3"], ids          # c1,c2 都留(None 不折叠);c3 留、c4 同 s1 去重


def _hit_kind(cid, doc, sid, kind):
    return Hit(chunk_id=cid, doc_id=doc, kind=kind, text="t", score=0.9, payload={"section_id": sid, "kind": kind})


def test_asset_not_section_deduped():
    # ③:资产块(图表/表格)不参与 section 去重 —— 否则与散文兄弟同 section 被折叠,而 big-block 不含其数据(_gather
    # 不取 asset_content)= 彻底丢失("召回到却答不出表/图里的数")。资产按 chunk_id 独立成键。
    pub = {"tenant": "t", "visibility": "public", "allow": []}
    hits = [_hit_kind("c1", "d", "s1", "text"), _hit_kind("c2", "d", "s1", "chart"), _hit_kind("c3", "d", "s1", "text")]
    out = _ret(hits, lambda h, u, cache=None: _Ctx(pub)).search_with_context("q", User(tenant="t", principals=[]))
    ids = [o["hit"].chunk_id for o in out]
    assert ids == ["c1", "c2"], ids        # c1散文留;c2图表不去重也留;c3散文同s1被去重


class _Win:
    """窗口块 mock(windowed=True),用于 R2#1/#5。"""
    def __init__(self, rs, anchor):
        self.acl = {"tenant": "t", "visibility": "public", "allow": []}
        self.text = "windowed text"
        self.anchor = anchor
        self.climbed = 0
        self.windowed = True
        self.resolved_section = rs


def test_window_block_status_section_window():
    # R2#1:窗口块(windowed)必须标 section_window,不能冒充 full_section(否则 agent 不去 expand)
    out = _ret([_hit("c1", "d", "s1")], lambda h, u, cache=None: _Win("s0#window", [0, 3])).search_with_context(
        "q", User(tenant="t", principals=[]))
    assert out[0]["context_status"] == "section_window", out[0]["context_status"]


def test_window_dedup_by_resolved_section():
    # R2#5:同 bound 的两个窗口(anchor 不同但 resolved_section 相同)应折叠,第二个降级
    ctxs = {"c1": _Win("s0#window", [0, 3]), "c2": _Win("s0#window", [1, 4])}
    out = _ret([_hit("c1", "d", "s1"), _hit("c2", "d", "s2")],
               lambda h, u, cache=None: ctxs[h.chunk_id]).search_with_context("q", User(tenant="t", principals=[]))
    assert out[0]["context"] is not None and out[1]["context"] is None      # c2 同 resolved_section 被折叠
    assert out[1]["context_status"] == "deduped"


def test_empty_text_bigblock_status_asset_no_prose():
    # R2#4:big-block text 全空(资产页无散文)-> asset_no_prose,不冒充 full_section
    empty = _Ctx({"tenant": "t", "visibility": "public", "allow": []}, anchor=[2, 4])
    empty.text = ""
    out = _ret([_hit("c1", "d", "s1")], lambda h, u, cache=None: empty).search_with_context(
        "q", User(tenant="t", principals=[]))
    assert out[0]["context_status"] == "asset_no_prose", out[0]["context_status"]


def test_exit_acl_check_blocks():
    # seal#2/铁律5:big.acl 对 user 不可见 -> context 置 None(不交付),即便 hit 已过硬过滤
    bad = _Ctx({"tenant": "t2", "visibility": "restricted", "allow": []})   # user 无权
    out = _ret([_hit("c1", "d", "s")], lambda h, u, cache=None: bad).search_with_context(
        "q", User(tenant="t1", principals=["g"]))
    assert out[0]["context"] is None


def test_sidecar_missing_degrades():
    # seal#10:一条 hit 的 sidecar 丢 -> 该条降级(context=None),不拖垮整次查询
    def assemble(h, u, cache=None):
        if h.doc_id == "bad":
            raise FileNotFoundError("sidecar gone")
        return _Ctx({"tenant": "t", "visibility": "public", "allow": []})   # 正常 ctx 带可见 acl(acl=None 现被出口拒绝,review#4)
    out = _ret([_hit("c1", "bad", "s"), _hit("c2", "ok", "s")], assemble).search_with_context(
        "q", User(tenant="t", principals=[]))
    assert len(out) == 2 and out[0]["context"] is None and out[1]["context"] is not None


def test_dedup_same_anchor_bigblock():
    # 同 doc 同 anchor 的 big-block(climb 到同父)只留首个,其余降级裸 hit(lazy-tree review:多命中重叠)
    acl = {"tenant": "t", "visibility": "public", "allow": []}
    amap = {"c1": [0, 5], "c2": [0, 5], "c3": [9, 12]}     # c1,c2 同 anchor;c3 不同
    out = _ret([_hit("c1", "d", "s1"), _hit("c2", "d", "s2"), _hit("c3", "d", "s3")],
               lambda h, u, cache=None: _Ctx(acl, anchor=amap[h.chunk_id])).search_with_context(
        "q", User(tenant="t", principals=[]))
    ctxs = [o["context"] for o in out]
    assert ctxs[0] is not None and ctxs[1] is None and ctxs[2] is not None   # c2 重复 anchor -> 降级


def test_writer_stamps_version():
    # 写侧:_write_sidecar 必须盖上当前 SIDECAR_VERSION(与读侧校验同源)
    from dataclasses import dataclass

    from embedder.embed import Embedder       # 函数内 import:torch/qdrant 链路只在跑此测试时加载(import 不触 GPU)

    @dataclass
    class _El:
        idx: int
        text: str

    @dataclass
    class _Sec:
        sec_id: str

    class _Res:                                # 仿 ChunkResult:_write_sidecar 只读 sections/banners/acl_index()
        sections = [_Sec("s1")]
        banners = frozenset(["banner"])

        def acl_index(self):
            return {0: {"tenant": "t", "visibility": "public", "allow": []}}

    d = tempfile.mkdtemp()
    e = Embedder.__new__(Embedder)             # 不走 __init__(不建 Store/Dense)
    e.cfg = EmbedConfig(sidecar_dir=d)
    e._write_sidecar("doc", [_El(0, "x")], _Res())
    with open(os.path.join(d, "doc.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == SIDECAR_VERSION, data.get("version")


def _load_with_sidecar(payload: dict):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "doc.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    r = Retriever.__new__(Retriever)           # 只需 cfg;_load_sidecar 版本检查在 chunker import 前,不触 GPU
    r.cfg = EmbedConfig(sidecar_dir=d)
    return r._load_sidecar("doc")


def test_sidecar_version_mismatch_loud():
    # 读侧:显式旧版本号 -> ValueError(非 FileNotFoundError/JSONDecodeError,search_with_context 不会 catch -> 响亮)
    try:
        _load_with_sidecar({"version": 999, "elements": [], "sections": [], "banners": [], "acl_index": {}})
        assert False, "版本不符应抛 ValueError"
    except ValueError as ex:
        assert "版本" in str(ex), str(ex)


def test_sidecar_missing_version_loud():
    # 读侧:旧 sidecar 无 version 字段(grandfather 决策:缺失即不符,强制重建基线)-> 同样响亮失败
    try:
        _load_with_sidecar({"elements": [], "sections": [], "banners": [], "acl_index": {}})
        assert False, "缺 version 字段应抛 ValueError"
    except ValueError as ex:
        assert "版本" in str(ex), str(ex)


def test_load_sidecar_doc_acl_precheck():
    # Batch2.B(B2 对抗 review 修订):doc 级预检只给 **doc_id 直读工具**(传 user)用;空 acl_index 也 fail-closed;
    # user=None(hit 驱动)不预检——防 chunker _stricter 把命中块 acl 改写出 acl_index 后误伤 public 命中(HIGH)。
    d = tempfile.mkdtemp()

    def _write(name, acl_index):
        p = {"version": SIDECAR_VERSION, "elements": [], "sections": [], "banners": [], "acl_index": acl_index}
        with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(p, f)

    _write("doc", {"0": {"tenant": "t1", "visibility": "restricted", "allow": ["g_fin"], "unset": False}})
    _write("empty", {})                                   # 空 acl_index
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig(sidecar_dir=d)
    for label, kw in [("无权 user", dict(user=User("t1", ["g_hr"]))),
                      ("空 acl_index 直读", dict(user=User("t1", ["g_fin"])))]:
        name = "doc" if "无权" in label else "empty"
        try:
            r._load_sidecar(name, **kw)
            assert False, f"{label} 应 fail-closed 抛 PermissionError"
        except PermissionError:
            pass
    assert r._load_sidecar("doc", user=User("t1", ["g_fin"]))[3], "有权 user 直读 -> 正常"
    assert r._load_sidecar("doc")[3] is not None, "hit 驱动(user=None)不预检,正常加载"
    assert r._load_sidecar("empty"), "空 acl_index 无 user(hit 驱动)也加载"


_HR = {"tenant": "t1", "visibility": "restricted", "allow": ["g_hr"], "unset": False}
_PUB = {"tenant": "t1", "visibility": "public", "allow": [], "unset": False}


def test_visible_own_sections_no_parent_leak():
    # B3 review HIGH:受限父小节(own-body 全无权)不应因可见子节被列出 —— own-body 判定
    from chunker.types import Section
    secs = {  # 父 s0[0,4) 自身 body idx0,1=HR;子 s1[2,4) idx2,3=public
        "s0": Section("s0", "d", 1, "机密HR", ["机密HR"], 0, 4, None),
        "s1": Section("s1", "d", 2, "公开通知", ["机密HR", "公开通知"], 2, 4, "s0"),
    }
    acl_index = {0: _HR, 1: _HR, 2: _PUB, 3: _PUB}
    r = Retriever.__new__(Retriever)
    vis_pub = {s.sec_id for s in r._visible_own_sections(secs, acl_index, User("t1", ["g_pub"]))}
    assert vis_pub == {"s1"}, f"public-only 用户只该见子节 s1,受限父 s0 标题不泄;实得 {vis_pub}"
    vis_hr = {s.sec_id for s in r._visible_own_sections(secs, acl_index, User("t1", ["g_hr"]))}
    assert vis_hr == {"s0", "s1"}, "HR 用户两节都见"


def _write_full_sidecar(d, name, elements, sections, acl_index):
    p = {"version": SIDECAR_VERSION, "elements": elements, "sections": sections,
         "banners": [], "acl_index": {str(k): v for k, v in acl_index.items()}}
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(p, f)


def test_get_document_mixed_acl_and_titles():
    # B3 review:get_document 逐元素 ACL 门控 + 纳入可见小节标题(标题不在 acl_index)+ 不回 n_elements_total
    from types import SimpleNamespace
    d = tempfile.mkdtemp()
    els = [{"idx": 0, "kind": "text", "text": "机密HR"}, {"idx": 1, "kind": "text", "text": "HR-BODY"},
           {"idx": 2, "kind": "text", "text": "公开通知"}, {"idx": 3, "kind": "text", "text": "PUB-BODY"}]
    secs = [{"sec_id": "s0", "doc_id": "d", "level": 1, "title": "机密HR", "breadcrumb": ["机密HR"],
             "start_idx": 0, "end_idx": 4, "parent_sec_id": None},
            {"sec_id": "s1", "doc_id": "d", "level": 2, "title": "公开通知", "breadcrumb": ["机密HR", "公开通知"],
             "start_idx": 2, "end_idx": 4, "parent_sec_id": "s0"}]
    _write_full_sidecar(d, "doc", els, secs, {1: _HR, 3: _PUB})   # 标题 idx0/2 不在 acl_index(仿真实 chunker)
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig(sidecar_dir=d)
    r.store = SimpleNamespace(get_doc_lang=lambda doc_id: "en")
    out = r.get_document("doc", User("t1", ["g_pub"]))            # public-only
    assert "PUB-BODY" in out["text"] and "公开通知" in out["text"], "应含可见正文+其小节标题"
    assert "HR-BODY" not in out["text"] and "机密HR" not in out["text"], "不得含无权 HR 正文/标题"
    assert "n_elements_total" not in out, "不回含无权计数的 total(info_leak)"
    assert out["n_elements_visible"] == 2                          # 公开通知标题 + PUB-BODY
    out_hr = r.get_document("doc", User("t1", ["g_hr"]))
    assert "HR-BODY" in out_hr["text"] and "PUB-BODY" in out_hr["text"], "HR 用户见全文"
    # get_outline:public-only 只列子节 s1,不列受限父 s0;不回 breadcrumb
    ol = r.get_outline("doc", User("t1", ["g_pub"]))
    assert {s["sec_id"] for s in ol} == {"s1"} and "breadcrumb" not in ol[0]


def test_load_sidecar_sparse_idx_raises():
    # B3 review LEAK 修:elements 非密集有序(idx!=position)-> _load_sidecar 断言 raise(防按位置错配 ACL)
    d = tempfile.mkdtemp()
    _write_full_sidecar(d, "sparse", [{"idx": 5, "kind": "text", "text": "a"}, {"idx": 0, "kind": "text", "text": "b"}],
                        [], {})
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig(sidecar_dir=d)
    try:
        r._load_sidecar("sparse")                                 # 无 user 也断言(取材安全前置)
        assert False, "稀疏 idx 应抛 ValueError"
    except ValueError as ex:
        assert "密集" in str(ex)


def test_rerank_graceful_degrade():
    # 4.B:rerank 失败(reranker.rerank 抛)-> 降级返回 hybrid 召回 hits[:k],不崩;返回 hit 仍是 hybrid 量纲
    from types import SimpleNamespace
    hits = [Hit(f"c{i}", "d", "text", "t", 0.1, {}, "rrf") for i in range(3)]
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig()
    r.dense = SimpleNamespace(encode_query=lambda q: SimpleNamespace(tolist=lambda: [0.0] * 8))
    r.store = SimpleNamespace(hybrid_search=lambda *a, **k: list(hits))

    def boom(*a, **k):
        raise RuntimeError("reranker OOM")
    r._get_reranker = lambda: SimpleNamespace(rerank=boom)
    out = r.search("x", User("t", []), top_k=2, rerank=True)
    assert len(out) == 2 and [h.chunk_id for h in out] == ["c0", "c1"], "rerank 失败应降级返回 hybrid top-2"
    assert all(h.score_kind == "rrf" for h in out), "降级 hit 仍是 hybrid 量纲(_build 据此标 rerank_degraded)"


def test_dense_construct_no_gpu():
    # 4.D:Dense 构造不 load 不触 GPU(list_documents 等纯元数据路径 GPU 挂了也能用)
    from embedder.dense import Dense
    d = Dense(EmbedConfig())
    assert d._model is None and d._gpu_error is None, "构造期不应加载模型/触 GPU"


def test_dense_query_cache():
    # 5.C:同 query 第二次命中 LRU,不再 encode(免重复 8B 前向)
    import numpy as np
    from embedder.dense import Dense
    d = Dense(EmbedConfig())
    calls = []
    d.encode_text = lambda texts, instruction=None: (calls.append(texts), np.array([[0.1] * 4]))[1]
    d.encode_query("hello"); d.encode_query("hello")
    assert len(calls) == 1, "同 query 第二次应命中缓存"
    d.encode_query("world")
    assert len(calls) == 2, "不同 query 重新 encode"


if __name__ == "__main__":
    test_none_section_not_folded()
    test_asset_not_section_deduped()
    test_window_block_status_section_window()
    test_window_dedup_by_resolved_section()
    test_empty_text_bigblock_status_asset_no_prose()
    test_exit_acl_check_blocks()
    test_sidecar_missing_degrades()
    test_dedup_same_anchor_bigblock()
    test_writer_stamps_version()
    test_sidecar_version_mismatch_loud()
    test_sidecar_missing_version_loud()
    test_load_sidecar_doc_acl_precheck()
    test_visible_own_sections_no_parent_leak()
    test_get_document_mixed_acl_and_titles()
    test_load_sidecar_sparse_idx_raises()
    test_rerank_graceful_degrade()
    test_dense_construct_no_gpu()
    test_dense_query_cache()
    print("retrieve tests OK")
