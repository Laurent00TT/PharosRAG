"""Generator 单测(纯 CPU,MockLLM + fake retriever,duck typing 不依赖 embedder)。
覆盖:引用解析、ACL 出口降级(context=None 用 hit.text)、grounding 退路(无 context)、越界引用丢弃。"""
import os
import sys


from generator.generate import Generator
from generator.llm import MockLLM


class _Hit:
    def __init__(self, cid, doc, text, payload):
        self.chunk_id, self.doc_id, self.text, self.payload = cid, doc, text, payload


class _Ctx:
    def __init__(self, text):
        self.text = text


class _Ret:
    def __init__(self, results):
        self._r = results

    def search_with_context(self, query, user, top_k=None, rerank=False):
        return self._r


def test_answer_with_citations():
    results = [
        {"hit": _Hit("c1", "d1", "hit1", {"doc_meta": {"title": "Doc One"}, "section_path": "S1", "page_start": 3}),
         "context": _Ctx("big context one")},
        {"hit": _Hit("c2", "d2", "hit2", {"doc_meta": {"title": "Doc Two"}}), "context": _Ctx("big context two")},
    ]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.n_contexts == 2
    assert [c.marker for c in ans.citations] == [1, 2]          # MockLLM 引用 [1][2]
    assert ans.citations[0].title == "Doc One" and ans.citations[0].page == 3
    assert "big context one" in ans.raw_messages[1].content     # context 进了 prompt


def test_acl_exit_degrade_to_hit():
    # context=None(sidecar 丢/出口未过)-> 降级用 hit.text(单 chunk 已授权),不丢答案
    results = [{"hit": _Hit("c1", "d1", "hit text only", {}), "context": None}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.n_contexts == 1 and "hit text only" in ans.raw_messages[1].content


def test_no_context_grounding():
    # 无检索结果 -> 无 context -> MockLLM 走 grounding 退路(信息不足),0 引用
    ans = Generator(_Ret([]), MockLLM()).answer("q", user=None)
    assert ans.n_contexts == 0 and ans.citations == [] and "enough information" in ans.text.lower()


def test_citation_out_of_range_dropped():
    # 答案幻觉出越界 [cite:99] -> 丢弃,不映射错来源
    class _BadLLM:
        def complete(self, messages):
            return "answer [cite:1] and [cite:99]"
    results = [{"hit": _Hit("c1", "d1", "t", {}), "context": _Ctx("ctx")}]
    ans = Generator(_Ret(results), _BadLLM()).answer("q", user=None)
    assert [c.marker for c in ans.citations] == [1]            # [cite:99] 越界被丢


def test_context_bracket_not_polluting():
    # LLM 照抄正文里的裸 [1]/[99] 不被当引用;只有 [cite:n] 才算(review#1 核心:溯源不被正文污染)
    class _EchoLLM:
        def complete(self, messages):
            return "Doc says [1] and [99] are footnotes. Real cite [cite:2]."
    results = [{"hit": _Hit("c1", "d1", "t1", {}), "context": _Ctx("ctx1")},
               {"hit": _Hit("c2", "d2", "t2", {}), "context": _Ctx("ctx2")}]
    ans = Generator(_Ret(results), _EchoLLM()).answer("q", user=None)
    assert [c.marker for c in ans.citations] == [2]           # 正文 [1][99] 不污染,只认 [cite:2]


def test_asset_content_raw_appended():
    # ③ 表格/图表 grounding:big-block(ctx)不含资产数据 -> content_raw 补进 prompt;非资产块不补
    results = [
        {"hit": _Hit("c1", "d1", "chart caption", {"kind": "chart", "content_raw": "DATA revenue=42M"}),
         "context": _Ctx("surrounding prose without the number")},
        {"hit": _Hit("c2", "d2", "t2", {"kind": "text", "content_raw": "SHOULD_NOT_APPEAR"}),
         "context": _Ctx("plain text ctx")},
    ]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    prompt = ans.raw_messages[1].content
    assert "DATA revenue=42M" in prompt              # 图表 content_raw 补回(big-block 缺它)
    assert "SHOULD_NOT_APPEAR" not in prompt         # 非资产块 content_raw 不补


def test_asset_long_content_raw_not_duplicated():
    # 长 content_raw 已在 ctx(big-block)里 -> 不重复追加(去重只对长资产,避免重复喂长表 HTML)
    blob = "Quarterly revenue table: Q1 100 Q2 200 Q3 300 Q4 400 total 1000 units"   # >40 字
    results = [{"hit": _Hit("c1", "d1", "cap", {"kind": "table", "content_raw": blob}),
               "context": _Ctx("Preamble. " + blob + " Footnote.")}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.raw_messages[1].content.count("Q2 200") == 1     # 长 craw 已在 ctx,不重复


def test_asset_short_content_raw_always_appended():
    # R2#2:短资产数据(如 "42")即便恰为散文子串也总补回 —— 子串去重会误伤,重现 ③ 失败
    results = [{"hit": _Hit("c1", "d1", "cap", {"kind": "table", "content_raw": "42"}),
               "context": _Ctx("revenue grew by 42 percent overall")}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.raw_messages[1].content.count("42") >= 2         # 短数据总补回(否则被误抑制)


def test_empty_context_deterministic_grounding():
    # R3 E:零召回 -> 确定性返回"信息不足",根本不调 LLM(grounding 退路不靠模型听话)
    class _BoomLLM:
        def complete(self, messages):
            raise AssertionError("零召回不应调用 LLM")
    ans = Generator(_Ret([]), _BoomLLM()).answer("q", user=None)
    assert ans.n_contexts == 0 and ans.citations == [] and "enough information" in ans.text.lower()


def test_citation_title_falls_back_to_doc_id():
    # R3 D:无 doc_meta.title -> Citation.title 兜底 doc_id(非空串,与喂给 LLM 的 source 一致)
    results = [{"hit": _Hit("c1", "mydoc", "t", {}), "context": _Ctx("ctx")}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.citations and ans.citations[0].title == "mydoc"


def test_acl_check_filters():
    # 注入 acl_check 时,无权命中块被挡在 prompt 外(防御纵深 review#3)
    results = [{"hit": _Hit("c1", "d1", "ok text", {"acl": {"v": "pub"}}), "context": _Ctx("ctx1")},
               {"hit": _Hit("c2", "d2", "SECRET", {"acl": {"v": "secret"}}), "context": _Ctx("ctx2")}]
    g = Generator(_Ret(results), MockLLM(), acl_check=lambda acl, u: acl.get("v") == "pub")
    ans = g.answer("q", user=None)
    assert ans.n_contexts == 1 and "SECRET" not in ans.raw_messages[1].content   # secret 块被挡在 prompt 外


def test_filters_forwarded_only_when_set():
    # N3:过滤参数按需透传 —— 不设不传(老窄签名 retriever 兼容),设了原样到 retriever
    class _CapRet:
        def __init__(self):
            self.kw = None

        def search_with_context(self, query, user, top_k=None, rerank=False, **kw):
            self.kw = kw
            return []

    cap = _CapRet()
    gen = Generator(cap, MockLLM())
    gen.answer("q", user=None)                                   # 不设过滤
    assert cap.kw == {}
    gen.answer("q", user=None, kind="table", doc_ids=["d1"], strategy="sparse")
    assert cap.kw == {"kind": "table", "doc_ids": ["d1"], "strategy": "sparse"}
    # 老窄签名(无 **kw)在不传过滤时依然可用
    class _Narrow:
        def search_with_context(self, query, user, top_k=None, rerank=False):
            return []
    Generator(_Narrow(), MockLLM()).answer("q", user=None)       # 不抛 TypeError


if __name__ == "__main__":
    test_answer_with_citations()
    test_acl_exit_degrade_to_hit()
    test_no_context_grounding()
    test_citation_out_of_range_dropped()
    test_context_bracket_not_polluting()
    test_asset_content_raw_appended()
    test_asset_long_content_raw_not_duplicated()
    test_asset_short_content_raw_always_appended()
    test_empty_context_deterministic_grounding()
    test_citation_title_falls_back_to_doc_id()
    test_acl_check_filters()
    test_filters_forwarded_only_when_set()
    print("generate tests OK")


def test_system_numeric_scope_constraint():
    # 数值范围约束(窄靶,防"分部数字被引申成总体"——Pharos N3 实证错答):SYSTEM 进 prompt
    from generator.prompt import SYSTEM, PromptBuilder
    assert "scoped" in SYSTEM and "NEVER present it as the total" in SYSTEM
    msgs = PromptBuilder().build("q", [{"text": "Domestic streaming revenues 4,180,339", "source": "s"}])
    assert "NEVER present it as the total" in msgs[0].content


def test_extra_legs_union_dedup():
    # smart-ask 多路检索:leg 命中并集追加在主命中之后,按 chunk_id 去重;不传 legs 零影响
    class _CapRet:
        def __init__(self):
            self.calls = []

        def search_with_context(self, query, user, top_k=None, rerank=False, **kw):
            self.calls.append({"top_k": top_k, "rerank": rerank, **kw})
            if kw.get("kind") == "table":
                return [{"hit": _Hit("t1", "d1", "表格块", {"kind": "table"}), "context": _Ctx("五年表")},
                        {"hit": _Hit("c1", "d1", "重复的", {}), "context": _Ctx("dup")}]   # c1 与主路重复
            return [{"hit": _Hit("c1", "d1", "散文块", {}), "context": _Ctx("主路上下文")}]

    cap = _CapRet()
    from generator.signals import DEFAULT_TABLE_LEG
    ans = Generator(cap, MockLLM()).answer("2015 净利润?", user=None, extra_legs=[DEFAULT_TABLE_LEG])
    assert len(cap.calls) == 2 and cap.calls[1]["kind"] == "table" and cap.calls[1]["rerank"] is True
    assert cap.calls[1]["top_k"] == 5 and cap.calls[1]["rerank_top_n"] == 50
    assert ans.n_contexts == 2                            # 主路 c1 + leg 新增 t1;重复 c1 去重
    assert "主路上下文" in ans.raw_messages[1].content and "五年表" in ans.raw_messages[1].content


def test_looks_numeric():
    from generator.signals import looks_numeric
    assert looks_numeric("Netflix 2011 到 2015 年每一年的净利润分别是多少?")
    assert looks_numeric("公司的营收占比如何变化")
    assert looks_numeric("How much were total revenues?")
    assert not looks_numeric("这篇论文的方法论核心思想是什么")
    assert not looks_numeric("What is the main contribution of this paper?")
