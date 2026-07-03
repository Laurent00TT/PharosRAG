"""smart-ask(D9,失败驱动版)单测:第一轮纯净 / 拒答才触发表格腿重试 / auto 留痕 /
用户显式参数优先 / 开关 / hints。前置腿版本已被 88 题实测否决(误伤散文题),勿改回。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from _fakes import FakeRetriever, make_app, make_cfg, make_hit, make_res
from pharos import smart
from pharos.engine import bootstrap
from pharos import config as pconfig

bootstrap(pconfig._default_engine())
from generator import Generator, MockLLM  # noqa: E402


def _gen_factory(r, c):
    return Generator(r, MockLLM())


def _app(retriever, **cfg_kw):
    return make_app(retriever=retriever, cfg=make_cfg(**cfg_kw), generator_factory=_gen_factory)


def test_numeric_good_answer_no_retry():
    # 第一轮就答上(非拒答)-> 不触发重试,零额外调用(前置腿的教训)
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="净利润 122,641 千美元")])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "2015 年净利润多少?"}).json()
    assert r["status"] == "ok" and r["auto"] == [] and r["hints"] == []
    assert len(ret.calls) == 1


def test_numeric_refusal_triggers_retry_discarded_when_still_refusal():
    # 空召回 -> 第一轮拒答 -> 触发重试;重试仍拒答 -> **弃用**(保留第一轮诚实拒答),留痕 discarded
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "Netflix 2011 年净利润是多少?"}).json()
    assert r["auto"] == ["table_leg_retry_discarded"]
    assert len(ret.calls) == 3                       # 第一轮 1 次 + 重试轮(主 1 + 腿 1)
    leg = ret.calls[2]
    assert leg["kind"] == "table" and leg["top_k"] == 5 and leg["rerank"] is True


def test_retry_adopted_when_fully_answered():
    # 第一轮空召回拒答;重试轮检索有结果 -> 完整答出(非拒答)-> 采用重试答案
    state = {"n": 0}

    def factory():
        state["n"] += 1
        return [] if state["n"] == 1 else [make_res(make_hit(), ctx_text="Net income 226,126 thousand")]

    ret = FakeRetriever(results_factory=factory)
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "Netflix 2011 年净利润是多少?"}).json()
    assert r["auto"] == ["table_leg_retry"] and r["status"] == "ok"
    assert "[cite:1]" in r["answer"] and r["hints"] == []          # 完整答出,无 hints


def test_non_numeric_refusal_no_retry():
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "这篇论文的核心方法思想是什么"}).json()
    assert r["auto"] == [] and len(ret.calls) == 1


def test_explicit_kind_respected_no_retry():
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "2015 年净利润多少?", "kind": "text"}).json()
    assert r["auto"] == [] and len(ret.calls) == 1


def test_smart_off_pure_mode():
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret, smart_ask=False)) as c:
        r = c.post("/v1/ask", json={"query": "2015 年净利润多少?"}).json()
    assert r["auto"] == [] and r["hints"] == [] and len(ret.calls) == 1


def test_refusal_hints_present_and_no_duplicate_suggestion():
    # 重试后仍拒答 -> hints 出现;已自动做过表格腿 -> 不再建议 kind=table;中文 -> 语言提示
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "Netflix 2011 年净利润是多少?"}).json()
    assert r["auto"] == ["table_leg_retry_discarded"] and r["hints"]
    assert not any("kind" in h and "table" in h for h in r["hints"])
    assert any("英文" in h or "文档语言" in h for h in r["hints"])


def test_is_refusal_patterns():
    assert smart.is_refusal("对于 2011 年,我没有足够的信息。")
    assert smart.is_refusal("I don't have enough information to answer.")
    assert smart.is_refusal("上下文未提供该数据,无法回答。")
    assert smart.is_refusal("没有找到 Netflix 2011 年和 2012 年的净利润数据。")
    assert not smart.is_refusal("2015 年净利润为 122,641 千美元 [cite:1]。")
