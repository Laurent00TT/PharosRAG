"""团队服务面单测(D10 多身份 + D11 观测):keys 解析 fail-closed / 401 / 身份流到引擎 /
跨用户会话隔离 / stats admin 门控 / 非回环启动守卫 / 请求日志(不落 key、query 截断/可关)。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from types import SimpleNamespace

from _fakes import FakeRetriever, make_app, make_cfg, make_hit, make_res, make_user
from pharos import identity as I

ALICE = I.Identity(name="alice", tenant="demo", principals=["g_eng"], admin=True)
BOB = I.Identity(name="bob", tenant="other", principals=[])
KEYS = {"pk_alice_0123456789abcdef": ALICE, "pk_bob_0123456789abcdef": BOB}


# ---------- keys 文件解析:fail-closed ----------
def test_load_keys_validation(tmp_path):
    p = tmp_path / "keys.json"
    p.write_text(json.dumps({"keys": [{"key": "short", "name": "a", "tenant": "t"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="过短"):
        I.load_keys(str(p))
    p.write_text(json.dumps({"keys": [{"key": "x" * 20, "name": "", "tenant": "t"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="name/tenant"):
        I.load_keys(str(p))
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="JSON"):
        I.load_keys(str(p))
    with pytest.raises(SystemExit, match="不存在"):
        I.load_keys(str(tmp_path / "nope.json"))


def test_append_key_roundtrip(tmp_path):
    p = str(tmp_path / "keys.json")
    key = I.append_key(p, name="carol", tenant="demo", principals=["g_hr"], admin=False)
    loaded = I.load_keys(p)
    assert key in loaded and loaded[key].name == "carol" and loaded[key].principals == ["g_hr"]
    assert key.startswith("pk_carol_") and len(key) >= 32


# ---------- keys 模式:鉴权 + 身份流到引擎 ----------
def test_keys_mode_401_and_identity_reaches_engine():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="ctx")])
    with TestClient(make_app(retriever=ret, keys=dict(KEYS))) as c:
        assert c.post("/v1/retrieve", json={"query": "q"}).status_code == 401          # 缺 key
        assert c.post("/v1/retrieve", json={"query": "q"},
                      headers={"X-API-Key": "wrong"}).status_code == 401               # 错 key
        assert c.get("/healthz").status_code == 200                                    # healthz 豁免
        r = c.post("/v1/retrieve", json={"query": "q"},
                   headers={"X-API-Key": "pk_alice_0123456789abcdef"})
        assert r.status_code == 200 and r.json()["status"] == "ok"
        c.post("/v1/retrieve", json={"query": "q"}, headers={"X-API-Key": "pk_bob_0123456789abcdef"})
    # 身份逐请求流到引擎:alice 用 demo 租户,bob 用 other 租户(能看什么由引擎 ACL 兑现)
    assert ret.calls[0]["user_tenant"] == "demo" and ret.calls[0]["user_principals"] == ["g_eng"]
    assert ret.calls[1]["user_tenant"] == "other"


def test_keys_mode_session_isolated_across_users():
    # 两个用户伪造**相同**的 X-Pharos-Session:登记键带身份名前缀,互不可见
    def fresh():
        return [make_res(make_hit(cid="c1"), ctx_text="big", anchor=[1, 5])]
    ret = FakeRetriever(results_factory=fresh)
    with TestClient(make_app(retriever=ret, keys=dict(KEYS))) as c:
        a = c.post("/v1/retrieve", json={"query": "q"},
                   headers={"X-API-Key": "pk_alice_0123456789abcdef", "X-Pharos-Session": "S"}).json()
        b = c.post("/v1/retrieve", json={"query": "q"},
                   headers={"X-API-Key": "pk_bob_0123456789abcdef", "X-Pharos-Session": "S"}).json()
        a2 = c.post("/v1/retrieve", json={"query": "q"},
                    headers={"X-API-Key": "pk_alice_0123456789abcdef", "X-Pharos-Session": "S"}).json()
    assert a["hits"][0]["context_status"] == "full_section"
    assert b["hits"][0]["context_status"] == "full_section"       # bob 不受 alice 影响
    assert a2["hits"][0]["context_status"] == "already_returned"  # alice 自己的会话正常去重


# ---------- stats:admin 门控 ----------
def test_stats_admin_gate():
    with TestClient(make_app(keys=dict(KEYS))) as c:
        assert c.get("/v1/stats", headers={"X-API-Key": "pk_bob_0123456789abcdef"}).status_code == 403
        r = c.get("/v1/stats", headers={"X-API-Key": "pk_alice_0123456789abcdef"})
        assert r.status_code == 200 and r.json()["identity_mode"] == "keys"


def test_stats_open_mode_accessible_and_counts():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="x")])
    with TestClient(make_app(retriever=ret)) as c:
        c.post("/v1/retrieve", json={"query": "q"})
        snap = c.get("/v1/stats").json()
    assert snap["status"] == "ok" and snap["endpoints"]["/v1/retrieve"]["n"] == 1
    assert snap["endpoints"]["/v1/retrieve"]["p50_ms"] is not None


# ---------- 非回环启动守卫 ----------
def test_non_loopback_requires_keys():
    with pytest.raises(SystemExit, match="非回环"):
        make_app(cfg=make_cfg(host="0.0.0.0"))
    make_app(cfg=make_cfg(host="0.0.0.0"), keys=dict(KEYS))       # keys 模式放行,不抛


# ---------- 请求日志:身份名而非 key;query 截断;可关 ----------
def test_request_log_written_no_key_material(tmp_path):
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="x")])
    cfg = make_cfg(log_dir=str(tmp_path))
    app = make_app(retriever=ret, cfg=cfg, keys=dict(KEYS))
    with TestClient(app) as c:
        c.post("/v1/retrieve", json={"query": "秘密问题" * 40},
               headers={"X-API-Key": "pk_alice_0123456789abcdef"})
    app.state.reqlog.flush()                                       # 落盘走后台写线程(阶段F 审查),读前等它排干
    lines = open(os.path.join(str(tmp_path), "requests.jsonl"), encoding="utf-8").read().strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["user"] == "alice" and rec["ep"] == "/v1/retrieve" and rec["http"] == 200
    assert "pk_alice" not in json.dumps(rec)                       # key 本体绝不落盘
    assert len(rec["query"]) <= 120                                # 截断


def test_request_log_queries_off(tmp_path):
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="x")])
    cfg = make_cfg(log_dir=str(tmp_path), log_queries=False)
    app = make_app(retriever=ret, cfg=cfg)
    with TestClient(app) as c:
        c.post("/v1/retrieve", json={"query": "不该出现的问题文本"})
    app.state.reqlog.flush()                                       # 同上:等后台写线程落盘再读
    rec = json.loads(open(os.path.join(str(tmp_path), "requests.jsonl"), encoding="utf-8").readline())
    assert "query" not in rec                                      # 隐私边界在入队前执行(log_queries=off 删 query),异步不影响


# ---------- 评审修:name 唯一/无 '|';keys new 不裸抛;观测崩溃安全 + 结构化 errors 计入 ----------
def test_load_keys_rejects_dup_name_and_pipe(tmp_path):
    p = tmp_path / "keys.json"
    p.write_text(json.dumps({"keys": [
        {"key": "x" * 20, "name": "同名", "tenant": "t1"},
        {"key": "y" * 20, "name": "同名", "tenant": "t2"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="重复"):
        I.load_keys(str(p))
    p.write_text(json.dumps({"keys": [{"key": "x" * 20, "name": "a|b", "tenant": "t"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="'\\|'"):
        I.load_keys(str(p))


def test_append_key_rejects_dup_name_and_corrupt(tmp_path):
    p = str(tmp_path / "keys.json")
    I.append_key(p, name="alice", tenant="demo", principals=[])
    with pytest.raises(SystemExit, match="已存在"):
        I.append_key(p, name="alice", tenant="demo", principals=[])
    corrupt = str(tmp_path / "bad.json")
    open(corrupt, "w").write("{not json")
    with pytest.raises(SystemExit, match="JSON"):        # 复用 load_keys 校验,不裸抛 traceback
        I.append_key(corrupt, name="bob", tenant="demo", principals=[])


def test_structured_failure_counted_in_errors():
    # no_identity(HTTP 200 + status=no_identity)必须计入 stats.errors(此前只看 http>=400 漏计)
    with TestClient(make_app(user=make_user(tenant=""), cfg=make_cfg(tenant=""))) as c:
        c.post("/v1/retrieve", json={"query": "q"})
        snap = c.get("/v1/stats").json()
    assert snap["endpoints"]["/v1/retrieve"]["n"] == 1 and snap["endpoints"]["/v1/retrieve"]["errors"] == 1


def test_observe_records_on_handler_crash(tmp_path):
    # 处理器崩溃(未捕获异常传到中间件)仍记录日志与计数(try/finally),不在观测里隐形。
    # 用 list_documents 路径:toolcore._list_impl 无 try/except,store 抛会真传播(retrieve 会被吞)。
    def boom(user):
        raise RuntimeError("store boom")
    ret = FakeRetriever()
    ret.store = SimpleNamespace(list_documents=boom)
    cfg = make_cfg(log_dir=str(tmp_path))
    client = TestClient(make_app(retriever=ret, cfg=cfg), raise_server_exceptions=False)
    with client as c:
        assert c.get("/v1/documents").status_code == 500
        snap = c.get("/v1/stats").json()
    assert snap["endpoints"]["/v1/documents"]["errors"] == 1
    recs = [json.loads(l) for l in open(os.path.join(str(tmp_path), "requests.jsonl"), encoding="utf-8")]
    assert any(r.get("crashed") and r["http"] == 500 and r["ep"] == "/v1/documents" for r in recs)
