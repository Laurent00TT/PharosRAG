"""ACL 客户端逻辑单测(纯 CPU)。重点:空 tenant fail-closed(seal#1)+ admits 与 split 同语义 + RESTRICTED 默认拒绝。"""
import os
import sys


from embedder.acl import acl_admits, acl_split
from embedder.types import User


def test_empty_tenant_failclosed():
    # seal#1:无 tenant 的 public 文档无法安全归属租户 -> acl_split 标 unset=True(默认拒绝)
    assert acl_split({"visibility": "public", "allow": []})["acl_unset"] is True
    # 空 tenant 文档对空 tenant 用户也拒绝(杜绝空串自匹配的 fail-open)
    assert acl_admits({"visibility": "public"}, User(tenant="", principals=[])) is False
    # 空 tenant 文档对真实租户用户也拒绝
    assert acl_admits({"visibility": "public"}, User(tenant="t1", principals=[])) is False


def test_restricted_acl_denied():
    RESTRICTED = {"visibility": "restricted", "allow": [], "unset": True}   # chunker RESTRICTED_ACL
    assert acl_split(RESTRICTED)["acl_unset"] is True
    assert acl_admits(RESTRICTED, User(tenant="t1", principals=["g"])) is False
    assert acl_admits({}, User(tenant="t1", principals=["g"])) is False     # 空 dict 拒绝


def test_normal_visibility():
    pub = {"tenant": "t1", "visibility": "public", "allow": []}
    assert acl_admits(pub, User(tenant="t1", principals=["x"])) is True     # 同租户 public
    assert acl_admits(pub, User(tenant="t2", principals=["x"])) is False    # 跨租户 public 仍拒绝
    r = {"tenant": "t1", "visibility": "restricted", "allow": ["g_hr"]}
    assert acl_admits(r, User(tenant="t1", principals=["g_hr"])) is True    # 有权
    assert acl_admits(r, User(tenant="t1", principals=["g_x"])) is False    # 同租户无组
    assert acl_admits(r, User(tenant="t2", principals=["g_hr"])) is False   # 跨租户


def test_split_admits_same_semantics():
    # acl_admits 必须与 acl_split(进而 store.acl_filter)同语义,否则 small-to-big 成旁路
    r = {"tenant": "t1", "visibility": "restricted", "allow": ["g_hr"]}
    f = acl_split(r)
    u = User(tenant="t1", principals=["g_hr"])
    store_pass = (not f["acl_unset"] and f["acl_tenant"] == u.tenant
                  and (bool(set(f["acl_allow"]) & set(u.principals)) or f["acl_visibility"] == "public"))
    assert store_pass == acl_admits(r, u)


def test_principals_none_tolerated():
    # B2 review nit:principals=None 不应崩(set(None or []) 容错)。public 仍可见、restricted 拒。
    assert acl_admits({"tenant": "t1", "visibility": "public", "allow": []}, User("t1", None)) is True
    assert acl_admits({"tenant": "t1", "visibility": "restricted", "allow": ["g"]}, User("t1", None)) is False


if __name__ == "__main__":
    test_empty_tenant_failclosed()
    test_restricted_acl_denied()
    test_normal_visibility()
    test_split_admits_same_semantics()
    test_principals_none_tolerated()
    print("acl tests OK")
