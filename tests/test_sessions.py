"""SessionRegistry:同会话拿同一 set、LRU 逐出、容量有界。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pharos.sessions import SessionRegistry


def test_same_session_same_set():
    reg = SessionRegistry()
    s = reg.get("a")
    s.add(("d1", "c1"))
    assert ("d1", "c1") in reg.get("a")          # 同会话共享登记
    assert ("d1", "c1") not in reg.get("b")      # 跨会话隔离


def test_lru_eviction_bounded():
    reg = SessionRegistry(max_sessions=3)
    for i in range(5):
        reg.get(f"s{i}")
    assert len(reg) == 3
    # 最久未用的 s0/s1 被逐出;s2 之后重新 get 会得到新空 set(丢去重不影响正确性)
    reg.get("s2").add("x")
    reg.get("s0")                                # s0 重新进来,逐出下一个最老的
    assert len(reg) == 3


def test_touch_refreshes_order():
    reg = SessionRegistry(max_sessions=2)
    reg.get("a").add("ka")
    reg.get("b")
    reg.get("a")                                 # touch a -> b 变最老
    reg.get("c")                                 # 逐出 b
    assert "ka" in reg.get("a")                  # a 仍在(没被逐出、登记保留)
