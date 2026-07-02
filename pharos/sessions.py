"""per-session 已交付登记(returned_keys)。

引擎 stdio server 里"进程=会话",一个进程级 set 即可;Pharos 守护进程是多会话共享的,
必须按 X-Pharos-Session 隔离 —— 否则 A 会话取过的段落,B 会话会被误标 already_returned
(正文被清空,B 实际从没收到过)。这正是引擎 server.py B5.A 注释里预告的坑,在此兑现。

语义:**去重是 opt-in** —— 请求不带 X-Pharos-Session 头则不做跨调用去重(returned_keys=None),
适合 curl/脚本一次性调用;MCP 适配器每进程生成一个 uuid,天然获得会话级去重。
"""
from __future__ import annotations

import threading
from collections import OrderedDict


class SessionRegistry:
    """有界 LRU:最多 max_sessions 个会话,每会话一个 returned_keys set(set 内上限由 toolcore 兜底)。"""

    def __init__(self, max_sessions: int = 64):
        self._max = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, set] = OrderedDict()

    def get(self, session_id: str) -> set:
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
            else:
                self._sessions[session_id] = set()
                while len(self._sessions) > self._max:
                    self._sessions.popitem(last=False)   # 逐出最久未用会话(丢去重不影响正确性)
            return self._sessions[session_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
