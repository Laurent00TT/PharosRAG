"""可观测性(DESIGN D11):请求日志(JSONL)+ 进程内指标。够用为度,不引监控栈。

隐私与安全边界:日志**绝不落盘 key 本体**(只记身份 name);query 默认记录(截断 120 字,
内网调试价值优先),PHAROS_LOG_QUERIES=off 可关。观测失败绝不影响服务(写日志异常只计数)。
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque


class Stats:
    """进程内指标:每端点计数/错误数/延迟环形队列(p50/p95)。重启归零(当前可接受)。"""

    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self._n = defaultdict(int)
        self._err = defaultdict(int)
        self._lat = defaultdict(lambda: deque(maxlen=window))
        self.started = time.time()
        self.log_write_failures = 0

    def record(self, ep: str, ms: float, error: bool) -> None:
        with self._lock:
            self._n[ep] += 1
            if error:
                self._err[ep] += 1
            self._lat[ep].append(ms)

    @staticmethod
    def _pct(sorted_vals, p):
        if not sorted_vals:
            return None
        i = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
        return round(sorted_vals[i], 1)

    def snapshot(self) -> dict:
        with self._lock:
            eps = {}
            for ep, n in sorted(self._n.items()):
                lat = sorted(self._lat[ep])
                eps[ep] = {"n": n, "errors": self._err[ep],
                           "p50_ms": self._pct(lat, 50), "p95_ms": self._pct(lat, 95),
                           "max_ms": round(lat[-1], 1) if lat else None}
            return {"uptime_s": round(time.time() - self.started, 1),
                    "log_write_failures": self.log_write_failures, "endpoints": eps}


class RequestLog:
    """JSONL 追加日志。dir 为空 = 关闭。单文件(团队规模);滚动交给 logrotate(见 OPERATIONS)。"""

    def __init__(self, log_dir: str, log_queries: bool = True):
        self.enabled = bool(log_dir)
        self.log_queries = log_queries
        self._lock = threading.Lock()
        self.path = ""
        if self.enabled:
            d = os.path.expanduser(log_dir)
            os.makedirs(d, exist_ok=True)
            self.path = os.path.join(d, "requests.jsonl")

    def write(self, rec: dict, stats: Stats | None = None) -> None:
        if not self.enabled:
            return
        # 隐私边界收在本层(评审:截断此前只在 _observe 做,任何新写入方都会漏):
        # log_queries=off 删除 query;否则统一截断 120 字。落盘方=隐私边界唯一执行点。
        if not self.log_queries:
            rec.pop("query", None)
        elif rec.get("query") is not None:
            rec["query"] = str(rec["query"])[:120]
        try:
            line = json.dumps(rec, ensure_ascii=False)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            if stats is not None:                     # 观测失败不影响服务,只计数供 /v1/stats 暴露
                stats.log_write_failures += 1
