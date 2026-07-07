"""可观测性(DESIGN D11):请求日志(JSONL)+ 进程内指标。够用为度,不引监控栈。

隐私与安全边界:日志**绝不落盘 key 本体**(只记身份 name);query 默认记录(截断 120 字,
内网调试价值优先),PHAROS_LOG_QUERIES=off 可关。观测失败绝不影响服务(写日志异常只计数)。
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict, deque

log = logging.getLogger("pharos")


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
    """JSONL 追加日志。dir 为空 = 关闭。单文件(团队规模);滚动交给 logrotate(见 OPERATIONS)。

    落盘走**后台单写线程 + 有界队列**(阶段F 审查):`write` 从 async 中间件 `_observe` 里调用,若在事件
    循环线程内做同步 open/append/close,共享 bind-mount 卷一旦 IO 卡顿会冻结整个事件循环 → 副本上所有端点
    (含健康探针分发)同时停摆。改为 `write` 只做无阻塞入队(队满即丢弃并计数,保持"观测绝不拖垮服务"语义),
    真正落盘在独立 daemon 线程 —— 事件循环永不因磁盘阻塞。"""

    def __init__(self, log_dir: str, log_queries: bool = True, queue_max: int = 4096):
        self.enabled = bool(log_dir)
        self.log_queries = log_queries
        self.path = ""
        self._q: queue.Queue | None = None
        self._writer: threading.Thread | None = None
        if self.enabled:
            d = os.path.expanduser(log_dir)
            os.makedirs(d, exist_ok=True)
            self.path = os.path.join(d, "requests.jsonl")
            self._q = queue.Queue(maxsize=queue_max)
            self._writer = threading.Thread(target=self._drain, name="reqlog-writer", daemon=True)
            self._writer.start()

    def _drain(self) -> None:
        """后台单写线程:串行落盘(单线程天然串行,无需锁)。stats 只用于计写失败数。"""
        while True:
            rec, stats = self._q.get()
            try:
                line = json.dumps(rec, ensure_ascii=False)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                if stats is not None:
                    stats.log_write_failures += 1
            finally:
                self._q.task_done()

    def flush(self, timeout: float | None = None) -> None:
        """等待队列中已入队记录全部落盘(测试/优雅停机用;生产热路径不调)。无队列(关闭)则 no-op。
        timeout=None 无限等(测试同步用法);给定时带 deadline 等待 —— stdlib queue.join() 不支持超时,
        写线程卡在 open/write(正是本类注释承认的 bind-mount IO 挂起场景)时无限 join 会把优雅停机
        冻死到 stop_grace_period 被 SIGKILL。超时则 warning 后放弃返回(挂死的盘那批日志本来也写不出去)。"""
        if self._q is None:
            return
        if timeout is None:
            self._q.join()
            return
        deadline = time.monotonic() + timeout
        with self._q.all_tasks_done:              # queue.join() 的同款条件变量,只是 wait 带剩余 deadline
            while self._q.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("reqlog flush 超时(%.1fs)放弃,%d 条记录未落盘",
                                timeout, self._q.unfinished_tasks)
                    return
                self._q.all_tasks_done.wait(remaining)

    def write(self, rec: dict, stats: Stats | None = None) -> None:
        if not self.enabled:
            return
        # 隐私边界收在本层(评审:截断此前只在 _observe 做,任何新写入方都会漏):
        # log_queries=off 删除 query;否则统一截断 120 字。入队方=隐私边界唯一执行点(在入队前做,不占写线程)。
        if not self.log_queries:
            rec.pop("query", None)
        elif rec.get("query") is not None:
            rec["query"] = str(rec["query"])[:120]
        try:
            self._q.put_nowait((rec, stats))         # 无阻塞入队:队满立即抛 Full,绝不阻塞调用方(事件循环)
        except queue.Full:
            if stats is not None:                     # 观测失败不影响服务,只计数供 /v1/stats 暴露(丢弃最新记录)
                stats.log_write_failures += 1
