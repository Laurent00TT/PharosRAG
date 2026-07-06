"""M1 锁下沉的并发回归测试(纯 CPU,不碰真 GPU/Qdrant/HTTP)。经对抗审查修订(修 test 假绿):
 ① remote encode 退避在 **search_with_context**(旧 LockedRetriever 真正串行的方法面)之外 -> 不阻塞其他查询;
 ②/②b Store._lock 串行 Qdrant 单 client 的**读(hybrid_search)与写(upsert)**路径(峰值==1);
 ③/③b Dense/Reranker._fwd_lock 串行 local GPU 前向(峰值==1);
 ⑦ encode_query 双段:锁外 encode 不阻塞其他 key 的读段(cache 锁的真实价值——CPython GIL 下单 dict 操作已原子,
    不存在 corruption 可测,见 scratchpad/diag_sensitivity.py 反证1;本锁额外保双段一致 + 面向 free-threading);
 ⑤/⑨ Dense/Reranker._load single-flight;⑥ RemoteDense/RemoteReranker 前向/加载锁是 nullcontext。
并发峰值探针用 threading.Barrier 强制重叠 -> 删锁必然 peak==N(确定性红),非靠 sleep 概率。"""
import threading
import time
import types
from types import SimpleNamespace

import numpy as np

from embedder.config import EmbedConfig
from embedder.dense import Dense
from embedder.types import User


class _OverlapProbe:
    """并发峰值探针。Barrier(n) 强制 n 个线程在入口集合:无锁 -> n 个同时进 -> peak==n;
    有锁 -> 只 1 个能进、barrier 凑不齐超时 -> peak==1。删锁必然红(确定性,非靠 sleep 概率)。"""
    def __init__(self, n: int, result, hold: float = 0.05, timeout: float = 0.4):
        self.n = n
        self._result = result
        self.hold = hold
        self.timeout = timeout
        self.peak = self.cur = 0
        self._lk = threading.Lock()
        self._barrier = threading.Barrier(n)

    def __call__(self, *a, **k):
        try:
            self._barrier.wait(timeout=self.timeout)   # 无锁:n 线程集合;有锁串行:凑不齐 -> BrokenBarrier
        except threading.BrokenBarrierError:
            pass
        with self._lk:
            self.cur += 1
            self.peak = max(self.peak, self.cur)
        time.sleep(self.hold)
        with self._lk:
            self.cur -= 1
        return self._result()


# ---------- ① M1 核心:remote encode 退避不阻塞其他查询(走 search_with_context —— 旧大锁真正串行的方法面)----------
def test_remote_encode_backoff_not_block_others():
    from embedder.retrieve import Retriever
    started, release = threading.Event(), threading.Event()

    class SlowRemoteDense:                       # 模拟 RemoteDense:第一个查询的 encode 卡在退避(HTTP,锁外)
        def __init__(self):
            self.n = 0
            self._lk = threading.Lock()

        def encode_query(self, q):
            with self._lk:
                self.n += 1
                first = self.n == 1
            if first:
                started.set()
                release.wait(3)                  # 卡住(模拟 remote 退避 sleep)
            return SimpleNamespace(tolist=lambda: [0.0] * 8)

    r = Retriever(EmbedConfig(inference_url="http://x"),
                  store=SimpleNamespace(hybrid_search=lambda *a, **k: []),
                  dense=SlowRemoteDense(), reranker=SimpleNamespace())
    fast_done = threading.Event()

    def slow():
        r.search_with_context("q1", User("t", []), assemble=False)   # 旧 LockedRetriever 串行的正是此方法面

    def fast():
        started.wait(2)
        r.search_with_context("q2", User("t", []), assemble=False)   # 修复后:不被 slow 退避阻塞
        fast_done.set()

    ts, tf = threading.Thread(target=slow), threading.Thread(target=fast)
    ts.start(); tf.start()
    try:
        assert fast_done.wait(2), "fast 的 search_with_context 被 slow 的 encode 退避阻塞(M1:大锁串行整个方法面)"
    finally:
        release.set(); ts.join(3); tf.join(1)


# ---------- ②/②b Store._lock 串行读(hybrid_search)与写(upsert)路径 ----------
def _fresh_store():
    from embedder.store import Store
    s = Store.__new__(Store)                     # 不走 __init__(不开真 Qdrant)
    s.cfg = EmbedConfig()
    s._lock = threading.Lock()
    return s


def test_store_lock_serializes_hybrid_search():
    s = _fresh_store()
    probe = _OverlapProbe(4, lambda: SimpleNamespace(points=[]))
    s.client = SimpleNamespace(query_points=probe)
    ts = [threading.Thread(target=lambda: s.hybrid_search([0.0] * 8, None, User("t", []), strategy="dense")) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"Store 锁未串行 hybrid_search,峰值={probe.peak}"


def test_store_lock_serializes_upsert():        # 写路径(最危险:np.append 重绑数组),审查缺口④
    s = _fresh_store()
    probe = _OverlapProbe(4, lambda: None)
    s.client = SimpleNamespace(upsert=probe)
    ts = [threading.Thread(target=lambda: s.upsert([])) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"Store 锁未串行 upsert(写路径!),峰值={probe.peak}"


# ---------- ③/③b Dense/Reranker._fwd_lock 串行 GPU 前向 ----------
def test_dense_fwd_lock_serializes():
    import torch
    d = Dense(EmbedConfig(dense_dim=1024))
    probe = _OverlapProbe(4, lambda: torch.zeros(1, 4096))   # _mrl 截到 1024 + renorm(真 torch CPU)
    d._model = SimpleNamespace(process=probe)                # 已设 _model -> _load 快返回,不碰真 GPU
    ts = [threading.Thread(target=lambda: d.encode_text(["x"])) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"Dense._fwd_lock 未串行 GPU 前向,峰值={probe.peak}"


def test_reranker_fwd_lock_serializes():        # 审查缺口③:rerank 是独立加的锁,单独测
    from embedder.rerank import Reranker
    rr = Reranker(EmbedConfig())
    probe = _OverlapProbe(4, lambda: [0.5])     # score 契约:len(scores)==len(docs)==1
    rr._model = SimpleNamespace(process=probe)
    ts = [threading.Thread(target=lambda: rr.score("q", ["d"])) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert probe.peak == 1, f"Reranker._fwd_lock 未串行 GPU 前向,峰值={probe.peak}"


# ---------- ⑦ encode_query 双段:锁外 encode 不阻塞其他 key 的读段(cache 锁的真实守门,M1 dense 层核心不变量)----------
# (原 test④"cache 并发 corrupt"已删:CPython GIL 下单 dict 操作原子,删锁也不 corrupt——反证证明是假绿。)
def test_cache_read_not_blocked_by_slow_encode():
    d = Dense(EmbedConfig())
    d.encode_text = lambda texts, instruction=None: np.zeros((1, 8), dtype=np.float32)
    d.encode_query("cached")                     # 预热:存入缓存
    slow_in, release = threading.Event(), threading.Event()

    def slow_encode(texts, instruction=None):    # 换成慢 encode(模拟退避,在 _cache_lock 外)
        slow_in.set(); release.wait(3)
        return np.zeros((1, 8), dtype=np.float32)
    d.encode_text = slow_encode
    fast_done = threading.Event()

    def slow():
        d.encode_query("newkey")                 # miss -> 读段(锁)释放 -> 锁外 encode_text 卡住
    def fast():
        slow_in.wait(2)
        d.encode_query("cached")                 # hit -> 读段(锁):若双段生效,slow 未持锁 -> 立即返回
        fast_done.set()

    ts, tf = threading.Thread(target=slow), threading.Thread(target=fast)
    ts.start(); tf.start()
    try:
        assert fast_done.wait(2), "已缓存 key 的读被慢 encode 阻塞(encode 在 _cache_lock 内 = 双段没生效)"
    finally:
        release.set(); ts.join(3); tf.join(1)


# ---------- ⑤/⑨ _load single-flight ----------
def _load_single_flight(model_cls_setter, load_fn):
    count = {"n": 0}

    def build(**k):
        count["n"] += 1
        time.sleep(0.05)                         # 慢加载,放大并发窗口
        return object()
    mod_name, mod = model_cls_setter(build)
    import sys
    sys.modules[mod_name] = mod
    try:
        ts = [threading.Thread(target=load_fn) for _ in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        return count["n"]
    finally:
        sys.modules.pop(mod_name, None)


def test_dense_load_single_flight():
    d = Dense(EmbedConfig())
    d._assert_gpu = lambda: None

    def setter(build):
        m = types.ModuleType("qwen3_vl_embedding")
        m.Qwen3VLEmbedder = build
        return "qwen3_vl_embedding", m
    assert _load_single_flight(setter, d._load) == 1, "Dense._load 未 single-flight(并发首查重复 load 8B)"


def test_reranker_load_single_flight():         # 审查缺口③:reranker _load 也 double-check
    from embedder.rerank import Reranker
    rr = Reranker(EmbedConfig())
    rr._assert_gpu = lambda: None

    def setter(build):
        m = types.ModuleType("qwen3_vl_reranker")
        m.Qwen3VLReranker = build
        return "qwen3_vl_reranker", m
    assert _load_single_flight(setter, rr._load) == 1, "Reranker._load 未 single-flight"


# ---------- ⑥ remote 前向/加载锁是 nullcontext ----------
def test_remote_forward_locks_are_nullcontext():
    import contextlib

    from embedder.remote import RemoteDense, RemoteReranker
    rd = RemoteDense(EmbedConfig(inference_url="http://x"))
    assert isinstance(rd._fwd_lock, contextlib.nullcontext)
    assert isinstance(rd._load_lock, contextlib.nullcontext)
    assert not isinstance(rd._cache_lock, contextlib.nullcontext)   # cache 锁仍是真锁(继承 Dense,remote 也走 LRU)
    rr = RemoteReranker(EmbedConfig(inference_url="http://x"))
    assert isinstance(rr._fwd_lock, contextlib.nullcontext)
    assert isinstance(rr._load_lock, contextlib.nullcontext)
