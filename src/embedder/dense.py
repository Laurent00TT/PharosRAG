"""Qwen3-VL-Embedding-8B 封装:embedder 里唯一吃 GPU 的部分。

复用模型自带的官方 `scripts/qwen3_vl_embedding.py` 的 `Qwen3VLEmbedder`(last-token pooling + 图文同空间),
不自行重写 pooling/forward。text 与 image 编码进同一 4096 空间(官方示例直接 query@doc.T 算相似度);
MRL:截前 `cfg.dense_dim` 维 + 重 L2 normalize(Qwen3-Embedding 系列标准做法,README 称支持 64–4096 自定义维度)。
按名锁定 4090——FASTEST_FIRST 下 torch 设备编号会漂移,只认卡名不认 device 0 这个编号。
"""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict

import numpy as np

from .config import EmbedConfig

_QUERY_CACHE_CAP = 256


class Dense:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self._model = None
        self._gpu_error: Exception | None = None   # B4:缓存永久性 GPU 配置错(断言失败),避免每次 retry 重跑
        self._query_cache: OrderedDict = OrderedDict()   # B5.C:query 文本 -> 向量 LRU(用户无关,ACL 在召回后才施加,安全)
        # M1 锁下沉(RemoteDense 把前向/加载锁 override 成 nullcontext -> HTTP 天然并发,退避不持锁):
        self._fwd_lock = threading.Lock()      # GPU 前向串行(仅 local;单卡 model.process 非线程安全)
        self._load_lock = threading.Lock()     # lazy load single-flight(并发首查只 load 一次 8B,独立于 _fwd_lock 防嵌套死锁)
        self._cache_lock = threading.Lock()    # query LRU:真实价值=双段结构让 encode 在锁外(退避不阻塞其他查询);
        #                                        CPython GIL 下单 dict 操作已原子(删锁也不 corrupt),本锁额外保双段一致 + 面向 free-threading(3.13+)

    def _assert_gpu(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用:Qwen3-VL dense 必须 GPU")
        name = torch.cuda.get_device_name(0)   # Qwen3VLEmbedder 固定用 torch device 0
        if self.cfg.gpu_name_must_contain not in name:
            raise RuntimeError(
                f"torch device 0 = {name!r},期望含 {self.cfg.gpu_name_must_contain!r}。"
                f"FASTEST_FIRST 编号会漂移——拒绝在错卡上跑;可用 CUDA_VISIBLE_DEVICES 锁定后重试。")

    def _load(self) -> None:
        if self._model is not None:            # 快路径:已加载免锁
            return
        with self._load_lock:                  # single-flight:并发首查只一个线程真 load(独立锁,不与 _fwd_lock 嵌套)
            if self._model is not None:        # double-check
                return
            if self._gpu_error is not None:    # B4:永久 GPU 配置错已缓存 -> 快速失败,不重试断言/加载
                raise self._gpu_error
            try:
                self._assert_gpu()
            except RuntimeError as e:          # 缺 CUDA / 错卡 = 本会话内永久,缓存(模型加载错不缓存,可能瞬态)
                self._gpu_error = e
                raise
            import torch
            scripts = os.path.join(self.cfg.dense_model_path, "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)     # 复用官方 Qwen3VLEmbedder(随模型一起下发)
            from qwen3_vl_embedding import Qwen3VLEmbedder
            self._model = Qwen3VLEmbedder(
                model_name_or_path=self.cfg.dense_model_path, torch_dtype=torch.bfloat16)

    def _mrl(self, emb) -> np.ndarray:
        """Matryoshka:截前 dense_dim 维 + 重 L2 normalize(dense_dim≥全维则不截)。"""
        import torch.nn.functional as F
        if emb.shape[-1] > self.cfg.dense_dim:
            emb = F.normalize(emb[:, :self.cfg.dense_dim], p=2, dim=-1)
        return emb.cpu().float().numpy()

    def encode_text(self, texts: list[str], instruction: str | None = None) -> np.ndarray:
        self._load()
        with self._fwd_lock:       # local GPU 前向串行(含 _mrl 的 GPU 上 normalize);RemoteDense override encode_text 走 HTTP,不经此
            emb = self._model.process([{"text": t, "instruction": instruction} for t in texts], normalize=True)
            return self._mrl(emb)

    def encode_image(self, image_paths: list[str], instruction: str | None = None) -> np.ndarray:
        """image_paths 必须是**绝对路径**(官方 format 会转 file://)。相对路径的拼接在 embed.py 完成。"""
        self._load()
        with self._fwd_lock:       # 同 encode_text:local GPU 前向串行
            emb = self._model.process([{"image": p, "instruction": instruction} for p in image_paths], normalize=True)
            return self._mrl(emb)

    def encode_query(self, query: str) -> np.ndarray:
        """查询端:加 retrieval instruction(README:query 用 task-specific 英文 instruction 提升 1–5%)。
        B5.C:LRU 缓存(agent 多跳常重发同/近似 query,免重复 8B 前向);返回的向量只读复用安全。"""
        c = self._query_cache
        with self._cache_lock:                 # 读段:只包 dict 操作
            if query in c:
                c.move_to_end(query)
                return c[query]
        # encode_text(HTTP/GPU + remote 退避)在锁外 —— 这是 M1 的关键:退避不持任何检索锁。
        # 良性竞态:两线程同 query 同时 miss 各算一次、后写覆盖,结果相同(MRL 确定性),可接受;绝不为消除它把整段包锁。
        v = self.encode_text([query], instruction=self.cfg.query_instruction)[0]
        with self._cache_lock:                 # 写段:只包 dict 操作
            c[query] = v
            c.move_to_end(query)
            if len(c) > _QUERY_CACHE_CAP:
                c.popitem(last=False)
        return v
