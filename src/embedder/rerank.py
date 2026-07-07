"""Qwen3-VL-Reranker-8B 封装:cross-encoder 精排(可选第二阶段)。

复用模型自带 `scripts/qwen3_vl_reranker.py` 的 `Qwen3VLReranker`(不自写):把 (query, document) 整个喂进模型,
用 yes/no token logits + sigmoid 出 0~1 相关性分,比 bi-encoder(dense)精,但慢——只对 hybrid 召回的 top-N 重排。
吃 GPU,按名锁 4090(同 dense)。

注:image_only 纯图 chunk 的**图像** rerank 是 TODO —— 检索期只有相对 image_path、缺绝对路径,当前按 text 重排,
纯图占位 text 会被低估。评估语料 image_only=0 故不影响验证;生产若有纯图需补检索期图路径方案。
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import replace

from .config import EmbedConfig
from .types import Hit


class Reranker:
    def __init__(self, cfg: EmbedConfig | None = None):
        self.cfg = cfg or EmbedConfig()
        self._model = None
        self._gpu_error: Exception | None = None   # P3(阶段F):缓存永久性 GPU 配置错(错卡/缺 CUDA),与 Dense 对称,避免每次 retry 重跑断言
        # M1 锁下沉(RemoteReranker override 成 nullcontext -> HTTP 并发):
        self._fwd_lock = threading.Lock()      # GPU 前向串行(仅 local)
        self._load_lock = threading.Lock()     # lazy load single-flight(独立锁,防与 _fwd_lock 嵌套)

    def _assert_gpu(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用:Qwen3-VL-Reranker 必须 GPU")
        name = torch.cuda.get_device_name(0)   # Qwen3VLReranker 固定用 torch device 0
        if self.cfg.gpu_name_must_contain not in name:
            raise RuntimeError(
                f"torch device 0 = {name!r},期望含 {self.cfg.gpu_name_must_contain!r};"
                f"FASTEST_FIRST 编号会漂移——拒绝在错卡上跑。")

    def _load(self) -> None:
        if self._model is not None:            # 快路径:已加载免锁
            return
        with self._load_lock:                  # single-flight(独立锁,不与 _fwd_lock 嵌套)
            if self._model is not None:        # double-check
                return
            if self._gpu_error is not None:    # P3:永久 GPU 配置错已缓存 -> 快速失败,不重试断言/加载
                raise self._gpu_error
            try:
                self._assert_gpu()
            except RuntimeError as e:          # 缺 CUDA / 错卡 = 本会话内永久,缓存(模型加载错不缓存,可能瞬态)
                self._gpu_error = e
                raise
            import torch
            scripts = os.path.join(self.cfg.rerank_model_path, "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)    # 复用官方 Qwen3VLReranker(随模型下发)
            from qwen3_vl_reranker import Qwen3VLReranker
            self._model = Qwen3VLReranker(
                model_name_or_path=self.cfg.rerank_model_path, torch_dtype=torch.bfloat16)

    def score(self, query: str, docs_text: list[str], instruction: str | None = None) -> list[float]:
        """对 (query, 每个 doc text) 打 cross-encoder 相关性分(0~1),顺序对应 docs_text。
        instruction(D3/P2-3):None 时用服务端 cfg 默认;非 None 时用调用方传入(inference_server 据此
        消费客户端 payload 的 instruction,收敛"客户端唯一真相",消除服务端曾忽略客户端值的双源 footgun)。"""
        self._load()
        inputs = {"instruction": instruction or self.cfg.rerank_instruction, "query": {"text": query},
                  "documents": [{"text": t or ""} for t in docs_text], "fps": 1.0}
        with self._fwd_lock:                # local GPU 前向串行;RemoteReranker override score 走 HTTP,不经此
            scores = self._model.process(inputs)
        if len(scores) != len(docs_text):   # 官方 API 契约:分数与文档一一对应。破坏则 fail-loud,不静默错排/越界
            raise RuntimeError(f"reranker 返回 {len(scores)} 分 != {len(docs_text)} 文档,API 契约破坏")
        return scores

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        """对 hybrid 召回的 hits 用 cross-encoder 重排,返回前 top_k(默认全部)。空 hits 原样返回。"""
        if not hits:
            return hits
        scores = self.score(query, [h.text or "" for h in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        # Batch1.B:把 cross-encoder 分写回 score 并标 score_kind='rerank' —— 否则名次按 rerank、Hit.score 仍是 RRF,
        # 二者脱节误导 agent 的置信判断。replace 生成新 Hit(不原地改,避免共享对象副作用)。
        out = [replace(hits[i], score=float(scores[i]), score_kind="rerank") for i in order]
        return out[:top_k] if top_k is not None else out   # is not None:top_k=0 应空,不能 falsy 当全部
