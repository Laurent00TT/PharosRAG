"""embedder 层语义异常(纯定义,无第三方依赖)。

**为什么单独一个文件**:`pharos.toolcore` 是 stdlib-only 的工具语义层(见 toolcore.py 顶部约定:无 GPU 环境
的薄适配器也要能 import 它),不能 import embedder(会连带触发 embedder/__init__ 拉起 jieba/dense/qdrant)。
所以本层异常用一个 **marker 类属性** 让 toolcore 用 duck-typing(getattr)识别,而非按类型 import ——
既能把"推理不可用"从通用 backend_unavailable 里分流,又不给 toolcore 引入 embedder 依赖。
"""
from __future__ import annotations


class InferenceUnavailable(RuntimeError):
    """远程推理服务不可用 —— 客户端重试耗尽(503 预热中 / 连接失败 / 读超时)后抛出。

    上层据此返回**可重试**的结构化错误(`inference_unavailable`),区别于:
      - 4xx 客户端错(remote.py 立即抛 httpx.HTTPStatusError,不重试、不转本异常);
      - 其它后端故障(Qdrant/sidecar 等,仍归通用 backend_unavailable)。

    `inference_unavailable = True` 是给 stdlib-only 的 toolcore 用的 duck-typing marker:
    `getattr(e, "inference_unavailable", False)` 为真即分流,无需 import 本模块。
    """
    inference_unavailable = True
