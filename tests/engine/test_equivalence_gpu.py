"""GPU 等价性测试(P1-2:local↔remote 检索向量数学等价的根据)。需真 GPU + Qwen3-VL 模型,CI 无 GPU 自动 skip。
手动跑:conda activate navikb && pytest tests/engine/test_equivalence_gpu.py -v

**为什么只需 local 模型、不起 inference 服务**:local 与 remote 用同一个 Qwen3-VL 权重,全维输出必然一致;
唯一差异在**截维路径**——local 走 Dense._mrl(torch,fp32),remote 走 RemoteDense._mrl_np(numpy,fp32)。
本测试对同一份真模型 bf16 全维输出跑两条截维路径,证明逐元素等价 => local 建库向量 == remote 查询向量。
端到端(起服务、混建混查 E2)见 docs/SCALE_OUT.md §5-B 的手动脚本(分时避 OOM:remote 存档→停服务→local 对比,
实测 encode cosine=1.0000000 / maxdiff 2.98e-08 / rerank maxdiff 0.00)。"""
import os

import numpy as np
import pytest

from embedder.config import EmbedConfig


def _gpu_ready() -> bool:
    try:
        import torch
        return (torch.cuda.is_available()
                and os.path.isdir(os.path.expanduser("~/models/Qwen3-VL-Embedding-8B/scripts")))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="需真 GPU + Qwen3-VL 模型(navikb 环境);CI 跳过")


def test_mrl_paths_equivalent_on_real_bf16_output():
    """真模型 bf16 全维输出上,Dense._mrl(torch,fp32) 与 RemoteDense._mrl_np(numpy,fp32) 逐元素等价(P1-2)。
    这是"同库 local 建、remote 查不错位"的数学根据。也守 fp32 修复:两路 norm 都 = 1(bf16 normalize 会 ≈1.002)。"""
    import torch

    from embedder.dense import Dense
    from embedder.remote import RemoteDense

    D = 1024
    texts = ["Netflix 2015 年营收是多少", "The company reported strong quarterly growth.",
             "混合 中英文 test 123 !@#$%", "a"]
    # 拿真模型全维(dense_dim=极大 -> _mrl 不截,返回 bf16->fp32 全维)。只加载一次模型,不 OOM。
    full = Dense(EmbedConfig(dense_dim=10 ** 9)).encode_text(texts)          # (n, 4096) fp32
    assert full.shape[0] == len(texts) and full.shape[1] > D

    v_torch = Dense(EmbedConfig(dense_dim=D))._mrl(torch.from_numpy(full))   # local 截维(torch);此 Dense 不加载模型
    rd = RemoteDense.__new__(RemoteDense)                                    # 不 __init__(不建 httpx client)
    rd.cfg = EmbedConfig(dense_dim=D)
    v_numpy = rd._mrl_np(full)                                              # remote 截维(numpy)

    assert v_torch.shape == v_numpy.shape == (len(texts), D)
    assert np.allclose(v_torch, v_numpy, atol=1e-6), \
        f"torch/numpy 截维路径不等价,maxdiff={np.abs(v_torch - v_numpy).max():.2e}"
    # fp32 normalize -> norm 都 = 1(若 _mrl 回退到 bf16 normalize,local norm≈1.002,本断言会红)
    assert np.allclose(np.linalg.norm(v_torch, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(v_numpy, axis=-1), 1.0, atol=1e-5)


def test_mrl_np_lower_bound_assertion():
    """P1-1:客户端 dense_dim > 服务端全维 -> fail-loud(不静默返回过短向量)。纯 CPU,顺带在此文件归档。"""
    from embedder.remote import RemoteDense
    rd = RemoteDense.__new__(RemoteDense)
    rd.cfg = EmbedConfig(dense_dim=8192)                      # > 4096 全维
    with pytest.raises(RuntimeError, match="配置错位"):
        rd._mrl_np(np.zeros((2, 4096), dtype=np.float32))
