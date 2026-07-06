"""GPU 等价性测试(P1-2:local↔remote 检索向量数学等价的根据)。需真 GPU + Qwen3-VL 模型,CI 无 GPU 自动 skip。
手动跑:conda activate navikb && pytest tests/engine/test_equivalence_gpu.py -v

**这里只覆盖截维实现的一致性(torch vs numpy),不是完整 local↔remote 等价**(阶段B审查 M1 澄清):
- `_mrl`(torch)与 `_mrl_np`(numpy)对同一份 fp32 全维截维,验证两实现逐元素一致;
- **bf16→fp32 环节**(真实建库入口:model.process 出 bf16 → _mrl.float())的守护在 test_remote.py(纯 CPU);
- **端到端**(起服务、bf16→JSON→fp32 传输、混建混查 E2)见 docs/SCALE_OUT.md §5-B 的 scripts/equiv_gpu.py 分时脚本
  (实测 encode cosine=1.0000000 / maxdiff 2.98e-08 / rerank maxdiff 0.00)。"""
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


def test_mrl_torch_numpy_consistent_on_fp32_output():
    """真模型输出(经 encode_text 已 .float() 成 fp32)上,Dense._mrl(torch) 与 RemoteDense._mrl_np(numpy) 逐元素一致。
    ⚠ 诚实边界(阶段B审查 M1):本测试从**同一份 fp32** 出发,只验"fp32 截维 torch≈numpy",**不含 bf16→fp32 环节**。
    - 真实建库入口(bf16 张量进 _mrl)的守护 -> test_remote.py::test_mrl_fp32_normalize_on_bf16_input(纯 CPU);
    - 端到端 local↔remote(含 bf16→JSON→fp32 传输)的实测 -> scripts/equiv_gpu.py 分时(cosine=1.0/maxdiff 2.98e-8,SCALE_OUT §5-B)。"""
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
