"""10 深度优化版（解耦后的 fla GLA chunk kernel）正确性测试。

两层校验：
  1. 忠实性 test_faithful_vs_fla：本地解耦 kernel 与 fla 原版同输入近 bitwise 一致
     （证明"拷贝 + 仅改 import 指向薄适配层"没改任何计算）；需装 fla 作对照（深度优化版本身不依赖 fla）。
  2. 正确性 test_chunk_vs_recurrent：本地 GLA chunk kernel vs 简要版 recurrent ground truth（fwd），
     外加 fwd+bwd 跑通。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close  # noqa: E402
from gla_triton import gla_chunk  # noqa: E402
from _fla_gla_chunk import chunk_gla as local_chunk_gla  # noqa: E402
from linear import gla_recurrent  # noqa: E402


def _mk(B, H, T, K, V, *, seed=0, dt=torch.bfloat16, grad=False):
    """fla layout [B,T,H,D]。"""
    g = torch.Generator("cuda").manual_seed(seed)
    q = torch.randn(B, T, H, K, device="cuda", dtype=dt, generator=g)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dt, generator=g)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dt, generator=g)
    gate = F.logsigmoid(torch.randn(B, T, H, K, device="cuda", dtype=torch.float32, generator=g))
    if grad:
        return [x.requires_grad_(True) for x in (q, k, v, gate)]
    return q, k, v, gate


def test_faithful_vs_fla():
    """解耦后的本地 kernel 与 fla 原版近 bitwise 一致。"""
    fla_gla = pytest.importorskip("fla.ops.gla")
    q, k, v, g = _mk(2, 4, 512, 64, 64, seed=1)
    o_local, _ = local_chunk_gla(q, k, v, g)
    o_fla, _ = fla_gla.chunk_gla(q, k, v, g)
    assert_close(o_local, o_fla, name="local==fla", atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("T", [256, 512])
def test_chunk_vs_recurrent(T):
    """本地 GLA chunk kernel vs 简要版 recurrent ground truth（[B,H,T,D] layout）。"""
    q, k, v, g = _mk(2, 4, T, 64, 64, seed=2)
    qr, kr, vr, gr = (x.transpose(1, 2) for x in (q, k, v, g))   # [B,H,T,D]
    o_tri = gla_chunk(qr, kr, vr, gr)
    o_ref = gla_recurrent(qr, kr, vr, gr)
    # bf16 chunk kernel vs fp32 recurrent，门控+长序列累积，容差放宽
    assert_close(o_tri, o_ref, name=f"chunk==recurrent (T={T})", atol=0.2, rtol=0.1)


def test_fwd_bwd():
    """fwd+bwd 跑通、梯度非空。"""
    q, k, v, g = _mk(2, 4, 256, 64, 64, seed=3, grad=True)
    o, _ = local_chunk_gla(q, k, v, g)
    o.sum().backward()
    assert all(x.grad is not None for x in (q, k, v, g))


def test_varlen_vs_fla():
    """变长（cu_seqlens / sequence packing）：本地解耦 kernel 与 fla 原版近 bitwise 一致。

    3 条变长序列拼成一个 batch（batch=1, T=总长），cu_seqlens 标记边界；kernel 据此分块、
    不跨序列 attend。验证解耦的 varlen 索引（prepare_chunk_indices 等）也忠实。
    """
    fla_gla = pytest.importorskip("fla.ops.gla")
    cu = torch.tensor([0, 128, 328, 512], device="cuda", dtype=torch.int32)  # 长度 128/200/184
    T, H, K, V = 512, 4, 64, 64
    gen = torch.Generator("cuda").manual_seed(5)
    q = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(1, T, H, V, device="cuda", dtype=torch.bfloat16, generator=gen)
    g = F.logsigmoid(torch.randn(1, T, H, K, device="cuda", dtype=torch.float32, generator=gen))
    o_local, _ = local_chunk_gla(q, k, v, g, cu_seqlens=cu)
    o_fla, _ = fla_gla.chunk_gla(q, k, v, g, cu_seqlens=cu)
    assert_close(o_local, o_fla, name="varlen local==fla", atol=1e-3, rtol=1e-3)
