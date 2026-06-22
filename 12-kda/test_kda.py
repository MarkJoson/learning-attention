"""12 KDA（Kimi Delta Attention）正确性测试。

  1. 忠实性：本地解耦 kernel ≡ fla 原版（定长 + 变长，近 bitwise）；
  2. kernel ≡ 简要版 gated-delta-rule recurrent（ground truth）；fwd+bwd。
KDA 需 use_qk_l2norm_in_kernel=True（delta rule 标配）。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close  # noqa: E402
from kda import kda_recurrent  # noqa: E402
from _fla_kda_chunk import chunk_kda as local_chunk_kda  # noqa: E402


def _mk(B, H, T, D, *, seed=0, dt=torch.bfloat16, grad=False):
    """fla layout [B,T,H,D]，g log-space decay，beta [B,T,H]。"""
    gen = torch.Generator("cuda").manual_seed(seed)
    q = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    k = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    v = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    g = F.logsigmoid(torch.randn(B, T, H, D, device="cuda", dtype=torch.float32, generator=gen))
    beta = torch.rand(B, T, H, device="cuda", dtype=dt, generator=gen)
    if grad:
        return [x.requires_grad_(True) for x in (q, k, v, g, beta)]
    return q, k, v, g, beta


def test_faithful_vs_fla():
    """本地解耦 kernel ≡ fla 原版（定长）。"""
    fla = pytest.importorskip("fla.ops.kda")
    q, k, v, g, beta = _mk(2, 4, 512, 64, seed=1)
    ol, _ = local_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    of, _ = fla.chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert_close(ol, of, name="local==fla", atol=1e-3, rtol=1e-3)


def test_kernel_vs_recurrent():
    """本地 KDA chunk kernel ≡ 简要版 gated-delta-rule recurrent ground truth。"""
    q, k, v, g, beta = _mk(2, 4, 512, 64, seed=2)        # [B,T,H,D]
    o_tri, _ = local_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    qr, kr, vr, gr = (x.transpose(1, 2) for x in (q, k, v, g))   # [B,H,T,D]
    o_ref = kda_recurrent(qr, kr, vr, gr, beta.transpose(1, 2), l2norm=True).transpose(1, 2)
    assert_close(o_tri, o_ref, name="kernel==recurrent", atol=3e-2, rtol=1e-2)


def test_varlen_vs_fla():
    """变长（cu_seqlens）：本地解耦 kernel ≡ fla 原版。"""
    fla = pytest.importorskip("fla.ops.kda")
    cu = torch.tensor([0, 128, 328, 512], device="cuda", dtype=torch.int32)
    T, H, D = 512, 4, 64
    gen = torch.Generator("cuda").manual_seed(5)
    q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    g = F.logsigmoid(torch.randn(1, T, H, D, device="cuda", dtype=torch.float32, generator=gen))
    beta = torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16, generator=gen)
    ol, _ = local_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    of, _ = fla.chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    assert_close(ol, of, name="varlen local==fla", atol=1e-3, rtol=1e-3)


def test_fwd_bwd():
    """fwd+bwd 跑通，梯度非空。"""
    q, k, v, g, beta = _mk(2, 4, 256, 64, seed=3, grad=True)
    o, _ = local_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    o.sum().backward()
    assert all(x.grad is not None for x in (q, k, v, g, beta))
