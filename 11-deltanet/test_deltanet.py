"""11 DeltaNet 正确性测试。

  1. 简要版自洽：delta rule recurrent ≡ chunked（WY 表示）；
  2. 深度优化版忠实性：本地解耦 kernel ≡ fla 原版（定长 + 变长，近 bitwise）；
  3. 深度优化版 ≡ 简要版 recurrent（ground truth）；fwd+bwd。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close  # noqa: E402
from deltanet import delta_rule_recurrent, delta_rule_chunked  # noqa: E402
from _fla_delta_chunk import chunk_delta_rule as local_chunk_delta_rule  # noqa: E402


def _mk(B, H, T, D, *, seed=0, dt=torch.bfloat16, grad=False):
    """fla layout [B,T,H,D] + beta [B,T,H]。"""
    g = torch.Generator("cuda").manual_seed(seed)
    q = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=g)
    k = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=g)
    v = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=g)
    beta = torch.rand(B, T, H, device="cuda", dtype=dt, generator=g)
    if grad:
        return [x.requires_grad_(True) for x in (q, k, v, beta)]
    return q, k, v, beta


@pytest.mark.parametrize("T", [256, 512])
def test_recurrent_vs_chunked(T):
    """简要版自洽：delta rule 逐步 recurrent ≡ WY chunked（fp32）。"""
    q, k, v = (torch.randn(2, 4, T, 64, device="cuda") for _ in range(3))
    beta = torch.rand(2, 4, T, device="cuda")
    rec = delta_rule_recurrent(q, k, v, beta)
    chk = delta_rule_chunked(q, k, v, beta, chunk_size=64)
    assert_close(chk, rec, name=f"chunked==recurrent (T={T})", atol=3e-3, rtol=1e-2)


def test_triton_faithful_vs_fla():
    """深度优化版忠实性：本地解耦 kernel ≡ fla 原版（定长）。"""
    fla = pytest.importorskip("fla.ops.delta_rule")
    q, k, v, beta = _mk(2, 4, 512, 64, seed=1)
    ol, _ = local_chunk_delta_rule(q, k, v, beta, use_qk_l2norm_in_kernel=True)
    of, _ = fla.chunk_delta_rule(q, k, v, beta, use_qk_l2norm_in_kernel=True)
    assert_close(ol, of, name="local==fla", atol=1e-3, rtol=1e-3)


def test_triton_vs_recurrent():
    """深度优化版（本地 kernel）≡ 简要版 recurrent ground truth。"""
    q, k, v, beta = _mk(2, 4, 512, 64, seed=2)            # [B,T,H,D]
    o_tri, _ = local_chunk_delta_rule(q, k, v, beta, use_qk_l2norm_in_kernel=True)
    qr, kr, vr = (x.transpose(1, 2) for x in (q, k, v))   # [B,H,T,D]
    o_ref = delta_rule_recurrent(qr, kr, vr, beta.transpose(1, 2), l2norm=True).transpose(1, 2)
    assert_close(o_tri, o_ref, name="triton==recurrent", atol=2e-2, rtol=1e-2)


def test_varlen_vs_fla():
    """变长（cu_seqlens / sequence packing）：本地解耦 kernel ≡ fla 原版。"""
    fla = pytest.importorskip("fla.ops.delta_rule")
    cu = torch.tensor([0, 128, 328, 512], device="cuda", dtype=torch.int32)
    T, H, D = 512, 4, 64
    g = torch.Generator("cuda").manual_seed(5)
    q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    beta = torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16, generator=g)
    ol, _ = local_chunk_delta_rule(q, k, v, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    of, _ = fla.chunk_delta_rule(q, k, v, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    assert_close(ol, of, name="varlen local==fla", atol=1e-3, rtol=1e-3)


def test_fwd_bwd():
    """fwd+bwd 跑通，梯度非空。"""
    q, k, v, beta = _mk(2, 4, 256, 64, seed=3, grad=True)
    o, _ = local_chunk_delta_rule(q, k, v, beta, use_qk_l2norm_in_kernel=True)
    o.sum().backward()
    assert all(x.grad is not None for x in (q, k, v, beta))
