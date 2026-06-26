"""15 Mamba 测试。

  1. SSD 对偶：recurrent ≡ attention-dual（半可分矩阵，SSD 的灵魂）；
  2. 忠实性：本地解耦 kernel ≡ fla simple_gla（定长 + 变长，bitwise）；kernel ≡ SSD recurrent；fwd+bwd；
  3. Mamba1 selective SSM：形状 + 块级 causal sanity。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close  # noqa: E402
from ssd import ssd_recurrent, ssd_attention_dual  # noqa: E402
from ssd_triton import ssd_chunk  # noqa: E402
from mamba1 import selective_ssm_recurrent  # noqa: E402
from _fla_ssd_chunk import chunk_simple_gla as local_ssd  # noqa: E402


def _mk(B, H, T, D, *, seed=0, dt=torch.bfloat16, grad=False):
    """fla 格式 [B,T,H,D]，g 为 [B,T,H] per-head 标量 log decay。"""
    gen = torch.Generator("cuda").manual_seed(seed)
    q = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    k = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    v = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    g = F.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32, generator=gen))
    if grad:
        return [x.requires_grad_(True) for x in (q, k, v, g)]
    return q, k, v, g


def test_ssd_dual_equivalence():
    # SSD 的灵魂：线性 recurrent ≡ 注意力对偶（半可分矩阵），float32 精确
    B, H, T, D = 2, 4, 128, 64
    gen = torch.Generator("cuda").manual_seed(0)
    q = torch.randn(B, H, T, D, device="cuda", generator=gen)
    k = torch.randn(B, H, T, D, device="cuda", generator=gen)
    v = torch.randn(B, H, T, D, device="cuda", generator=gen)
    g = F.logsigmoid(torch.randn(B, H, T, device="cuda", generator=gen))
    o_rec = ssd_recurrent(q, k, v, g)
    o_dual = ssd_attention_dual(q, k, v, g)
    assert_close(o_rec, o_dual, name="SSD recurrent==attention-dual", atol=1e-4, rtol=1e-4)


def test_ssd_faithful_vs_fla():
    fla = pytest.importorskip("fla.ops.simple_gla")
    q, k, v, g = _mk(2, 4, 512, 64, seed=1)
    ol, _ = local_ssd(q, k, v, g)
    of, _ = fla.chunk_simple_gla(q, k, v, g)
    assert_close(ol, of, name="local==fla")


def test_ssd_varlen_vs_fla():
    fla = pytest.importorskip("fla.ops.simple_gla")
    cu = torch.tensor([0, 128, 328, 512], device="cuda", dtype=torch.int32)
    q, k, v, g = _mk(1, 4, 512, 64, seed=5)
    ol, _ = local_ssd(q, k, v, g, cu_seqlens=cu)
    of, _ = fla.chunk_simple_gla(q, k, v, g, cu_seqlens=cu)
    assert_close(ol, of, name="varlen local==fla")


def test_ssd_kernel_vs_recurrent():
    q, k, v, g = _mk(2, 4, 256, 64, seed=2)
    o_tri, _ = local_ssd(q, k, v, g)                                       # [B,T,H,D]
    qr, kr, vr, gr = (x.transpose(1, 2) for x in (q, k, v, g))            # -> [B,H,T,D] / g [B,H,T]
    o_ref = ssd_recurrent(qr, kr, vr, gr).transpose(1, 2)
    assert_close(o_tri, o_ref, name="kernel==SSD recurrent", atol=3e-2, rtol=2e-2)


def test_ssd_chunk_wrapper():
    # [B,H,T,D] 封装与 fla 原生 [B,T,H,D] 一致
    q, k, v, g = _mk(2, 4, 256, 64, seed=4)
    o_native, _ = local_ssd(q, k, v, g)                                   # [B,T,H,D]
    o_wrap = ssd_chunk(*(x.transpose(1, 2) for x in (q, k, v)), g.transpose(1, 2))  # [B,H,T,D]
    assert_close(o_wrap.transpose(1, 2), o_native, name="ssd_chunk wrapper==native")


def test_ssd_fwd_bwd():
    q, k, v, g = _mk(2, 4, 256, 64, seed=3, grad=True)
    o, _ = local_ssd(q, k, v, g)
    o.sum().backward()
    assert all(x.grad is not None for x in (q, k, v, g))


def test_mamba1_selective_ssm():
    bsz, L, D, N = 2, 64, 16, 8
    gen = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(bsz, L, D, device="cuda", generator=gen)
    A = -torch.rand(D, N, device="cuda", generator=gen)               # 负衰减（稳定）
    B = torch.randn(bsz, L, N, device="cuda", generator=gen)
    C = torch.randn(bsz, L, N, device="cuda", generator=gen)
    dt = F.softplus(torch.randn(bsz, L, D, device="cuda", generator=gen))
    y = selective_ssm_recurrent(x, A, B, C, dt)
    assert y.shape == (bsz, L, D) and not torch.isnan(y).any()
    # causal：扰动末尾输入，靠前输出不变
    x2 = x.clone(); x2[:, -1] += 10.0
    y2 = selective_ssm_recurrent(x2, A, B, C, dt)
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5)
