"""13 GDN（Gated DeltaNet）正确性测试。

  1. 忠实性：本地解耦 kernel ≡ fla 原版（定长 + 变长）；
  2. kernel ≡ 简要版 gated-delta recurrent（ground truth）；fwd+bwd。
GDN 需 use_qk_l2norm_in_kernel=True。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close  # noqa: E402
from gdn import gated_delta_recurrent  # noqa: E402
from gdn2 import gdn2_recurrent  # noqa: E402
from _fla_gdn_chunk import chunk_gated_delta_rule as local_gdn  # noqa: E402
from _fla_gdn2_chunk import chunk_gdn2 as local_gdn2  # noqa: E402


def _mk(B, H, T, D, *, seed=0, dt=torch.bfloat16, grad=False):
    gen = torch.Generator("cuda").manual_seed(seed)
    q = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    k = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    v = torch.randn(B, T, H, D, device="cuda", dtype=dt, generator=gen)
    g = F.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32, generator=gen))  # per-head 标量门控
    beta = torch.rand(B, T, H, device="cuda", dtype=dt, generator=gen)
    if grad:
        return [x.requires_grad_(True) for x in (q, k, v, g, beta)]
    return q, k, v, g, beta


def test_faithful_vs_fla():
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    q, k, v, g, beta = _mk(2, 4, 512, 64, seed=1)
    ol, _ = local_gdn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    of, _ = fla.chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert_close(ol, of, name="local==fla")


def test_kernel_vs_recurrent():
    q, k, v, g, beta = _mk(2, 4, 512, 64, seed=2)
    o_tri, _ = local_gdn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    qr, kr, vr, gr = (x.transpose(1, 2) for x in (q, k, v, g))
    o_ref = gated_delta_recurrent(qr, kr, vr, gr, beta.transpose(1, 2), l2norm=True).transpose(1, 2)
    assert_close(o_tri, o_ref, name="kernel==recurrent", atol=3e-2, rtol=2e-2)


def test_varlen_vs_fla():
    fla = pytest.importorskip("fla.ops.gated_delta_rule")
    cu = torch.tensor([0, 128, 328, 512], device="cuda", dtype=torch.int32)
    T, H, D = 512, 4, 64
    gen = torch.Generator("cuda").manual_seed(5)
    q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    g = F.logsigmoid(torch.randn(1, T, H, D, device="cuda", dtype=torch.float32, generator=gen))
    beta = torch.rand(1, T, H, device="cuda", dtype=torch.bfloat16, generator=gen)
    ol, _ = local_gdn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    of, _ = fla.chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    assert_close(ol, of, name="varlen local==fla")


def test_fwd_bwd():
    q, k, v, g, beta = _mk(2, 4, 256, 64, seed=3, grad=True)
    o, _ = local_gdn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    o.sum().backward()
    assert all(x.grad is not None for x in (q, k, v, g, beta))


# ============================ GDN-2（erase/write 双门控解耦）============================
def _mk2(B, H, T, K, V, *, seed=0, dt=torch.bfloat16, grad=False, normed=False):
    """GDN-2 输入：g/b 是 [B,T,H,K]，w 是 [B,T,H,V]（per-channel 双门控）。
    normed=True 时 q/k 预先 L2norm 并转 float32（用于和不含 norm 的 recurrent 对齐）。"""
    gen = torch.Generator("cuda").manual_seed(seed)
    dev = "cuda"
    q = torch.randn(B, T, H, K, device=dev, dtype=dt, generator=gen)
    k = torch.randn(B, T, H, K, device=dev, dtype=dt, generator=gen)
    v = torch.randn(B, T, H, V, device=dev, dtype=dt, generator=gen)
    g = F.logsigmoid(torch.rand(B, T, H, K, device=dev, dtype=torch.float32, generator=gen))  # per-channel decay
    b = torch.rand(B, T, H, K, device=dev, dtype=dt, generator=gen)   # erase 门（key 轴）
    w = torch.rand(B, T, H, V, device=dev, dtype=dt, generator=gen)   # write 门（value 轴）
    if normed:
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        v, b, w = v.float(), b.float(), w.float()
    if grad:
        return [x.requires_grad_(True) for x in (q, k, v, g, b, w)]
    return q, k, v, g, b, w


def test_gdn2_faithful_vs_fla():
    fla = pytest.importorskip("fla.ops.gdn2")
    q, k, v, g, b, w = _mk2(2, 4, 512, 64, 64, seed=1)
    ol, _ = local_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True)
    of, _ = fla.chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True)
    assert_close(ol, of, name="gdn2 local==fla")


def test_gdn2_kernel_vs_recurrent():
    # 预先 L2norm + float32，与不含 norm 的 ground-truth recurrent 对齐（否则谱半径爆炸 → nan）
    q, k, v, g, b, w = _mk2(2, 4, 256, 64, 64, seed=2, normed=True)
    o_tri, _ = local_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=False)
    qr, kr, vr, gr, br, wr = (x.transpose(1, 2) for x in (q, k, v, g, b, w))
    o_ref = gdn2_recurrent(qr, kr, vr, gr, br, wr, l2norm=False).transpose(1, 2)
    assert_close(o_tri, o_ref, name="gdn2 kernel==recurrent", atol=3e-2, rtol=2e-2)


def test_gdn2_varlen_vs_fla():
    fla = pytest.importorskip("fla.ops.gdn2")
    cu = torch.tensor([0, 128, 328, 512], device="cuda", dtype=torch.int32)
    q, k, v, g, b, w = _mk2(1, 4, 512, 64, 64, seed=5)
    ol, _ = local_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    of, _ = fla.chunk_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    assert_close(ol, of, name="gdn2 varlen local==fla")


def test_gdn2_fwd_bwd():
    q, k, v, g, b, w = _mk2(2, 4, 256, 64, 64, seed=3, grad=True)
    o, _ = local_gdn2(q, k, v, g, b, w, use_qk_l2norm_in_kernel=True)
    o.sum().backward()
    assert all(x.grad is not None for x in (q, k, v, g, b, w))
