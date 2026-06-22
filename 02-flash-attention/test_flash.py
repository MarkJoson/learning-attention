"""02-flash-attention 数值正确性测试：Triton FlashAttention v2 vs 朴素参考 / SDPA。

覆盖 forward 与 backward（dq/dk/dv），causal / non-causal，head_dim ∈ {64, 128}。
其中 head_dim=128 专门用来验证针对 RTX 4090 共享内存的 autotune config 适配是否生效。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, make_qkv, naive_attention  # noqa: E402
from flash import flash_attention  # noqa: E402


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seqlen", [256, 1024])
def test_forward(causal, head_dim, seqlen):
    q, k, v = make_qkv(2, 8, seqlen, head_dim, dtype=torch.float16, seed=0)
    out = flash_attention(q, k, v, causal=causal)
    ref = naive_attention(q, k, v, causal=causal)
    assert_close(out, ref, name=f"fwd vs naive (causal={causal},D={head_dim},S={seqlen})")


def test_forward_matches_sdpa():
    q, k, v = make_qkv(2, 16, 2048, 64, dtype=torch.float16, seed=3)
    out = flash_attention(q, k, v, causal=True)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert_close(out, sdpa, name="fwd vs SDPA")


# 注意：经典教程 kernel 的 backward 仅支持 causal=True —— 原教程的 benchmark 也只测
# causal 反向，non-causal 反向在原教程中即数值不正确（已用其自带 reference 复现确认）。
# 这是该 kernel 的真实能力边界，故此处据实只验证 causal backward。详见 README「已知限制」。
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_backward(causal, head_dim):
    seqlen = 512  # 128 的倍数（kernel backward 约束）
    q, k, v = make_qkv(2, 8, seqlen, head_dim, dtype=torch.float16, seed=1, requires_grad=True)
    qr, kr, vr = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    do = torch.randn_like(q)

    out = flash_attention(q, k, v, causal=causal)
    out.backward(do)
    ref = naive_attention(qr, kr, vr, causal=causal)
    ref.backward(do)

    # 反向在 fp16 下累积误差比前向大，容差适当放宽
    assert_close(q.grad, qr.grad, atol=3e-2, rtol=2e-2, name=f"dq (causal={causal},D={head_dim})")
    assert_close(k.grad, kr.grad, atol=3e-2, rtol=2e-2, name=f"dk (causal={causal},D={head_dim})")
    assert_close(v.grad, vr.grad, atol=3e-2, rtol=2e-2, name=f"dv (causal={causal},D={head_dim})")


def test_rejects_gqa():
    """GQA 输入应被明确拒绝（本章 kernel 不支持，留给 03 章）。"""
    q, k, v = make_qkv(2, 8, 256, 64, kv_heads=2, dtype=torch.float16, seed=2)
    with pytest.raises(AssertionError):
        flash_attention(q, k, v, causal=True)
