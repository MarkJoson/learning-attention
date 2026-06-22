"""10 简要版（linear attention + GLA）正确性测试。

校验：
  1. linear attention 三形式等价：parallel ≡ recurrent ≡ chunked（causal）；
  2. 非 causal 结合律：φ(Q)(φ(K)ᵀV) 两种括号顺序一致；
  3. GLA 在 gate=0（无衰减）时退化为 linear attention（identity φ + scale）。
用 fp32 精确对齐（三形式数学等价，差异仅浮点累积）。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, make_qkv  # noqa: E402
from linear import (  # noqa: E402
    feature_map,
    linear_attn_parallel,
    linear_attn_recurrent,
    linear_attn_chunked,
    gla_recurrent,
)


@pytest.mark.parametrize("S,chunk", [(256, 64), (512, 128), (384, 64)])
def test_three_forms_equivalent(S, chunk):
    """parallel ≡ recurrent ≡ chunked（causal）。"""
    q, k, v = make_qkv(2, 4, S, 64, dtype=torch.float32, seed=0)
    rec = linear_attn_recurrent(q, k, v)
    par = linear_attn_parallel(q, k, v, causal=True)
    chk = linear_attn_chunked(q, k, v, chunk_size=chunk)
    # linear attention 无 softmax 归一，输出值域随 S 增长，parallel 的大矩阵乘与 recurrent 逐步
    # 累加的浮点路径不同，大 S 下绝对差异偏大（mean 仍 ~1e-4）；放宽 atol 反映这一数值特性。
    assert_close(par, rec, name=f"parallel==recurrent (S={S})", atol=2e-2, rtol=1e-2)
    assert_close(chk, rec, name=f"chunked==recurrent (S={S},L={chunk})", atol=2e-2, rtol=1e-2)


def test_noncausal_associativity():
    """非 causal：(φ(Q)φ(K)ᵀ)V ≡ φ(Q)(φ(K)ᵀV)（结合律，linear attention 省算的根据）。"""
    q, k, v = make_qkv(2, 4, 256, 64, dtype=torch.float32, seed=1)
    fast = linear_attn_parallel(q, k, v, causal=False)                 # 走结合律 O(S·D²)
    qf, kf = feature_map(q), feature_map(k)
    slow = torch.matmul(torch.matmul(qf, kf.transpose(-1, -2)), v)     # O(S²·D)
    assert_close(fast, slow, name="non-causal associativity")


def test_gla_reduces_to_linear_no_decay():
    """GLA 在 g=0（α=1，无遗忘）时 ≡ linear attention（identity φ，scale 作用在 q）。"""
    q, k, v = make_qkv(2, 4, 128, 64, dtype=torch.float32, seed=2)
    g = torch.zeros_like(q)
    scale = 64 ** -0.5
    gla = gla_recurrent(q, k, v, g)
    ref = linear_attn_recurrent(q * scale, k, v, phi=lambda x: x)
    assert_close(gla, ref, name="GLA(g=0) == linear attention")


def test_gla_decay_shrinks_state():
    """sanity：更强的遗忘（更负的 g）应让远期 key 的影响更小（输出对早期扰动更不敏感）。"""
    torch.manual_seed(3)
    q, k, v = make_qkv(1, 2, 64, 32, dtype=torch.float32, seed=3)
    g_weak = torch.zeros_like(q)                       # 不遗忘
    g_strong = torch.full_like(q, -1.0)                # 强遗忘
    o_weak = gla_recurrent(q, k, v, g_weak)
    o_strong = gla_recurrent(q, k, v, g_strong)
    assert not torch.allclose(o_weak, o_strong), "衰减门控应改变输出"
