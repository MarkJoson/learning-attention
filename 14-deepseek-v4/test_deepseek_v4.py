"""14 DeepSeek V4 Hybrid Attention 机制测试（自写简要版，纯 torch）。

验证 compress→index→attend 三步的正确性：压缩缩长 + softmax 加权、indexer 选 top-k 且块级 causal、
CSA/HCA 形状、整体 causal、CSA(top_k=全部) 退化为 HCA 稠密。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepseek_v4 import (softmax_compress, lightning_indexer,  # noqa: E402
                         csa_attention, hca_attention)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_compress_shape_meanpool():
    x = torch.randn(2, 1, 64, 16, device=DEV)
    c = softmax_compress(x, 4, z=None)
    assert c.shape == (2, 1, 16, 16)                       # 序列缩 4 倍
    ref = x.reshape(2, 1, 16, 4, 16).mean(3)               # z=None → 均值池化
    assert torch.allclose(c, ref, atol=1e-6)


def test_compress_softmax_picks_position():
    # z 在某位置极大 → softmax 几乎只取该位置（学习式池化能逼近 max/选择）
    x = torch.randn(1, 1, 8, 4, device=DEV)
    z = torch.zeros(1, 1, 8, 4, device=DEV)
    z[0, 0, 1] = 1e4                                        # block0 内位置 1 主导
    c = softmax_compress(x, 4, z=z)
    assert torch.allclose(c[0, 0, 0], x[0, 0, 1], atol=1e-4)


def test_indexer_topk_and_causal():
    B, Hi, T, d, nb = 2, 2, 64, 8, 16                      # m = T//nb = 4
    q = torch.randn(B, Hi, T, d, device=DEV)
    w = torch.rand(B, Hi, T, device=DEV)
    kc = torch.randn(B, Hi, nb, d, device=DEV)
    sel, causal = lightning_indexer(q, w, kc, top_k=3)
    assert sel.shape == (B, T, 3)
    # 选中的块必须块级 causal 可见（s*m+m-1 < t）。用可见块数 ≥ top_k 的 t（否则 topk 必然带回被掩块，
    # 由下游 compressed_attention 的 causal 再屏蔽，无害）。t=30/63 时可见块 7/16 个 ≥ 3。
    for t in [30, 63]:
        vis = set((causal[t]).nonzero().flatten().tolist())
        chosen = set(sel[0, t].tolist())
        assert chosen <= vis, f"t={t} 选中了不可见块: {chosen - vis}"


def test_csa_hca_shapes():
    B, Hq, Hkv, T, D = 2, 4, 1, 256, 32
    q = torch.randn(B, Hq, T, D, device=DEV)
    k = torch.randn(B, Hkv, T, D, device=DEV)
    v = torch.randn(B, Hkv, T, D, device=DEV)
    # CSA：需要 indexer 输入（Hi 个索引头）
    Hi, di = 2, 16
    nb = T // 4
    q_idx = torch.randn(B, Hi, T, di, device=DEV)
    w_idx = torch.rand(B, Hi, T, device=DEV)
    k_idx = torch.randn(B, Hi, nb, di, device=DEV)
    o_csa = csa_attention(q, k, v, m=4, top_k=8, q_idx=q_idx, w_idx=w_idx, k_idx=k_idx)
    o_hca = hca_attention(q, k, v, m=128)
    assert o_csa.shape == (B, Hq, T, D)
    assert o_hca.shape == (B, Hq, T, D)
    assert not torch.isnan(o_csa).any() and not torch.isnan(o_hca).any()


def test_causality():
    # 扰动末尾 token 的 K/V，靠前位置输出不应改变（块级 causal）
    B, Hq, T, D = 1, 2, 128, 32
    q = torch.randn(B, Hq, T, D, device=DEV)
    k = torch.randn(B, 1, T, D, device=DEV)
    v = torch.randn(B, 1, T, D, device=DEV)
    o1 = hca_attention(q, k, v, m=8)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, -8:] += 10.0; v2[:, :, -8:] += 10.0           # 改最后一个压缩块
    o2 = hca_attention(q, k2, v2, m=8)
    # 前 T-8 个 query 看不到最后一块 → 输出不变
    assert torch.allclose(o1[:, :, :T - 8], o2[:, :, :T - 8], atol=1e-5)


def test_csa_full_topk_equals_hca():
    # top_k >= nb 时，indexer 选中所有块 → CSA 退化为 HCA 稠密（同一压缩比 m）
    B, Hq, T, D = 2, 4, 128, 32
    q = torch.randn(B, Hq, T, D, device=DEV)
    k = torch.randn(B, 1, T, D, device=DEV)
    v = torch.randn(B, 1, T, D, device=DEV)
    m, nb = 4, 128 // 4
    Hi, di = 2, 16
    q_idx = torch.randn(B, Hi, T, di, device=DEV)
    w_idx = torch.rand(B, Hi, T, device=DEV)
    k_idx = torch.randn(B, Hi, nb, di, device=DEV)
    o_csa_full = csa_attention(q, k, v, m=m, top_k=nb, q_idx=q_idx, w_idx=w_idx, k_idx=k_idx)
    o_hca = hca_attention(q, k, v, m=m)                     # 同压缩比、稠密
    assert torch.allclose(o_csa_full, o_hca, atol=1e-5)
