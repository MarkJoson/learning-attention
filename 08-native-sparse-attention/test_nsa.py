"""08-native-sparse-attention 正确性测试。

NSA 是个带学习权重的模块、没有外部 ground truth，所以我们验证它的**结构性质**：
  - 三条分支各自正确（滑窗≡naive window；selected 全选退化为 full）；
  - 门控归一（softmax 和为 1）；
  - 整体严格 causal（输出不泄露未来 token）。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, naive_attention  # noqa: E402
from nsa import NativeSparseAttention  # noqa: E402


def _qkv(nsa, x):
    B, S, _ = x.shape
    H, D = nsa.n_heads, nsa.head_dim
    qkv = nsa.to_qkv(x).view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def test_sliding_branch_matches_naive_window():
    torch.manual_seed(0)
    nsa = NativeSparseAttention(256, n_heads=4, head_dim=64, sliding_window_size=64).cuda().half()
    x = torch.randn(2, 256, 256, dtype=torch.float16, device="cuda")
    q, k, v = _qkv(nsa, x)
    out = nsa.sliding_branch(q, k, v)
    ref = naive_attention(q, k, v, causal=True, window=nsa.window)
    assert_close(out, ref, name="sliding branch == naive window")


def test_selected_branch_fullselect_equals_full():
    """selected 分支选满所有块时，退化为普通 full causal 注意力。"""
    torch.manual_seed(0)
    nb = 256 // 32
    nsa = NativeSparseAttention(256, n_heads=4, head_dim=64, block_size=32,
                                num_selected_blocks=nb).cuda().half()
    x = torch.randn(2, 256, 256, dtype=torch.float16, device="cuda")
    q, k, v = _qkv(nsa, x)
    _, blk_sim = nsa.compressed_branch(q, k, v)
    out = nsa.selected_branch(q, k, v, blk_sim)
    ref = naive_attention(q, k, v, causal=True)
    assert_close(out, ref, name="selected fullselect == full")


def test_forward_shape_and_gates():
    nsa = NativeSparseAttention(256, n_heads=4).cuda().half()
    x = torch.randn(2, 256, 256, dtype=torch.float16, device="cuda")
    out, branches = nsa(x, return_branches=True)
    assert out.shape == x.shape, f"输出形状 {out.shape} != 输入 {x.shape}"
    gate_sums = branches["gates"].float().sum(-1)
    assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-2), "门控未归一"


def test_causal_no_future_leak():
    """改 t 之后的输入，不应影响 <= t 的输出（NSA 三分支都必须严格 causal）。"""
    torch.manual_seed(0)
    nsa = NativeSparseAttention(128, n_heads=4, head_dim=32, block_size=16,
                                num_selected_blocks=2, sliding_window_size=32).cuda().double()
    x = torch.randn(1, 128, 128, dtype=torch.float64, device="cuda")
    t = 64
    out1 = nsa(x)
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:])
    out2 = nsa(x2)
    assert_close(out1[:, : t + 1], out2[:, : t + 1], atol=1e-8, rtol=1e-6, name="no future leak")
