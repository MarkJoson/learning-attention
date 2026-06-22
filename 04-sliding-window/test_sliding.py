"""04-sliding-window 数值正确性测试：滑窗 kernel vs 朴素带窗参考。

验证 common.naive_attention 的 window 掩码与 kernel 的滑窗严格一致，并覆盖 GQA+滑窗、
以及"窗口 >= 序列长度时退化为普通因果注意力"这一边界。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, make_qkv, naive_attention  # noqa: E402
from sliding import sliding_window_attention  # noqa: E402


@pytest.mark.parametrize("window", [16, 32, 64, 128, 257])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sliding_vs_naive(window, head_dim):
    q, k, v = make_qkv(2, 8, 512, head_dim, dtype=torch.float16, seed=0)
    out = sliding_window_attention(q, k, v, window_size=window, causal=True)
    ref = naive_attention(q, k, v, causal=True, window=window)
    assert_close(out, ref, name=f"window={window} D={head_dim}")


@pytest.mark.parametrize("window", [32, 96])
def test_sliding_with_gqa(window):
    """滑窗叠加 GQA（Hq=8, Hkv=2）。"""
    q, k, v = make_qkv(2, 8, 512, 64, kv_heads=2, dtype=torch.float16, seed=1)
    out = sliding_window_attention(q, k, v, window_size=window, causal=True)
    ref = naive_attention(q, k, v, causal=True, window=window)
    assert_close(out, ref, name=f"GQA+window={window}")


def test_window_ge_seqlen_is_full_causal():
    """窗口不小于序列长度时，滑窗退化为普通因果注意力。"""
    q, k, v = make_qkv(2, 8, 256, 64, dtype=torch.float16, seed=2)
    out = sliding_window_attention(q, k, v, window_size=512, causal=True)
    ref = naive_attention(q, k, v, causal=True)  # 无窗
    assert_close(out, ref, name="window>=seqlen == full causal")


def test_smallest_window():
    """最小可用窗口 window=2：每个 token 只看自己和前一个。"""
    q, k, v = make_qkv(1, 4, 64, 64, dtype=torch.float16, seed=3)
    out = sliding_window_attention(q, k, v, window_size=2, causal=True)
    ref = naive_attention(q, k, v, causal=True, window=2)
    assert_close(out, ref, name="window=2")


def test_window_one_rejected():
    """window=1 是 kernel 不支持的退化边界，应被明确拒绝。"""
    q, k, v = make_qkv(1, 4, 64, 64, dtype=torch.float16, seed=3)
    with pytest.raises(AssertionError):
        sliding_window_attention(q, k, v, window_size=1, causal=True)
