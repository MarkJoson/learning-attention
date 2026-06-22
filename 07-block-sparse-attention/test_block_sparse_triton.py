"""07 深度优化版（复用 08 NSA selected kernel）正确性测试。

语义 = 对角块必看 + top-k 历史块。两条校验：
  1. anchor：选满所有历史块 → 退化为 full causal（验证 kernel 复用接口正确）；
  2. NSA kernel 复用 == 匹配语义的 PyTorch 参考（对角块 + top-k 历史块）。

依赖 08 的 nsa_triton（einx/triton），无需外部包。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, make_qkv, naive_attention  # noqa: E402
from block_sparse_triton import (  # noqa: E402
    block_sparse_attention_triton,
    block_sparse_nsa_reference,
)


@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("S", [256, 512])
def test_fullhistory_equals_full_causal(block_size, S):
    """选满所有历史块 → 退化为 full causal（对角 Part1 + 全历史 Part2）。"""
    q, k, v = make_qkv(1, 2, S, 64, dtype=torch.float16, seed=0)
    nb = S // block_size
    out = block_sparse_attention_triton(q, k, v, block_size, top_k=nb, causal=True)
    full = naive_attention(q, k, v, causal=True)
    assert_close(out, full, name=f"fullhistory==full (bs={block_size},S={S})")


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
def test_triton_matches_nsa_reference(top_k, block_size):
    """NSA kernel 复用 == 匹配语义的 PyTorch 参考（对角块 + top-k 历史块）。"""
    q, k, v = make_qkv(2, 4, 512, 64, dtype=torch.float16, seed=1)
    out = block_sparse_attention_triton(q, k, v, block_size, top_k, causal=True)
    ref = block_sparse_nsa_reference(q, k, v, block_size, top_k, causal=True)
    assert_close(out, ref, name=f"triton==nsa_ref (top_k={top_k},bs={block_size})")
