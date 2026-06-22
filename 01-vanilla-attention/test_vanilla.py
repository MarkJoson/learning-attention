"""01-vanilla-attention 数值正确性测试。

核心验证：online softmax 分块实现在各种配置下都与朴素参考实现 / SDPA 数值一致，
且**与分块大小无关**（online softmax 的关键性质）。
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, make_qkv, naive_attention  # noqa: E402
from vanilla import online_softmax_attention  # noqa: E402


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", [128, 384, 512])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_online_vs_naive(causal, seqlen, head_dim):
    q, k, v = make_qkv(2, 4, seqlen, head_dim, dtype=torch.float16, seed=0)
    ref = naive_attention(q, k, v, causal=causal)
    out = online_softmax_attention(q, k, v, causal=causal, block_q=128, block_k=64)
    assert_close(out, ref, name=f"online vs naive (causal={causal},S={seqlen},D={head_dim})")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_q,block_k", [(64, 32), (128, 64), (256, 128), (32, 256)])
def test_blocksize_invariance(causal, block_q, block_k):
    """不同分块大小结果必须一致 —— online softmax 与分块无关。"""
    q, k, v = make_qkv(2, 4, 512, 64, dtype=torch.float16, seed=1)
    ref = naive_attention(q, k, v, causal=causal)
    out = online_softmax_attention(q, k, v, causal=causal, block_q=block_q, block_k=block_k)
    assert_close(out, ref, name=f"blocks ({block_q},{block_k}) causal={causal}")


@pytest.mark.parametrize("causal", [False, True])
def test_gqa(causal):
    q, k, v = make_qkv(2, 8, 320, 64, kv_heads=2, dtype=torch.float16, seed=2)
    ref = naive_attention(q, k, v, causal=causal)
    out = online_softmax_attention(q, k, v, causal=causal)
    assert_close(out, ref, name=f"GQA (causal={causal})")


@pytest.mark.parametrize("causal", [False, True])
def test_cross_attention_unequal_len(causal):
    """Sq != Sk（decode / cross-attention）；causal 下 query 末端对齐 key 末端。"""
    q, k, v = make_qkv(2, 4, 64, 64, seqlen_k=256, dtype=torch.float16, seed=3)
    ref = naive_attention(q, k, v, causal=causal)
    out = online_softmax_attention(q, k, v, causal=causal)
    assert_close(out, ref, name=f"unequal len (causal={causal})")


def test_matches_sdpa():
    q, k, v = make_qkv(2, 8, 512, 64, dtype=torch.float16, seed=4)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = online_softmax_attention(q, k, v, causal=True)
    assert_close(out, sdpa, name="online vs SDPA")
