"""07-block-sparse-attention 数值正确性测试。

两条关键校验：
  1. top-k 取满（选中所有块）时，块稀疏退化为普通 full 注意力；
  2. gather 省算实现与 mask 参考实现数值一致（不同 top-k / causal）。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, make_qkv, naive_attention  # noqa: E402
from block_sparse import (  # noqa: E402
    block_sparse_attention,
    block_sparse_reference,
    select_topk_blocks,
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("S,block_size", [(256, 32), (512, 64), (384, 32)])
def test_fullselect_equals_full(causal, S, block_size):
    """选中所有块时，块稀疏 == 普通 full 注意力。"""
    q, k, v = make_qkv(2, 4, S, 64, dtype=torch.float16, seed=0)
    nb = S // block_size
    ref = block_sparse_reference(q, k, v, block_size, top_k=nb, causal=causal)
    full = naive_attention(q, k, v, causal=causal)
    assert_close(ref, full, name=f"fullselect==full (causal={causal},S={S})")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
def test_gather_equals_reference(causal, top_k, block_size):
    """gather 省算实现 == mask 参考实现。"""
    q, k, v = make_qkv(2, 4, 512, 64, dtype=torch.float16, seed=1)
    ref = block_sparse_reference(q, k, v, block_size, top_k, causal=causal)
    gat = block_sparse_attention(q, k, v, block_size, top_k, causal=causal)
    assert_close(gat, ref, name=f"gather==ref (top_k={top_k},bs={block_size},causal={causal})")


def test_causal_selection_respects_order():
    """causal 下，候选充足的 query 块（i >= top_k）选中的 key 块都 <= i。

    当块预算超过可选块数时（i < top_k），topk 会顺带选到一些"未来块"，但它们在注意力里会被
    causal 掩码全部排除、不影响结果（数值正确性由 test_gather_equals_reference 覆盖）。
    """
    q, k, v = make_qkv(1, 2, 256, 64, dtype=torch.float16, seed=2)
    block_size, nb, top_k = 32, 256 // 32, 4
    idx = select_topk_blocks(q, k, block_size, top_k=top_k, causal=True)
    for i in range(top_k, nb):  # 只看候选充足的 query 块
        assert (idx[:, :, i] <= i).all(), f"query 块 {i} 选了未来的 key 块"
