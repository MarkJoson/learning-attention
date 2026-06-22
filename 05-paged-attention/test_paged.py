"""05-paged-attention 数值正确性测试。

验证：分页存储 + block table 间接寻址后，decode 注意力的结果与"把 KV 当连续张量"的朴素
参考完全一致——无论物理 block 怎么分配（包括多序列交错分配导致的物理不连续）。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, naive_attention  # noqa: E402
from paged import PagedKVCache, paged_decode_attention  # noqa: E402


def _ref_decode(q_r, k_seq, v_seq):
    """单序列 decode 参考：q_r (Hq,D) attend 到 k_seq/v_seq (L,Hkv,D)。"""
    q4 = q_r.unsqueeze(0).unsqueeze(2)               # (1,Hq,1,D)
    k4 = k_seq.permute(1, 0, 2).unsqueeze(0)          # (1,Hkv,L,D)
    v4 = v_seq.permute(1, 0, 2).unsqueeze(0)
    return naive_attention(q4, k4, v4, causal=False)[0, :, 0, :]  # (Hq,D)


@pytest.mark.parametrize("block_size", [1, 16, 64])
@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (8, 2), (16, 4)])
def test_paged_decode(block_size, Hq, Hkv):
    """逐序列填充：不同 block_size、MHA/GQA 都应与朴素参考一致。"""
    torch.manual_seed(0)
    D = 64
    seqlens = [300, 500, 128]
    total = sum(seqlens)
    cache = PagedKVCache((total + block_size - 1) // block_size + 8, block_size, Hkv, D)
    q = torch.randn(len(seqlens), Hq, D, dtype=torch.float16, device="cuda")
    refs = []
    for r, L in enumerate(seqlens):
        k_seq = torch.randn(L, Hkv, D, dtype=torch.float16, device="cuda")
        v_seq = torch.randn(L, Hkv, D, dtype=torch.float16, device="cuda")
        for t in range(L):
            cache.append(r, k_seq[t], v_seq[t])
        refs.append(_ref_decode(q[r], k_seq, v_seq))

    out = paged_decode_attention(q, cache, list(range(len(seqlens))))
    for r in range(len(seqlens)):
        assert_close(out[r], refs[r], name=f"req {r} block_size={block_size} Hq={Hq} Hkv={Hkv}")


def test_interleaved_allocation():
    """多序列交替生成：物理 block 交错分配（真实推理场景），寻址仍须正确。"""
    torch.manual_seed(1)
    Hq, Hkv, D, block_size = 8, 2, 64, 4
    seqlens = [10, 14, 7]
    cache = PagedKVCache(64, block_size, Hkv, D)
    ks = [torch.randn(L, Hkv, D, dtype=torch.float16, device="cuda") for L in seqlens]
    vs = [torch.randn(L, Hkv, D, dtype=torch.float16, device="cuda") for L in seqlens]

    # 交替 append：t=0 时三条各加一个 token，t=1 时再各加一个……物理 block 因此交错
    for t in range(max(seqlens)):
        for r, L in enumerate(seqlens):
            if t < L:
                cache.append(r, ks[r][t], vs[r][t])

    q = torch.randn(len(seqlens), Hq, D, dtype=torch.float16, device="cuda")
    out = paged_decode_attention(q, cache, list(range(len(seqlens))))
    for r, L in enumerate(seqlens):
        assert_close(out[r], _ref_decode(q[r], ks[r], vs[r]), name=f"interleaved req {r}")

    # 确认物理上确实交错：序列 0 的 block 编号不是 0,1,2... 连续
    assert cache.block_table[0] != list(range(len(cache.block_table[0]))), "应发生交错分配"


def test_memory_accounting():
    """按需分配：用掉的 block 数应等于各序列所需 block 数之和。"""
    block_size = 16
    cache = PagedKVCache(128, block_size, num_kv_heads=2, head_dim=64)
    seqlens = [10, 33, 64]
    for r, L in enumerate(seqlens):
        for t in range(L):
            kt = torch.randn(2, 64, dtype=torch.float16, device="cuda")
            cache.append(r, kt, kt)
    expected = sum((L + block_size - 1) // block_size for L in seqlens)
    assert cache.used_blocks() == expected, f"{cache.used_blocks()} != {expected}"
