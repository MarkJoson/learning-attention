"""MoBA（Kimi 动态块稀疏）对照测试 —— 只验证可跑的纯 PyTorch 参考 moba_naive。

MoBA 的 chunk 级动态 top-k 选块 + "当前块必选"，与 07 的"对角块必看"思路一致。
高效版 moba_efficient.py 依赖 flash-attn（本环境未装），仅供阅读，不在此测试。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, naive_attention  # noqa: E402
from moba_naive import moba_attn_varlen_naive  # noqa: E402


@pytest.mark.parametrize("chunk,topk", [(64, 2), (32, 4)])
def test_moba_naive_runs(chunk, topk):
    """MoBA naive 能跑、形状对、无 nan。"""
    torch.manual_seed(0)
    S, H, D = 256, 2, 64
    q = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
    out = moba_attn_varlen_naive(q, k, v, cu, S, moba_chunk_size=chunk, moba_topk=topk)
    assert out.shape == (S, H, D)
    assert not out.isnan().any()


def test_moba_naive_fullselect_equals_full_causal():
    """选满所有 chunk（moba_topk≥块数）→ 退化为 full causal。"""
    torch.manual_seed(0)
    S, H, D = 256, 2, 64
    chunk = 64
    nb = S // chunk
    q = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
    out = moba_attn_varlen_naive(q, k, v, cu, S, moba_chunk_size=chunk, moba_topk=nb)

    # full causal 参考：(S,H,D) → (1,H,S,D) → 回 (S,H,D)
    full = naive_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], causal=True
    )[0].transpose(0, 1)
    assert_close(out, full, name="moba 全选 == full causal")
