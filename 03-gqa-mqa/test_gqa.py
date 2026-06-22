"""03-gqa-mqa 数值正确性测试：MHA / GQA / MQA 共用一份 kernel，全部对齐参考实现。

重点：同一个 kernel 通过 kv_group_num 自然涵盖三种 head 共享模式；并验证 varlen（不等长序列
拼接）这一真实推理格式。
"""
import sys
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close, make_qkv, naive_attention  # noqa: E402
from gqa import gqa_attention, gqa_attention_varlen  # noqa: E402


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (8, 2), (8, 1), (16, 4), (32, 8)])
@pytest.mark.parametrize("D", [64, 128])
def test_mha_gqa_mqa(causal, Hq, Hkv, D):
    """同一 kernel 覆盖 MHA(8/8) / GQA(8/2,16/4,32/8) / MQA(8/1)。"""
    q, k, v = make_qkv(2, Hq, 512, D, kv_heads=Hkv, dtype=torch.float16, seed=0)
    out = gqa_attention(q, k, v, causal=causal)
    ref = naive_attention(q, k, v, causal=causal)
    assert_close(out, ref, name=f"Hq={Hq} Hkv={Hkv} D={D} causal={causal}")


def test_matches_sdpa_gqa():
    """与 PyTorch 官方 SDPA 的原生 GQA（enable_gqa=True）对齐。"""
    q, k, v = make_qkv(2, 16, 1024, 64, kv_heads=4, dtype=torch.float16, seed=1)
    out = gqa_attention(q, k, v, causal=True)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    assert_close(out, sdpa, name="GQA vs SDPA(enable_gqa)")


@pytest.mark.parametrize("causal", [False, True])
def test_varlen_unequal_seqlens(causal):
    """varlen：把长度不同的序列首尾相接（真实推理格式），逐条对齐 naive。"""
    Hq, Hkv, D = 8, 2, 64
    seqlens = [300, 500, 128]
    total = sum(seqlens)
    torch.manual_seed(2)
    qp = torch.randn(total, Hq, D, dtype=torch.float16, device="cuda")
    kp = torch.randn(total, Hkv, D, dtype=torch.float16, device="cuda")
    vp = torch.randn(total, Hkv, D, dtype=torch.float16, device="cuda")
    o = torch.empty_like(qp)

    starts, s = [], 0
    for L in seqlens:
        starts.append(s); s += L
    b_start = torch.tensor(starts, device="cuda", dtype=torch.int32)
    b_seqlen = torch.tensor(seqlens, device="cuda", dtype=torch.int32)

    gqa_attention_varlen(qp, kp, vp, o, b_start_loc=b_start, b_seq_len=b_seqlen,
                         max_seqlen=max(seqlens), causal=causal, sm_scale=1.0 / math.sqrt(D))

    # 逐条序列单独用 naive 求参考，再拼接对比
    for st, L in zip(starts, seqlens):
        q1 = qp[st:st + L].permute(1, 0, 2).unsqueeze(0)  # (1,Hq,L,D)
        k1 = kp[st:st + L].permute(1, 0, 2).unsqueeze(0)
        v1 = vp[st:st + L].permute(1, 0, 2).unsqueeze(0)
        ref = naive_attention(q1, k1, v1, causal=causal)[0].permute(1, 0, 2)  # (L,Hq,D)
        assert_close(o[st:st + L], ref, name=f"varlen seq L={L} causal={causal}")
