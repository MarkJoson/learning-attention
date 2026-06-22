"""GQA / MQA / MHA 的统一调用封装。

真正的 kernel 在 `gqa_triton.py`（提取自 vLLM，见其文件头与 SOURCES.md）。kernel 原生支持
分组：`kv_group_num = Hq // Hkv`，kernel 内 `cur_kv_head = cur_head // kv_group_num`，
于是同一份 kernel 自然涵盖三种情形：

  - MHA：Hq == Hkv（每个 query 头独享一组 KV）
  - GQA：Hq  > Hkv 且能整除（每 group 个 query 头共享一组 KV）
  - MQA：Hkv == 1（所有 query 头共享同一组 KV）

kernel 采用 varlen packed 布局 `(总 token 数, head, head_dim)` + `b_start_loc/b_seq_len`
（真实推理就是这样把不同长度的序列拼在一起、不做 padding）。本文件提供两层接口：
varlen 原生接口，以及把标准 `(B,H,S,D)` 转过去的便捷接口，方便和前两章对照。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gqa_triton import context_attention_fwd  # noqa: E402

__all__ = ["gqa_attention", "gqa_attention_varlen"]


def gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """标准 (B, H, S, D) 接口的 GQA/MQA/MHA。

    Hq=q.shape[1] 与 Hkv=k.shape[1] 的关系决定走 MHA / GQA / MQA。
    `sliding_window` 暂留接口（kernel 已支持），04 章滑动窗口会用到。
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert Hq % Hkv == 0, f"Hq={Hq} 必须能被 Hkv={Hkv} 整除（GQA 分组要求）"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # (B,H,S,D) → varlen packed (B*S, H, D)：把 batch 维拼进 token 维
    qp = q.permute(0, 2, 1, 3).reshape(B * S, Hq, D).contiguous()
    kp = k.permute(0, 2, 1, 3).reshape(B * S, Hkv, D).contiguous()
    vp = v.permute(0, 2, 1, 3).reshape(B * S, Hkv, D).contiguous()
    o = torch.empty_like(qp)

    out = gqa_attention_varlen(
        qp, kp, vp, o,
        b_start_loc=torch.arange(0, B * S, S, device=q.device, dtype=torch.int32),
        b_seq_len=torch.full((B,), S, device=q.device, dtype=torch.int32),
        max_seqlen=S, causal=causal, sm_scale=sm_scale, sliding_window=sliding_window,
    )
    return out.reshape(B, S, Hq, D).permute(0, 2, 1, 3)


def gqa_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    *,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_seqlen: int,
    causal: bool = False,
    sm_scale: float | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """varlen 原生接口：q/k/v 为 (总 token 数, head, head_dim)，写入并返回 o。

    这是真实推理里的格式 —— 多条不同长度的序列首尾相接，用 b_start_loc/b_seq_len 标界，
    避免 padding 浪费。
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    context_attention_fwd(
        q, k, v, o, b_start_loc, b_seq_len, max_seqlen,
        is_causal=causal, softmax_scale=sm_scale,
        sliding_window_q=sliding_window, sliding_window_k=sliding_window,
    )
    return o
