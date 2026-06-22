"""标准（朴素）注意力参考实现 —— 数值 ground truth。

这里的实现刻意写得直白、逐步，与数学公式一一对应，只追求"看得懂"，不追求性能。
所有变体的优化实现（包括从外部仓库拷贝来的 Triton kernel）都以本文件的输出作为
正确性基准（reference）。

形状约定（与 PyTorch SDPA / FlashAttention 一致）：
    q: (B, Hq,  Sq, D)
    k: (B, Hkv, Sk, D)
    v: (B, Hkv, Sk, D)
其中 Hkv 可以小于 Hq —— 此时按 GQA/MQA 规则把每个 KV head 广播给一组 Q head。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 (B, Hkv, S, D) 的 KV 沿 head 维重复 n_rep 次 -> (B, Hkv*n_rep, S, D)。

    这是 GQA/MQA 的"等价稠密展开"：每个 KV head 复制给同一组里的 n_rep 个 Q head。
    高效 kernel 不会真的复制，而是在 index 上做映射；这里为了参考实现的清晰直接展开。
    """
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, S, D)
        .reshape(B, H * n_rep, S, D)
    )


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    dropout_p: float = 0.0,
    window: int | None = None,
) -> torch.Tensor:
    """最朴素的缩放点积注意力，每一步都对应一行数学公式。

    O = softmax(QK^T / sqrt(d) + mask) @ V

    刻意全程在 float32 上累加再转回原 dtype，避免 fp16/bf16 的中间精度损失干扰
    "数学正确性"，使其成为可靠的 ground truth。

    Args:
        causal: 是否使用因果掩码（下三角可见）。当 Sq != Sk 时，让 query 末端对齐
            key 末端（decode 场景：Sq 个新 query 接在 Sk-Sq 个历史 token 之后）。
        sm_scale: softmax 温度缩放，默认 1/sqrt(D)。
        dropout_p: 注意力权重 dropout（训练用，对比数值时设 0）。
    """
    _, Hq, Sq, D = q.shape  # 第一维 batch 不直接用到
    Hkv = k.shape[1]
    Sk = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # GQA / MQA：把 KV heads 展开到与 Q heads 对齐
    if Hkv != Hq:
        assert Hq % Hkv == 0, f"Hq={Hq} 必须能被 Hkv={Hkv} 整除"
        k = repeat_kv(k, Hq // Hkv)
        v = repeat_kv(v, Hq // Hkv)

    # 1) 打分：S = QK^T / sqrt(d)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale  # (B,Hq,Sq,Sk)

    # 2) 因果掩码（可选）：query i 只能看到 key <= i（末端对齐）
    if causal:
        causal_mask = torch.ones(Sq, Sk, dtype=torch.bool, device=q.device).tril(
            diagonal=Sk - Sq
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    # 2b) 滑动窗口（可选）：query 只看最近 window 个 key（含自己），更早的屏蔽
    if window is not None:
        q_pos = (torch.arange(Sq, device=q.device) + (Sk - Sq))[:, None]
        k_pos = torch.arange(Sk, device=q.device)[None, :]
        scores = scores.masked_fill(k_pos <= q_pos - window, float("-inf"))

    # 3) softmax（float32，数值稳定）
    attn = torch.softmax(scores, dim=-1)

    # 4) dropout（可选）
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    # 5) 加权求和：O = attn @ V
    o = torch.matmul(attn, v.float())
    return o.to(q.dtype)
