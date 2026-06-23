"""13 GDN（Gated DeltaNet，Qwen3-Next 的线性注意力主干）—— gated delta rule。

GDN = DeltaNet 的 **delta rule 纠错** + 指数 **门控衰减**（自适应遗忘）。和 12-KDA 同属 gated delta
rule，递归公式一致：

    Sₜ = diag(exp(gₜ)) Sₜ₋₁ + βₜ kₜ (vₜ − Ŝᵀkₜ)ᵀ

区别在**门控粒度与配套**：GDN 的门控 + L2norm(Q/K) + Causal Conv1D（局部），是 Qwen3-Next 的设计
（每 4 层 1 层 full attention，~75% 层用 GDN）；KDA（Kimi）用更细的 per-channel 门控。本文件聚焦
核心递归 ground truth（conv1d/混合层是模型层面的封装，不在此）。

GDN-2（gdn2，Qwen3.5）进一步把 delta rule 的 **erase 和 write 解耦**成两个门 —— 见 README/notebook。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gated_delta_recurrent(q, k, v, g, beta, *, scale=None, l2norm=True):
    """GDN gated delta rule 逐步 recurrent（causal，ground truth）。

    q/k: (B,H,T,K)，v: (B,H,T,V)，**g: (B,H,T) 是 per-head 标量门控**（log-space），beta: (B,H,T)。
    这是 GDN 与 KDA 的关键区别：GDN 用 per-head 标量衰减（整个状态统一遗忘），KDA 用 per-channel
    细粒度门控。与 fla kernel 对齐：q 缩放 1/√d、q/k 做 L2 归一化。
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = q.float(), k.float(), v.float(), g.float(), beta.float()
    if l2norm:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
    q = q * scale
    S = torch.zeros(B, H, K, V, device=q.device)
    outs = []
    for t in range(T):
        k_t, v_t, q_t, b_t = k[:, :, t], v[:, :, t], q[:, :, t], beta[:, :, t, None]
        S = g[:, :, t].exp()[..., None, None] * S              # per-head 标量门控衰减（整个状态统一遗忘）
        v_hat = (k_t.unsqueeze(-1) * S).sum(-2)                # 预测 v̂ = Ŝᵀ kₜ
        delta = b_t * (v_t - v_hat)                            # delta rule 纠错
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((q_t.unsqueeze(-1) * S).sum(-2))
    return torch.stack(outs, dim=2).to(v.dtype)
