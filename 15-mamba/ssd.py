r"""15 Mamba2 SSD（State Space Duality）—— SSM 与注意力的对偶。

Mamba2 的核心洞察：**selective SSM 可以等价写成一种带衰减的 masked 注意力**。把 SSM 的标量衰减 $a_t$、
输入投影 $B_t=k_t$、输出投影 $C_t=q_t$、输入 $x_t=v_t$ 代进去，得到 **SSD**。它有两种等价计算形式：

1. **recurrent（线性形式）**：状态 $S\in\mathbb R^{K\times V}$ 逐步递推 —— 推理 $O(T)$、内存 $O(1)$：

       Sₜ = exp(gₜ)·Sₜ₋₁ + kₜ vₜᵀ,      yₜ = qₜ Sₜ

   这**就是标量衰减的线性注意力 / GLA**（GLA 的 g 是 per-channel 向量，SSD 的 g 是 per-head 标量）。

2. **attention（对偶形式）**：把递推展开成一个**半可分矩阵**（1-semiseparable）乘法 —— 训练可并行 $O(T^2)$：

       M = L ∘ (Q Kᵀ),   L_{ij} = exp(g^cum_i − g^cum_j)（i≥j，否则 0）,   Y = M V

   其中 $g^{\mathrm{cum}}$ 是 $g$ 的前缀和。$L$ 是"累积衰减下三角"，$L\circ(QK^\top)$ 就是带衰减的 causal 注意力分数。

两种形式数学等价（SSD 对偶），这正是 Mamba2 既能像 RNN 一样 $O(T)$ 推理、又能像注意力一样并行训练的原因。
深度优化版（生产 chunk kernel）解耦自 fla 的 `simple_gla`（见 ssd_triton.py / SOURCES.md）；介于两种形式之间，
块内走对偶（矩阵乘）、块间走递推。
"""
from __future__ import annotations

import torch


def ssd_recurrent(q, k, v, g, *, scale=None):
    """SSD 线性（recurrent）形式 —— 标量衰减的线性注意力（ground truth）。

    q/k: (B,H,T,K)，v: (B,H,T,V)，**g: (B,H,T) 是 per-head 标量 log decay**（$\\le 0$，exp(g)∈(0,1]）。
    Sₜ = exp(gₜ) Sₜ₋₁ + kₜ vₜᵀ，yₜ = (qₜ·scale) Sₜ。scale 默认 1/√K。
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    S = torch.zeros(B, H, K, V, device=q.device)
    outs = []
    for t in range(T):
        S = g[:, :, t].exp()[..., None, None] * S + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        outs.append((q[:, :, t].unsqueeze(-1) * S).sum(-2) * scale)
    return torch.stack(outs, dim=2).to(v.dtype)


def ssd_attention_dual(q, k, v, g, *, scale=None):
    """SSD 注意力（对偶）形式 —— 半可分矩阵 $M=L\\circ(QK^\\top)$，$Y=MV$（与 recurrent 等价）。

    把 recurrent 展开：$y_i=\\sum_{j\\le i} e^{g^{\\mathrm{cum}}_i-g^{\\mathrm{cum}}_j}(q_i^\\top k_j)\\,v_j$。
    其中 $g^{\\mathrm{cum}}$ 是 g 的前缀和，衰减因子 $L_{ij}=e^{g^{\\mathrm{cum}}_i-g^{\\mathrm{cum}}_j}$ 构成累积衰减下三角。
    """
    K = q.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    g_cum = g.cumsum(-1)                                            # (B,H,T) 累积衰减
    A = (q @ k.transpose(-1, -2)) * scale                          # (B,H,T,T) 注意力分数 QKᵀ
    L = (g_cum[..., :, None] - g_cum[..., None, :]).exp()          # L_ij = exp(g_cum_i - g_cum_j)
    L = L.tril()                                                    # causal：只保留 i>=j（i<j 置 0）
    M = A * L                                                       # 半可分矩阵 M = L ∘ (QKᵀ)
    return (M @ v).to(v.dtype)
