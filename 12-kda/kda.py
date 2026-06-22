"""12 KDA（Kimi Delta Attention）—— gated delta rule：GLA 门控 + DeltaNet 纠错。

KDA 是 Kimi Linear 的核心，把前两章合二为一：
  - 10 章 **GLA** 的逐通道衰减门控（选择性遗忘）：`Sₜ = diag(exp(gₜ)) Sₜ₋₁`
  - 11 章 **DeltaNet** 的 delta rule 纠错写入：`+ βₜ kₜ (vₜ − Ŝᵀkₜ)ᵀ`

合起来（先门控衰减、再 delta 纠错）：

    Ŝ = diag(exp(gₜ)) Sₜ₋₁                       # 先按 per-channel 门控遗忘
    v̂ₜ = Ŝᵀ kₜ                                   # 用 kₜ 查询（门控后的）旧状态
    Sₜ = Ŝ + βₜ kₜ (vₜ − v̂ₜ)ᵀ                    # delta rule 纠错写入

gₜ（log-space，per-channel）让状态选择性遗忘，βₜ 控制纠错写入强度。比单独的 GLA（只有衰减、
写入仍只加不减）或 DeltaNet（纠错但无遗忘）都更强 —— 既能定向擦写、又能按通道遗忘。

`kda_recurrent` 是 ground truth（对齐 fla kernel：q 缩放 1/√d、q/k 做 L2 归一化）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kda_recurrent(q, k, v, g, beta, *, scale=None, l2norm=True):
    """KDA = gated delta rule 逐步 recurrent（causal，ground truth）。

    q/k/g: (B,H,T,K)，v: (B,H,T,V)，g 为 log-space per-channel 衰减门控，beta: (B,H,T)。
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = q.float(), k.float(), v.float(), g.float(), beta.float()
    if l2norm:                      # fla 的 use_qk_l2norm_in_kernel
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
    q = q * scale
    S = torch.zeros(B, H, K, V, device=q.device)
    outs = []
    for t in range(T):
        k_t, v_t, q_t, b_t = k[:, :, t], v[:, :, t], q[:, :, t], beta[:, :, t, None]
        S = g[:, :, t].exp().unsqueeze(-1) * S                 # ① GLA 门控：per-channel 衰减
        v_hat = (k_t.unsqueeze(-1) * S).sum(-2)                # ② 预测 v̂ = Ŝᵀ kₜ
        delta = b_t * (v_t - v_hat)                            # ③ 纠错 βₜ(vₜ − v̂ₜ)
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)        #    delta rule 写入
        outs.append((q_t.unsqueeze(-1) * S).sum(-2))          # ④ 读出 oₜ = qₜᵀ Sₜ
    return torch.stack(outs, dim=2).to(v.dtype)
