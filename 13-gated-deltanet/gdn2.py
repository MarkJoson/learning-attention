"""13 GDN-2（Gated DeltaNet 2，Qwen3.5）—— erase / write **双门控解耦**的 gated delta rule。

GDN（gdn.py）把 delta rule 的「擦除旧值」和「写入新值」**绑在同一个标量 β** 上：

    Sₜ = diag(exp(gₜ)) Sₜ₋₁ + βₜ kₜ (vₜ − Ŝᵀkₜ)ᵀ
                                 └─ β 同时控制擦多少、写多少 ─┘

GDN-2 的洞察：擦除和写入是**两件事**，该由两个门独立控制。于是拆成 per-channel 的
**erase 门 b∈R^K**（key 轴，擦多少旧状态）和 **write 门 w∈R^V**（value 轴，写多少新值）：

    Sₜ = (I − kₜ (bₜ⊙kₜ)ᵀ) diag(exp(gₜ)) Sₜ₋₁ + kₜ (wₜ⊙vₜ)ᵀ

逐项看（⊙ 是逐元素积）：
  · diag(exp(gₜ)) Sₜ₋₁          —— per-channel 衰减（与 KDA 同，g∈R^K）；
  · (I − kₜ(bₜ⊙kₜ)ᵀ)·…          —— 用 **erase 门 b** 调制的投影，沿 kₜ 方向擦除旧内容；
  · + kₜ (wₜ⊙vₜ)ᵀ              —— 用 **write 门 w** 缩放后写入新值。

退化关系（一图记住整条谱系）：
    b=w=β（标量）         → KDA（gdn.py 的 per-channel 门控版）
    再令 g 退化为 per-head 标量 → GDN v1（gdn.py）
    再令 g≡0、b=w=β        → DeltaNet（第 11 章，无门控）

本文件是**自写的 ground truth recurrent**（与 fla `naive_recurrent_gdn2` 数学等价，见 gdn2_naive.py），
深度优化的 chunk kernel 在 gdn2_triton.py（解耦自 fla）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gdn2_recurrent(q, k, v, g, b, w, *, scale=None, l2norm=True):
    """GDN-2 erase/write 双门控 delta rule 逐步 recurrent（causal，ground truth）。

    形状（`[B,H,T,D]` layout）：
      q/k: (B,H,T,K)，v: (B,H,T,V)
      g:   (B,H,T,K)  per-channel **decay** 门（log-space，exp(g)∈(0,1] 是遗忘率）
      b:   (B,H,T,K)  per-channel **erase** 门（key 轴，擦除旧状态的强度，典型 [0,1]）
      w:   (B,H,T,V)  per-channel **write** 门（value 轴，写入新值的强度，典型 [0,1]）

    与 fla kernel 对齐：q 缩放 1/√K、q/k 做 L2 归一化（delta rule 标配，否则
    (I−k(b⊙k)ᵀ) 谱半径爆炸 → 状态发散为 nan）。
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    if l2norm:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
    q = q * scale
    S = torch.zeros(B, H, K, V, device=q.device)
    outs = []
    for t in range(T):
        k_t, v_t, q_t = k[:, :, t], v[:, :, t], q[:, :, t]          # (B,H,K)/(B,H,V)
        b_t, w_t, g_t = b[:, :, t], w[:, :, t], g[:, :, t]
        S = g_t.exp().unsqueeze(-1) * S                             # per-channel 衰减 diag(exp(g))S
        erase = ((b_t * k_t).unsqueeze(-1) * S).sum(-2)            # erase 门读出旧值 (b⊙k)ᵀS  → (B,H,V)
        v_new = w_t * v_t - erase                                   # write 门写入新值，减去被擦除部分
        S = S + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)            # rank-1 外积更新 + kₜ v_newᵀ
        outs.append((q_t.unsqueeze(-1) * S).sum(-2))               # 读出 oₜ = Sᵀ qₜ
    return torch.stack(outs, dim=2).to(v.dtype)
