"""11 DeltaNet —— delta rule（纠错式状态更新 / 快速权重）。

linear attention 的状态**只加不减**：`Sₜ = Sₜ₋₁ + kₜvₜᵀ`，写入新记忆从不擦除旧的 —— 键冲突时
旧值残留、互相干扰。DeltaNet 用 **delta rule**（Widrow-Hoff 学习规则 / 快速权重）：写入前先用 kₜ
"查询"旧状态、算预测误差，只写入**误差**，等价于先擦掉 kₜ 方向的旧记忆、再写新值：

    v̂ₜ = Sₜ₋₁ᵀ kₜ                       # 用 kₜ 查询旧状态得到的"预测值"
    Sₜ = Sₜ₋₁ + βₜ kₜ (vₜ − v̂ₜ)ᵀ
       = Sₜ₋₁ (I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ   # 纠错式更新：先擦 kₜ 方向、再写 vₜ

βₜ ∈ (0,1) 是 data-dependent 的写入强度（类似学习率）。比 GLA 的逐维衰减更接近"在线学习"：
不是按固定速率遗忘，而是按"新旧冲突程度"**定向擦写**。

两种等价形式（与 fla 对齐：q 缩放 1/√d、可选对 q/k 做 L2 归一化）：
  - `delta_rule_recurrent` ：逐步纠错更新（ground truth）；
  - `delta_rule_chunked`   ：**WY 表示**把块内 t 步的 delta 连乘 ∏(I−βkkᵀ) 重排成"一次下三角求逆
                            T=(I−tril(βKKᵀ))⁻¹ + 块间状态递归"，训练高效（深度优化版 kernel 用这个）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _prep(q, k, v, beta, scale, l2norm):
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    if l2norm:                       # fla 的 use_qk_l2norm_in_kernel：对 q/k 做 L2 归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
    q = q * (scale if scale is not None else q.shape[-1] ** -0.5)
    return q, k, v, beta


def delta_rule_recurrent(q, k, v, beta, *, scale=None, l2norm=True):
    """delta rule 逐步纠错更新（causal，ground truth）。q/k/v: (B,H,T,D)，beta: (B,H,T)。"""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    q, k, v, beta = _prep(q, k, v, beta, scale, l2norm)
    S = torch.zeros(B, H, Dk, Dv, device=q.device)              # 状态矩阵 (d_k × d_v)
    outs = []
    for t in range(T):
        k_t, v_t, q_t, b_t = k[:, :, t], v[:, :, t], q[:, :, t], beta[:, :, t, None]
        v_hat = (S * k_t[..., None]).sum(-2)                    # 预测 v̂ = Sᵀ kₜ
        delta = (v_t - v_hat) * b_t                            # 误差 (vₜ − v̂ₜ) × 写入强度 βₜ
        S = S + k_t[..., None] * delta[..., None, :]           # 外积写入 S += βₜ kₜ (vₜ−v̂ₜ)ᵀ
        outs.append((q_t[..., None] * S).sum(-2))              # 读出 oₜ = qₜᵀ Sₜ
    return torch.stack(outs, dim=2).to(v.dtype)


def delta_rule_chunked(q, k, v, beta, *, chunk_size=32, scale=None, l2norm=True):
    """WY 表示的 chunk-parallel delta rule（训练高效；深度优化版 kernel 的算法骨架）。

    关键：块内连续 t 步的 delta 更新连乘 ∏(I − βₜkₜkₜᵀ) 形成一个**下三角变换**，可一次性求逆
    T = (I − strict_tril(diag(β) K Kᵀ))⁻¹（前代法），把"逐步擦写"重排成块内矩阵运算；块间只传一个
    状态 S 做递归。于是既有并行度（块内 GEMM），又是 O(T) 复杂度（块间递归）。
    """
    B, H, L, Dk = q.shape
    Dv = v.shape[-1]
    q, k, v, beta = _prep(q, k, v, beta, scale, l2norm)
    assert L % chunk_size == 0, "演示用：序列长度需为 chunk_size 整数倍"
    C, N = chunk_size, L // chunk_size

    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    qr, kr, vr, kbr = (x.view(B, H, N, C, x.shape[-1]) for x in (q, k, v, k_beta))

    # T = (I − strict_lower_tri(diag(β) K Kᵀ))⁻¹，用前代法逐行解
    incl = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), 0)
    Tmat = -(kbr @ kr.transpose(-1, -2)).masked_fill(incl, 0)
    for i in range(1, C):
        Tmat[..., i, :i] = Tmat[..., i, :i] + (Tmat[..., i, :, None].clone() * Tmat[..., :, :i].clone()).sum(-2)
    Tmat = Tmat + torch.eye(C, device=q.device)
    u = Tmat @ vr           # WY 表示的 u（已"解耦"块内依赖）
    w = Tmat @ kbr          # WY 表示的 w

    S = q.new_zeros(B, H, Dk, Dv)
    strict = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), 1)
    outs = []
    for i in range(N):
        qi, ki = qr[:, :, i], kr[:, :, i]
        a = (qi @ ki.transpose(-1, -2)).masked_fill(strict, 0)     # 块内严格下三角
        ui = u[:, :, i] - w[:, :, i] @ S                            # 减去块间状态贡献
        outs.append(qi @ S + a @ ui)                                # 块间(q·S) + 块内(a·u)
        S = S + ki.transpose(-1, -2) @ ui                           # 更新块间状态
    return torch.cat(outs, dim=2).view(B, H, L, Dv).to(v.dtype)
