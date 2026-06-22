"""10 Linear Attention + GLA —— 把 O(S²) 的 attention 变成 O(S) 的线性递归。

标准 attention：`softmax(QKᵀ)V`，O(S²)。**linear attention** 去掉 softmax、换成 feature map φ：

    O = φ(Q) · (φ(K)ᵀ V)

靠**矩阵结合律**：先算 `φ(K)ᵀV`（一个 D×D 的"状态"矩阵），再 `φ(Q)·状态` —— 复杂度 O(S·D²)，
在序列长度上**线性**。代价是表达力弱于 softmax（没有 query-dependent 的归一化峰）。

本文件给 linear attention 的**三种等价形式**（causal）：
  - `linear_attn_parallel`  ：φ(Q)(φ(K)ᵀV) + causal mask —— 概念最清晰（但 causal 下仍是 O(S²)）；
  - `linear_attn_recurrent` ：Sₜ = Sₜ₋₁ + φ(kₜ)vₜᵀ, oₜ = φ(qₜ)Sₜ —— RNN 形式，O(S) 但串行（ground truth）；
  - `linear_attn_chunked`   ：块内 parallel + 块间传 state —— **训练高效形式**（fla 的 triton kernel 用这个）。

以及 **GLA（Gated Linear Attention）**：给状态加 data-dependent 的**衰减门控**：

    Sₜ = diag(αₜ) Sₜ₋₁ + kₜ vₜᵀ,   oₜ = qₜ Sₜ,   αₜ = exp(gₜ) ∈ (0,1)ᴷ

αₜ 由输入算出的遗忘门（log 空间 gₜ ≤ 0），让状态**选择性遗忘** —— 比固定衰减（RetNet）更强，
是 fla `chunk_gla` 的数学内核。`gla_recurrent` 是其 ground truth，与 fla 的 kernel 语义对齐。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """linear attention 常用的非负 feature map：elu(x)+1（保证 φ(x) > 0）。"""
    return F.elu(x) + 1.0


# ---------------- linear attention（无门控）----------------

def linear_attn_parallel(q, k, v, *, causal=True, phi=feature_map):
    """parallel 形式：φ(Q)(φ(K)ᵀV)。causal 用下三角 mask；非 causal 直接走结合律 O(S·D²)。"""
    q, k = phi(q).float(), phi(k).float()
    v = v.float()
    if causal:
        scores = torch.matmul(q, k.transpose(-1, -2)).tril()      # (B,H,S,S)
        out = torch.matmul(scores, v)
    else:
        kv = torch.matmul(k.transpose(-1, -2), v)                 # (B,H,D,Dv) 状态
        out = torch.matmul(q, kv)
    return out.to(v.dtype)


def linear_attn_recurrent(q, k, v, *, phi=feature_map):
    """recurrent 形式（causal，ground truth）：逐步更新状态 S = Σ φ(k)vᵀ。"""
    q, k = phi(q).float(), phi(k).float()
    v = v.float()
    B, H, S, D = q.shape
    Dv = v.shape[-1]
    state = torch.zeros(B, H, D, Dv, device=q.device, dtype=torch.float32)
    outs = []
    for t in range(S):
        state = state + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)   # 外积累加 φ(kₜ)vₜᵀ
        outs.append(torch.einsum("bhd,bhdv->bhv", q[:, :, t], state))         # oₜ = φ(qₜ)Sₜ
    return torch.stack(outs, dim=2).to(v.dtype)


def linear_attn_chunked(q, k, v, *, chunk_size=64, phi=feature_map):
    """chunked 形式（causal，训练高效）：块内 parallel + 块间传递累积状态 state。

    o_chunk = φ(Q_c)·state(前面所有块)  +  下三角(φ(Q_c)φ(K_c)ᵀ)·V_c
              └─ inter（跨块，O(L·D²)）      └─ intra（块内，O(L²·D)）
    """
    q, k = phi(q).float(), phi(k).float()
    v = v.float()
    B, H, S, D = q.shape
    Dv = v.shape[-1]
    L = chunk_size
    assert S % L == 0, "演示用：序列长度需为 chunk_size 整数倍"
    nc = S // L
    q, k, v = (t.view(B, H, nc, L, t.shape[-1]) for t in (q, k, v))
    state = torch.zeros(B, H, D, Dv, device=q.device, dtype=torch.float32)
    outs = []
    for c in range(nc):
        qc, kc, vc = q[:, :, c], k[:, :, c], v[:, :, c]
        inter = torch.matmul(qc, state)                              # 前面块的状态贡献
        intra = torch.matmul(torch.matmul(qc, kc.transpose(-1, -2)).tril(), vc)
        outs.append(inter + intra)
        state = state + torch.matmul(kc.transpose(-1, -2), vc)       # 累积本块状态
    return torch.cat(outs, dim=2).reshape(B, H, S, Dv).to(v.dtype)


# ---------------- GLA（带 data-dependent 衰减门控）----------------

def gla_recurrent(q, k, v, g, *, scale=None):
    """GLA recurrent（causal，ground truth）：Sₜ = diag(exp(gₜ)) Sₜ₋₁ + kₜvₜᵀ。

    q/k/g: (B,H,S,K)，v: (B,H,S,V)，g 为 log forget gate（≤0）。scale 默认 1/√K，作用在 q（对齐 fla）。
    """
    B, H, S, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    state = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    outs = []
    for t in range(S):
        alpha = g[:, :, t].float().exp().unsqueeze(-1)               # (B,H,K,1) 衰减
        kv = k[:, :, t].float().unsqueeze(-1) * v[:, :, t].float().unsqueeze(-2)  # kₜvₜᵀ
        state = alpha * state + kv
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t].float() * scale, state))
    return torch.stack(outs, dim=2).to(v.dtype)
