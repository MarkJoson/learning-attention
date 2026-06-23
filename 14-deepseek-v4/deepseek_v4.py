"""14 DeepSeek V4 Hybrid Attention（CSA + HCA）—— 自写简要版，讲机制。

DeepSeek V4（2026-04）把长上下文注意力做成**两种压缩注意力交替**的混合架构。它是第 08 章 NSA
（compress + select + sliding window 三分支）的进化：核心仍是"先把 KV 压缩、再稀疏地用"，但两处升级——

1. **学习式 softmax 压缩**（不是 NSA 的卷积/均值池化）：每 m 个 token 压成 1 个压缩块，权重由一个
   per-coordinate softmax 决定（每个特征维度在 m 个位置上独立加权）。压缩后的 `C` **同时充当 K 和 V**
   （每块只缓存一个向量）。
2. **Lightning indexer**（轻量索引器）做稀疏选块：用 ReLU 打分（是排序、不是概率分布）选 top-k 个压缩块，
   FP4 精度算，极省。

两种注意力是同一个「压缩注意力」CompAttn(m, k) 的不同配置：

    CSA（Compressed Sparse Attention）：m=4  压缩 + indexer 选 top-k=1024  →  **稀疏**
    HCA（Heavily Compressed Attention）：m=128 压缩 + 不选（k=all）稠密      →  **极致压缩**

V4-Pro 61 层 ≈ 30 层 CSA + 31 层 HCA 交替。1M token 上下文下，单 token 推理 FLOP 仅 V3.2 的 27%、
KV cache 仅 10%。**注意：DSv4 没有 sliding window 分支**，省全靠"压缩 + 稀疏索引"。

> 来源：DeepSeek V4 技术报告 / vLLM DeepSeek-V4 / NVIDIA Blackwell 部署博客（见 SOURCES.md）。本文件是
> **自写的机制简化版**（讲清 compress→index→attend 三步），不是 vLLM 生产 kernel 的提取——DSv4 不在 fla，
> 生产 kernel 在 vLLM/官方，本章按用户约定"自写简要版 + 指向来源"。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ============================ 1. 学习式 softmax 压缩 ============================
def softmax_compress(x, m, z=None):
    """每 m 个 token 压成 1 个压缩块 —— per-coordinate softmax pooling（DSv4 压缩核心）。

    x: (B,H,T,D) 原始待压缩序列（既当 K 又当 V）；z: (B,H,T,D) 打分 logits（None → 退化为均值池化）。
    返回 (B,H,nb,D)，nb=T//m。**每个特征维度 d 在 m 个位置上独立做 softmax 加权**（学习式池化，
    比 NSA 的固定均值/卷积更灵活）。
    """
    B, H, T, D = x.shape
    nb = T // m
    xb = x[:, :, :nb * m].reshape(B, H, nb, m, D)
    if z is None:
        w = x.new_full((B, H, nb, m, D), 1.0 / m)          # 均匀权重 = 均值池化（退化基线）
    else:
        zb = z[:, :, :nb * m].reshape(B, H, nb, m, D)
        w = torch.softmax(zb, dim=3)                        # 沿 m 个位置、逐坐标独立 softmax
    return (w * xb).sum(dim=3)                              # (B,H,nb,D)


# ============================ 2. Lightning indexer（选块）============================
def lightning_indexer(q_idx, w_idx, k_comp, top_k):
    """轻量索引器：给每个 query 选 top-k 个压缩块。返回选中块索引 (B,Hq,T,k)。

    打分 I_{t,s} = Σ_h w_{t,h} · ReLU(q_{t,h} · k_comp_s)  —— **ReLU 是排序信号、不是概率**（无 softmax）。
    q_idx: (B,Hi,T,d_i) 索引查询；w_idx: (B,Hi,T) 各索引头权重；k_comp: (B,Hi,nb,d_i) 压缩块的索引键。
    块级 causal：position t 的 query 只能看 block s < t//m（m 由 nb 与 T 反推）。
    """
    B, Hi, T, _ = q_idx.shape
    nb = k_comp.shape[2]
    m = T // nb
    # per-head 打分后按 w 加权求和 → (B,T,nb)
    raw = torch.einsum("bhtd,bhsd->bhts", q_idx, k_comp).relu()       # ReLU 排序
    score = (w_idx.unsqueeze(-1) * raw).sum(dim=1)                    # (B,T,nb)
    # 块级 causal mask：block s 覆盖原始位置 [s*m, s*m+m)，query t 只能看 s*m+m-1 < t，即 s < t//m
    t_idx = torch.arange(T, device=q_idx.device)
    s_idx = torch.arange(nb, device=q_idx.device)
    causal = (s_idx[None, :] * m + m - 1) < t_idx[:, None]            # (T,nb) True=可见
    score = score.masked_fill(~causal[None], float("-inf"))
    k = min(top_k, nb)
    sel = score.topk(k, dim=-1).indices                              # (B,T,k)
    return sel, causal


# ============================ 3. 统一的压缩注意力 CompAttn ============================
def compressed_attention(q, k, v, m, *, top_k=None, z=None, q_idx=None, w_idx=None, k_idx=None, scale=None):
    """DeepSeek V4 的统一压缩注意力。CSA / HCA 都是它的特例（见下方 csa_attention / hca_attention）。

    q: (B,Hq,T,D) 查询；k,v: (B,Hkv,T,D)（共享 KV，MQA：Hkv 通常=1 或少量）。
    m: 压缩比（CSA=4，HCA=128）。top_k: 选多少压缩块（None=全用，即 HCA 的稠密分支）。
    z: 压缩打分 logits（None=均值池化）。q_idx/w_idx/k_idx: lightning indexer 的查询/头权重/压缩索引键
    （top_k 给定时必需）。返回 (B,Hq,T,D)。
    """
    B, Hq, T, D = q.shape
    if scale is None:
        scale = D ** -0.5
    # —— 压缩：C 同时充当 K 和 V（每块一个向量）——
    k_comp = softmax_compress(k, m, z)                               # (B,Hkv,nb,D)
    v_comp = softmax_compress(v, m, z)
    nb = k_comp.shape[2]
    # MQA：把共享 KV 广播到 Hq 个查询头
    k_comp = k_comp.expand(B, Hq, nb, D)
    v_comp = v_comp.expand(B, Hq, nb, D)
    # 块级 causal（query t 只能看 s < t//m）
    t_idx = torch.arange(T, device=q.device)
    s_idx = torch.arange(nb, device=q.device)
    causal = (s_idx[None, :] * m + m - 1) < t_idx[:, None]           # (T,nb)

    attn = torch.einsum("bhtd,bhsd->bhts", q, k_comp) * scale        # (B,Hq,T,nb)
    attn = attn.masked_fill(~causal[None, None], float("-inf"))

    if top_k is not None:
        # —— 稀疏：lightning indexer 选 top-k 压缩块，只在选中的块上算注意力（CSA 分支）——
        sel, _ = lightning_indexer(q_idx, w_idx, k_idx, top_k)        # (B,T,k)
        keep = torch.zeros(B, T, nb, dtype=torch.bool, device=q.device)
        keep.scatter_(2, sel, True)
        attn = attn.masked_fill(~keep[:, None], float("-inf"))

    # 某些 query（靠前、无完整历史压缩块）整行 -inf。直接把 logits 置 0 会让 softmax 变成对**所有块
    # （含未来）均匀加权 → 泄漏未来。正确做法：softmax 前置 0 避免 nan，softmax 后把这些 query 的**输出**
    # 整体置 0（它们没有可用的压缩上下文）。简化版的取舍：DSv4 无 sliding window，最前面几个 token 暂无
    # 压缩历史；真实模型靠层间交替 + 局部细节弥补，这里聚焦 compress→index→attend 主干。
    all_masked = torch.isinf(attn).all(dim=-1, keepdim=True)          # (B,Hq,T,1)
    p = torch.softmax(attn.masked_fill(all_masked, 0.0), dim=-1)
    out = torch.einsum("bhts,bhsd->bhtd", p, v_comp)
    return out.masked_fill(all_masked, 0.0)                          # 无上下文的 query 输出置 0


def csa_attention(q, k, v, *, m=4, top_k=1024, z=None, q_idx=None, w_idx=None, k_idx=None, scale=None):
    """CSA（Compressed Sparse Attention）：m=4 压缩 + lightning indexer 选 top-k 稀疏。DSv4 的稀疏层。"""
    return compressed_attention(q, k, v, m, top_k=top_k, z=z,
                                q_idx=q_idx, w_idx=w_idx, k_idx=k_idx, scale=scale)


def hca_attention(q, k, v, *, m=128, z=None, scale=None):
    """HCA（Heavily Compressed Attention）：m=128 极致压缩 + 稠密（不选块）。DSv4 的稠密层。"""
    return compressed_attention(q, k, v, m, top_k=None, z=z, scale=scale)
