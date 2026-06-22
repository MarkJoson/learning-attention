"""01 vanilla attention —— 标准注意力 + online softmax 分块版（纯 PyTorch，教学用）。

本文件是理解 FlashAttention 的"地基"，提供两种实现：

1. `naive_attention`（复用 common）：一次性构造完整的 (Sq, Sk) 注意力矩阵。
   显存 O(Sq·Sk)，长序列会爆显存，但和数学公式一一对应，作为 ground truth。

2. `online_softmax_attention`：把 K/V 沿序列分块，用 **online softmax** 增量更新输出，
   全程**不显式构造完整注意力矩阵**，显存降到 O(Sq·D)。这正是 FlashAttention kernel
   的算法骨架 —— 这里用纯 PyTorch 双重循环写出来，便于逐行对照论文公式。

online softmax 的核心：维护三个 running 量（逐 query 行）
    m  —— 到目前为止见过的最大 score（running max）
    l  —— 归一化分母 Σ exp(score - m)（running sum）
    acc —— 未归一化的输出 Σ exp(score - m)·V（running output）
每来一个新的 key 块，先把 m 更新为 m_new，再用修正因子 exp(m_old - m_new) 把旧的
l、acc "缩放"到新基准下，最后并入新块的贡献。遍历完所有 key 块后 O = acc / l。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.reference import naive_attention, repeat_kv  # noqa: E402

__all__ = ["naive_attention", "online_softmax_attention"]


def online_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    block_q: int = 128,
    block_k: int = 64,
) -> torch.Tensor:
    """FlashAttention 风格的 online softmax，纯 PyTorch 分块实现（教学版）。

    数值上等价于 `naive_attention`，但峰值显存只与块大小有关，与序列长度平方无关。
    全程在 float32 上累加，最后转回输入 dtype。

    形状：q (B,Hq,Sq,D)，k/v (B,Hkv,Sk,D)，Hkv<Hq 时按 GQA 展开。
    """
    B, Hq, Sq, D = q.shape
    Hkv = k.shape[1]
    Sk = k.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    if Hkv != Hq:
        assert Hq % Hkv == 0, f"Hq={Hq} 必须能被 Hkv={Hkv} 整除"
        k = repeat_kv(k, Hq // Hkv)
        v = repeat_kv(v, Hq // Hkv)

    qf, kf, vf = q.float(), k.float(), v.float()
    o = torch.zeros((B, Hq, Sq, D), dtype=torch.float32, device=q.device)

    # causal 下 query 末端对齐 key 末端：query 局部位置 i 的全局 key 坐标要加上该偏移
    align = Sk - Sq

    for i in range(0, Sq, block_q):
        qi = qf[:, :, i : i + block_q, :]  # (B,H,bq,D)
        bq = qi.shape[2]

        # 三个 running 量，逐 query 行维护
        m = torch.full((B, Hq, bq), float("-inf"), device=q.device)
        l = torch.zeros((B, Hq, bq), device=q.device)
        acc = torch.zeros((B, Hq, bq, D), device=q.device)

        # query 块覆盖的全局行坐标（用于 causal 跳过 / 掩码）
        q_max_pos = i + bq - 1 + align

        for j in range(0, Sk, block_k):
            # causal 优化：整块落在上三角（所有 key 都在所有 query 之后）则跳过
            if causal and j > q_max_pos:
                continue

            kj = kf[:, :, j : j + block_k, :]  # (B,H,bk,D)
            vj = vf[:, :, j : j + block_k, :]
            bk = kj.shape[2]

            s = torch.matmul(qi, kj.transpose(-1, -2)) * sm_scale  # (B,H,bq,bk)

            if causal:
                q_idx = (torch.arange(bq, device=q.device) + i + align)[:, None]
                k_idx = (torch.arange(bk, device=q.device) + j)[None, :]
                s = s.masked_fill(k_idx > q_idx, float("-inf"))

            # —— online softmax 增量更新 ——
            m_new = torch.maximum(m, s.amax(dim=-1))  # (B,H,bq)
            p = torch.exp(s - m_new[..., None])  # (B,H,bq,bk) 当前块未归一化权重
            corr = torch.exp(m - m_new)  # (B,H,bq) 把旧累积修正到新基准
            l = l * corr + p.sum(dim=-1)
            acc = acc * corr[..., None] + torch.matmul(p, vj)
            m = m_new

        o[:, :, i : i + bq, :] = acc / l[..., None]

    return o.to(q.dtype)
