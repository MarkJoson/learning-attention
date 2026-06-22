"""07 块稀疏注意力（入门）—— 稀疏注意力的地基。

标准注意力让每个 query 看**所有** key，计算量 O(S²)。块稀疏的想法是：把 key 切成块，
为每个 query 块只挑出**最相关的 top-k 个 key 块**来算，其余块直接不看。这正是 NSA 等方法
"selected（选择）"分支的核心。

关键三步：
  1. **分块 + pooled 代表**：把 query/key 按块取均值，得到每块的"代表向量"；
  2. **块级重要性 + top-k 选块**：用代表向量两两打分，每个 query 块选出最相关的 top-k 个 key 块；
  3. **稀疏计算**：只对选中的块算注意力。

本文件给两条等价路径：
  - `block_sparse_reference`：块稀疏 mask + full 注意力（ground truth，清楚地"看见"哪些块被选）；
  - `block_sparse_attention`：gather 出选中的块、**只对它们**算注意力（真正省计算）。
两者数值一致（`test_block_sparse.py` 验证）。真正高性能的稀疏 kernel（如 NSA 的 Triton 实现）
是下一章 08 的主题。
"""
from __future__ import annotations

import math

import torch


def _pool_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """(B,H,S,D) 按 block_size 分块、每块取均值 → (B,H,nb,D) 的块代表。"""
    B, H, S, D = x.shape
    nb = S // block_size
    return x.view(B, H, nb, block_size, D).mean(dim=3)


def select_topk_blocks(q, k, block_size, top_k, *, causal=True):
    """为每个 query 块选出 top-k 个最相关的 key 块。

    返回 topk_idx: (B, H, nb, k_eff)，每个 query 块选中的 key 块索引（k_eff=min(top_k, 可选块数)）。
    """
    B, H, S, D = q.shape
    nb = S // block_size
    q_blk = _pool_blocks(q, block_size)                       # (B,H,nb,D)
    k_blk = _pool_blocks(k, block_size)
    imp = torch.matmul(q_blk.float(), k_blk.float().transpose(-1, -2)) / math.sqrt(D)  # (B,H,nb,nb)
    if causal:
        tri = torch.ones(nb, nb, device=q.device, dtype=torch.bool).tril()
        imp = imp.masked_fill(~tri, float("-inf"))            # query 块只能选 <= 自己的 key 块
    k_eff = min(top_k, nb)
    return imp.topk(k_eff, dim=-1).indices                    # (B,H,nb,k_eff)


def _block_mask_from_idx(topk_idx, nb):
    """topk_idx (B,H,nb,k) → block_mask (B,H,nb,nb) bool（块 i 是否选了块 j）。"""
    B, H, nbq, _ = topk_idx.shape
    bm = torch.zeros(B, H, nbq, nb, device=topk_idx.device, dtype=torch.bool)
    return bm.scatter(-1, topk_idx, True)


def block_sparse_reference(q, k, v, block_size, top_k, *, causal=True, sm_scale=None):
    """ground truth：把块选择展开成 token 级 mask，再做一次普通（full）注意力。

    不省计算，但最直观——能清楚看到"哪些块被选中、哪些被屏蔽"。
    """
    B, H, S, D = q.shape
    nb = S // block_size
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    topk_idx = select_topk_blocks(q, k, block_size, top_k, causal=causal)
    block_mask = _block_mask_from_idx(topk_idx, nb)           # (B,H,nb,nb)
    # 块级 mask 展开到 token 级
    token_mask = block_mask.repeat_interleave(block_size, 2).repeat_interleave(block_size, 3)
    if causal:
        token_mask &= torch.ones(S, S, device=q.device, dtype=torch.bool).tril()

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    scores = scores.masked_fill(~token_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v.float()).to(q.dtype)


def block_sparse_attention(q, k, v, block_size, top_k, *, causal=True, sm_scale=None):
    """省计算的实现：gather 出每个 query 块选中的 key 块，**只对它们**算注意力。

    计算量从 O(nb·nb) 个块降到 O(nb·top_k) 个块。全程向量化（无 Python 循环）。
    """
    B, H, S, D = q.shape
    nb = S // block_size
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    topk_idx = select_topk_blocks(q, k, block_size, top_k, causal=causal)  # (B,H,nb,kk)
    kk = topk_idx.shape[-1]

    k_blocks = k.view(B, H, nb, block_size, D)
    v_blocks = v.view(B, H, nb, block_size, D)
    # gather 每个 query 块选中的 key/value 块 → (B,H,nb,kk,block_size,D)
    idx = topk_idx[..., None, None].expand(B, H, nb, kk, block_size, D)
    k_sel = torch.gather(k_blocks[:, :, None].expand(B, H, nb, nb, block_size, D), 3, idx)
    v_sel = torch.gather(v_blocks[:, :, None].expand(B, H, nb, nb, block_size, D), 3, idx)
    k_sel = k_sel.reshape(B, H, nb, kk * block_size, D)
    v_sel = v_sel.reshape(B, H, nb, kk * block_size, D)

    q_blk = q.view(B, H, nb, block_size, D)
    # 每个 query 块只和自己选中的 kk 个块算注意力
    scores = torch.einsum("bhnqd,bhnkd->bhnqk", q_blk.float(), k_sel.float()) * sm_scale

    if causal:
        # 选中块在序列里的全局位置由 topk_idx 决定，据此构造 causal 掩码
        q_pos = (torch.arange(block_size, device=q.device)[None, :]
                 + torch.arange(nb, device=q.device)[:, None] * block_size)        # (nb, bs)
        k_pos = (topk_idx[..., None] * block_size
                 + torch.arange(block_size, device=q.device))                       # (B,H,nb,kk,bs)
        k_pos = k_pos.reshape(B, H, nb, kk * block_size)                            # (B,H,nb,kk*bs)
        causal_mask = q_pos[None, None, :, :, None] >= k_pos[:, :, :, None, :]      # (B,H,nb,bs,kk*bs)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)                                            # (B,H,nb,bs,kk*bs)
    out = torch.einsum("bhnqk,bhnkd->bhnqd", attn, v_sel.float())                   # (B,H,nb,bs,D)
    return out.reshape(B, H, S, D).to(q.dtype)
