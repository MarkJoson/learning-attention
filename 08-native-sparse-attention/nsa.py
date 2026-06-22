"""08 Native Sparse Attention (NSA) —— DeepSeek 的"三分支稀疏注意力"。

07 章的块稀疏只做了一件事：top-k 选块。NSA 更进一步，**三条稀疏分支并行**，再用学习的门控
把它们融合：

  1. **compressed（压缩）**：把 K/V 按块压缩成少量"压缩 token"，query 去 attend 全部压缩 token
     —— 一种粗粒度的全局视野。顺带，压缩注意力的分数还**直接拿来给分支 2 选块**。
  2. **selected（选择）**：用上面的压缩分数，为每个 query 选 top-k 个原始块，只 attend 选中块
     —— 细粒度地补回重要细节（这就是 07 的块稀疏，只是选块依据来自压缩分支）。
  3. **sliding window（滑窗）**：attend 最近的窗口 —— 保证局部性（这就是 04 的滑窗）。

三条分支的输出由一个从 query 算出的**门控**（3 个权重）加权合并。

本文件用清晰的纯 PyTorch 实现这套架构（三分支均为 mask 版，作为 ground truth）。NSA 真正高性能的
稀疏 Triton kernel（lucidrains 的实现近 2000 行、依赖众多）不在此提取——这一章像 06-MLA 一样
**聚焦架构**，把"三分支 + 门控"讲透。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class NativeSparseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        block_size: int = 32,        # 压缩 / 选择共用的块大小
        num_selected_blocks: int = 4,
        sliding_window_size: int = 64,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_selected = num_selected_blocks
        self.window = sliding_window_size
        self.scale = head_dim ** -0.5

        inner = n_heads * head_dim
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_gates = nn.Linear(dim, n_heads * 3, bias=False)   # 每个 head 三条分支的门控
        self.to_out = nn.Linear(inner, dim, bias=False)

    # ---------- 三条分支 ----------
    def compressed_branch(self, q, k, v):
        """把 K/V 按块 mean-pool 成压缩 token，query attend 全部压缩 token。

        返回 (out, blk_sim)：out 是该分支输出；blk_sim 是 query 对每个块的分数，供 selected 分支选块。
        """
        B, H, S, D = q.shape
        bs, nb = self.block_size, S // self.block_size
        k_cmp = k.view(B, H, nb, bs, D).mean(dim=3)              # (B,H,nb,D) 块代表
        v_cmp = v.view(B, H, nb, bs, D).mean(dim=3)

        sim = torch.matmul(q.float(), k_cmp.float().transpose(-1, -2)) * self.scale  # (B,H,S,nb)
        # causal：query i 只能看到"整块都在 i 及之前"的压缩块（块末位置 <= i）
        q_pos = torch.arange(S, device=q.device)[:, None]
        blk_last = (torch.arange(nb, device=q.device) * bs + bs - 1)[None, :]
        sim = sim.masked_fill(blk_last > q_pos, float("-inf"))
        # 早期 token（还不足一个完整压缩块）没有可见压缩块，整行 -inf → softmax 为 nan；
        # 这里置 0（这些 token 由 sliding 分支兜底），避免 nan 向后传播。
        attn = torch.softmax(sim, dim=-1).nan_to_num(0.0)
        out = torch.matmul(attn, v_cmp.float())  # (B,H,S,D)
        return out.to(q.dtype), sim

    def selected_branch(self, q, k, v, blk_sim):
        """用压缩分数 blk_sim 为每个 query 选 top-k 块，只 attend 选中块（块稀疏）。"""
        B, H, S, D = q.shape
        bs, nb = self.block_size, S // self.block_size
        k_eff = min(self.num_selected, nb)
        topk_idx = blk_sim.topk(k_eff, dim=-1).indices               # (B,H,S,k_eff) 每 token 选的块
        block_mask = torch.zeros(B, H, S, nb, dtype=torch.bool, device=q.device)
        block_mask.scatter_(-1, topk_idx, True)
        token_mask = block_mask.repeat_interleave(bs, dim=-1)        # (B,H,S,S)
        token_mask &= torch.ones(S, S, dtype=torch.bool, device=q.device).tril()

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~token_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)  # 早期 token 可能选不到有效块 → 置 0
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)

    def sliding_branch(self, q, k, v):
        """只 attend 最近 window 个 token（局部）。"""
        B, H, S, D = q.shape
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        q_pos = torch.arange(S, device=q.device)[:, None]
        k_pos = torch.arange(S, device=q.device)[None, :]
        mask = (k_pos <= q_pos) & (k_pos > q_pos - self.window)
        scores = scores.masked_fill(~mask, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), v.float())
        return out.to(q.dtype)

    # ---------- 组合 ----------
    def forward(self, x, *, return_branches=False):
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.to_qkv(x).view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)  # (3,B,H,S,D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out_cmp, blk_sim = self.compressed_branch(q, k, v)
        out_sel = self.selected_branch(q, k, v, blk_sim)
        out_slid = self.sliding_branch(q, k, v)

        # 门控：从 query 算每个 head 的三分支权重
        gates = self.to_gates(x).view(B, S, H, 3).softmax(dim=-1).permute(0, 2, 1, 3)  # (B,H,S,3)
        out = (gates[..., 0:1] * out_cmp
               + gates[..., 1:2] * out_sel
               + gates[..., 2:3] * out_slid)                                # (B,H,S,D)

        out = out.permute(0, 2, 1, 3).reshape(B, S, H * D)
        out = self.to_out(out)
        if return_branches:
            return out, dict(compressed=out_cmp, selected=out_sel, sliding=out_slid, gates=gates)
        return out
