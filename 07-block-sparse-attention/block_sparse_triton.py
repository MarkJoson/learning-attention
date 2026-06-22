"""07 块稀疏 · 深度优化版 —— 复用 08 NSA 的真实 triton kernel。

07 的"动态 top-k 块稀疏"正是 NSA 的 **selected 分支**。本文件不另造 kernel，而是直接复用
08 已提取并验证的 `nsa_triton.native_sparse_attend`（来源 lucidrains NSA，见 08/SOURCES.md），
把它配置成"纯块稀疏"：只算 selected + 对角块，关压缩、关滑窗。

语义（= NSA selected 的真实语义，已由实验 + test 验证）：
  每个 query 看 ——「对角块（自己所在块，块内 causal）**必看**」 + 「top-k 个**历史**块」。

  - "对角块必看"是真实动态块稀疏的通用设计：MoBA 也强制选当前块（见 moba_naive.py 里
    `gate[i块, i块] = inf`），NSA 的 kernel 用 Part1 专门处理对角块。比简要版 `block_sparse.py`
    的"纯 top-k"更贴近生产。
  - 复用关键：传给 kernel 的 `selected_block_indices` 必须**排除对角块**（对角块由 kernel 的
    Part1 自动算），否则对角块会被在线 softmax 重复累加。
  - 当 top-k 取满所有历史块时，退化为 full causal（`test_block_sparse_triton.py` 的 anchor 验证）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# 复用 08 的 NSA kernel（同源 lucidrains，已在 08 验证忠实 + 正确）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08-native-sparse-attention"))
from nsa_triton import native_sparse_attend  # noqa: E402


def select_topk_history_blocks(q, k, block_size, top_k):
    """每个 query 块选 top-k 个**历史**块（严格 j < i，排除对角块 i）。

    返回 token 级 (idx, fmask)，形状 (B,H,S,sel)：块内所有 query token 共享同一选块。
    历史不足 top-k 的早期块，多余槽位由 fmask=False 屏蔽。
    """
    B, H, S, D = q.shape
    nb = S // block_size
    sel = max(min(top_k, nb - 1), 1)  # 至少 1 个槽位（kernel 要求），不足由 fmask 屏蔽

    q_blk = q.view(B, H, nb, block_size, D).mean(3).float()  # 块代表（mean-pool）
    k_blk = k.view(B, H, nb, block_size, D).mean(3).float()
    imp = torch.matmul(q_blk, k_blk.transpose(-1, -2)) / (D ** 0.5)  # (B,H,nb,nb)

    # 只允许严格历史块（j < i）；对角与未来屏蔽为 -inf
    i_idx = torch.arange(nb, device=q.device)
    hist_mask = i_idx[:, None] > i_idx[None, :]  # (nb,nb) True 当 j < i
    imp = imp.masked_fill(~hist_mask, float("-inf"))

    topv, topi = imp.topk(sel, dim=-1)                 # (B,H,nb,sel)
    fmask_blk = topv > float("-inf")                   # 选到 -inf 的是无效（历史不足）
    idx_tok = topi.repeat_interleave(block_size, dim=2).to(torch.int32)   # (B,H,S,sel)
    fmask_tok = fmask_blk.repeat_interleave(block_size, dim=2)
    return idx_tok, fmask_tok


def block_sparse_attention_triton(q, k, v, block_size, top_k, *, causal=True):
    """用 08 NSA 的 selected kernel 算块稀疏：对角块必看 + top-k 历史块。

    q/k/v: (B,H,S,D)，须 fp16/bf16（NSA kernel 约束）；block_size 须为 16 的倍数。
    """
    assert causal, "复用的 NSA kernel 仅支持 causal"
    assert block_size % 16 == 0 and block_size >= 16, "NSA kernel 要求 block_size 是 16 的倍数"
    assert q.dtype in (torch.float16, torch.bfloat16), "NSA kernel 仅支持 fp16/bf16"
    idx, fmask = select_topk_history_blocks(q, k, block_size, top_k)
    return native_sparse_attend(
        q, k, v, block_size, idx, fmask,
        include_block_causal=True, return_sliding_window_out=False,
    )


def block_sparse_nsa_reference(q, k, v, block_size, top_k, *, causal=True):
    """匹配 NSA selected 语义的纯 PyTorch 参考（ground truth）：对角块 + top-k 历史块。

    与 `block_sparse_attention_triton` 用同一套选块逻辑，只把"算注意力"换成朴素 mask 版，
    用来验证 kernel 复用的数值正确性。
    """
    B, H, S, D = q.shape
    nb = S // block_size
    idx, fmask = select_topk_history_blocks(q, k, block_size, top_k)   # token 级历史选块

    # token 级 → 块级（块内共享，取每块首 token 即可）
    idx_blk = idx[:, :, ::block_size, :].long()            # (B,H,nb,sel)
    fmask_blk = fmask[:, :, ::block_size, :]

    # 块级允许矩阵：选中历史块。无效选块用 sentinel 槽(nb)接住后丢弃
    block_allow = torch.zeros(B, H, nb, nb + 1, dtype=torch.bool, device=q.device)
    safe_idx = torch.where(fmask_blk, idx_blk, nb)        # 无效 → 第 nb 槽（丢弃）
    block_allow.scatter_(-1, safe_idx, True)
    block_allow = block_allow[..., :nb]
    # 对角块必看
    diag = torch.arange(nb, device=q.device)
    block_allow[:, :, diag, diag] = True

    # 展开 token 级 + causal
    token_allow = block_allow.repeat_interleave(block_size, 2).repeat_interleave(block_size, 3)
    token_allow &= torch.ones(S, S, device=q.device, dtype=torch.bool).tril()

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (D ** 0.5)
    scores = scores.masked_fill(~token_allow, float("-inf"))
    return torch.matmul(scores.softmax(dim=-1), v.float()).to(q.dtype)
