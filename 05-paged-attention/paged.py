"""Paged Attention —— KV cache 的分页内存管理。

paged attention 的精髓**不在注意力计算**（那还是普通的 QK^T·softmax·V），而在
**KV cache 怎么在显存里摆放**。本文件用一个最小的 `PagedKVCache` 把这套机制讲清楚，
真正的高效 decode kernel 复用 `paged_decode_triton.py`（提取自 lightllm）。

核心思想（借鉴操作系统的虚拟内存分页）：
  - 把 KV cache 切成固定大小的物理 **block（页）**，放进一个统一的池子；
  - 每个序列维护一张 **block table**，记录"我的 token 存在哪些物理 block"；
  - 逻辑位置 → 物理 slot 的映射由 block table 完成，于是序列在物理上**可以不连续**，
    显存按需分配、几乎零浪费（没有"为每个序列预留最大长度"的碎片）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paged_decode_triton import gqa_decode_attention_fwd  # noqa: E402

__all__ = ["PagedKVCache", "paged_decode_attention"]


class PagedKVCache:
    """最小可用的分页 KV cache。

    物理池形如 (num_blocks * block_size, num_kv_heads, head_dim)，即一共 num_blocks 个
    block、每个 block 容纳 block_size 个 token。空闲 block 用一个 free list 管理。
    """

    def __init__(self, num_blocks, block_size, num_kv_heads, head_dim,
                 dtype=torch.float16, device="cuda"):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.k_cache = torch.zeros(num_blocks * block_size, num_kv_heads, head_dim,
                                   dtype=dtype, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.device = device
        self._free = list(range(num_blocks))     # 空闲物理 block 编号
        self.block_table: dict[int, list[int]] = {}   # req_id → [物理 block, ...]
        self.length: dict[int, int] = {}              # req_id → 已存 token 数

    def append(self, req_id: int, k_tok: torch.Tensor, v_tok: torch.Tensor) -> None:
        """给序列 req_id 追加一个 token 的 K/V（形状均为 (num_kv_heads, head_dim)）。

        当当前 block 满了，就从 free list 取一个新 block——这正是"按需分配"。
        """
        if req_id not in self.block_table:
            self.block_table[req_id] = []
            self.length[req_id] = 0
        pos = self.length[req_id]
        if pos % self.block_size == 0:          # 当前没有空位 → 申请新 block
            assert self._free, "物理 block 池已耗尽"
            self.block_table[req_id].append(self._free.pop(0))
        slot = self._slot(req_id, pos)
        self.k_cache[slot] = k_tok
        self.v_cache[slot] = v_tok
        self.length[req_id] += 1

    def _slot(self, req_id: int, pos: int) -> int:
        """逻辑位置 pos → 物理 slot：先按 block_size 定位到哪个 block，再加上块内偏移。"""
        block = self.block_table[req_id][pos // self.block_size]
        return block * self.block_size + pos % self.block_size

    def build_req_to_tokens(self, req_ids: list[int]) -> torch.Tensor:
        """把 block table 展开成 kernel 需要的 (num_req, max_len) 的 token→物理slot 映射。"""
        max_len = max(self.length[r] for r in req_ids)
        rtt = torch.zeros(len(req_ids), max_len, dtype=torch.int32, device=self.device)
        for i, r in enumerate(req_ids):
            for pos in range(self.length[r]):
                rtt[i, pos] = self._slot(r, pos)
        return rtt

    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)


def paged_decode_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    req_ids: list[int],
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """对一批序列做一步 decode 注意力：每个序列用 1 个 query，attend 到它在 cache 里的全部历史。

    q: (num_req, num_q_heads, head_dim)，与 req_ids 一一对应。
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    rtt = cache.build_req_to_tokens(req_ids)
    b_req_idx = torch.arange(len(req_ids), dtype=torch.int32, device=cache.device)
    b_seq_len = torch.tensor([cache.length[r] for r in req_ids],
                             dtype=torch.int32, device=cache.device)
    o = torch.empty_like(q)
    # 注：kernel 内部已写死 sm_scale=1/sqrt(d)，与默认一致；此处保留 sm_scale 仅为接口完整。
    gqa_decode_attention_fwd(q, cache.k_cache, cache.v_cache, o, rtt, b_req_idx, b_seq_len)
    return o
