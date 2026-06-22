"""薄适配层（本仓库自写）—— 复现 fla GLA kernel 依赖的少量框架辅助符号，使拷贝来的三个
kernel 文件（_fla_gla_chunk / _fla_chunk_h / _fla_cumsum，来源见 SOURCES.md）**脱离 fla 包独立运行**。

fla 的 GLA kernel 计算逻辑本身只用到 torch/triton，框架侧依赖都是这类"工具/适配"符号：
  - exp2 / RCP_LN2          —— 数学（libdevice exp2 + 换底常量），直接复现；
  - fla_cache_autotune      —— fla 扩展的 autotune（带磁盘 config 缓存），签名与 triton.autotune
                               一致 → 退化为标准 triton.autotune（放弃缓存，不影响计算）；
  - check_shared_mem        —— 按 GPU shared memory 选 autotune 的 block 候选大小；这里用 torch
                               查实际 smem 复现（4090≈100KB），即便不精确 triton.autotune 也会
                               跳过 OOM config 兜底；
  - input_guard             —— 确保输入 contiguous（fla 原版还设 device，这里简化）；
  - autotune_cache_kwargs   —— 磁盘缓存开关，本地置空；
  - prepare_chunk_indices/offsets/_segmented_arange/prepare_lens —— 变长（cu_seqlens / sequence
    packing）的分块索引，**拷自 fla `ops/utils/index.py`（纯 torch，计算逻辑未改）**，支持变长。
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

RCP_LN2 = 1.4426950408889634          # 1/ln(2)
autotune_cache_kwargs: dict = {}      # fla 磁盘 config 缓存开关；本地不缓存


@triton.jit
def exp2(x):
    return tldevice.exp2(x.to(tl.float32))


def fla_cache_autotune(configs, key=None, **_ignored):
    """fla 的 autotune（扩展 triton.autotune + 从 fla/configs 读缓存）→ 标准 triton.autotune。
    用法 `@fla_cache_autotune(configs=[...], key=[...], **autotune_cache_kwargs)` 与 triton 一致。"""
    return triton.autotune(configs=configs, key=key or [])


def _device_max_smem(idx: int = 0) -> int:
    p = torch.cuda.get_device_properties(idx)
    for attr in ("shared_memory_per_block_optin", "max_shared_memory_per_block", "sharedMemPerBlockOptin"):
        if hasattr(p, attr):
            return int(getattr(p, attr))
    return 49152  # 保守 48KB


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """当前 GPU 的最大 shared memory 是否达到某架构水平（决定 autotune 的 block 候选）。
    4090(sm_89,~100KB)：check_shared_mem()→True（BK 候选含 64）、check_shared_mem('ampere')→False。"""
    thresh = {"none": 0, "ampere": 166912, "hopper": 232448, "blackwell": 232448}
    return _device_max_smem(tensor_idx) >= thresh.get(arch, 0)


def input_guard(fn=None, *, no_guard_contiguous=False):
    """确保所有输入张量 contiguous（fla 原版还做 device 绑定，这里简化为 contiguous-only）。"""
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            return f(*args, **kwargs)
        return wrapper
    return deco(fn) if fn is not None else deco


# --- 变长（varlen / cu_seqlens）分块索引：拷自 fla ops/utils/index.py（纯 torch，逻辑未改）---
# sequence packing 把多条变长序列拼成一个 batch（如 [seqA|seqB|seqC]），cu_seqlens 记录各序列边界；
# 下面的函数据此算出"每个 chunk 属于哪条序列、是序列内第几个 chunk"，让 kernel 不跨序列错误 attend。

def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """各序列长度：cu_seqlens=[0,3,7] → [3,4]。"""
    return torch.diff(cu_seqlens)


def _segmented_arange(counts: torch.Tensor):
    """per-segment counts → (seg_id, intra_idx)：counts=[2,3] → seg_id=[0,0,1,1,1], intra=[0,1,0,1,2]。"""
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype), counts)
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    intra_idx = torch.arange(seg_id.shape[0], device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return seg_id, intra_idx


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int, cu_seqlens_cpu=None) -> torch.Tensor:
    """每个 chunk 的 (序列号, 序列内 chunk 序号)，供 kernel 在 packed 多序列里正确定位。"""
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """各序列的 chunk 起始偏移（前缀和），供块间状态在 packed batch 里按序列对齐。"""
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)
