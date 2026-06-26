"""薄适配层（本仓库自写）—— 让拷贝来的 Mamba2 SSD（fla `simple_gla`）kernel 脱离 fla 独立运行。

`simple_gla` 就是 Mamba2 的 **SSD**：head-wise 标量衰减门控 `g`（对比 GLA 的 per-channel elementwise）。
kernel 闭包（chunk + common.chunk_h / common.chunk_o + utils.cumsum）依赖一组 fla 工具符号，这里全部
复现；**no-op dispatch** 绕过 `chunk_o` 的后端分派。模块名用 `_fla_ssd_` 唯一前缀（与 10-13 章隔离，
避免全库 pytest 同名模块冲突）。比 KDA/GDN 简单：不依赖多卡 context-parallel，故无 cp stub。
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

RCP_LN2 = 1.4426950408889634
autotune_cache_kwargs: dict = {}

# ---- GPU 能力标志（4090 = Ada sm_89）----
IS_NVIDIA_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9   # False
TRITON_ABOVE_3_4_0 = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 4)        # True

autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


def dispatch(operation: str):
    """fla 后端分派 → no-op（直接用被装饰的 triton 实现）。绕过 CP/TileLang 后端。"""
    def deco(fn):
        return fn
    return deco


def fla_cache_autotune(configs, key=None, **_ignored):
    return triton.autotune(configs=configs, key=key or [])


def _device_max_smem(idx: int = 0) -> int:
    p = torch.cuda.get_device_properties(idx)
    for a in ("shared_memory_per_block_optin", "max_shared_memory_per_block", "sharedMemPerBlockOptin"):
        if hasattr(p, a):
            return int(getattr(p, a))
    return 49152


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    thresh = {"none": 0, "ampere": 166912, "hopper": 232448, "blackwell": 232448}
    return _device_max_smem(tensor_idx) >= thresh.get(arch, 0)


def input_guard(fn=None, *, no_guard_contiguous=False):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            return f(*args, **kwargs)
        return wrapper
    return deco(fn) if fn is not None else deco


# ---- 变长（cu_seqlens）分块索引：拷自 fla ops/utils/index.py（纯 torch）----
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.diff(cu_seqlens)


def _segmented_arange(counts: torch.Tensor):
    seg_id = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=counts.dtype), counts)
    seg_start = F.pad(counts.cumsum(0), (1, 0))[:-1]
    intra = torch.arange(seg_id.shape[0], device=counts.device, dtype=counts.dtype) - seg_start[seg_id]
    return seg_id, intra


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int, cu_seqlens_cpu=None) -> torch.Tensor:
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
    seg_id, intra = _segmented_arange(counts)
    return torch.stack([seg_id, intra], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


# chunk_local_cumsum 实为拷贝的 kernel（_fla_ssd_cumsum，门控的块内 prefix-sum）；某些 kernel 以
# `from fla.ops.utils import chunk_local_cumsum`（已重写为本 compat）取用，这里 re-export。
# _fla_ssd_cumsum 只依赖本 compat 上方已定义的符号，循环 import 安全。
from _fla_ssd_cumsum import chunk_local_cumsum  # noqa: E402
