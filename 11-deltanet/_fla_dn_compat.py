"""薄适配层（本仓库自写）—— 让拷贝来的 DeltaNet kernel 脱离 fla 包独立运行。

比 10 章的 compat 多几样（DeltaNet 的 kernel 用到）：
  - dispatch        —— fla 的后端分派装饰器（triton / TileLang / context-parallel）→ **no-op**：
                       直接用被装饰的 triton 实现。这样不 import fla.ops.backends，CP/TileLang/attn
                       后端全部不进依赖闭包（把完整解耦的闭包从 ~27 个文件收敛到 8 个 triton 文件）。
  - autocast_*      —— autograd Function 的 fwd/bwd 的 amp 包装；
  - IS_* / TRITON_* —— GPU 能力标志（4090 = Ada sm_89：非 Hopper、非 AMD、无 TMA）；
  - safe_dot        —— tl.dot 包装（非 Blackwell 直接 tl.dot）；
  - make_tensor_descriptor —— TMA descriptor（triton 3.7 原生；4090 IS_TMA_SUPPORTED=False 不会真用）。
其余（exp2/autotune/check_shared_mem/input_guard/varlen 索引）与 10 章相同。
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice  # noqa: F401

RCP_LN2 = 1.4426950408889634
autotune_cache_kwargs: dict = {}

# ---- GPU 能力标志（4090 = Ada sm_89）----
IS_NVIDIA_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9  # False
IS_AMD = torch.version.hip is not None                                                       # False
IS_TMA_SUPPORTED = False                          # TMA 是 Hopper+；4090 不支持
IS_NVIDIA_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10  # False
IS_GATHER_SUPPORTED = True                        # triton 3.7 支持 tl.gather
TRITON_ABOVE_3_4_0 = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 4)      # True
USE_CUDA_GRAPH = False                            # fla autotune 是否用 cuda graph；本地关掉更稳

# ---- autograd Function 的 amp 包装 ----
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def safe_dot(a, b, allow_tf32: tl.constexpr = None):
    return tl.dot(a, b, allow_tf32=allow_tf32)


# TMA descriptor：triton 3.4+ 原生；否则 stub（4090 IS_TMA_SUPPORTED=False，不会真正调用）
make_tensor_descriptor = getattr(triton.language, "make_tensor_descriptor", None) \
    or getattr(triton.language, "_experimental_make_tensor_descriptor", None)
if make_tensor_descriptor is None:
    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        return None


def dispatch(operation: str):
    """fla 后端分派装饰器 → no-op（直接用被装饰的 triton 实现）。绕过 CP/TileLang 后端。"""
    def deco(fn):
        return fn
    return deco


def fla_cache_autotune(configs, key=None, **_ignored):
    """fla 的 autotune（带磁盘 config 缓存）→ 标准 triton.autotune。"""
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
    """确保输入张量 contiguous（fla 原版还做 device 绑定，这里简化）。"""
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            return f(*args, **kwargs)
        return wrapper
    return deco(fn) if fn is not None else deco


# ---- 变长（cu_seqlens / sequence packing）分块索引：拷自 fla ops/utils/index.py（纯 torch）----
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
