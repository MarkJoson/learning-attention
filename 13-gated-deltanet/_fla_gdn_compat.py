"""薄适配层（本仓库自写）—— 让拷贝来的 GDN kernel 脱离 fla 包独立运行。

GDN（Kimi Delta Attention）的 kernel 比 DeltaNet 更复杂，依赖更多工具符号。本文件在 11 章
compat 的基础上多提供：
  - exp / gather / softplus —— 数学/门控用的 triton 算子；
  - IS_TF32_SUPPORTED / tensor_cache —— GPU 标志 / memoize（no-op）；
  - **cp stub** —— context-parallel（多卡）的 FLACPContext + 预处理函数。GDN 的 chunk_fwd/bwd 静态
    import 它们，但单卡 `cp_context=None` 不会调用；这里 stub 成占位，使闭包不引入 cp 多卡代码。
模块名用 _fla_kda_ 唯一前缀（与 10/11 的 _fla_* 隔离，避免全库测试时同名模块冲突）。
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
IS_NVIDIA_HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9   # False
IS_AMD = torch.version.hip is not None                                                        # False
IS_TMA_SUPPORTED = False
IS_NVIDIA_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10  # False
IS_GATHER_SUPPORTED = True
IS_TF32_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8   # True (Ada)
TRITON_ABOVE_3_4_0 = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 4)       # True
USE_CUDA_GRAPH = False

autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type="cuda")


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def softplus(x):
    # 数值稳定的 softplus：x 大时 ≈ x，避免 exp 溢出
    x = x.to(tl.float32)
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.jit
def safe_dot(a, b, allow_tf32: tl.constexpr = None):
    return tl.dot(a, b, allow_tf32=allow_tf32)


gather = tl.gather  # triton 3.7 支持

make_tensor_descriptor = getattr(triton.language, "make_tensor_descriptor", None) \
    or getattr(triton.language, "_experimental_make_tensor_descriptor", None)
if make_tensor_descriptor is None:
    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        return None


def dispatch(operation: str):
    """fla 后端分派 → no-op（直接用被装饰的 triton 实现）。绕过 CP/TileLang 后端。"""
    def deco(fn):
        return fn
    return deco


def tensor_cache(fn):
    """fla 的张量结果 memoize（性能优化，功能无关）→ no-op。"""
    return fn


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


# ---- context-parallel（多卡）stub —— 单卡 cp_context=None 不会调用 ----
class FLACPContext:  # noqa: D401
    """cp 多卡上下文占位（单卡用不到；GDN 仅用它做类型注解 + None 判断）。"""
    pass


def _cp_unavailable(*args, **kwargs):
    raise NotImplementedError("context-parallel（多卡）未移植；单卡请用 cp_context=None。")


chunk_gated_delta_rule_fwd_h_pre_process = _cp_unavailable
chunk_gated_delta_rule_bwd_dhu_pre_process = _cp_unavailable
compress_h0 = _cp_unavailable
expand_h0 = _cp_unavailable


# chunk_local_cumsum 实为拷贝的 kernel（_fla_gdn_cumsum，门控的块内 prefix-sum）。某些 kernel 以
# `from fla.ops.utils import chunk_local_cumsum`（已重写为本 compat）取用，这里 re-export。
# _fla_gdn_cumsum 只依赖本 compat 上方已定义的符号（fla_cache_autotune 等），循环 import 安全。


# chunk_local_cumsum 实为拷贝的 kernel（_fla_gdn_cumsum）；re-export 供其他 kernel 取用。
from _fla_gdn_cumsum import chunk_local_cumsum  # noqa: E402
