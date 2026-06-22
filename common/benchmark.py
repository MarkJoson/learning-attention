"""性能基准工具：延迟、显存、TFLOP/s。

优先使用 triton.testing.do_bench（自动 warmup、取中位数、对 L2 cache 友好），
没有 triton 时退化到手动 CUDA event 计时。
"""
from __future__ import annotations

from typing import Callable

import torch

try:
    import triton

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


def bench_ms(fn: Callable[[], object], *, warmup: int = 25, rep: int = 100) -> float:
    """返回 fn() 的中位数耗时（毫秒）。fn 应为无参可调用（用 lambda 闭包传参）。"""
    if _HAS_TRITON:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # 退化路径：手动 CUDA event 计时
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def attention_flops(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    *,
    causal: bool = False,
    mode: str = "fwd",
) -> float:
    """标准注意力的 FLOPs 估算。

    主体是两个矩阵乘：QK^T 与 PV，各 2*B*H*Sq*Sk*D FLOPs（乘加各算一次 = ×2）。
    causal 平均只算下三角，约 ×0.5。
    bwd 的计算量约为 fwd 的 2.5 倍，fwd+bwd 约 3.5 倍（与 FlashAttention 论文 /
    Triton 教程口径一致）。
    """
    f = 2.0 * 2.0 * batch * heads * seqlen_q * seqlen_k * head_dim  # 两个 matmul
    if causal:
        f *= 0.5
    factor = {"fwd": 1.0, "bwd": 2.5, "fwd_bwd": 3.5}[mode]
    return factor * f


def tflops(flops: float, ms: float) -> float:
    """由 FLOPs 和毫秒耗时换算 TFLOP/s。"""
    return flops / (ms * 1e-3) / 1e12


def peak_memory_mb(reset: bool = True) -> float:
    """返回当前 CUDA 峰值显存（MB）；reset=True 时顺便重置统计。"""
    mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return mb
