"""learning-attention 公共工具库。

- reference: 标准注意力的朴素参考实现（数值 ground truth）
- testing:   数值对比与测试张量构造
- benchmark: 延迟 / 显存 / TFLOP-s 基准
"""
from .benchmark import attention_flops, bench_ms, peak_memory_mb, tflops
from .reference import naive_attention, repeat_kv
from .testing import assert_close, error_report, make_qkv

__all__ = [
    "naive_attention",
    "repeat_kv",
    "assert_close",
    "error_report",
    "make_qkv",
    "bench_ms",
    "attention_flops",
    "tflops",
    "peak_memory_mb",
]
