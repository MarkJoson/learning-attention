"""01-vanilla-attention 基准：naive（完整矩阵） vs online（分块） vs SDPA（融合 kernel）。

直观展示三件事：
  - naive 的注意力矩阵占 O(S^2) 显存，长序列直接 OOM；
  - online softmax 把显存降到 O(S·D)，但纯 PyTorch 双重循环延迟极高（Python 开销）；
  - SDPA（PyTorch 内置融合 kernel）又快又省显存。

结论：光有 online softmax *算法* 还不够，还需要把它**融合进一个 GPU kernel**，
避免反复读写 HBM 和 Python 循环开销 —— 这正是 FlashAttention 要做的事（见 02）。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    attention_flops,
    bench_ms,
    make_qkv,
    naive_attention,
    peak_memory_mb,
    tflops,
)
from vanilla import online_softmax_attention  # noqa: E402


def _measure(fn, flops, *, warmup, rep):
    """返回 (ms, MB, TFLOPs)；OOM 时返回 nan。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        ms = bench_ms(fn, warmup=warmup, rep=rep)
        mb = peak_memory_mb()
        return ms, mb, tflops(flops, ms)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), float("nan"), float("nan")


def run(seqlens=(512, 1024, 2048, 4096), batch=4, heads=16, head_dim=64,
        causal=True, dtype=torch.float16):
    dev = torch.cuda.get_device_name(0)
    print(f"# device={dev}  B={batch} H={heads} D={head_dim} causal={causal} dtype={dtype}")
    print(f"{'seqlen':>7} | {'naive ms':>9} {'MB':>7} | {'online ms':>10} {'MB':>7} | "
          f"{'sdpa ms':>8} {'MB':>7} {'TFLOP/s':>8}")
    print("-" * 78)
    for S in seqlens:
        q, k, v = make_qkv(batch, heads, S, head_dim, dtype=dtype)
        flops = attention_flops(batch, heads, S, S, head_dim, causal=causal, mode="fwd")

        ms_n, mb_n, _ = _measure(lambda: naive_attention(q, k, v, causal=causal),
                                 flops, warmup=3, rep=10)
        # online 是纯 Python 双重循环，很慢，用更少的重复次数
        ms_o, mb_o, _ = _measure(lambda: online_softmax_attention(q, k, v, causal=causal),
                                 flops, warmup=1, rep=3)
        ms_s, mb_s, tf_s = _measure(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
            flops, warmup=10, rep=50)

        print(f"{S:>7} | {ms_n:>9.3f} {mb_n:>7.0f} | {ms_o:>10.2f} {mb_o:>7.0f} | "
              f"{ms_s:>8.3f} {mb_s:>7.0f} {tf_s:>8.1f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
