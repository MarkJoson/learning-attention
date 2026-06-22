"""02-flash-attention 基准：Triton FlashAttention v2 vs PyTorch SDPA vs 朴素实现。

观察点：
  - FlashAttention 显存随 seqlen **线性**增长（不构造 S×S 矩阵），naive 是平方增长；
  - Triton kernel 的 TFLOP/s 与 PyTorch 内置 SDPA（同为融合 kernel）在同一量级；
  - 相对纯 PyTorch naive 的加速比随序列变长而显著拉大。

实现说明：Triton autotune 首次调用时会异步 benchmark 多个 config。若让它与交错的
SDPA/naive kernel 同时跑，会在大 seqlen 下因抢显存触发异步非法访存。故本脚本分两阶段：
先在显存干净时一次性跑完所有 seqlen 的 autotune，再正式测量。
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
from flash import flash_attention  # noqa: E402


def run(seqlens=(1024, 2048, 4096, 8192), batch=2, heads=16, head_dim=64,
        causal=True, dtype=torch.float16):
    print(f"# device={torch.cuda.get_device_name(0)}  "
          f"B={batch} H={heads} D={head_dim} causal={causal} {dtype}")
    print(f"{'seqlen':>7} | {'flash ms':>9} {'TFLOP/s':>8} {'MB':>6} | "
          f"{'sdpa ms':>8} {'TFLOP/s':>8} | {'naive ms':>9} | {'flash vs naive':>14}")
    print("-" * 92)

    # 阶段一：显存干净时逐个完成各 seqlen 的 Triton autotune，每个之间同步并清缓存，
    # 避免 do_bench 的 L2-flush 缓冲在多次 autotune 间累积显存。
    for S in seqlens:
        q, k, v = make_qkv(batch, heads, S, head_dim, dtype=dtype)
        flash_attention(q, k, v, causal=causal)
        torch.cuda.synchronize()
        del q, k, v
        torch.cuda.empty_cache()

    # 阶段二：正式测量（此时 flash 直接命中 autotune 缓存，不再异步探测 config）。
    for S in seqlens:
        q, k, v = make_qkv(batch, heads, S, head_dim, dtype=dtype)
        flops = attention_flops(batch, heads, S, S, head_dim, causal=causal, mode="fwd")

        torch.cuda.reset_peak_memory_stats()
        ms_f = bench_ms(lambda: flash_attention(q, k, v, causal=causal))
        mb_f = peak_memory_mb()

        ms_s = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))

        # naive 会构造 batch*heads*S*S 的注意力矩阵；预估其显存，过大则跳过 —— 这种
        # 平方级显存正是 FlashAttention 要消除的（跳过不是缺陷，而是论点本身）。
        naive_scores_gb = batch * heads * S * S * 4 / 1e9
        if naive_scores_gb < 2.5:
            ms_n = bench_ms(lambda: naive_attention(q, k, v, causal=causal), warmup=3, rep=10)
        else:
            ms_n = float("nan")

        del q, k, v
        torch.cuda.empty_cache()

        speedup = f"{ms_n / ms_f:.1f}x" if ms_n == ms_n else "  n/a"
        print(f"{S:>7} | {ms_f:>9.3f} {tflops(flops, ms_f):>8.1f} {mb_f:>6.0f} | "
              f"{ms_s:>8.3f} {tflops(flops, ms_s):>8.1f} | {ms_n:>9.3f} | {speedup:>14}")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== head_dim = 64 ==")
    run(head_dim=64)
    print("\n== head_dim = 128 ==")
    run(head_dim=128)
