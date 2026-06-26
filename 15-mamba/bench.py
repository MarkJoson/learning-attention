"""15 Mamba2 SSD 基准：SSD chunk（= 标量衰减线性注意力）的 O(S) vs full attention O(S²)。

SSD 用 head-wise 标量衰减（比 GLA 的 per-channel 门控更轻），深度优化版解耦自 fla simple_gla 的 chunk kernel。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from ssd_triton import ssd_chunk  # noqa: E402


def run():
    B, H, D = 4, 8, 128
    print(f"RTX 4090 · B={B} H={H} D={D} causal\n")
    print(f"{'S':>6} | {'full SDPA(ms)':>14} | {'SSD(ms)':>9} | {'加速':>7}")
    print("-" * 44)
    for S in [1024, 2048, 4096, 8192]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        qb, kb, vb = (x.to(torch.bfloat16) for x in (q, k, v))
        g = F.logsigmoid(torch.randn(B, H, S, device="cuda", dtype=torch.float32))   # per-head 标量衰减
        t_full = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        t_ssd = bench_ms(lambda: ssd_chunk(qb, kb, vb, g))
        print(f"{S:>6} | {t_full:>14.3f} | {t_ssd:>9.3f} | {t_full / t_ssd:>6.2f}×")
    print("\n→ Mamba2 SSD = 标量衰减线性注意力，O(S) 复杂度；长序列优于 full attention O(S²)。")
    print("  标量衰减比 GLA 的 per-channel 门控更轻，是 Mamba2 既能 O(T) 推理又能并行训练的基础。")


if __name__ == "__main__":
    run()
