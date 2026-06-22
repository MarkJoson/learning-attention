"""11 DeltaNet 基准：delta rule 的 O(S) vs full attention O(S²)。

DeltaNet 仍属线性注意力家族（状态递归，O(S)），但每步多了"纠错擦写"。对比 full attention（SDPA），
看长序列下的线性优势。深度优化版用解耦自 fla 的 chunk-parallel kernel。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from deltanet_triton import delta_chunk  # noqa: E402


def run():
    B, H, D = 4, 8, 128
    print(f"RTX 4090 · B={B} H={H} D={D} causal\n")
    print(f"{'S':>6} | {'full SDPA(ms)':>14} | {'DeltaNet(ms)':>13} | {'加速':>7}")
    print("-" * 48)
    for S in [1024, 2048, 4096, 8192]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        qb, kb, vb = (x.to(torch.bfloat16) for x in (q, k, v))
        beta = torch.rand(B, H, S, device="cuda", dtype=torch.bfloat16)
        t_full = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        t_delta = bench_ms(lambda: delta_chunk(qb, kb, vb, beta))
        print(f"{S:>6} | {t_full:>14.3f} | {t_delta:>13.3f} | {t_full / t_delta:>6.2f}×")
    print("\n→ full O(S²)、DeltaNet O(S)：序列越长 DeltaNet 越占优；delta rule 比 GLA 每步多一次")
    print("  '查询旧状态+纠错'，常数更大，但复杂度仍线性。")


if __name__ == "__main__":
    run()
