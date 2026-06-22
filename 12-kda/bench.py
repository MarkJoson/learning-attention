"""12 KDA 基准：gated delta rule 的 O(S) vs full attention O(S²)。

KDA 每步比 GLA/DeltaNet 更重（门控衰减 + delta 纠错都做），但仍是线性复杂度。
深度优化版用解耦自 fla 的 chunk-parallel kernel。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from kda_triton import kda_chunk  # noqa: E402


def run():
    B, H, D = 4, 8, 128
    print(f"RTX 4090 · B={B} H={H} D={D} causal\n")
    print(f"{'S':>6} | {'full SDPA(ms)':>14} | {'KDA(ms)':>10} | {'加速':>7}")
    print("-" * 44)
    for S in [1024, 2048, 4096, 8192]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        qb, kb, vb = (x.to(torch.bfloat16) for x in (q, k, v))
        g = F.logsigmoid(torch.randn(B, H, S, D, device="cuda", dtype=torch.float32))
        beta = torch.rand(B, H, S, device="cuda", dtype=torch.bfloat16)
        t_full = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        t_kda = bench_ms(lambda: kda_chunk(qb, kb, vb, g, beta))
        print(f"{S:>6} | {t_full:>14.3f} | {t_kda:>10.3f} | {t_full / t_kda:>6.2f}×")
    print("\n→ KDA O(S)：gated delta（门控衰减 + delta 纠错）每步比 GLA/DeltaNet 更重，常数更大，")
    print("  但复杂度仍线性；长序列优于 full attention O(S²)。")


if __name__ == "__main__":
    run()
