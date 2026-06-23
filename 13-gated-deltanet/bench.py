"""13 GDN / GDN-2 基准：gated delta rule 的 O(S) vs full attention O(S²)。

GDN（Qwen3-Next）用 per-head 标量门控；GDN-2（Qwen3.5）拆成 per-channel 的 erase/write 双门控，
每步更重（多两个门 + 更细粒度），但复杂度仍线性。深度优化版均为解耦自 fla 的 chunk-parallel kernel。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from gdn_triton import gdn_chunk  # noqa: E402
from gdn2_triton import gdn2_chunk  # noqa: E402


def run():
    B, H, D = 4, 8, 128
    print(f"RTX 4090 · B={B} H={H} D={D} causal\n")
    print(f"{'S':>6} | {'full SDPA(ms)':>14} | {'GDN(ms)':>9} | {'GDN-2(ms)':>10} | {'GDN加速':>8} | {'GDN2加速':>9}")
    print("-" * 70)
    for S in [1024, 2048, 4096, 8192]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        qb, kb, vb = (x.to(torch.bfloat16) for x in (q, k, v))
        # GDN：per-head 标量门控 g/beta
        g1 = F.logsigmoid(torch.randn(B, H, S, device="cuda", dtype=torch.float32))
        beta = torch.rand(B, H, S, device="cuda", dtype=torch.bfloat16)
        # GDN-2：per-channel decay g、erase b、write w
        g2 = F.logsigmoid(torch.randn(B, H, S, D, device="cuda", dtype=torch.float32))
        b = torch.rand(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        w = torch.rand(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        t_full = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        t_gdn = bench_ms(lambda: gdn_chunk(qb, kb, vb, g1, beta))
        t_gdn2 = bench_ms(lambda: gdn2_chunk(qb, kb, vb, g2, b, w))
        print(f"{S:>6} | {t_full:>14.3f} | {t_gdn:>9.3f} | {t_gdn2:>10.3f} | "
              f"{t_full / t_gdn:>7.2f}× | {t_full / t_gdn2:>8.2f}×")
    print("\n→ GDN / GDN-2 均 O(S)：长序列优于 full attention O(S²)。GDN-2 多了 per-channel 双门控，")
    print("  每步常数更大（略慢于 GDN），换来更强的 erase/write 表达力。")


if __name__ == "__main__":
    run()
