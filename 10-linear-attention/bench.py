"""10 linear attention 基准：GLA chunk kernel 的 O(S) vs full attention 的 O(S²)。

序列越长，linear/GLA 的线性复杂度优势越大 —— 这正是它存在的理由（长上下文）。
GLA triton 用解耦自 fla 的 chunk-parallel kernel；full 用 PyTorch SDPA（FlashAttention 后端）。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from gla_triton import gla_chunk  # noqa: E402


def run():
    B, H, D = 4, 8, 128
    print(f"RTX 4090 · B={B} H={H} D={D} causal fp16/bf16\n")
    print(f"{'S':>6} | {'full SDPA(ms)':>14} | {'GLA triton(ms)':>15} | {'加速':>8}")
    print("-" * 52)
    for S in [1024, 2048, 4096, 8192]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        # GLA 用 bf16 + log forget gate
        qb, kb, vb = (x.to(torch.bfloat16) for x in (q, k, v))
        g = F.logsigmoid(torch.randn(B, H, S, D, device="cuda", dtype=torch.float32))

        t_full = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        t_gla = bench_ms(lambda: gla_chunk(qb, kb, vb, g))
        print(f"{S:>6} | {t_full:>14.3f} | {t_gla:>15.3f} | {t_full / t_gla:>7.2f}×")

    print("\n→ full attention O(S²)：S 翻倍延迟约 ×4；GLA O(S)：S 翻倍延迟约 ×2。")
    print("  序列越长，linear/GLA 的线性复杂度优势越明显（长上下文场景的意义所在）。")


if __name__ == "__main__":
    run()
