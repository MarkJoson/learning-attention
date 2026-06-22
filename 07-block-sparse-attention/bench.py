"""07-block-sparse-attention 基准：稀疏度与计算量。

块稀疏的价值在长序列：每个 query 块只算 top-k 个 key 块，而非全部 nb 个。计算量按块数线性缩减。
下面对照三者：朴素 full（O(S²)）、SDPA full（融合 kernel）、块稀疏 gather（纯 PyTorch）。

要看清的一点：块稀疏 gather 因为"少算块"确实远快于朴素 full；但它是纯 PyTorch（gather/scatter
有开销），和融合的 SDPA 比并不占便宜。**要既稀疏又快，得把"选块 + 稀疏算"融进一个 Triton
kernel——那正是 08-NSA 的主题。** 这里重点看稀疏度（选中比例）与计算量缩减。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms, make_qkv, naive_attention  # noqa: E402
from block_sparse import block_sparse_attention  # noqa: E402


def run(S=2048, block_size=64, B=4, H=8, D=64, causal=True):
    nb = S // block_size
    q, k, v = make_qkv(B, H, S, D, dtype=torch.float16, seed=0)
    print(f"# S={S} block_size={block_size} nb={nb} 块  B={B} H={H} causal={causal}")

    ms_naive = bench_ms(lambda: naive_attention(q, k, v, causal=causal))
    ms_sdpa = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    print(f"full 朴素 (PyTorch O(S²)) : {ms_naive:.3f} ms")
    print(f"full SDPA (融合 kernel)   : {ms_sdpa:.3f} ms\n")

    print(f"{'top_k':>6} {'稀疏度':>8} {'选中块':>9} {'gather ms':>10} {'vs 朴素':>8} {'vs SDPA':>8}")
    print("-" * 56)
    for top_k in [1, 2, 4, 8, nb]:
        ms_gat = bench_ms(lambda: block_sparse_attention(q, k, v, block_size, top_k, causal=causal))
        kept = top_k / nb
        print(f"{top_k:>6} {kept:>7.0%} {top_k:>6}/{nb} {ms_gat:>9.3f} "
              f"{ms_naive/ms_gat:>6.1f}× {ms_sdpa/ms_gat:>6.1f}×")

    print(f"\n块稀疏只算选中块——top_k 越小越快，选 2/{nb} 块就比朴素 full 快一大截。")
    print("但 gather 是纯 PyTorch，和融合的 SDPA 比并不占优。要既稀疏又快，")
    print("得把'选块 + 稀疏算'融进一个 Triton kernel —— 那就是下一章 08-NSA。")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
