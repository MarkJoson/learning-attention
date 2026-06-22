"""08-native-sparse-attention 基准：三分支的"信息覆盖"与延迟。

NSA 的三条分支各看一部分上下文、互补：compressed 看全局粗粒度、selected 看最重要的细节、
sliding 看局部。本教学版三分支都是 full + mask 实现（不省计算，只为讲清机制）；真正的 NSA
用稀疏 Triton kernel 只算选中部分才省。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from nsa import NativeSparseAttention  # noqa: E402


def run(S=2048, dim=512, block_size=32, num_selected=4, window=128, B=2, H=8, D=64):
    nb = S // block_size
    nsa = NativeSparseAttention(dim, H, D, block_size, num_selected, window).cuda().half()
    x = torch.randn(B, S, dim, dtype=torch.float16, device="cuda")
    print(f"# S={S} block_size={block_size} nb={nb} 块  num_selected={num_selected}  window={window}")

    print("\n三分支的信息覆盖（每个 query 看到哪些 key）：")
    print(f"  compressed : {nb:>4} 个压缩 token   —— {S} token 的粗粒度全局视野")
    print(f"  selected   : top-{num_selected} 块 = {num_selected * block_size:>4} token   —— 最重要的细节（07 块稀疏）")
    print(f"  sliding    : 最近 {window:>4} token        —— 局部（04 滑窗）")

    qkv = nsa.to_qkv(x).view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    _, blk_sim = nsa.compressed_branch(q, k, v)

    print("\n各分支延迟（本教学版均为 full+mask，不省计算）：")
    print(f"  compressed : {bench_ms(lambda: nsa.compressed_branch(q, k, v)):.3f} ms")
    print(f"  selected   : {bench_ms(lambda: nsa.selected_branch(q, k, v, blk_sim)):.3f} ms")
    print(f"  sliding    : {bench_ms(lambda: nsa.sliding_branch(q, k, v)):.3f} ms")
    print(f"  NSA 整体   : {bench_ms(lambda: nsa(x)):.3f} ms（三分支 + 门控合并）")

    print("\n真实 NSA 把三分支都做成稀疏 Triton kernel（只算选中部分），才能把'看得少'变成'算得快'；")
    print("本章聚焦讲清三分支架构，kernel 细节见 lucidrains 的实现（近 2000 行）。")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
