"""05-paged-attention 基准：paged 的价值 —— 显存利用率（消除碎片）。

传统做法要为每个请求预留"最大上下文长度"的连续 KV cache，可真实请求长度参差不齐，
大量预留的显存其实是空的——这就是碎片浪费，也限制了能并发的请求数。

paged 按 block 粒度**按需分配**，几乎不浪费。下面用一批长度参差的请求把差距量化出来，
并实测 paged decode kernel 的延迟（功能正常的佐证）。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from paged_decode_triton import gqa_decode_attention_fwd  # noqa: E402


def run():
    print(f"# device={torch.cuda.get_device_name(0)}")

    # ---- 1) 显存利用率：传统预分配 vs paged 按需 ----
    Hkv, D, block_size = 8, 128, 16
    max_len = 8192                       # 模型支持的最大上下文
    g = torch.Generator().manual_seed(0)
    seqlens = torch.randint(128, 2048, (64,), generator=g).tolist()  # 64 个请求，长度参差
    bytes_per_token = 2 * Hkv * D * 2    # K+V，fp16

    trad = len(seqlens) * max_len * bytes_per_token / 1024**2                       # 每请求按 max_len 预留
    paged = sum(((L + block_size - 1) // block_size) * block_size for L in seqlens) \
        * bytes_per_token / 1024**2                                                 # 按 block 粒度
    actual = sum(seqlens) * bytes_per_token / 1024**2                              # 真正用到

    print(f"\n64 个请求，长度 128~2048，模型 max_len={max_len}，block_size={block_size}")
    print(f"  传统预分配（每请求按 max_len）: {trad:>7.0f} MB")
    print(f"  paged   （按需分配 block）   : {paged:>7.0f} MB   省 {trad/paged:.0f}×")
    print(f"  实际所需                     : {actual:>7.0f} MB   （paged 利用率 {actual/paged*100:.0f}%）")

    # ---- 2) decode 延迟实测（连续布局，专注 kernel 本身）----
    Hq, Hkv2, D2 = 32, 8, 128
    batch, seqlen = 64, 2048
    total = batch * seqlen
    k_cache = torch.randn(total, Hkv2, D2, dtype=torch.float16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    rtt = torch.arange(total, device="cuda", dtype=torch.int32).reshape(batch, seqlen)
    b_req_idx = torch.arange(batch, device="cuda", dtype=torch.int32)
    b_seq_len = torch.full((batch,), seqlen, device="cuda", dtype=torch.int32)
    q = torch.randn(batch, Hq, D2, dtype=torch.float16, device="cuda")
    o = torch.empty_like(q)

    ms = bench_ms(lambda: gqa_decode_attention_fwd(q, k_cache, v_cache, o, rtt, b_req_idx, b_seq_len))
    print(f"\ndecode 一步：{batch} 个序列 × {seqlen} 历史 token（Hq={Hq},Hkv={Hkv2},D={D2}）: {ms:.3f} ms")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
