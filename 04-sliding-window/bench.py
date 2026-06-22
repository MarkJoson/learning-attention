"""04-sliding-window 基准：滑动窗口的价值到底在哪？

**诚实地说**：本章 kernel 用「掩码」实现滑窗——窗口外的 key 块照样被读进来计算、再屏蔽掉，
循环范围并没有因窗口而缩小。所以它**不会加速 prefill**（下方第 2 张表会看到延迟与 full 相近）。

滑窗真正的价值在 **decode**：每个新 token 只看最近 W 个，于是 KV cache 只需保留最近 W 个 token，
**固定大小、不随上下文增长**。这是 Mistral 等能在固定显存下吞下超长上下文的关键（第 1 张表）。

（想在 prefill 也省算力，需要 kernel 主动跳过窗口外的整块——那是更高级的 block-sparse 实现，
 属于后面稀疏注意力章节的范畴。）
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03-gqa-mqa"))

from common import bench_ms, make_qkv  # noqa: E402
from gqa import gqa_attention  # noqa: E402
from sliding import sliding_window_attention  # noqa: E402


def run(B=2, Hq=8, Hkv=2, D=128, window=4096):
    dev = torch.cuda.get_device_name(0)
    print(f"# device={dev}  Hq={Hq} Hkv={Hkv} D={D} window={window}")

    print(f"\n== decode KV cache（单序列, Hkv={Hkv}, D={D}, fp16）：滑窗固定，full 线性增长 ==")
    print(f"{'上下文长度':>12} {'full KV':>12} {'滑窗 KV':>12} {'省':>8}")
    for ctx in [4096, 16384, 65536, 262144, 1048576]:
        full = 2 * Hkv * ctx * D * 2 / 1024**2
        sw = 2 * Hkv * min(ctx, window) * D * 2 / 1024**2
        print(f"{ctx:>12} {full:>10.0f}MB {sw:>10.0f}MB {full/sw:>6.0f}×")

    print(f"\n== prefill 延迟：full causal vs 滑窗(W={window})——掩码实现，二者相近 ==")
    print(f"{'seqlen':>8} {'full ms':>10} {'滑窗 ms':>10}")
    for S in [4096, 8192, 16384]:
        q, k, v = make_qkv(B, Hq, S, D, kv_heads=Hkv, dtype=torch.float16, seed=0)
        ms_full = bench_ms(lambda: gqa_attention(q, k, v, causal=True))
        ms_sw = bench_ms(lambda: sliding_window_attention(q, k, v, window_size=window, causal=True))
        del q, k, v
        torch.cuda.empty_cache()
        print(f"{S:>8} {ms_full:>8.3f} {ms_sw:>9.3f}")

    print("\n滑窗省的是 decode 的 KV cache（上表可达上千倍），而非 prefill 算力。")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
