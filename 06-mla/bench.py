"""06-mla 基准：MLA 的 KV cache vs MHA / GQA，以及 absorb 路径省下的重建开销。

MLA 的全部意义在 KV cache：它只缓存一个低维 latent（+ 一个共享 RoPE key），而不是每个 head 的
完整 K/V。按 DeepSeek-V2 的规模算一笔账，差距非常惊人。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms  # noqa: E402
from mla import MLA, MLAConfig  # noqa: E402


def run():
    # ---- 1) KV cache per token：MHA vs GQA vs MLA（DeepSeek-V2 规模）----
    H, head_dim, kv_lora, qk_rope = 128, 128, 512, 64
    schemes = {
        "MHA  (128 KV heads)": 2 * H * head_dim,          # K、V 各 H*head_dim
        "GQA  (8 KV heads)  ": 2 * 8 * head_dim,
        "MLA  (latent 512+64)": kv_lora + qk_rope,        # 只缓存 latent + 共享 RoPE key
    }
    base = schemes["MHA  (128 KV heads)"]
    print("# DeepSeek-V2 规模：128 头, head_dim=128, kv_lora_rank=512, qk_rope=64\n")
    print(f"{'方案':<22}{'每 token 缓存':>14}{'相对 MHA':>10}")
    for name, per in schemes.items():
        print(f"{name:<22}{per:>10} 个值 {base/per:>8.1f}×")

    # 长上下文的总 KV cache（60 层, batch=32, 上下文 8192, fp16）
    layers, batch, ctx = 60, 32, 8192
    print(f"\n整模型 KV cache（{layers} 层, batch={batch}, 上下文={ctx}, fp16）：")
    for name, per in schemes.items():
        gb = layers * batch * ctx * per * 2 / 1024**3
        print(f"  {name:<22}{gb:>8.1f} GB")

    # ---- 2) naive（重建 K/V）vs absorb（不重建）的前向延迟 ----
    torch.manual_seed(0)
    cfg = MLAConfig()
    mla = MLA(cfg).cuda().half()
    h = torch.randn(4, 1024, cfg.d_model, dtype=torch.float16, device="cuda")
    pos = torch.arange(1024, device="cuda")
    ms_naive = bench_ms(lambda: mla.forward_naive(h, pos))
    ms_absorb = bench_ms(lambda: mla.forward_absorb(h, pos))
    ms_triton = bench_ms(lambda: mla.forward_absorb_triton(h, pos))
    print(f"\n前向延迟（B=4 S=1024）:")
    print(f"  naive (重建 K/V, PyTorch)     : {ms_naive:.3f} ms")
    print(f"  absorb (latent, PyTorch)      : {ms_absorb:.3f} ms")
    print(f"  absorb (latent, triton kernel): {ms_triton:.3f} ms   ← 提取自 lightllm")
    print("（triton 是 MLA prefill 的真实 kernel；DeepSeek 的 CUDA FlashMLA 会更快。）")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
