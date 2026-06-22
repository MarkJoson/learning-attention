"""03-gqa-mqa 基准：MHA / GQA / MQA 的 KV cache 显存与延迟对比。

GQA/MQA 的真正收益在**推理**：KV cache 的大小正比于 KV head 数 Hkv。把 Hkv 从 32（MHA）
降到 8（GQA）甚至 1（MQA），KV cache 直接缩小 4×/32×，而模型质量几乎不掉——这正是当下
几乎所有大模型都用 GQA 的原因。

注意看：因为 query 头数 Hq 不变，三者的**注意力计算量基本一样**（延迟接近），省的纯是 KV 显存。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import bench_ms, make_qkv  # noqa: E402
from gqa import gqa_attention  # noqa: E402


def kv_cache_mb(batch, n_kv_heads, seqlen, head_dim, bytes_per=2):
    """推理时需缓存的 K 与 V 的总显存（MB）：2 · batch · Hkv · S · D。"""
    return 2 * batch * n_kv_heads * seqlen * head_dim * bytes_per / 1024**2


def run(Hq=32, head_dim=128, seqlen=4096, batch=8, causal=True, dtype=torch.float16):
    print(f"# device={torch.cuda.get_device_name(0)}  "
          f"Hq={Hq} D={head_dim} seqlen={seqlen} batch={batch} {dtype}")
    print(f"{'模式':>5} {'Hkv':>4} {'KV cache':>11} {'vs MHA':>7} {'attn 延迟':>10}")
    print("-" * 46)

    hkv_list = sorted(set([Hq, Hq // 4, Hq // 8, 1]), reverse=True)
    kv_mha = kv_cache_mb(batch, Hq, seqlen, head_dim)
    for Hkv in hkv_list:
        name = "MHA" if Hkv == Hq else ("MQA" if Hkv == 1 else "GQA")
        kv = kv_cache_mb(batch, Hkv, seqlen, head_dim)
        q, k, v = make_qkv(batch, Hq, seqlen, head_dim, kv_heads=Hkv, dtype=dtype, seed=0)
        ms = bench_ms(lambda: gqa_attention(q, k, v, causal=causal))
        del q, k, v
        torch.cuda.empty_cache()
        print(f"{name:>5} {Hkv:>4} {kv:>9.0f} MB {kv_mha/kv:>6.0f}× {ms:>8.3f} ms")

    print(f"\nKV cache 从 MHA 的 {kv_mha:.0f}MB 一路缩到 MQA 的 {kv_cache_mb(batch,1,seqlen,head_dim):.0f}MB，")
    print("而注意力延迟几乎不变——省显存几乎是免费的。")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
