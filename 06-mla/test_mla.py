"""06-mla 数值正确性测试。

核心验证：absorb（推理高效路径）与 naive（重建 K/V 路径）**数值等价** —— 这是 absorb 这套
代数重排正确性的试金石。此外验证 MLA 的 KV cache 远小于 MHA。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import assert_close  # noqa: E402
from mla import MLA, MLAConfig  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_absorb_equals_naive(dtype):
    torch.manual_seed(0)
    cfg = MLAConfig()
    mla = MLA(cfg).cuda().to(dtype)
    h = torch.randn(2, 128, cfg.d_model, dtype=dtype, device="cuda")
    pos = torch.arange(128, device="cuda")
    o_naive = mla.forward_naive(h, pos)
    o_absorb = mla.forward_absorb(h, pos)
    # float32 下两条路径应几乎逐位相等；float16 放宽
    atol = 1e-4 if dtype == torch.float32 else 2e-2
    assert_close(o_absorb, o_naive, atol=atol, rtol=1e-3, name=f"absorb vs naive ({dtype})")


@pytest.mark.parametrize("kv_lora_rank,qk_rope,v_dim", [(128, 32, 64), (256, 64, 128), (64, 16, 32)])
def test_absorb_equals_naive_various_dims(kv_lora_rank, qk_rope, v_dim):
    torch.manual_seed(1)
    cfg = MLAConfig(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope, v_head_dim=v_dim)
    mla = MLA(cfg).cuda().double()
    h = torch.randn(2, 96, cfg.d_model, dtype=torch.float64, device="cuda")
    pos = torch.arange(96, device="cuda")
    o_naive = mla.forward_naive(h, pos)
    o_absorb = mla.forward_absorb(h, pos)
    # double 下两路径仅差浮点累积误差（~1e-7），足以证明 absorb 的代数重排正确
    assert_close(o_absorb, o_naive, atol=1e-6, rtol=1e-5,
                 name=f"dims kv={kv_lora_rank},rope={qk_rope},v={v_dim}")


def test_kv_cache_much_smaller_than_mha():
    cfg = MLAConfig()
    mla = MLA(cfg)
    mla_kv = mla.kv_cache_per_token()                # kv_lora_rank + qk_rope_head_dim
    # 同等规模的标准 MHA：每 token 要缓存 K 和 V，各 H * head_dim
    mha_kv = 2 * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) // 2
    mha_kv = 2 * cfg.n_heads * cfg.qk_nope_head_dim   # K、V 各 H*head_dim（head_dim≈qk_nope）
    assert mla_kv < mha_kv, f"MLA {mla_kv} 应远小于 MHA {mha_kv}"
    assert mla_kv == cfg.kv_lora_rank + cfg.qk_rope_head_dim


@pytest.mark.parametrize("kv_lora_rank", [128, 256])
def test_absorb_triton_matches_naive(kv_lora_rank):
    """提取自 lightllm 的 MLA triton kernel（在 latent 上算注意力）应与 naive 一致。"""
    torch.manual_seed(0)
    cfg = MLAConfig(kv_lora_rank=kv_lora_rank)
    mla = MLA(cfg).cuda().half()
    h = torch.randn(2, 256, cfg.d_model, dtype=torch.float16, device="cuda")
    pos = torch.arange(256, device="cuda")
    o_naive = mla.forward_naive(h, pos)
    o_triton = mla.forward_absorb_triton(h, pos)
    assert_close(o_triton, o_naive, atol=3e-2, rtol=1e-2,
                 name=f"MLA triton kernel vs naive (kv_lora={kv_lora_rank})")
