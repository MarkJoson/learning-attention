"""FlashAttention v2（Triton）调用封装。

真正的 kernel 在 `flash_triton.py`（拷贝自 triton 官方教程，见其文件头来源标注）。
本文件**不含任何 kernel 逻辑**，只做默认 scale、输入校验、统一接口。

这个经典教程 kernel 刻意保持精简（这也是它适合教学的原因），因此有如下限制：
  - 标准 MHA：要求 Hq == Hkv（**不支持 GQA**，GQA/MQA 见 03 章）；
  - self-attention：要求 Sq == Sk（q/k/v 同形）；
  - head_dim ∈ {16, 32, 64, 128, 256}；
  - backward 要求 seqlen 是 128 的倍数（kernel 内部 PRE_BLOCK=128 的约束）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flash_triton import attention as _triton_attention  # noqa: E402

__all__ = ["flash_attention"]


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """标准多头 FlashAttention v2。

    形状：q/k/v 均为 (B, H, S, D)，且三者同形。dtype 建议 fp16/bf16。
    """
    assert q.shape == k.shape == v.shape, (
        f"经典教程 kernel 要求 q/k/v 同形 (MHA, Sq==Sk, Hq==Hkv)，"
        f"实际 {tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}"
    )
    D = q.shape[-1]
    assert D in {16, 32, 64, 128, 256}, f"head_dim 必须 ∈ {{16,32,64,128,256}}，实际 {D}"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    return _triton_attention(q, k, v, causal, sm_scale)
