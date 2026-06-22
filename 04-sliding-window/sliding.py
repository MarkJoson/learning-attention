"""滑动窗口注意力封装 —— 复用 03 提取的 vLLM kernel（它本就自带滑窗能力）。

本章**不引入新 kernel**：`03-gqa-mqa/gqa_triton.py` 的 `_fwd_kernel` 已支持
`SLIDING_WINDOW_Q/K`，我们只是把它点亮。窗口语义：每个 query 只看**最近 window_size 个
token**（含自己），这是 Mistral / Qwen 等用的因果滑窗。

off-by-one：kernel 的 `SLIDING_WINDOW_Q = W` 表示 `key >= query - W`（窗口含 W+1 个位置）。
所以"看最近 window_size 个 token"需要传 `window_size - 1`。本文件已替你处理好。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03-gqa-mqa"))
from gqa import gqa_attention  # noqa: E402

__all__ = ["sliding_window_attention"]


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: int,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """因果滑动窗口注意力：每个 query 只看最近 window_size 个 token（含自己）。

    形状 (B, H, S, D)；同样支持 GQA（Hq > Hkv）。底层是 03 的 kernel。
    """
    assert window_size >= 2, (
        "window_size 需 >= 2：底层 kernel 以 SLIDING_WINDOW>0 来启用滑窗，"
        "无法表达 window=1（仅看自身）这一退化情形。"
    )
    return gqa_attention(
        q, k, v, causal=causal, sm_scale=sm_scale, sliding_window=window_size - 1
    )
