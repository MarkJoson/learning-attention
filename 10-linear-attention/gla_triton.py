"""10 GLA 深度优化版入口 —— 复用解耦后的 fla chunk-parallel triton kernel（已脱离 fla 包）。

底层 kernel（拷贝自 fla、仅改 import 指向本地薄适配层，计算逻辑一字未改，见 SOURCES.md）：
  - `_fla_gla_chunk.py`  chunk-parallel 主体（块内并行 + 块间递归）
  - `_fla_chunk_h.py`    块间状态传递（chunk_fwd_h / chunk_bwd_dh）
  - `_fla_cumsum.py`     门控的块内累积（chunk_local_cumsum）
  - `_fla_compat.py`     薄适配层（exp2 / autotune / check_shared_mem / input_guard ...）

对外提供：
  - `chunk_gla`：fla 原生接口，layout `[B,T,H,D]`，返回 `(o, final_state)`；
  - `gla_chunk`：`[B,H,T,D]` layout 的便捷封装（与简要版 `linear.gla_recurrent` 对齐）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fla_gla_chunk import chunk_gla  # noqa: E402  解耦后的 GLA chunk-parallel kernel


def gla_chunk(q, k, v, g, *, scale=None):
    """`[B,H,T,K]`/`[B,H,T,V]` layout 的 GLA（对接简要版）。内部转 fla 的 `[B,T,H,D]` 调 chunk_gla。"""
    qt, kt, vt, gt = (x.transpose(1, 2).contiguous() for x in (q, k, v, g))
    o, _ = chunk_gla(qt, kt, vt, gt, scale=scale)
    return o.transpose(1, 2)
