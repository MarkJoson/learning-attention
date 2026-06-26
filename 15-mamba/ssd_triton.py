"""15 Mamba2 SSD 深度优化版入口 —— 复用解耦后的 fla simple_gla chunk kernel（脱离 fla）。

simple_gla = Mamba2 的 SSD（head-wise 标量衰减）。底层 4 个 triton 文件（chunk / common.chunk_h /
common.chunk_o / utils.cumsum，拷贝自 fla、仅改 import 指向本地，计算逻辑一字未改，见 SOURCES.md），
入口 `_fla_ssd_chunk.chunk_simple_gla`。no-op dispatch 脱离 fla。介于 SSD 两种形式之间：块内走对偶
（矩阵乘），块间走递推。

- `chunk_simple_gla`：fla 原生接口，layout `[B,T,H,D]`，g `[B,T,H]`（per-head 标量 log decay）。
- `ssd_chunk`：`[B,H,T,D]` 便捷封装（对接简要版 ssd.py）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fla_ssd_chunk import chunk_simple_gla  # noqa: E402


def ssd_chunk(q, k, v, g, *, scale=None):
    """`[B,H,T,D]` layout 封装。g 为 `[B,H,T]` per-head 标量 log decay。内部转 fla 的 `[B,T,H,D]`。"""
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    gt = g.transpose(1, 2).contiguous()                       # [B,H,T] -> [B,T,H]
    o, _ = chunk_simple_gla(qt, kt, vt, gt, scale=scale)
    return o.transpose(1, 2)
