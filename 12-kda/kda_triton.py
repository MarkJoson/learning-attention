"""12 KDA 深度优化版入口 —— 复用解耦后的 fla KDA chunk-parallel triton kernel（脱离 fla）。

底层 kernel（拷贝自 fla、仅改 import 指向本地，计算逻辑一字未改，见 SOURCES.md）共 14 个文件，
入口 `_fla_kda_chunk.chunk_kda`。no-op dispatch 绕过后端分派、cp stub 绕过多卡 context-parallel。

- `chunk_kda`：fla 原生接口，layout `[B,T,H,D]`，g `[B,T,H,K]`、beta `[B,T,H]`。
- `kda_chunk`：`[B,H,T,D]` 便捷封装（对接简要版 `kda.kda_recurrent`）。
注意 KDA 需 `use_qk_l2norm_in_kernel=True`（delta rule 标配；否则未归一化的 k 会让 bf16 状态发散）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fla_kda_chunk import chunk_kda  # noqa: E402  解耦后的 KDA chunk kernel


def kda_chunk(q, k, v, g, beta, *, scale=None, l2norm=True):
    """`[B,H,T,D]` layout 封装。内部转 fla 的 `[B,T,H,D]` 调 chunk_kda。"""
    qt, kt, vt, gt = (x.transpose(1, 2).contiguous() for x in (q, k, v, g))
    bt = beta.transpose(1, 2).contiguous()
    o, _ = chunk_kda(qt, kt, vt, gt, bt, scale=scale, use_qk_l2norm_in_kernel=l2norm)
    return o.transpose(1, 2)
