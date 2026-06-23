"""13 GDN 深度优化版入口 —— 复用解耦后的 fla Gated DeltaNet chunk kernel（脱离 fla）。

底层 10 个 triton 文件（拷贝自 fla、仅改 import 指向本地，计算逻辑一字未改，见 SOURCES.md），入口
`_fla_gdn_chunk.chunk_gated_delta_rule`。no-op dispatch + cp stub 脱离 fla。

- `chunk_gated_delta_rule`：fla 原生接口，layout `[B,T,H,D]`，g `[B,T,H,K]`、beta `[B,T,H]`。
- `gdn_chunk`：`[B,H,T,D]` 便捷封装（对接简要版）。需 `use_qk_l2norm_in_kernel=True`（delta rule 标配）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fla_gdn_chunk import chunk_gated_delta_rule  # noqa: E402


def gdn_chunk(q, k, v, g, beta, *, scale=None, l2norm=True):
    """`[B,H,T,D]` layout 封装。内部转 fla 的 `[B,T,H,D]` 调 chunk_gated_delta_rule。"""
    qt, kt, vt, gt = (x.transpose(1, 2).contiguous() for x in (q, k, v, g))
    bt = beta.transpose(1, 2).contiguous()
    o, _ = chunk_gated_delta_rule(qt, kt, vt, gt, bt, scale=scale, use_qk_l2norm_in_kernel=l2norm)
    return o.transpose(1, 2)
