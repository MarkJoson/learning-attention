"""13 GDN-2 深度优化版入口 —— 复用解耦后的 fla Gated DeltaNet 2 chunk kernel（脱离 fla）。

底层 13 个 triton 文件（拷贝自 fla、仅改 import 指向本地，计算逻辑一字未改，见 SOURCES.md），入口
`_fla_gdn2_chunk.chunk_gdn2`。GDN-2 复用了 KDA 的 chunk_intra/gate/wy 与 gla.chunk（退化关系见 gdn2.py）。
no-op dispatch + cp stub 脱离 fla。

- `chunk_gdn2`：fla 原生接口，layout `[B,T,H,D]`；g/b `[B,T,H,K]`、w `[B,T,H,V]`（erase/write 双门控）。
- `gdn2_chunk`：`[B,H,T,D]` 便捷封装（对接简要版 gdn2.py）。需 `use_qk_l2norm_in_kernel=True`（delta rule 标配）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fla_gdn2_chunk import chunk_gdn2  # noqa: E402


def gdn2_chunk(q, k, v, g, b, w, *, scale=None, l2norm=True):
    """`[B,H,T,D]` layout 封装。内部转 fla 的 `[B,T,H,D]` 调 chunk_gdn2（erase 门 b / write 门 w）。"""
    qt, kt, vt, gt = (x.transpose(1, 2).contiguous() for x in (q, k, v, g))
    bt, wt = (x.transpose(1, 2).contiguous() for x in (b, w))
    o, _ = chunk_gdn2(qt, kt, vt, gt, bt, wt, scale=scale, use_qk_l2norm_in_kernel=l2norm)
    return o.transpose(1, 2)
