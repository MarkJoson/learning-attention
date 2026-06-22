"""11 DeltaNet 深度优化版入口 —— 复用解耦后的 fla delta_rule chunk-parallel triton kernel。

底层 kernel（拷贝自 fla、仅改 import 指向本地 _fla_compat，计算逻辑一字未改，见 SOURCES.md）：
  _fla_delta_chunk（主体）/ _fla_wy_fast（WY 表示）/ _fla_chunk_delta_h（块间状态）/
  _fla_chunk_o（输出）/ _fla_chunk_scaled_dot_kkt / _fla_l2norm / _fla_solve_tril。
no-op dispatch（_fla_compat）绕过 fla 的 CP/TileLang 后端，使其脱离 fla 包独立运行。

- `chunk_delta_rule`：fla 原生接口，layout `[B,T,H,D]`，beta `[B,T,H]`，返回 `(o, final_state)`。
- `delta_chunk`：`[B,H,T,D]` layout 便捷封装（对接简要版 `deltanet.delta_rule_recurrent`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fla_delta_chunk import chunk_delta_rule  # noqa: E402  解耦后的 DeltaNet chunk kernel


def delta_chunk(q, k, v, beta, *, scale=None, l2norm=True):
    """`[B,H,T,D]` layout 封装。内部转 fla 的 `[B,T,H,D]` 调 chunk_delta_rule。"""
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    bt = beta.transpose(1, 2).contiguous()           # (B,H,T) -> (B,T,H)
    o, _ = chunk_delta_rule(qt, kt, vt, bt, scale=scale, use_qk_l2norm_in_kernel=l2norm)
    return o.transpose(1, 2)
