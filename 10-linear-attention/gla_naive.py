# =============================================================================
# 来源标注 (Provenance) —— 本仓库 10-linear-attention（GLA 深度优化版）
# -----------------------------------------------------------------------------
# 本文件的 Triton kernel 完整拷贝自 fla-org/flash-linear-attention，**计算逻辑一字未改**，
# 仅把对 fla 框架的 import 改指向本地薄适配层（_fla_compat.py / _fla_chunk_h.py / _fla_cumsum.py），
# 使其脱离 fla 包独立运行。适配层复现了 exp2/RCP_LN2/autotune/check_shared_mem/input_guard 等
# 工具符号（见 _fla_compat.py）；定长与变长（cu_seqlens / sequence packing）序列均支持（变长索引函数亦拷自 fla）。
#   source repo : https://github.com/fla-org/flash-linear-attention
#   source file : fla/ops/gla/naive.py
#   commit      : 0b27f7b
#   license     : MIT (Copyright fla-org)
# 详见同目录 SOURCES.md 与仓库根 NOTICE。
# =============================================================================
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def ceildiv(a, b):
    return -(a // -b)


def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    dtype = q.dtype
    q, k, v, gk = map(lambda x: x.transpose(1, 2).float(), (q, k, v, gk))
    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    scale = K ** -0.5

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        h += initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o[:, :, i] = (q_i[..., None] * h).sum(-2)

    if not output_final_state:
        h = None
    return o.transpose(1, 2).to(dtype), h
