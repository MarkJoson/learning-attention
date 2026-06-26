# 15-mamba · 外部来源登记

本章覆盖 **Mamba1（selective SSM）** 与 **Mamba2（SSD = State Space Duality）**。Mamba2 SSD 自写简要版（recurrent +
attention 对偶）+ 深度优化版（**完整解耦自 fla `simple_gla`**）；Mamba1 自写简要版讲机制 + 指向官方（CUDA 不提取）。
汇总见仓库根 [`NOTICE`](../NOTICE)。

## 深度优化版：Mamba2 SSD 完整解耦自 fla simple_gla

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/fla-org/flash-linear-attention · commit `0b27f7b` · MIT |
| 拷贝文件（4 triton + 1 参考） | `simple_gla/chunk`（入口）；common：`chunk_h`/`chunk_o`；`utils/cumsum`；`simple_gla/naive`（参考，3 个 ground truth） |
| 适配层 | `_fla_ssd_compat.py`（本仓库自写：RCP_LN2/exp2/autotune/check_shared_mem/input_guard/prepare_chunk_* + no-op dispatch） |

**为什么 `simple_gla` 就是 Mamba2 SSD**：fla `chunk_simple_gla` 的 docstring 明确——门控 `g` 是 **head-wise 标量**
（"the gating is head-wise instead of elementwise"，对比 GLA 的 per-channel）。head-wise 标量衰减正是 Mamba2 SSD 的
$a_t$（标量 $A$），所以 `simple_gla` 的 chunk kernel = Mamba2 的 SSD chunk scan。

**解耦**：no-op dispatch 绕过 `chunk_o` 的后端分派（CP/TileLang 不进闭包），模块用 `_fla_ssd_` 唯一前缀（与 10-13 隔离）。
比 KDA/GDN 简单——不依赖多卡 context-parallel，无 cp stub。

> `test_mamba.py::test_ssd_faithful_vs_fla` / `test_ssd_varlen_vs_fla` 验证本地解耦 ≡ fla 原版（定长 + 变长，**bitwise，
> max_abs=0**）；`test_ssd_kernel_vs_recurrent` 验证 ≡ SSD recurrent ground truth。

## Mamba1（selective SSM）—— 仅指向来源，未提取 kernel

| 项 | 值 |
|---|---|
| 来源 | https://github.com/state-spaces/mamba （`mamba_ssm`，Gu & Dao 2023） |
| 机制 | selective SSM（S6）：`B/C/Δ` data-dependent，A 对角；硬件感知的 **selective_scan** |
| 落地 | 本仓库 `mamba1.py` 自写纯 PyTorch recurrent 讲机制；真实 `selective_scan` 是 **CUDA kernel**（需编译，`mamba_ssm` 本环境未装），按"讲机制 + 指向来源"不提取（定位同 06-MLA decode / 14-DSv4） |

## 本仓库自写

- `ssd.py`：`ssd_recurrent`（SSD 线性形式 = 标量衰减线性注意力）+ `ssd_attention_dual`（SSD 注意力对偶 = 半可分矩阵 $M=L\circ(QK^\top)$）。
- `mamba1.py`：`selective_ssm_recurrent`（Mamba1 S6 selective SSM，ground truth）。
- `ssd_triton.py`：深度优化版入口（`[B,H,T,D]` 封装）；`_fla_ssd_compat.py`：薄适配层。
- `test_mamba.py`（7 测试：SSD 对偶等价 + 解耦 bitwise + kernel≡recurrent + bwd + Mamba1 causal）/ `bench.py` / `mamba.ipynb`。

## 算法来源

- **Mamba**：Gu & Dao 2023《Mamba: Linear-Time Sequence Modeling with Selective State Spaces》（selective SSM / S6）。
- **Mamba2 / SSD**：Dao & Gu 2024《Transformers are SSMs: ... State Space Duality》—— 证明 selective SSM 与 masked
  注意力的对偶（1-semiseparable matrix），SSD 是其高效算法。与本仓库线性注意力线（10 GLA / 11 DeltaNet / 12 KDA）同源：
  SSD = 标量衰减 GLA。
