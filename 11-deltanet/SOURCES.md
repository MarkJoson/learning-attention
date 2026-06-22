# 11-deltanet · 外部来源登记

自写简要版（delta rule recurrent + WY chunked）+ 深度优化版（**完整解耦自 fla** 的 DeltaNet
chunk-parallel triton kernel）。汇总见仓库根 [`NOTICE`](../NOTICE)。

## 深度优化版：完整解耦自 fla 的 DeltaNet kernel

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/fla-org/flash-linear-attention |
| commit | `0b27f7b` · license MIT (Copyright fla-org) |
| 拷贝文件（8 个 triton 核心） | `delta_rule/chunk.py`→`_fla_delta_chunk.py`；`delta_rule/wy_fast.py`→`_fla_wy_fast.py`；`common/chunk_delta_h.py`→`_fla_chunk_delta_h.py`；`common/chunk_o.py`→`_fla_chunk_o.py`；`common/chunk_scaled_dot_kkt.py`→`_fla_chunk_scaled_dot_kkt.py`；`modules/l2norm.py`→`_fla_l2norm.py`；`ops/utils/solve_tril.py`→`_fla_solve_tril.py`；`delta_rule/naive.py`→`deltanet_naive.py` |

### 完整解耦的关键：no-op dispatch 收敛依赖闭包

DeltaNet 的 kernel 比 GLA 耦合深得多：用到 WY 表示（`wy_fast` + `solve_tril` 解下三角）、块间状态
（`chunk_delta_h`）、输出（`chunk_o`）、L2norm，并且通过 `backends.dispatch` **静态引入多卡 context-
parallel、TileLang 后端、full attention**——朴素的完整依赖闭包达 **~27 个文件且不收敛**（每层还在引入
新后端）。

**关键洞察**：那些 CP/TileLang/attn 后端全部经 `backends.dispatch` 引入，而 dispatch 仅以
`@dispatch('common')` 装饰 triton 实现。本仓库用 **no-op dispatch**（`_fla_compat.py`，直接返回被装饰
的 triton 函数）替换它 → 不再 import `fla.ops.backends`，CP/TileLang/attn 后端**全部不进闭包**，完整
解耦收敛到 **8 个 triton 核心文件**。`_fla_compat.py` 另复现 ~17 个工具符号（dispatch / autocast /
`IS_NVIDIA_HOPPER` 等能力标志 / `safe_dot` / `exp2` / varlen 索引 …）。

`test_deltanet.py::test_triton_faithful_vs_fla` 与 `test_varlen_vs_fla` 验证"本地解耦 kernel == fla
原版"（定长 + 变长，**bitwise 0.0**），证明解耦没改任何计算。

## 本仓库自写

- `deltanet.py`：`delta_rule_recurrent`（逐步纠错更新，ground truth）+ `delta_rule_chunked`（WY 表示）。
- `deltanet_triton.py`：深度优化版入口（封装解耦 kernel，`[B,H,T,D]` layout）。
- `_fla_compat.py`：薄适配层（no-op dispatch + 工具符号）。
- `test_deltanet.py` / `bench.py` / `deltanet.ipynb`。

## 算法来源

- **DeltaNet**：Yang, Wang, Zhang et al. 2024,《Parallelizing Linear Transformers with the Delta Rule
  over Sequence Length》（DeltaNet 的 chunk-parallel + WY 表示）。
- **delta rule**：Widrow & Hoff 1960（最小均方 / LMS）；快速权重（Schmidhuber 1992；Schlag et al. 2021）。
