# 12-kda · 外部来源登记

自写简要版（gated delta rule recurrent）+ 深度优化版（**完整解耦自 fla** 的 KDA chunk-parallel
triton kernel）。汇总见仓库根 [`NOTICE`](../NOTICE)。

## 深度优化版：完整解耦自 fla 的 KDA kernel（本仓库最复杂的一章）

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/fla-org/flash-linear-attention · commit `0b27f7b` · MIT |
| 拷贝文件（14 个 triton） | kda：`chunk/chunk_bwd/chunk_fwd/chunk_intra/chunk_intra_token_parallel/gate/wy_fast`；common：`chunk_delta_h/chunk_h/gate`；`gla.chunk`（复用）；`l2norm`；`cumsum`；`naive`（参考） |

### 完整解耦的三道关收敛

KDA 的朴素依赖闭包 ~27 文件且发散，比 DeltaNet 复杂得多。收敛靠三招：

1. **no-op dispatch**：绕过 `backends.dispatch` 的后端分派（CP / TileLang / full attention 全不进闭包）；
2. **cp stub**：KDA 的 `chunk_fwd/bwd` **直接** import context-parallel（多卡）的 `FLACPContext` 与
   `chunk_gated_delta_rule_*_pre_process` 等；单卡 `cp_context=None` 不会调用，故 `_fla_kda_compat.py`
   把它们 stub 成占位 → 多卡代码不进闭包；
3. **复用 gla.chunk**：KDA 借 GLA 的输出函数 `chunk_gla_fwd_o_gk`（连同 `gla.chunk` 拷入本地）。

收敛到 **14 个 triton 文件**，脱离 fla 独立运行（计算逻辑一字未改）。模块名用 `_fla_kda_` 唯一前缀
（与 10/11 隔离，避免全库测试时同名冲突）。

> 注意：KDA 需 `use_qk_l2norm_in_kernel=True`（delta rule 标配；否则未归一化的 k 让 bf16 状态发散出 nan）。
> `test_kda.py::test_faithful_vs_fla` / `test_varlen_vs_fla` 验证本地解耦 ≡ fla 原版（定长 + 变长，**bitwise**）。

## 本仓库自写

- `kda.py`：`kda_recurrent` —— gated delta rule（GLA 门控 + DeltaNet 纠错），ground truth。
- `kda_triton.py`：深度优化版入口；`_fla_kda_compat.py`：薄适配（11 超集 + exp/gather/softplus/cp stub）。
- `test_kda.py` / `bench.py` / `kda.ipynb`。

## 算法来源

- **KDA（Kimi Delta Attention）**：Moonshot AI《Kimi Linear》(2025) 的核心线性注意力。
- **gated delta rule**：Yang et al. 2024《Gated Delta Networks》（delta rule + 门控）；KDA 用**细粒度
  per-channel 门控**。
