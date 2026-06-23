# 13-gated-deltanet · 外部来源登记

本章覆盖 **GDN（Gated DeltaNet v1，Qwen3-Next）** 与 **GDN-2（Gated DeltaNet 2，Qwen3.5）** 两代。
自写简要版（gated delta rule recurrent）+ 深度优化版（**完整解耦自 fla** 的 chunk-parallel triton
kernel）。汇总见仓库根 [`NOTICE`](../NOTICE)。

## 深度优化版：完整解耦自 fla 的 GDN / GDN-2 kernel

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/fla-org/flash-linear-attention · commit `0b27f7b` · MIT |
| GDN 拷贝文件（10 个，`_fla_gdn_*`） | gated_delta_rule：`chunk/chunk_fwd/gate/wy_fast`；common：`chunk_delta_h/chunk_o/gate`；`l2norm`；`cumsum`；`naive`（参考） |
| GDN-2 特有文件（13 个，`_fla_gdn2_*`） | gdn2：`chunk/chunk_bwd/chunk_fwd/chunk_intra/chunk_intra_token_parallel/wy_fast`；复用 kda：`chunk_bwd/chunk_intra/chunk_intra_token_parallel/gate/wy_fast`；`gla.chunk`；common：`chunk_h`；`naive`（参考） |
| GDN-2 复用 GDN 已拷共享文件 | `_fla_gdn_chunk_delta_h` / `_fla_gdn_l2norm` / `_fla_gdn_cumsum` / `_fla_gdn_compat` |

### 退化谱系（一条线理解四代）

```
DeltaNet（第 11 章）          Sₜ = (I − βkkᵀ) Sₜ₋₁ + βkvᵀ              无门控
   └─ + per-channel 门控 g →  KDA（第 12 章）       diag(exp g) + 标量 β       Kimi
   └─ + per-head 标量门控 g →  GDN v1（本章）        diag(exp g) + 标量 β       Qwen3-Next
        └─ erase/write 解耦 → GDN-2（本章）  (I − k(b⊙k)ᵀ)diag(exp g)S + k(w⊙v)ᵀ  Qwen3.5
```

令 GDN-2 的 `b=w=β`（标量）即退化为 KDA；再令 `g` 退化为 per-head 标量即 GDN v1；再令 `g≡0` 即 DeltaNet。

### 完整解耦的三道关收敛

GDN/GDN-2 的朴素依赖闭包发散（GDN-2 还跨 gdn2 + kda + gla + common 四处）。收敛靠三招（与 12-KDA 同）：

1. **no-op dispatch**：绕过 `backends.dispatch` 的后端分派（CP / TileLang / full attention 全不进闭包）；
2. **cp stub**：`chunk_fwd/bwd` 直接 import context-parallel（多卡）的 `FLACPContext` 与
   `chunk_gated_delta_rule_*_pre_process`；单卡 `cp_context=None` 不会调用，`_fla_gdn_compat.py`
   把它们 stub 成占位 → 多卡代码不进闭包；
3. **跨章复用**：GDN-2 借 KDA 的 `chunk_intra/gate/wy_fast` 与 GLA 的 `gla.chunk`（连同源文件拷入本地）。

模块名用 `_fla_gdn_` / `_fla_gdn2_` 唯一前缀（与 10/11/12 隔离，避免全库测试时同名冲突）。

> 注意：GDN/GDN-2 均需 `use_qk_l2norm_in_kernel=True`（delta rule 标配；否则未归一化的 k 让
> `(I−k(b⊙k)ᵀ)` 谱半径爆炸 → 状态发散出 nan）。
> `test_gdn.py` 的 `test_*_faithful_vs_fla` / `test_*_varlen_vs_fla` 验证本地解耦 ≡ fla 原版（定长 + 变长，
> **bitwise，max_abs=0**）。

## 本仓库自写

- `gdn.py`：`gated_delta_recurrent` —— GDN v1（per-head 标量门控 + delta 纠错），ground truth。
- `gdn2.py`：`gdn2_recurrent` —— GDN-2（per-channel erase/write 双门控解耦），ground truth。
- `gdn_triton.py` / `gdn2_triton.py`：深度优化版入口；`_fla_gdn_compat.py`：薄适配（KDA 超集 + cp stub）。
- `test_gdn.py`（GDN 4 + GDN-2 4）/ `bench.py` / `gdn.ipynb`。

## 算法来源

- **GDN（Gated DeltaNet）**：Yang et al. 2024《Gated Delta Networks: Improving Mamba2 with Delta Rule》
  （delta rule + 门控衰减）；Qwen3-Next 采用（每 4 层 1 层 full attention，~75% 层用 GDN）。
- **GDN-2**：Qwen3.5 引入，把 delta rule 的 erase（擦除）与 write（写入）解耦成 per-channel 双门控
  `b∈R^K`（key 轴）/ `w∈R^V`（value 轴），表达力更强。
