# 12 · KDA（Kimi Delta Attention）—— gated delta rule

> 线性注意力四章的终点。KDA 把前面的机制**合二为一**：用 **GLA 的逐通道衰减门控**（选择性遗忘）+
> **DeltaNet 的 delta rule 纠错**（定向擦写），是 Moonshot **Kimi Linear** 的核心。
>
> 两个版本：`kda.py`（自写 gated delta rule recurrent，ground truth）+ 深度优化版（**完整解耦自 fla**
> 的 chunk-parallel triton kernel，14 个文件，见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. 四章演进：状态更新规则一张表

线性注意力的全部差异，就在"状态矩阵 S 每步怎么更新"这一行：

| 章 | 机制 | 状态更新 `Sₜ =` | 缺陷/特点 |
|---|---|---|---|
| 10 | linear attn | `Sₜ₋₁ + kₜvₜᵀ` | 只加不减，键冲突混叠 |
| 10 | **GLA** | `diag(exp(gₜ)) Sₜ₋₁ + kₜvₜᵀ` | 加 per-channel 衰减门控（选择性遗忘），但写入仍只加 |
| 11 | **DeltaNet** | `Sₜ₋₁(I − βₜkₜkₜᵀ) + βₜvₜkₜᵀ` | delta 纠错（定向擦写），但无遗忘 |
| 12 | **KDA** | `diag(exp(gₜ)) Sₜ₋₁ + βₜkₜ(vₜ − v̂ₜ)ᵀ` | **门控遗忘 + delta 纠错**，二者兼得 |

KDA = GLA（门控）⊕ DeltaNet（纠错）。

---

## 2. KDA = gated delta rule

展开看：每步先**门控衰减**，再**delta 纠错写入**：

$$\hat S = \operatorname{diag}(\exp(g_t))\,S_{t-1},\qquad
  \hat v_t = \hat S^\top k_t,\qquad
  S_t = \hat S + \beta_t\, k_t (v_t - \hat v_t)^\top.$$

- `exp(gₜ)`（gₜ 是 log-space、**per-channel** 的衰减）让状态按通道选择性遗忘（来自 GLA）；
- `βₜ kₜ(vₜ − v̂ₜ)ᵀ` 用 kₜ 查询门控后的旧状态、算误差、定向擦写（来自 DeltaNet delta rule）。

于是 KDA 既能"按通道遗忘旧信息"，又能"在 key 方向纠错覆盖" —— 比单独的 GLA（不纠错）或 DeltaNet
（不遗忘）记忆管理都更强。这正是 Kimi Linear 用它做长上下文骨干的原因。`kda.py:kda_recurrent`
逐步实现这四步（与 fla kernel 对齐：q 缩放 1/√d、q/k 做 L2 归一化）。

---

## 3. 深度优化版：完整解耦自 fla（本仓库最复杂的解耦）

KDA 的 fla kernel 由 **14 个 triton 文件**组成，朴素依赖闭包 ~27 文件且发散。本仓库**完整解耦**
（计算逻辑一字未改），靠三招收敛（详见 `SOURCES.md`）：

1. **no-op dispatch** 绕过后端分派（CP / TileLang）；
2. **cp stub** 绕过多卡 context-parallel（单卡 `cp_context=None` 不用）；
3. **复用 gla.chunk**（KDA 借 GLA 的输出函数）。

```
_fla_kda_chunk.py   入口 + chunk-parallel 主体
_fla_kda_chunk_intra.py / _intra_tp.py   块内（chunk_intra，KDA 最大的 kernel）
_fla_kda_gate.py    门控（fine-grained gate）
_fla_kda_wy_fast.py delta 的 WY 表示
_fla_kda_chunk_delta_h.py / _chunk_h.py  块间状态
_fla_kda_gla_chunk.py  复用的 GLA 输出函数
_fla_kda_cumsum.py / _l2norm.py / _common_gate.py / _fla_kda_compat.py
```

入口：

```python
from kda_triton import chunk_kda   # fla 原生接口 [B,T,H,D]
from kda_triton import kda_chunk   # [B,H,T,D] 便捷封装（对接简要版）
# 注意 use_qk_l2norm_in_kernel=True
```

`test_faithful_vs_fla` / `test_varlen_vs_fla` 验证本地解耦 ≡ fla 原版（定长 + 变长，**bitwise**），
`test_kernel_vs_recurrent` 验证它与简要版 gated-delta recurrent 对齐。

---

## 4. 测试与运行

```bash
pytest 12-kda/test_kda.py -v   # 忠实(vs fla bitwise) + kernel≡recurrent + varlen + bwd
python 12-kda/bench.py         # KDA 的 O(S) vs full attention
```

> 学习路径：先读 `kda.py` + §1 的演进表，看清 KDA 怎么把 GLA 门控与 DeltaNet 纠错合一；再对照 §3
> 读解耦的真实 kernel（它是四章里最复杂的 fla kernel）。

**上一章** ← 11-deltanet · 线性注意力线（10 linear/GLA → 11 DeltaNet → 12 KDA）到此完成。
