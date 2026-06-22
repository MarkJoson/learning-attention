# 11 · DeltaNet（delta rule：纠错式状态更新 / 快速权重）

> linear attention（10 章）的状态**只加不减**：`Sₜ = Sₜ₋₁ + kₜvₜᵀ`。写入新记忆时从不擦除旧的，
> 键冲突（不同时刻用相近的 key）会让旧值残留、互相污染。**DeltaNet** 用 **delta rule**（Widrow-Hoff
> 学习规则 / 快速权重）做**纠错式更新**：写入前先用当前 key 查询旧状态、算预测误差，只写入误差 ——
> 等价于"先擦掉这个 key 方向的旧记忆，再写入新值"。
>
> 两个版本：`deltanet.py`（自写 recurrent + WY chunked，ground truth）+ 深度优化版（**完整解耦自 fla**
> 的 chunk-parallel triton kernel，见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. 把状态矩阵看作一个"在线学习的线性回归器"

把状态 `S ∈ ℝ^{d_k×d_v}` 看作一个线性映射：给一个 key，它输出一个 value 的预测 `v̂ = Sᵀk`。
attention 的"记忆"本质就是"key→value 的关联"。那么"写入 (kₜ, vₜ)"应该让 S 在 kₜ 处输出 vₜ。

- **linear attention** 的写法是 `Sₜ = Sₜ₋₁ + kₜvₜᵀ` —— 直接叠加，不管 S 在 kₜ 处原来输出什么。若
  之前已写过相近的 key，新旧 value 叠加在一起，读出时混叠。
- **delta rule** 把它当成一步**梯度下降 / 最小二乘的在线更新**：最小化预测误差 `½‖Sᵀkₜ − vₜ‖²`，
  对 S 求梯度得 `kₜ(Sᵀkₜ − vₜ)ᵀ`，按步长 βₜ 更新：

$$\hat v_t = S_{t-1}^\top k_t,\qquad S_t = S_{t-1} - \beta_t\, k_t(\hat v_t - v_t)^\top
  = S_{t-1} + \beta_t\, k_t (v_t - \hat v_t)^\top$$

这就是 **Widrow-Hoff / LMS 规则**，也是"快速权重"（fast weights）里把权重当记忆在线改写的思路。

---

## 2. delta rule = 定向擦写

把上式展开，能看清它在"擦"什么：

$$S_t = S_{t-1} + \beta_t k_t v_t^\top - \beta_t k_t (S_{t-1}^\top k_t)^\top
     = S_{t-1}\underbrace{(I - \beta_t k_t k_t^\top)}_{\text{擦掉 }k_t\text{ 方向}} + \underbrace{\beta_t v_t k_t^\top}_{\text{写入新值}}$$

`(I − βₜkₜkₜᵀ)` 是一个沿 kₜ 方向的**收缩算子**（householder-like）：它把 S 中与 kₜ 对齐的分量按 βₜ
缩掉，再加上 βₜvₜkₜᵀ。所以 delta rule 不是 GLA 那种"所有维度按固定门控衰减"，而是**只在当前 key
的方向上定向擦写** —— βₜ=1 时完全覆盖该方向的旧记忆，βₜ=0 时不写。这更接近"在线学习/检索"。

`deltanet.py:delta_rule_recurrent` 逐步实现这三行（预测 `v̂` → 误差 ×βₜ → 外积写入 → 读出 `qᵀS`）。
配置上与 fla 对齐：q 缩放 `1/√d`、可选对 q/k 做 L2 归一化（`use_qk_l2norm`）。

---

## 3. WY 表示：把"逐步擦写"变成块内可并行

recurrent 形式是严格串行的（Sₜ 依赖 Sₜ₋₁）。要在 GPU 上训练高效，需要 chunk-parallel。难点在于：一个
chunk 内连续 t 步的更新是**矩阵连乘** `∏ₜ(I − βₜkₜkₜᵀ)`，不能简单并行。

**WY 表示**（来自 Householder 变换的 WY 形式）解决了它。对一个长度 C 的 chunk，把 C 步更新累积成一个
变换，可证明它等价于一个**下三角线性系统的逆**：

$$T = \big(I - \operatorname{tril}(\operatorname{diag}(\beta)\,KK^\top,\,-1)\big)^{-1}$$

（K 是 chunk 内的 key 矩阵，`tril(·,−1)` 取严格下三角。）有了 T，就能把"逐步擦写"一次算出来：

$$u = T\,(\beta\odot V),\qquad w = T\,(\beta\odot K)$$

`deltanet.py:delta_rule_chunked` 里：先用**前代法**（forward substitution）O(C²) 解出 T（`for i in
range(1,C)` 那段），得到 `u, w`；然后**块间**只传一个状态 S 做递归：

```
o_i = q_i·S(块间)  +  tril(q_i k_iᵀ)·(u_i − w_i·S)(块内)
S  += k_iᵀ·(u_i − w_i·S)
```

于是块内是稠密 GEMM（并行度高），块间是 O(块数) 递归（线性复杂度）——既快又是 O(T)。这正是深度优化版
triton kernel 的算法骨架（`_fla_wy_fast` 算 T/u/w，`_fla_solve_tril` 解下三角，`_fla_chunk_delta_h`
传块间状态）。`test_deltanet.py` 验证 recurrent ≡ chunked。

---

## 4. 深度优化版：完整解耦自 fla（脱离 fla 独立运行）

DeltaNet 的 fla kernel 由 8 个 triton 文件组成（WY 表示 / 块间状态 / 输出 / L2norm / 解三角）。它比 GLA
耦合深，还通过 `backends.dispatch` 静态引入多卡 CP、TileLang 后端、full attention（朴素闭包 ~27 文件）。

本仓库**完整解耦**（计算逻辑一字未改）：关键是用 **no-op dispatch** 绕过后端分派（那些 CP/TileLang 全
经 dispatch 引入），把闭包收敛到 8 个 triton 核心文件 + 一层薄适配 `_fla_compat.py`。详见 `SOURCES.md`。

```
_fla_delta_chunk.py   入口 + chunk-parallel 主体（块内 + 块间合并）
_fla_wy_fast.py       WY 表示：算 T=(I−tril(βKKᵀ))⁻¹、u、w
_fla_solve_tril.py    批量解下三角线性系统（WY 求逆）
_fla_chunk_delta_h.py 块间递归状态 H 的前向/反向
_fla_chunk_o.py       输出计算
_fla_chunk_scaled_dot_kkt.py / _fla_l2norm.py / _fla_compat.py
```

入口：

```python
from deltanet_triton import chunk_delta_rule   # fla 原生接口 [B,T,H,D]
from deltanet_triton import delta_chunk         # [B,H,T,D] 便捷封装（对接简要版）
```

`test_triton_faithful_vs_fla` / `test_varlen_vs_fla` 验证本地解耦 kernel 与 fla 原版 **bitwise 一致**
（定长 + 变长 cu_seqlens），`test_triton_vs_recurrent` 验证它与 recurrent ground truth 对齐。

---

## 5. 测试与运行

```bash
pytest 11-deltanet/test_deltanet.py -v   # 自洽(recurrent≡chunked) + 忠实(vs fla) + vs recurrent + varlen + bwd
python 11-deltanet/bench.py              # delta rule 的 O(S) vs full attention
```

> 学习路径：先读 `deltanet.py` + §1–§3 弄懂"delta rule 为什么是纠错式擦写、WY 表示如何让它块内并行"，
> 再对照 §4 读解耦的真实 kernel。

**上一章** ← 10-linear-attention（GLA） · **下一章** → 12-kda（Kimi Delta Attention：gated delta + 细粒度门控）
