# 13 · Gated DeltaNet（GDN / GDN-2）—— 门控 delta rule 的两代

> 把第 11 章的 **DeltaNet（delta rule 纠错）** 与第 12 章的 **KDA（门控遗忘）** 推到生产线注意力的
> 最新形态：**GDN v1**（Qwen3-Next 主干）与 **GDN-2**（Qwen3.5，erase/write 双门控解耦）。
>
> 每代两个版本：自写 recurrent（`gdn.py` / `gdn2.py`，ground truth）+ 深度优化版（**完整解耦自 fla**
> 的 chunk-parallel triton kernel，GDN 10 文件 / GDN-2 13 文件，见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. 退化谱系：一条线看懂四代

线性注意力的全部差异，就在"状态矩阵 S 每步怎么更新"这一行：

| 章 | 机制 | 状态更新 `Sₜ =` | 门控粒度 |
|---|---|---|---|
| 10 | GLA | `diag(exp gₜ) Sₜ₋₁ + kₜvₜᵀ` | per-channel 衰减，但写入只加不纠错 |
| 11 | DeltaNet | `(I − βₜkₜkₜᵀ) Sₜ₋₁ + βₜkₜvₜᵀ` | delta 纠错，但无遗忘 |
| 12 | KDA | `diag(exp gₜ) Sₜ₋₁ + βₜkₜ(vₜ−v̂ₜ)ᵀ` | per-channel 门控 + delta 纠错 |
| **13** | **GDN v1** | `diag(exp gₜ) Sₜ₋₁ + βₜkₜ(vₜ−v̂ₜ)ᵀ` | **per-head 标量**门控 + delta 纠错 |
| **13** | **GDN-2** | `(I − kₜ(bₜ⊙kₜ)ᵀ) diag(exp gₜ) Sₜ₋₁ + kₜ(wₜ⊙vₜ)ᵀ` | **erase/write 双门控**（per-channel） |

```
DeltaNet  ──+per-channel 门控 g──►  KDA（Kimi）
   │                                  │ g 退化为 per-head 标量
   │                                  ▼
   └──────────────────────────►  GDN v1（Qwen3-Next）
                                      │ erase/write 解耦
                                      ▼
                                  GDN-2（Qwen3.5）
```

退化验证：GDN-2 令 `b=w=β`（标量）→ KDA；再令 `g` 退化为 per-head 标量 → GDN v1；再令 `g≡0` → DeltaNet。

---

## 2. GDN v1：per-head 标量门控的 gated delta rule

$$\hat S=\exp(g_t)\,S_{t-1},\qquad \hat v_t=\hat S^\top k_t,\qquad
  S_t=\hat S+\beta_t\,k_t\,(v_t-\hat v_t)^\top.$$

与 KDA 同属 gated delta rule，**唯一区别是门控粒度**：GDN 的 `gₜ` 是 **per-head 标量**（整个状态矩阵
统一遗忘），KDA 是 per-channel 向量（每个 key 通道独立遗忘）。标量门控更省（每 head 1 个数 vs D 个数），
是 Qwen3-Next 的工程选择（配 Causal Conv1D + L2norm(Q/K)，每 4 层插 1 层 full attention）。
`gdn.py:gated_delta_recurrent` 逐步实现（注意 g 形状 `[B,H,T]`，比 KDA 少一维 D）。

---

## 3. GDN-2：把 erase 与 write 解耦成两个门

GDN/KDA 的 delta rule 用**同一个标量 β** 同时控制「擦除旧值」和「写入新值」。GDN-2 的洞察是这两件事
该分开：

$$S_t=\underbrace{\big(I-k_t\,(b_t\odot k_t)^\top\big)}_{\text{erase 门 }b\in\mathbb R^K}
        \operatorname{diag}(\exp g_t)\,S_{t-1}
      +\underbrace{k_t\,(w_t\odot v_t)^\top}_{\text{write 门 }w\in\mathbb R^V}.$$

- **erase 门 `b∈R^K`**（key 轴，逐通道）：`(I − k(b⊙k)ᵀ)` 沿 kₜ 方向擦除旧内容，b 控制擦多少；
- **write 门 `w∈R^V`**（value 轴，逐通道）：`k(w⊙v)ᵀ` 写入新值，w 控制写多少；
- **decay 门 `g∈R^K`**：per-channel 衰减（与 KDA 同）。

三个门各管一件事，比单标量 β 表达力更强。`gdn2.py:gdn2_recurrent` 逐步实现这套 erase→write→读出
（与 fla `gdn2_naive.py` 数学等价）。**数学推导、WY 表示、chunk 并行的完整拆解见 `gdn.ipynb`。**

---

## 4. 深度优化版：完整解耦自 fla

GDN 10 文件、GDN-2 13 文件（GDN-2 还复用 KDA 的 `chunk_intra/gate/wy` 与 GLA 的 `gla.chunk`）。完整
解耦（计算逻辑一字未改）靠三招收敛（详见 `SOURCES.md`）：**no-op dispatch**（绕后端分派）+ **cp stub**
（绕多卡 context-parallel）+ **跨章复用** kernel。

```python
from gdn_triton import gdn_chunk     # GDN  [B,H,T,D] 封装（g/beta 是 [B,H,T]）
from gdn2_triton import gdn2_chunk   # GDN-2 [B,H,T,D] 封装（g/b [B,H,T,K]、w [B,H,T,V]）
# 两者都需 use_qk_l2norm_in_kernel=True（delta rule 标配，否则 bf16 状态发散为 nan）
```

`test_gdn.py` 的 `*_faithful_vs_fla` / `*_varlen_vs_fla` 验证本地解耦 ≡ fla 原版（定长 + 变长，
**bitwise，max_abs=0**），`*_kernel_vs_recurrent` 验证 kernel 与简要版 recurrent 对齐。

---

## 5. 测试与运行

```bash
pytest 13-gated-deltanet/test_gdn.py -v   # GDN 4 + GDN-2 4：忠实(bitwise) + kernel≡recurrent + varlen + bwd
python 13-gated-deltanet/bench.py         # GDN/GDN-2 的 O(S) vs full attention O(S²)
jupyter notebook 13-gated-deltanet/gdn.ipynb
```

RTX 4090 实测（B=4 H=8 D=128 causal）：GDN 在 S=8192 达 **1.69×** full attention；GDN-2 双门控每步更重，
交叉点更靠后（长序列才显优势）。

> 学习路径：先读 §1 退化谱系表，把 DeltaNet→KDA→GDN→GDN-2 串成一条线；再读 `gdn.py`/`gdn2.py` 看
> 两代 recurrent 差异；最后用 `gdn.ipynb` 啃 GDN-2 的 erase/write 数学推导与 chunk 并行。

**上一章** ← 12-kda · gated delta rule 线（11 DeltaNet → 12 KDA → 13 GDN/GDN-2）的生产级终点。
