# 15 · Mamba / Mamba2：SSM 与注意力的对偶

> 状态空间模型（SSM）把序列建模成一个**线性动态系统**，本是另一条技术路线。**Mamba2** 的核心贡献 **SSD
> （State Space Duality）** 却证明了：selective SSM 等价于一种**带衰减的 masked 注意力**——于是它和本仓库的线性
> 注意力线（GLA/DeltaNet/KDA）汇成一条河。
>
> 本章：`mamba1.py`（自写 selective SSM 讲机制）+ `ssd.py`（Mamba2 SSD 的 recurrent 与 attention 两种对偶形式）+
> 深度优化版（**完整解耦自 fla `simple_gla`** 的 chunk kernel，见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. SSM 基础：把序列看成线性动态系统

连续 SSM：$h'(t)=A\,h(t)+B\,x(t),\; y(t)=C\,h(t)$。离散化（ZOH，步长 $\Delta$）后变成 RNN 式递推：

$$h_t=\bar A\,h_{t-1}+\bar B\,x_t,\qquad y_t=C\,h_t,\qquad \bar A=e^{\Delta A},\;\bar B\approx \Delta B.$$

S4 等早期 SSM 的 $A,B,C,\Delta$ **与输入无关**（LTI），可用卷积并行，但无法按内容选择性记忆。

## 2. Mamba1：selective（S6）

Mamba1 的关键改动：让 $B,C,\Delta$ **随输入变化**（data-dependent），模型据 token 内容决定"记多少、用哪部分状态"。
代价是不再 LTI、不能卷积，得用**硬件感知的 selective scan**。每个特征通道 $d$ 维护独立的 $N$ 维对角 SSM：

$$h_t[d]=e^{\Delta_t[d]A[d]}\odot h_{t-1}[d]+(\Delta_t[d]B_t)\,x_t[d],\qquad y_t[d]=C_t\cdot h_t[d].$$

`mamba1.py:selective_ssm_recurrent` 是其纯 PyTorch ground truth（真实 `selective_scan` 是 CUDA kernel，在官方
`state-spaces/mamba`，本章不提取——见 SOURCES）。

## 3. Mamba2 SSD：SSM ↔ 注意力的对偶

Mamba2 把 $A$ 简化为**标量** $a_t$（per-head），并令 $B_t=k_t,\,C_t=q_t,\,x_t=v_t$。于是 SSM 有**两种等价形式**：

**① 线性（recurrent）形式** —— $O(T)$ 推理、$O(1)$ 内存：

$$S_t=e^{g_t}\,S_{t-1}+k_tv_t^\top,\qquad y_t=q_t\,S_t.$$

这**就是标量衰减的线性注意力**（GLA 的 $g$ 是 per-channel 向量，SSD 的 $g$ 是 per-head 标量）。

**② 注意力（对偶）形式** —— $O(T^2)$ 但可并行训练。把 ① 展开：

$$y_i=\sum_{j\le i}e^{\,g^{\mathrm{cum}}_i-g^{\mathrm{cum}}_j}\,(q_i^\top k_j)\,v_j
   \;\Longrightarrow\; Y=\big(\underbrace{L\circ(QK^\top)}_{M}\big)V,\quad L_{ij}=e^{\,g^{\mathrm{cum}}_i-g^{\mathrm{cum}}_j}\ (i\ge j).$$

$M$ 是一个**半可分矩阵（1-semiseparable）**，$L$ 是"累积衰减下三角"。两种形式数学等价（**SSD 对偶**），这正是
Mamba2 既能像 RNN 一样 $O(T)$ 推理、又能像注意力一样并行训练的原因。`ssd.py` 的 `ssd_recurrent` 与
`ssd_attention_dual` 实现这两形式，`test_ssd_dual_equivalence` 数值验证它们相等。

---

## 4. 与线性注意力线的关系

| 章 | 机制 | 状态更新 / 衰减 |
|---|---|---|
| 10 | GLA | `diag(exp gₜ) Sₜ₋₁ + kₜvₜᵀ`，g **per-channel** |
| **15** | **Mamba2 SSD** | `exp(gₜ) Sₜ₋₁ + kₜvₜᵀ`，g **per-head 标量** |
| 11 | DeltaNet | 加 delta 擦除 `(I−βkkᵀ)` |
| 12 | KDA | 门控 + delta（per-channel） |

**SSD = 标量衰减 GLA**。fla 把它实现为 `simple_gla`（"simplified GLA"），所以本章的深度优化 kernel 与第 10 章 GLA
同源、同样的解耦流程。SSD 的"半可分矩阵"视角，也给了线性注意力一个统一的矩阵语言。

---

## 5. 深度优化版：完整解耦自 fla simple_gla

4 个 triton 文件（`chunk` + common `chunk_h`/`chunk_o` + `cumsum`）拷自 fla、计算逻辑一字未改，靠 **no-op dispatch**
脱离 fla 独立运行（比 KDA/GDN 简单，无 cp stub）。

```python
from ssd_triton import ssd_chunk        # [B,H,T,D] 封装，g [B,H,T] per-head 标量
# 或 from _fla_ssd_chunk import chunk_simple_gla  # fla 原生 [B,T,H,D]
```

`test_ssd_faithful_vs_fla` / `test_ssd_varlen_vs_fla` 验证本地解耦 ≡ fla 原版（定长 + 变长，**bitwise，max_abs=0**）。

---

## 6. 测试与运行

```bash
pytest 15-mamba/test_mamba.py -v   # SSD 对偶等价 + 解耦 bitwise + kernel≡recurrent + bwd + Mamba1 causal
python 15-mamba/bench.py           # SSD 的 O(S) vs full attention O(S²)
jupyter notebook 15-mamba/mamba.ipynb
```

> 学习路径：先读 §1–§2 建立 SSM/selective 的概念（`mamba1.py`）；再读 §3 + `ssd.py` 看 SSD 的两种对偶形式；最后用
> `mamba.ipynb` 把"recurrent ≡ 半可分矩阵 ≡ 标量衰减 GLA"的对偶完整推一遍。

**关联** ← 第 10 章 GLA（SSD = 标量衰减 GLA）。Mamba 把"状态空间"与"注意力"两条线在 SSD 处合一。
