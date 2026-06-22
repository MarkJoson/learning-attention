# 10 · Linear Attention & GLA（线性注意力 / 门控线性注意力）

> 标准 attention 是 `softmax(QKᵀ)V`，复杂度 O(S²)。**linear attention** 去掉 softmax、换成 feature
> map φ，靠**矩阵结合律**把它变成 O(S) 的线性递归 —— 既能像 transformer 一样并行训练，又能像 RNN
> 一样 O(1) 推理。**GLA** 给它加一个 data-dependent 的衰减门控，补回"选择性遗忘"的表达力。
>
> 本章两个版本：`linear.py`（自写三形式 + GLA，纯 PyTorch ground truth）+ 深度优化版（**解耦自
> fla 的 GLA chunk-parallel triton kernel**，见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. 从 softmax 到 linear：一个结合律的事

标准注意力对每个 query 算 `softmax(qᵢ·Kᵀ)·V`，softmax 让它必须**先**算出 S×S 的分数矩阵 → O(S²)。
linear attention 把 `exp(qᵢ·kⱼ)` 换成 `φ(qᵢ)·φ(kⱼ)`（φ 是非负 feature map），于是：

$$o_i = \frac{\sum_{j\le i} (\phi(q_i)\cdot\phi(k_j))\, v_j}{\sum_{j\le i}\phi(q_i)\cdot\phi(k_j)}
      = \frac{\phi(q_i)\sum_{j\le i}\phi(k_j)v_j^\top}{\phi(q_i)\sum_{j\le i}\phi(k_j)}$$

关键在分子：`Σ φ(kⱼ)vⱼᵀ` 是一个 **D×D 的"状态矩阵" S**，与 query 无关。先把它累起来，再让每个 query
去乘 —— **复杂度 O(S·D²)，在序列长度上线性**。（`linear.py:feature_map` 用 elu(x)+1 保证 φ>0。本仓库
的 reference 省略了分母归一，聚焦"结合律省算"这一核心机制。）

---

## 2. 三种等价形式（`linear.py`）

同一个 causal linear attention，三种算法、同一结果（`test_linear.py` 验证）：

| 形式 | 怎么算 | 复杂度 | 用途 |
|---|---|---|---|
| `linear_attn_parallel` | φ(Q)(φ(K)ᵀV) + 下三角 mask | O(S²·D) | 概念最清晰 |
| `linear_attn_recurrent` | Sₜ = Sₜ₋₁ + φ(kₜ)vₜᵀ, oₜ = φ(qₜ)Sₜ | O(S·D²) | **推理**：O(1)/步，RNN 形式（ground truth）|
| `linear_attn_chunked` | 块内 parallel + 块间传 state | O(S·D²) | **训练**：并行度高，GPU 友好（kernel 用这个）|

**chunked 是训练高效的关键**：把序列切成块，块内用矩阵乘并行算（`下三角(φQ_c φK_cᵀ)·V_c`），块间只
传一个累积状态 state（`φ(Q_c)·state`）。既有并行度（块内 GEMM），又是线性复杂度（块间 O(块数)）。
这正是下面 triton kernel 的算法骨架。

> 数值提示：linear attention 没有 softmax 归一，输出值域随 S 增长，parallel 的大矩阵乘与 recurrent
> 的逐步累加在大 S 下有可见的浮点差异（`test_linear.py` 对此放宽了容差）。

---

## 3. GLA：给状态加"选择性遗忘"

linear attention 的状态**只加不减**，远古信息永不衰减 —— 表达力受限。**GLA** 加一个 data-dependent
的衰减门控 αₜ（由输入算出的遗忘门，每个 key 维度一个）：

$$S_t = \operatorname{diag}(\alpha_t)\, S_{t-1} + k_t v_t^\top,\qquad o_t = q_t S_t,\qquad \alpha_t=\exp(g_t)\in(0,1)^K$$

gₜ ≤ 0 是 log 空间的门控。αₜ 让状态**选择性遗忘**：某些维度快速衰减（关注近期）、某些维度保留
（记住远期）。比固定衰减（RetNet 的标量 γ）更强。`linear.py:gla_recurrent` 是其 ground truth，
与 fla 的 kernel 语义对齐（scale=1/√K 作用在 q）。gate=0（不遗忘）时 GLA 退化为 linear attention。

---

## 4. 深度优化版：解耦自 fla 的 GLA chunk-parallel kernel

GLA 的 chunk-parallel 形式 = §2 的 chunked linear attention **+ 门控衰减**：块内 attention 要按
`exp(bᵢ - bⱼ)` 加权（bᵢ 是块内累积衰减 cumsum(g)），块间状态 H 也要带衰减传递。这套 kernel 由
fla 提供，本仓库把它**解耦到本地**（核心三件套，计算逻辑一字未改）：

| 文件 | 职责 |
|---|---|
| `_fla_gla_chunk.py` | chunk-parallel 主体：块内并行 attention + 块间合并 + 门控 |
| `_fla_chunk_h.py` | 块间递归状态 H 的前向/反向（带门控衰减） |
| `_fla_cumsum.py` | 门控的块内累积 `chunk_local_cumsum`（log decay 的 prefix-sum） |
| `_fla_compat.py` | **薄适配层（自写）**：复现 fla 框架辅助符号，使上面三件套脱离 fla |

**怎么脱离 fla**：fla 的 kernel 计算本身只用 torch/triton，对框架的依赖都是"工具/适配"符号
（`exp2`、autotune 缓存、`check_shared_mem`、`input_guard`、变长索引）。我们只改 import 指向
`_fla_compat.py`（`fla_cache_autotune`→标准 `triton.autotune`、`check_shared_mem`→torch 查 smem、
变长索引→从 fla 拷贝的纯 torch 函数，支持 cu_seqlens），计算逻辑零改动。`test_gla_triton.py::test_faithful_vs_fla` 验证本地解耦
kernel 与 fla 原版**近 bitwise 一致**（max diff 0.0）。

入口：

```python
from gla_triton import chunk_gla    # fla 原生接口，layout [B,T,H,D]
from gla_triton import gla_chunk    # [B,H,T,D] 便捷封装（对接简要版 linear.gla_recurrent）
```

---

## 5. 测试与运行

```bash
# 简要版三形式等价 + GLA 性质（纯 PyTorch）
pytest 10-linear-attention/test_linear.py -v

# 深度优化版（解耦 GLA kernel）：忠实性 vs fla + 对齐 recurrent + fwd/bwd
pytest 10-linear-attention/test_gla_triton.py -v

python 10-linear-attention/bench.py        # linear vs full attention 的复杂度对比
```

- 简要版 `linear.py`：零额外依赖，讲清"结合律省算 + 门控"机制。
- 深度优化版：解耦后仅需 `torch/triton`（+`einops`）即可运行；`test_faithful_vs_fla` 用 `fla` 作对照
  （`pip install --no-deps flash-linear-attention`，仅测试需要）。

> 学习路径：先读 `linear.py` + §1–§3 弄懂"linear attention 为何 O(S)、GLA 的门控加在哪"，再对照
> §4 读解耦的 chunk-parallel kernel，弄懂"训练时怎么既并行又线性"。

**上一章** ← 09-dynamic-sparse · **下一章** → 11-deltanet（DeltaNet：用 delta rule 让状态做"增量更新"）
