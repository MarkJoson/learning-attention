# 01 · Vanilla Attention 与 Online Softmax

> 一切高效 Attention 的地基。本章不碰任何 GPU kernel，只用纯 PyTorch 把
> **标准注意力**和 **online softmax 分块算法**讲透 —— 后者正是 FlashAttention 的算法内核。

---

## 1. 标准缩放点积注意力

给定 query $Q\in\mathbb{R}^{S_q\times d}$、key $K\in\mathbb{R}^{S_k\times d}$、value $V\in\mathbb{R}^{S_k\times d}$：

$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right)V
$$

其中 $M$ 是掩码（causal 时上三角为 $-\infty$）。逐步拆开就是 `common/reference.py` 里的 `naive_attention`：

```
S = Q @ K^T * scale     # (Sq, Sk)  打分
S = S + mask            # 因果掩码
P = softmax(S, dim=-1)  # 注意力权重
O = P @ V               # 加权求和
```

**问题**：中间矩阵 $S, P$ 的大小是 $S_q\times S_k$，显存 $O(S^2)$。序列一长就爆显存 ——
本章 `bench.py` 实测 `seqlen=4096` 时 naive 占用 **8696 MB**，而下面的 online 版本只要 **654 MB**。

---

## 2. 数值稳定的 Softmax：减去最大值

直接 $e^{x_i}$ 会溢出。标准做法是减去每行最大值 $m=\max_i x_i$（不改变结果）：

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

这一步是 online softmax 的前提：我们要维护这个 "running max" $m$。

---

## 3. Online Softmax：不构造完整矩阵，增量更新

关键洞察：softmax 的分子分母都是**对 key 维求和**，可以**分块累加**，无需一次性看到所有 key。

把 key/value 沿序列切成若干块 $K^{(1)},V^{(1)},K^{(2)},V^{(2)},\dots$。对每个 query 行维护三个 running 量：

| 量 | 含义 |
|---|---|
| $m$ | 到目前为止见过的最大 score（running max） |
| $\ell$ | 归一化分母 $\sum e^{s-m}$（running sum） |
| $\mathbf{o}$ | 未归一化输出 $\sum e^{s-m}\,\mathbf{v}$（running output） |

处理新块、其块内 score 为 $s^{(j)}$、最大值 $\tilde m=\max s^{(j)}$ 时，先更新全局最大：

$$
m^{\text{new}} = \max(m,\ \tilde m)
$$

旧的 $\ell,\mathbf{o}$ 是以旧基准 $m$ 算的，需用**修正因子** $\alpha=e^{m-m^{\text{new}}}$ 缩放到新基准，再并入新块贡献：

$$
\ell^{\text{new}} = \alpha\,\ell + \textstyle\sum_k e^{s^{(j)}_k - m^{\text{new}}}
$$

$$
\mathbf{o}^{\text{new}} = \alpha\,\mathbf{o} + \textstyle\sum_k e^{s^{(j)}_k - m^{\text{new}}}\,\mathbf{v}^{(j)}_k
$$

遍历完所有块后归一化：

$$
O = \mathbf{o}\,/\,\ell
$$

这套更新对应 `vanilla.py::online_softmax_attention` 的内循环，逐行可对照：

```python
m_new = torch.maximum(m, s.amax(dim=-1))   # m^new
p     = torch.exp(s - m_new[..., None])     # e^{s - m^new}
corr  = torch.exp(m - m_new)                # α 修正因子
l     = l * corr + p.sum(dim=-1)            # ℓ^new
acc   = acc * corr[..., None] + p @ vj      # o^new
m     = m_new
```

> 这就是 **FlashAttention v2** 的算法骨架。FlashAttention 的"魔法"不在算法本身（就是上面的
> online softmax），而在于把它**融合进单个 GPU kernel**，让 $S,P$ 只活在片上 SRAM 里、
> 从不落地到 HBM。算法见本章，kernel 见 [02-flash-attention](../02-flash-attention/)。

---

## 4. 因果掩码与"跳过上三角块"

causal 下 query $i$ 只能看到 key $\le i$。分块时若**整个 key 块都在 query 块之后**（落在上三角），
该块对结果零贡献，可以直接跳过 —— `vanilla.py` 里：

```python
if causal and j > q_max_pos:   # 整块在上三角
    continue
```

这既是正确性需要（避免 running max 仍是 $-\infty$ 时对全 $-\infty$ 行求 softmax 出 NaN），
也是 FlashAttention causal 模式下省掉约一半计算的来源。

> 当 $S_q\ne S_k$（decode / cross-attention）时，让 query 末端对齐 key 末端：
> query 的全局坐标加上偏移 $S_k-S_q$。`reference.py` 用 `tril(diagonal=Sk-Sq)` 实现，
> `vanilla.py` 用 `align = Sk - Sq` 实现，两者一致。

---

## 5. 实测：算法对了，还不够快

`python bench.py`（RTX 4090, B=4 H=16 D=64 causal fp16）：

| seqlen | naive 延迟 | naive 显存 | online 延迟 | online 显存 | SDPA 延迟 | SDPA 显存 |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 0.79 ms | 420 MB | 1.90 ms | 320 MB | 0.054 ms | 280 MB |
| 1024 | 3.16 ms | 833 MB | 6.62 ms | 366 MB | 0.112 ms | 296 MB |
| 2048 | 12.6 ms | 2428 MB | 26.7 ms | 462 MB | 0.302 ms | 329 MB |
| 4096 | 49.6 ms | 8696 MB | 98.9 ms | **654 MB** | 0.998 ms | 393 MB |

读这张表：

1. **online softmax 确实省显存**：4096 时 654 MB vs naive 8696 MB（≈13×）。
2. **但纯 Python 分块反而更慢**：98.9 ms vs naive 49.6 ms —— Python 双重循环 + 反复读写 HBM 的开销，盖过了省显存的好处。
3. **SDPA 融合 kernel 又快又省**：0.998 ms（比 online 快 ~100×）、393 MB。

> 这正是 FlashAttention 的动机：**online softmax 算法 + GPU kernel 融合**，缺一不可。

---

## 文件

| 文件 | 作用 |
|---|---|
| `vanilla.py` | `online_softmax_attention` 分块实现（教学核心）；`naive_attention` 复用自 `common` |
| `test_vanilla.py` | 数值正确性：online vs naive / SDPA，含分块无关性、GQA、不等长序列 |
| `bench.py` | naive / online / SDPA 的延迟与显存对比 |

```bash
pytest 01-vanilla-attention/ -v     # 31 个用例
python 01-vanilla-attention/bench.py
```

**下一章** → [02-flash-attention](../02-flash-attention/)：把本章的 online softmax 用 Triton kernel 融合实现（拷贝自 triton 官方教程）。
