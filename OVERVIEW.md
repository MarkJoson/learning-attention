# 全景导览：一页看懂 15 个变体

> 这是仓库的"地图"——先在这里建立全局视角（每个变体在做什么、彼此什么关系、从哪读起），再钻进各章细读。
> 各章细节见对应目录的 `README.md` 与 `*.ipynb`；本仓库定位与同类对比见根 [`README.md`](./README.md)。

---

## 1. 学习路径与难度

```
基础（必读地基）   01 vanilla ─▶ 02 flash
                                  │
推理/部署          03 GQA/MQA ─▶ 04 滑窗 ─▶ 05 paged ─▶ 06 MLA
                                  │
稀疏               07 块稀疏 ─▶ 08 NSA ─▶ 14 DeepSeek V4
                                  │
线性 / SSM         10 GLA ┬─▶ 11 DeltaNet ─▶ 12 KDA ─▶ 13 GDN/GDN-2
                          └─▶ 15 Mamba2 SSD
```

**章节依赖**（读 B 前最好先读 A，记作 `A → B`）：
`01→02`（online softmax 是 flash 的内核）、`03→04`（滑窗复用 GQA kernel）、`07→08→14`（NSA 含块稀疏、DSv4 是 NSA 进化）、
`10→11→12→13`（状态矩阵 → 加 delta 擦除 → 加门控 → 门控 delta）、`10→15`（SSD = 标量衰减 GLA）。

**难度**（⭐ 入门 → ⭐⭐⭐⭐ 硬核）：

| ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
|---|---|---|---|
| 01 02 03 04 05 | 06 07 10 15 | 08 11 12 14 | 13 |

> 推荐三条读法：**只想懂 FlashAttention** → 01→02；**做推理/部署** → 02→03→04→05→06；
> **追线性注意力前沿** → 10→11→12→13 +（15 与 10 并行）+ 08→14。

---

## 2. 全景对比（一）：softmax 注意力家族（基础 / 推理 / 稀疏）

这一族都还是"$\operatorname{softmax}(QK^\top)V$"，差别在**算哪些 KV、KV 怎么存**。

| # | 变体 | 核心做法 | 复杂度 | 省了什么 | kernel 来源 |
|---|---|---|---|---|---|
| 01 | vanilla | $o=\operatorname{softmax}(QK^\top/\sqrt d)\,V$；online softmax 单遍稳定 | $O(S^2)$ | —（地基） | 自写 |
| 02 | **FlashAttention** | 同上，但**分块 + online softmax + IO-aware**，不落地 $S\times S$ 分数 | $O(S^2)$ 算，$O(S)$ 显存 | HBM 读写（10–20×） | triton 官方 |
| 03 | GQA / MQA | $G$ 个 query 头**共享 1 组 KV** | $O(S^2)$ | KV cache ÷ 组数 | vLLM |
| 04 | sliding window | 只 attend 最近 $w$ 个 key | $O(S\,w)$ | 长程算量 | vLLM（复用 03） |
| 05 | paged | KV cache **分页 + block table 间接寻址** | $O(S^2)$ | 显存碎片（利用率↑） | lightllm |
| 06 | MLA | KV **低秩压缩**到潜变量 + absorb 吸收 | $O(S^2)$ | KV cache（低秩） | lightllm |
| 07 | block-sparse | 块代表打分 → **top-$k$ 选块**，只算选中块 | $O(S\,k\,B)$ | 非重点块算量 | 自写（复用 08） |
| 08 | **NSA** | **压缩 + 选择 + 滑窗**三分支，门控融合 | $O(S(n_b{+}kB{+}w))$ | 算量 + KV | lucidrains（1987 行） |
| 14 | DeepSeek V4 | **CSA**（压缩4×+DSA选块）+ **HCA**（压缩128×稠密）层间交替 | core ≈ $O(k)$ | FLOP/KV → V3.2 的 27%/10% | 自写（指向 vLLM） |

---

## 3. 全景对比（二）：状态矩阵递推家族（线性 / SSM）

这一族把历史压进**固定大小的状态矩阵** $S\in\mathbb R^{K\times V}$，每步递推、读出 $o_t=q_tS_t$，复杂度 $O(S)$。
**全部差异只在"状态 $S_t$ 每步怎么更新"这一行**——加不加衰减门、加不加 delta 擦除、门控什么粒度：

| # | 变体 | 状态更新 $S_t=$ | 衰减门控 | delta 擦除 | kernel 来源 |
|---|---|---|---|---|---|
| 10 | linear | $S_{t-1}+k_tv_t^\top$ | 无 | 无 | fla |
| 10 | **GLA** | $\operatorname{diag}(e^{g_t})\,S_{t-1}+k_tv_t^\top$ | **per-channel** $g\in\mathbb R^K$ | 无 | fla |
| 15 | **Mamba2 SSD** | $e^{g_t}\,S_{t-1}+k_tv_t^\top$ | **per-head 标量** $g\in\mathbb R$ | 无 | fla `simple_gla` |
| 11 | **DeltaNet** | $(I-\beta_t k_tk_t^\top)\,S_{t-1}+\beta_t k_tv_t^\top$ | 无 | delta（标量 $\beta$） | fla |
| 12 | **KDA** | $(I-\beta_t k_tk_t^\top)\operatorname{diag}(e^{g_t})\,S_{t-1}+\beta_t k_tv_t^\top$ | per-channel $g$ | delta（标量 $\beta$） | fla |
| 13 | **GDN** | $(I-\beta_t k_tk_t^\top)\,e^{g_t}\,S_{t-1}+\beta_t k_tv_t^\top$ | per-head 标量 $g$ | delta（标量 $\beta$） | fla |
| 13 | **GDN-2** | $\bigl(I-k_t(b_t\odot k_t)^\top\bigr)\operatorname{diag}(e^{g_t})\,S_{t-1}+k_t(w_t\odot v_t)^\top$ | per-channel $g$ | **erase $b$ / write $w$ 双门** | fla `gdn2` |

**两条退化链**（令某门退化即得上一代，仓库各章都有数值验证）：

$$\textbf{GDN-2}\xrightarrow{\,b=w=\beta\,}\textbf{KDA}\xrightarrow{\,g\to\text{per-head}\,}\textbf{GDN}\xrightarrow{\,g\equiv0\,}\textbf{DeltaNet},
\qquad \textbf{GLA}\xrightarrow{\,g\to\text{标量}\,}\textbf{Mamba2 SSD}.$$

**SSD 对偶**（第 15 章）：状态递推 $S_t=e^{g_t}S_{t-1}+k_tv_t^\top$ 等价于半可分矩阵 $Y=\bigl(L\circ(QK^\top)\bigr)V$，
$L_{ij}=e^{g^{\mathrm{cum}}_i-g^{\mathrm{cum}}_j}$——SSM 既是递推也是注意力。线性线的 chunk kernel 都用同一招：**块内走对偶（矩阵乘并行）、块间走递推（线性）**。

---

## 4. 统一符号表

| 记号 | 含义 |
|---|---|
| $B,H,T/S,K,V,D,N$ | batch、heads、序列长、key 维、value 维、head 维、SSM state 维 |
| **layout** | 自写简要版/recurrent 用 `[B,H,T,D]`；fla kernel 用 `[B,T,H,D]`（两者差一个 transpose，封装里转换） |
| $S$（或 $h$） | 状态矩阵，形状 $K\times V$（把历史压成固定大小内存） |
| $g$ | log 空间衰减门，$g_t\le0$、$e^{g_t}\in(0,1]$；$\alpha=e^{g}$ 是线性衰减率 |
| $\beta$ | delta rule 的写入/擦除强度（标量，$\in[0,1]$，像学习率） |
| $b,\;w$ | GDN-2 的 erase 门（key 轴 $\in\mathbb R^K$）/ write 门（value 轴 $\in\mathbb R^V$） |
| $\odot$ | 逐元素（Hadamard）积 |
| $g^{\mathrm{cum}}$ | $g$ 的块内前缀和（chunk 内累积衰减） |
| $L$ | 累积衰减下三角 $L_{ij}=e^{g^{\mathrm{cum}}_i-g^{\mathrm{cum}}_j}\,(i\ge j)$，即"半可分矩阵"的衰减部分 |
| $T$ / WY | chunk 内把串行擦除解成一次三角求逆 $(I+T)^{-1}$（WY/UT transform，见 11/12/13） |
| `cu_seqlens` | 变长（packed）序列的分块偏移 `[N+1]`，batch=1 下拼接多条序列 |
| `use_qk_l2norm_in_kernel` | delta-rule 系（11/12/13）标配：对 q/k 做 L2 归一化，否则 bf16 状态谱半径爆炸出 nan |

---

## 5. 每章产物一览

每个变体目录统一含：`<name>.py`（自写简要版 / ground truth）、`<name>_triton.py`（深度优化版入口，封装解耦的 kernel）、
`_fla_*` 或 `*_triton.py`（拷贝并解耦的真实 kernel，文件头标 provenance）、`<name>_naive.py`（fla 参考，拆段精读对象）、
`test_*.py`（bitwise vs 原版 + ≡ recurrent + fwd/bwd）、`bench.py`、`SOURCES.md`、`README.md`、`*.ipynb`（数学深入 notebook）。

> 例外：自写机制章（01/04/07/14、Mamba1）无解耦 kernel，拆段精读对象是自写实现本身 + 指向官方来源。
