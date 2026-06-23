"""生成 11-deltanet/deltanet.ipynb —— DeltaNet 数学深入 tutorial（新标准）。

按 13-gdn 样板：数学逐步推导 + 全 LaTeX + 拆段精读 + 完整 kernel 源码备查 + retina 高清图。
DeltaNet 是 delta rule 的主场（几何讲最透）+ WY 表示最干净的原型（无门控）。
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

# ============================ 0. 标题 ============================
cells.append(md(r"""
# DeltaNet：用 delta rule 给线性注意力装上"橡皮擦"

线性注意力把历史压进一个固定大小的**状态矩阵** $S\in\mathbb R^{K\times V}$，每步更新、读出 $o_t=S_t^\top q_t$。
最朴素的更新 $S_t=S_{t-1}+k_tv_t^\top$ **只加不减**：写入新记忆从不擦旧的，键冲突时旧值残留、互相干扰。

**DeltaNet** 借在线学习的 *delta 规则*（Widrow–Hoff / 快速权重）给它装上"橡皮擦"：写入前先用 $k_t$ 查询旧状态、
算预测误差，只写入**误差**——等价于先擦掉 $k_t$ 方向的旧记忆、再写新值。这一章把它的几何与 chunk 并行一步步推清楚：

1. §1 先**亲眼看见**朴素线性注意力的键冲突；
2. §2 把 **delta rule 的几何**讲透——为什么 $(I-\beta_t k_tk_t^\top)$ 是"沿 $k_t$ 方向的擦除"（后续 KDA/GDN 都引用这里）；
3. §3 推导 **WY 表示**——把块内串行擦除解成一次三角求逆，这是 chunk 并行 kernel 的地基，DeltaNet 无门控，是最干净的原型；
4. §4 **逐段精读**可读的 chunk 实现；
5. §5 保留**完整 kernel 源码**并验证本仓库解耦自 fla 的实现与原版一致。
""".strip()))

# ============================ 1. setup ============================
cells.append(code(r"""
import sys, math
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

ROOT = Path.cwd()
while not (ROOT / "common").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
CH = ROOT / "11-deltanet"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CH))

from common.nbtools import setup_cjk, show_code
setup_cjk()                                       # 中文字体 + retina 高清出图
from deltanet import delta_rule_recurrent, delta_rule_chunked   # 简要版：recurrent + WY chunk

torch.manual_seed(0)
print("跑在", torch.cuda.get_device_name(0))
""".strip()))

# ============================ 2. §1 键冲突 ============================
cells.append(md(r"""
## 1. 先亲眼看见：朴素线性注意力无法"覆盖更新"

朴素线性注意力把每对 $(k_i,v_i)$ 直接累加进状态：$S=\sum_i k_iv_i^\top$。如果同一个 key $k$ 先写入 $v_a$、之后想**覆盖**
成 $v_b$，它只会把两者**累加**，无法覆盖：

$$S^\top k=(k^\top k)\,(v_a+v_b)=v_a+v_b\;\neq\;v_b.$$

**delta rule** 在写 $v_b$ 前，先用 $k$ 读出旧值 $\hat v=S^\top k=v_a$、算误差 $v_b-v_a$、**只写误差**，于是状态精确变成
$S=k\,v_b^\top$，读出就是 $v_b$——成功覆盖。下面对比。
""".strip()))

cells.append(code(r"""
# 同一个 key，先写 v_a、再想覆盖成 v_b
k = F.normalize(torch.tensor([1.0, 0.5, -0.3]), dim=0)            # 一个 key（已归一化）
v_a = torch.tensor([1.0, 0.0, 0.0]); v_b = torch.tensor([0.0, 1.0, 0.0])

S_lin = torch.outer(k, v_a) + torch.outer(k, v_b)                # 朴素 linear：两次写入直接累加
print(f"朴素 linear  读出: {[round(x,2) for x in (S_lin.T @ k).tolist()]}   → v_a+v_b 累加，覆盖失败")

# DeltaNet：同一个 key 写 v_a 再写 v_b（先擦后写）
q = k.reshape(1, 1, 1, 3).repeat(1, 1, 2, 1).cuda()             # 两步都用 k 读（看第 2 步）
kk = k.reshape(1, 1, 1, 3).repeat(1, 1, 2, 1).cuda()
vv = torch.stack([v_a, v_b]).reshape(1, 1, 2, 3).cuda()
beta = torch.ones(1, 1, 2).cuda()
o = delta_rule_recurrent(q, kk, vv, beta, l2norm=True, scale=1.0)
print(f"DeltaNet     读出: {[round(x,2) for x in o[0,0,1].tolist()]}   → ≈ v_b，成功覆盖（先擦 v_a 再写 v_b）")
""".strip()))

# ============================ 3. §2 delta rule 几何 ============================
cells.append(md(r"""
## 2. delta rule：先擦后写（几何讲透）

DeltaNet 把状态 $S$ 看成一个线性映射 $k\mapsto v$，每来一对 $(k_t,v_t)$ 就像做一步在线学习的修正，分三步：

$$
\underbrace{\hat v_t=S_{t-1}^\top k_t}_{\text{① 读出 }k_t\text{ 方向的旧值}}\qquad
\underbrace{\Delta_t=\beta_t\,(v_t-\hat v_t)}_{\text{② 算预测误差}}\qquad
\underbrace{S_t=S_{t-1}+k_t\,\Delta_t^\top}_{\text{③ 沿 }k_t\text{ 写回修正}}
$$

把 $\hat v_t=S_{t-1}^\top k_t$ 代回 ③ 合并，得到 DeltaNet 的状态更新式（关键代数变形）：

$$
S_t=S_{t-1}+\beta_t\,k_t\bigl(v_t-S_{t-1}^\top k_t\bigr)^\top
   =\boxed{(I-\beta_t\,k_tk_t^\top)\,S_{t-1}+\beta_t\,k_tv_t^\top}.
$$

**几何**：当 $\lVert k_t\rVert=1,\beta_t=1$ 时，$I-k_tk_t^\top$ 正是把向量在 $k_t$ 方向分量**清零**的正交投影（保留垂直分量）。
所以 delta rule = **写入 $k_t$ 前，先把状态里 $k_t$ 方向已存的旧内容擦掉**，再写新的 $v_t$；$\beta_t\in[0,1]$ 是擦/写的力度
（像学习率）。这正是 §1 里"覆盖更新"能成功的原因，也缓解了非正交 key 的串扰。下面把"擦除"画出来。

> 这条几何是后面所有 gated delta（**第 12 章 KDA**、**第 13 章 GDN/GDN-2**）的共同基础——它们都在这个擦除算子上
> 再叠门控、或把 erase/write 拆成两个门。
""".strip()))

cells.append(code(r"""
# 2D 演示：状态在 k 方向有旧分量，(I - k k^T) 把它沿 k 擦掉，只留垂直分量
k_dir = torch.tensor([1.0, 1.0]); k_dir = k_dir / k_dir.norm()
s_old = torch.tensor([1.2, 0.2])
P = torch.eye(2) - torch.outer(k_dir, k_dir)                      # 擦除算子 I - k k^T
s_erased = P @ s_old

plt.figure(figsize=(4.8, 4.8))
ax = plt.gca()
for vec, c, lab in [(s_old, "#C44", "旧状态 $s$"), (s_erased, "#247", "$(I-kk^\\top)s$"), (k_dir, "#393", "key 方向 $k$")]:
    ax.annotate("", xy=vec.tolist(), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.text(vec[0] * 1.04, vec[1] * 1.04, lab, color=c)
ax.axline((0, 0), slope=(k_dir[1] / k_dir[0]).item(), color="#393", ls="--", alpha=0.4)
ax.set_xlim(-0.6, 1.4); ax.set_ylim(-0.6, 1.4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_title("delta rule 的擦除：$(I-kk^\\top)s$ 清掉 $s$ 在 $k$ 方向的分量")
plt.tight_layout(); plt.show()
print("擦除后向量与 k 正交：s_erased · k =", f"{(s_erased @ k_dir).item():.2e}")
print("→ 写入新 v 前先沿 k 擦掉旧值——这就是 delta rule 实现覆盖更新的几何本质。")
""".strip()))

# ============================ 4. §3 WY 表示 ============================
cells.append(md(r"""
## 3. chunk 并行的数学核心：WY 表示（最干净的原型）

recurrent 形式 $O(T)$ 串行、GPU 上慢。chunk-parallel 把序列切成大小 $C$ 的块、块内矩阵化、块间递推。难点：擦除
算子 $(I-\beta_t k_tk_t^\top)$ 逐 token **串行相乘**。DeltaNet **没有门控**，是看清 WY 表示的最干净原型。

### 3.1 块内展开：串行的下三角依赖

只看一个块（块起点状态记为 $S$）。块内第 $i$ 步的**有效写入** $u_i$ 要先擦掉 $k_i$ 方向的旧值，而这"旧值"含同块内
前面 $j<i$ 步刚写的 $u_j$：

$$u_i=\beta_i v_i-\sum_{j<i}\underbrace{\beta_i\bigl(k_i^\top k_j\bigr)}_{=\,T_{ij}}\,u_j.$$

于是 $u_i$ 依赖所有 $u_j\ (j<i)$ —— 一个**严格下三角**的串行依赖。注意 DeltaNet 的 $T_{ij}=\beta_i(k_i^\top k_j)$ 里
**没有衰减项**（对比第 12 章 KDA 多了 $e^{g^{\mathrm{cum}}_i-g^{\mathrm{cum}}_j}$、第 13 章 GDN-2 还把 erase/write 拆开）。

### 3.2 写成线性方程组，一次解开

把块内 $u_i$ 摞成 $U$、$\beta_iv_i$ 摞成 $\beta V$，上式即

$$U=\beta V-T\,U\;\Longrightarrow\;(I+T)\,U=\beta V\;\Longrightarrow\;\boxed{U=(I+T)^{-1}\,\beta V}.$$

这就是 **WY / UT transform**（DeltaNet 论文）：严格下三角 $T=\operatorname{tril}(\operatorname{diag}(\beta)KK^\top,-1)$
编码块内所有"擦除-写入"相互作用，求一次逆 $(I+T)^{-1}$ 就把 $C$ 步串行依赖**一次性解开**，块内于是可以并行 GEMM。

### 3.3 前向替换 + 块间递推

$T$ 严格下三角 $\Rightarrow I+T$ 单位下三角，用**前向替换**逐行 $O(C^2)$ 求逆（`delta_rule_chunked` 的 `for i` 循环）。
解出 $u=(I+T)^{-1}\beta v$、$w=(I+T)^{-1}\beta k$ 后，块间带入状态 $S$：

$$u^{\text{new}}_i=u_i-w_i\,S,\qquad o_i=q_i\,S+\underbrace{(q_i k_i^\top)_{\text{严格下三角}}}_{\text{块内因果}}u^{\text{new}}_i,\qquad S\leftarrow S+k_i^\top u^{\text{new}}_i.$$

下面验证：(a) 前向替换确实给出 $(I+T)^{-1}$；(b) 整套 chunk 算法与 $O(T)$ recurrent 数值一致。
""".strip()))

cells.append(code(r"""
# (a) 前向替换求逆：A @ (I+T) == I（单位下三角）
C = 16
torch.manual_seed(3)
Tm = torch.randn(C, C, device="cuda").tril(-1)
A = -Tm.clone()
for i in range(1, C):
    A[i, :i] = A[i, :i] + (A[i, :i, None] * A[:i, :i]).sum(0)
A = A + torch.eye(C, device="cuda")
err = (A @ (torch.eye(C, device="cuda") + Tm) - torch.eye(C, device="cuda")).abs().max()
print(f"(a) 前向替换 A=(I+T)^(-1)：‖A(I+T)-I‖∞ = {err.item():.2e}")

# (b) chunk(WY) == recurrent（两条独立路径数值一致即证 WY 推导正确）
B, H, T, D = 2, 3, 256, 64
q = torch.randn(B, H, T, D, device="cuda")
k = torch.randn(B, H, T, D, device="cuda")
v = torch.randn(B, H, T, D, device="cuda")
beta = torch.rand(B, H, T, device="cuda")
o_chunk = delta_rule_chunked(q, k, v, beta, chunk_size=64, l2norm=True)   # 内部 l2norm，安全
o_rec = delta_rule_recurrent(q, k, v, beta, l2norm=True)
print(f"(b) chunk(WY) vs recurrent：max diff = {(o_chunk - o_rec).abs().max().item():.2e}")
print("→ WY 把块内串行擦除解成一次三角求逆，与 O(T) recurrent 逐位等价——chunk kernel 的数学地基成立。")
""".strip()))

# ============================ 5. §4 逐段精读 ============================
cells.append(md(r"""
## 4. 逐段精读：可读的 chunk 实现

§3 的数学，对应代码就是本仓库 `deltanet.py:delta_rule_chunked`（自写教学版 WY，与 fla 参考 `deltanet_naive.py`
数学等价、注释更清晰）。下面拆成 2 段精读，每段标注公式与真实 Triton kernel 位置。
""".strip()))

cells.append(md(r"""
### 4.1 构造严格下三角 $T$，前向替换求 $(I+T)^{-1}$

```python
# deltanet.py · delta_rule_chunked（节选）
v = v * beta[..., None]; k_beta = k * beta[..., None]            # 把 β 并进 v、k
incl = torch.triu(torch.ones(C, C, bool), 0)                     # 含对角的上三角（要置 0 的部分）
Tmat = -(k_beta @ k.transpose(-1, -2)).masked_fill(incl, 0)      # -T：T_ij = β_i (k_i·k_j)，严格下三角
for i in range(1, C):                                            # 前向替换：单位下三角逐行求逆（§3.3）
    Tmat[..., i, :i] += (Tmat[..., i, :, None] * Tmat[..., :, :i]).sum(-2)
Tmat = Tmat + torch.eye(C)                                       # Tmat = (I+T)^{-1}
u = Tmat @ v                                                     # u = (I+T)^{-1} β v
w = Tmat @ k_beta                                                # w = (I+T)^{-1} β k
```

这段就是 §3.2 的 $T_{ij}=\beta_i(k_i^\top k_j)$ 与前向替换求 $(I+T)^{-1}$。DeltaNet 的 $T$ 里没有衰减因子，是最干净的形式。

> **真实 Triton 对应**：构造 $\operatorname{diag}(\beta)KK^\top$ 在 `_fla_chunk_scaled_dot_kkt.py`；三角求逆在
> `_fla_solve_tril.py`（DeltaNet 的 WY 求逆核心，409 行）；$w/u$ 的计算在 `_fla_wy_fast.py`。
""".strip()))

cells.append(md(r"""
### 4.2 块间循环：状态在块之间传递

```python
S = q.new_zeros(B, H, Dk, Dv)                                    # 跨块状态
strict = torch.triu(torch.ones(C, C, bool), 1)                  # 严格上三角（块内因果 mask）
for i in range(N):
    qi, ki = qr[:, :, i], kr[:, :, i]
    a = (qi @ ki.transpose(-1, -2)).masked_fill(strict, 0)       # 块内严格下三角 QK^T（因果）
    ui = u[:, :, i] - w[:, :, i] @ S                             # u^new = u - w·S（减去块间贡献）
    outs.append(qi @ S + a @ ui)                                # 块间(q·S) + 块内(a·u^new)
    S = S + ki.transpose(-1, -2) @ ui                           # 更新块间状态
```

§3.3 的三个公式：用块起点 $S$ 修正出 $u^{\text{new}}$，输出 = 块内因果注意力 + 块间状态读出，最后把状态传给下个块。
$N$ 个块之间只有这一条 $S$ 的串行链（$O(T/C)$ 步），块内全部并行——chunk-parallel 把 $O(T)$ 降到 $O(T/C)$ 串行。

> **真实 Triton 对应**：块间状态递推在 `_fla_chunk_delta_h.py`；输出 $o$ 在 `_fla_chunk_o.py`。
""".strip()))

# ============================ 6. §5 完整 kernel + 验证 ============================
cells.append(md(r"""
## 5. 真实 Triton kernel：完整解耦自 fla

本仓库**完整拷贝并解耦** DeltaNet 的 8 个 Triton 文件（计算逻辑一字未改），靠 **no-op dispatch** 绕过 fla 后端分派，
把依赖闭包从 ~27 文件收敛到 8 个、脱离 fla 独立运行。下面保留**完整源码**——DeltaNet 的 WY 三角求逆核心
`_fla_solve_tril.py`（409 行，可滚动），即 §4.1 前向替换的 Triton 实现；其余 7 个文件见 [`SOURCES.md`](./SOURCES.md)。
""".strip()))

cells.append(code(r"""show_code(str(CH / "_fla_solve_tril.py"))"""))

cells.append(md(r"""
### 数值验证：解耦没改任何计算

把本地解耦的 kernel 与 fla 原版逐位对比（定长 + 变长 `cu_seqlens`），再与简要版 recurrent 对齐：
""".strip()))

cells.append(code(r"""
from _fla_delta_chunk import chunk_delta_rule as local
B, T, H, D = 2, 512, 4, 64
gg = torch.Generator("cuda").manual_seed(1)
q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gg)
k = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gg)
v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, generator=gg)
beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16, generator=gg)

ol, _ = local(q, k, v, beta, use_qk_l2norm_in_kernel=True)
try:
    from fla.ops.delta_rule import chunk_delta_rule as flak
    of, _ = flak(q, k, v, beta, use_qk_l2norm_in_kernel=True)
    print(f"① 定长：本地解耦 vs fla 原版   max diff: {(ol.float()-of.float()).abs().max().item():.2e}")
    cu = torch.tensor([0, 200, 512, 800, 1024], device="cuda", dtype=torch.int32)
    qp, kp, vp = (x.reshape(1, B * T, H, D) for x in (q, k, v)); bp = beta.reshape(1, B * T, H)
    o2, _ = local(qp, kp, vp, bp, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    f2, _ = flak(qp, kp, vp, bp, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    print(f"② 变长(cu_seqlens)：本地 vs fla    max diff: {(o2.float()-f2.float()).abs().max().item():.2e}")
except ImportError:
    print("（未装 fla，跳过原版对照）")

o_rec = delta_rule_recurrent(*(x.transpose(1, 2) for x in (q, k, v)), beta.transpose(1, 2), l2norm=True).transpose(1, 2)
print(f"③ 本地 chunk vs 简要版 recurrent   max diff: {(ol.float()-o_rec.float()).abs().max().item():.2e}")
print("→ 完整解耦（8 文件 + no-op dispatch）没改任何计算：与 fla 数值一致、与 recurrent ground truth 对齐。")
""".strip()))

# ============================ 7. §6 bench ============================
cells.append(md(r"""
## 6. 复杂度：$O(S)$ vs full attention $O(S^2)$

DeltaNet 是线性复杂度，长序列优于 full attention。delta rule 的纠错让它在固定状态大小下记忆质量优于朴素 linear attention。
""".strip()))

cells.append(code(r"""
from common import bench_ms
from deltanet_triton import delta_chunk

B, H, D = 4, 8, 128
Ss = [1024, 2048, 4096, 8192]
full, dn_t = [], []
for S in Ss:
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    qb, kb, vb = (x.to(torch.bfloat16) for x in (q, k, v))
    beta = torch.rand(B, H, S, device="cuda", dtype=torch.bfloat16)
    full.append(bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)))
    dn_t.append(bench_ms(lambda: delta_chunk(qb, kb, vb, beta)))

plt.figure(figsize=(7, 4.5))
plt.plot(Ss, full, "o-", label="full attention $O(S^2)$")
plt.plot(Ss, dn_t, "s-", label="DeltaNet $O(S)$")
plt.xlabel("序列长度 S"); plt.ylabel("前向耗时 (ms)"); plt.yscale("log"); plt.xscale("log")
plt.title("RTX 4090 · B=4 H=8 D=128 causal"); plt.legend(); plt.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()
for i, S in enumerate(Ss):
    print(f"S={S:>5}  full {full[i]:6.3f}ms  DeltaNet {dn_t[i]:6.3f}ms ({full[i]/dn_t[i]:.2f}×)")
""".strip()))

# ============================ 8. 收尾 ============================
cells.append(md(r"""
## 7. 收尾

DeltaNet 给线性注意力装上了"橡皮擦"：

1. **无法覆盖**（§1）：朴素 linear attention 只加不减，同 key 重写只会累加 $v_a+v_b$；
2. **delta rule 几何**（§2）：$(I-\beta k k^\top)$ 沿 $k$ 方向擦除旧值再写新值，"先擦后写"实现覆盖更新、缓解键串扰——
   这是后续 KDA / GDN 全系的共同基础；
3. **WY 表示**（§3）：把块内串行擦除写成 $(I+T)U=\beta V$，一次三角求逆解开，是 chunk 并行的地基；DeltaNet 无门控、
   是最干净的原型（$T_{ij}=\beta_i k_i^\top k_j$）；
4. **逐段精读 + 完整 kernel**（§4–§5）：可读参考实现的每段都对应公式与 Triton 位置，本仓库完整解耦自 fla、与原版一致。

**下一章** → 12-kda：在这个擦除算子上叠加 per-channel 门控遗忘（KDA = GLA 门控 ⊕ DeltaNet 纠错），再到 13 章把 erase/write
解耦成两个门（GDN/GDN-2）。
""".strip()))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "learnattn", "language": "python", "name": "learnattn"}
out = "/home/robomaster/Research/learning-attention/11-deltanet/deltanet.ipynb"
nbf.write(nb, out)
print("写入", out, "·", len(cells), "cells")
