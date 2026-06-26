r"""15 Mamba1（S6）—— selective State Space Model（自写简要版，讲机制）。

SSM（状态空间模型）把序列建模成一个**线性动态系统**：状态 $h$ 随时间演化，由输入驱动、被读出。
连续形式 $h'(t)=A\,h(t)+B\,x(t),\; y(t)=C\,h(t)$；离散化（ZOH，步长 $\Delta$）后变成递推：

    h_t = Ā h_{t-1} + B̄ x_t,    y_t = C h_t,    Ā = exp(ΔA),  B̄ ≈ Δ B

S4 等早期 SSM 的 $A,B,C,\Delta$ 都是**与输入无关**的固定参数（LTI 系统），可以用卷积并行，但**不能按内容
选择性地记忆/遗忘**。Mamba1 的关键改动叫 **selective（S6）**：让 $B,C,\Delta$ **随输入 $x$ 变化**（data-dependent），
于是模型能根据 token 内容决定"这一步记多少、用哪部分状态"——代价是不再是 LTI、不能用卷积，必须用
**selective scan**（硬件感知的并行扫描）。

每个特征通道 $d$ 维护一个独立的 $N$ 维状态（$A$ 是 $(D,N)$ 的对角矩阵，每通道一个 $N$ 维对角 SSM）：

    h_t[d] = exp(Δ_t[d]·A[d]) ⊙ h_{t-1}[d] + (Δ_t[d]·B_t) x_t[d],   y_t[d] = C_t · h_t[d]

本文件用纯 PyTorch 写出这套 selective 递推（ground truth，讲机制）。Mamba1 真正高性能的 **selective_scan
CUDA kernel** 在官方 `state-spaces/mamba`（`mamba_ssm`），本仓库不提取（CUDA、需编译，定位同 06-MLA/14-DSv4
的"讲机制 + 指向来源"）。Mamba2 把 $A$ 进一步简化为**标量**，从而得到可对偶成注意力的 SSD —— 见 ssd.py。
"""
from __future__ import annotations

import torch


def selective_ssm_recurrent(x, A, B, C, dt):
    """Mamba1 (S6) selective SSM 逐步递推（causal，ground truth）。

    形状：
      x:  (batch, L, D)        输入序列（D 个特征通道）
      A:  (D, N)               每通道一个 N 维对角 SSM 的状态衰减（通常 < 0，与输入无关）
      B:  (batch, L, N)        输入→状态 投影（**data-dependent**，selective）
      C:  (batch, L, N)        状态→输出 投影（**data-dependent**，selective）
      dt: (batch, L, D)        每通道的离散步长 Δ（**data-dependent**，softplus 后 > 0）
    返回 y: (batch, L, D)。

    离散化（ZOH）：Ā = exp(Δ⊙A)（逐通道、逐 state 维），B̄ = Δ⊙B；递推 h_t = Ā h_{t-1} + B̄ x_t，y_t = C_t h_t。
    """
    bsz, L, D = x.shape
    N = A.shape[-1]
    h = torch.zeros(bsz, D, N, device=x.device, dtype=torch.float32)
    A = A.float()
    ys = []
    for t in range(L):
        dt_t = dt[:, t].float()                                    # (bsz, D)
        dA = (dt_t[..., None] * A).exp()                           # Ā = exp(Δ⊙A)  → (bsz, D, N)
        dB_x = dt_t[..., None] * B[:, t].float()[:, None, :] * x[:, t].float()[..., None]  # B̄ x_t → (bsz,D,N)
        h = dA * h + dB_x                                          # 状态递推
        y_t = (h * C[:, t].float()[:, None, :]).sum(-1)           # y_t = C_t · h_t → (bsz, D)
        ys.append(y_t)
    return torch.stack(ys, dim=1).to(x.dtype)                     # (bsz, L, D)
