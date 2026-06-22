"""数值对比工具：把"优化实现 vs 参考实现"的误差量化并给出可读报告。

attention 涉及大量浮点累加，fp16/bf16 下逐元素相等不现实，所以统一用
"绝对+相对容差 + 误差报告"的方式判定数值是否一致。容差按参考张量的 dtype 自动选择。
"""
from __future__ import annotations

import torch

# 不同精度下的推荐容差（绝对 / 相对）。bf16 尾数位少，容差最宽。
_TOL = {
    torch.float32: dict(atol=1e-3, rtol=1e-4),
    torch.float16: dict(atol=2e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=3e-2, rtol=2e-2),
}


def error_report(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """统计 max/mean 的绝对误差与相对误差（在 float32 上计算）。"""
    a = actual.detach().float()
    e = expected.detach().float()
    diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "mean_rel": (diff / denom).mean().item(),
    }


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    name: str = "output",
) -> dict:
    """带可读误差报告的断言；容差按 expected.dtype 自动选择，可手动覆盖。

    返回误差报告 dict，方便调用方打印或记录。
    """
    tol = _TOL.get(expected.dtype, _TOL[torch.float16])
    atol = tol["atol"] if atol is None else atol
    rtol = tol["rtol"] if rtol is None else rtol
    rep = error_report(actual, expected)
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        raise AssertionError(
            f"[{name}] 数值不一致 (atol={atol}, rtol={rtol}):\n"
            f"  max_abs ={rep['max_abs']:.3e}  mean_abs ={rep['mean_abs']:.3e}\n"
            f"  max_rel ={rep['max_rel']:.3e}  mean_rel ={rep['mean_rel']:.3e}"
        )
    return rep


def make_qkv(
    batch: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    *,
    kv_heads: int | None = None,
    seqlen_k: int | None = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    requires_grad: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造一组 (q, k, v) 测试张量，形状遵循 (B, H, S, D) 约定。

    用固定 seed 保证可复现；kv_heads/seqlen_k 用于 GQA、cross-attention 等场景。
    """
    g = torch.Generator(device=device).manual_seed(seed)
    kv_heads = kv_heads or heads
    seqlen_k = seqlen_k or seqlen
    q = torch.randn(batch, heads, seqlen, head_dim, dtype=dtype, device=device, generator=g)
    k = torch.randn(batch, kv_heads, seqlen_k, head_dim, dtype=dtype, device=device, generator=g)
    v = torch.randn(batch, kv_heads, seqlen_k, head_dim, dtype=dtype, device=device, generator=g)
    if requires_grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v
