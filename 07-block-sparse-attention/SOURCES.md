# 07-block-sparse-attention · 外部来源登记

本章有三层：① 自写 PyTorch 简要版（讲机制）；② 深度优化版（复用 08 已提取的 NSA 真实 triton
kernel）；③ MoBA 对照（拷贝 Kimi 的另一种动态块稀疏实现）。汇总见仓库根 [`NOTICE`](../NOTICE)。

## ① 本仓库自写（简要版）

- `block_sparse.py`：`select_topk_blocks` / `block_sparse_reference` / `block_sparse_attention`
  —— 分块、块级 top-k 选择、gather 省算。纯 PyTorch，讲清"块稀疏"机制（纯 top-k 语义）。
- `test_block_sparse.py`：全选 ≡ full、gather ≡ mask 参考、causal 选块顺序。

## ② 深度优化版：复用 08 NSA 的 selected kernel

- `block_sparse_triton.py` + `test_block_sparse_triton.py`（自写胶水 + 参考 + 测试）。
  **不另造 kernel**，直接复用 08 `nsa_triton.native_sparse_attend`（真实 kernel 来源 lucidrains
  NSA，commit / license 见 [08/SOURCES.md](../08-native-sparse-attention/SOURCES.md) 与根 NOTICE）。
  - **为什么复用**：07 的"动态 top-k 块稀疏"正是 NSA 的 selected 分支；开源生态里没有独立、
    干净、4090 可跑的"动态块稀疏 triton kernel"（NSA 那种与压缩 / 滑窗耦合；MoBA 用 flash_attn；
    triton 旧版 `blocksparse` ops 已在 3.x 移除）。NSA 的 selected kernel 就是这套机制最权威的真实实现。
  - **语义**：对角块必看 + top-k 历史块（= NSA selected 的真实语义；复用时 selected_block_indices
    须排除对角块，对角块由 kernel 自动算）。`test_block_sparse_triton.py` 验证"全历史 → full causal"
    与"kernel == 匹配语义的 PyTorch 参考"。

## ③ MoBA 对照（拷贝 Kimi / Moonshot AI）

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/MoonshotAI/MoBA |
| commit | `b5d58363311d3ca946f1ec444182727c15e338b5` |
| license | MIT (Copyright © 2025 Moonshot AI) |
| 取用 | `moba/moba_naive.py`（可跑参考）+ `moba/moba_efficient.py`（仅阅读，依赖 flash-attn） |

- `moba_naive.py`：MoBA 纯 PyTorch 参考，**可跑**（`test_moba.py` 验证全选 ≡ full causal）。
  chunk 级动态 top-k + **当前块必选** —— 与本章深度优化版"对角块必看"异曲同工。
- `moba_efficient.py`：MoBA 高效版，**依赖 flash-attn==2.6.3（本环境未装），仅供阅读**。
  关键认知：它**没有自己的 triton kernel**，而是用 flash_attn varlen + 数据重排 + 在线 softmax
  合并来实现动态稀疏 —— 与 08 NSA"自写稀疏 triton kernel"形成两条路线的对照。

## 概念出处

块级 top-k 选择是 DeepSeek **NSA** 的 selected 分支、Kimi **MoBA**、以及众多块稀疏方法的共同地基。
