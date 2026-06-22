# 08-native-sparse-attention · 外部来源登记

本章既有自写 PyTorch 参考（`nsa.py`，三分支架构 ground truth），也**完整提取**了一份真实的 NSA
稀疏 attention triton kernel（`nsa_triton.py`，1987 行）。汇总见仓库根 [`NOTICE`](../NOTICE)。

## `nsa_triton.py`（完整提取的 NSA 稀疏 flash kernel）

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/lucidrains/native-sparse-attention-pytorch |
| 文件 | `native_sparse_attention_pytorch/triton_native_sparse_attention.py` |
| 版本 | pip `native-sparse-attention-pytorch==0.2.3` |
| License | MIT (Copyright Phil Wang / lucidrains) |
| 取用 | **整份文件**（forward + backward kernel、`NSA` autograd Function、`native_sparse_attend` 入口） |
| kernel 谱系 | 改编自 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的 triton flash kernel（见文件内原注释第 ~13 行） |

### 为什么是它

NSA 真正高性能的部分是"**给定每个 query 选中的 top-k 块，只对这些块（外加滑窗）做 flash 式注意力**"
的稀疏 kernel —— 这必须手写 triton 才快（标准 FlashAttention / SDPA 处理不了 **per-query 动态选块**）。
DeepSeek 官方未开源 NSA kernel；社区里 lucidrains 这份是**最完整、可独立运行、又能在消费级 GPU
编译跑通**的真实实现（forward + backward + GQA，已在 RTX 4090 验证 fwd/bwd 跑通）。

### 本仓库改动（**不改 kernel 计算逻辑**）

仅去除 1 个库内依赖，使本文件不再 import lucidrains 包本体：

```python
# 原: from native_sparse_attention_pytorch.tensor_typing import Float, Int, Bool
# 内联为零依赖占位（Float/Int/Bool 只是 jaxtyping 张量 shape 注解，from __future__ import
# annotations 下不在运行时求值，不参与任何计算）。
```

去依赖后仅依赖 `torch / triton / einx / einops`。`test_nsa_triton.py::test_faithful_vs_library`
逐位验证"拷贝 == 库原版"（`atol=0`），证明改动没碰任何计算。

### 这 1987 行在整个 NSA 里的位置

NSA = 三分支（compressed / selected / sliding）+ 门控。其中**只有 selected + sliding 的稀疏计算**
由这份 triton kernel 承担；"块压缩、算块分数、top-k 选块、门控合并"是 host 端 PyTorch 逻辑
（lucidrains 的 `SparseAttention` Module，或本章简要版 `nsa.py`）。入口：

```python
native_sparse_attend(fq, fk, fv, block_size, selected_block_indices, fmask, ...)
#   selected_block_indices 已是"每个 query 选哪些块"的结果，kernel 只负责把它们算得快
```

## 算法来源

DeepSeek《Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention》(2025)。

## 本仓库自写

- `nsa.py`：`NativeSparseAttention` 模块 —— 三分支（compressed mean-pool 压缩 + 复用压缩分数选块、
  selected top-k 块稀疏、sliding 滑窗）+ 门控合并，纯 PyTorch、mask 版，作为机制 ground truth。
- `test_nsa.py`：简要版三分支正确性（滑窗 ≡ naive window、全选 ≡ full、门控归一、严格 causal）。
- `test_nsa_triton.py`：拷贝忠实性（vs 库 `atol=0`）+ 端到端（替换库 triton 入口 vs 库 PyTorch 路径，fwd+bwd）。
- `bench.py`：三分支"信息覆盖"与延迟基准。
