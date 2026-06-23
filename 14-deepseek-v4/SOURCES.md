# 14-deepseek-v4 · 外部来源登记

本章是**自写简要版讲机制 + 指向来源**，**不提取生产 kernel**。DeepSeek V4 的 Hybrid Attention（CSA+HCA）
不在 fla，生产实现在 vLLM / DeepSeek 官方，是全新的混合压缩稀疏机制，与本仓库前面"拷贝外部 triton
kernel 解耦"的章节定位不同——这里按用户约定只用纯 PyTorch 把机制讲清楚。

## 机制来源（权威）

| 来源 | 内容 |
|---|---|
| DeepSeek V4 技术报告（deepseek-ai/DeepSeek-V4-Pro，HuggingFace，2026-04） | Hybrid Attention 原始定义：CSA / HCA、lightning indexer、压缩公式 |
| vLLM `DeepSeek-V4` 实现 | 生产推理 kernel（CSA 的 FP4 indexer、FP8 core、KV cache 布局），本章**指向不提取** |
| NVIDIA Blackwell 部署博客《Build with DeepSeek V4》 | GPU 加速端点与 Blackwell 上的 attention 优化 |

> 本章 `deepseek_v4.py` 是**自写的机制简化版**，刻意省略了生产实现的 FP4/FP8 量化、dual-stream 重叠压缩
> 的全部细节、以及 KV cache 工程，聚焦 **compress → index → attend** 主干，便于理解。

## DeepSeek V4 Hybrid Attention 要点（写入代码注释 + notebook）

**两种压缩注意力交替**（V4-Pro 61 层 ≈ 30 CSA + 31 HCA），都是统一的「压缩注意力」CompAttn(m, k) 的特例：

1. **CSA（Compressed Sparse Attention）**：压缩比 m=4 + lightning indexer 选 top-k=1024 压缩块 → **稀疏**。
   - 压缩：dual KV stream + per-coordinate softmax 池化（学习式，比 NSA 均值/卷积更灵活）；压缩后的 `C`
     **同时充当 K 和 V**（每块只缓存一个向量）。
   - indexer：打分 `I_{t,s}=Σ_h w_{t,h}·ReLU(q_{t,h}·K^IComp_s)`（**ReLU 排序、非概率**），FP4 精度，取 top-k。
   - core：shared-KV MQA，只在选中的压缩块上算。
2. **HCA（Heavily Compressed Attention）**：压缩比 m'=128 + **不选块**（稠密）。压缩到序列足够短，稠密注意力
   也便宜。是 CSA 去掉 indexer、加大压缩比的简化。

**性能**：1M token 上下文，单 token 推理 FLOP 仅 DeepSeek V3.2 的 **27%**（V4-Pro）、KV cache 仅 **10%**。

**与第 08 章 NSA 的关系**：NSA = 压缩分支 + 选块分支 + sliding window 三分支。DSv4 是它的进化——CSA ≈
NSA 的"压缩 + 选块"，但压缩换成学习式 softmax 池化、选块换成 ReLU lightning indexer；HCA 则是"压缩到极致
就不必再选"。**DSv4 没有 sliding window 分支**，省全靠压缩 + 稀疏索引。

## 本仓库自写

- `deepseek_v4.py`：`softmax_compress`（学习式压缩池化）/ `lightning_indexer`（ReLU 选块）/
  `compressed_attention`（统一 CompAttn）/ `csa_attention` / `hca_attention`。
- `test_deepseek_v4.py`（6 测试：压缩缩长、softmax 加权、indexer top-k+causal、CSA/HCA 形状、整体 causal、
  CSA(top_k=全)≡HCA 稠密）/ `deepseek_v4.ipynb`。

## 算法来源

- **DeepSeek V4**：DeepSeek-AI 2026-04 发布的长上下文模型，Hybrid Attention（CSA+HCA）是其核心。
- **DSA（DeepSeek Sparse Attention）**：V3.2-Exp 引入的 lightning-indexer 细粒度 token 稀疏；CSA 在**压缩后**
  的表示上应用 DSA。
- **NSA（Native Sparse Attention）**：第 08 章，DSv4 压缩稀疏思路的前身。
