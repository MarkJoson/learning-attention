## 先澄清几个缩写

你提到的几个缩写大概率对应：

| 缩写 | 常见含义 | 类型 |
|---|---|---|
| **DSA** | Dynamic Sparse Attention / Dynamic Sparse FlashAttention | 动态稀疏注意力 |
| **KDA** | Kimi Delta Attention | 线性注意力 / Delta-rule attention |
| **GDA** | Grouped Differential Attention 或 Gated Delta Attention/Network 相关变体 | 分组差分注意力或门控线性注意力 |
| **GQA** | Grouped Query Attention | KV head 共享 |
| **MQA** | Multi-Query Attention | 单 KV head |
| **MLA** | Multi-head Latent Attention，DeepSeek 系列使用 | KV 压缩 / latent KV |
| **NSA** | Native Sparse Attention，DeepSeek 提出 | 稀疏注意力 |
| **Paged Attention** | vLLM/FlashInfer 常用 | 推理 KV cache 管理 |
| **Sliding Window Attention** | Mistral、Qwen 等常见 | 局部窗口稀疏 |
| **Linear Attention** | RetNet、DeltaNet、GLA、KDA 等 | 线性复杂度注意力 |

其中 **KDA、GDA/GDN、GLA、DeltaNet** 这一类更接近 **linear attention / recurrent attention**，而 **DSA、NSA、Sliding Window、Paged Attention** 更接近 **稀疏或 KV-cache 优化 attention**。

---

# 最值得看的仓库清单

## 1. Flash Linear Attention：学习 KDA / GLA / DeltaNet / 线性注意力实现

### 推荐仓库：**fla-org/flash-linear-attention**

这个是我认为你必须看的仓库之一。

它的定位就是高性能 **linear attention** 算子库，里面有大量 Triton 实现，非常适合学习：

- GLA，Gated Linear Attention
- DeltaNet
- Gated DeltaNet
- Kimi Delta Attention / KDA 相关实现
- RetNet 类 recurrent attention
- RWKV 类 recurrence kernel
- chunkwise / recurrent / parallel scan 实现方式
- Triton fused kernel 写法

KDA 相关生态里，FLA 是重要参考实现；一些后续的 FlashKDA/CUTLASS 实现也会拿 FLA 的 Triton kernel 作为 baseline。FlashKDA 相关介绍也提到，原始 KDA reference kernels 在开源 `flash-linear-attention` 库中用 Triton 实现。([nerdleveltech.com](https://nerdleveltech.com/flashkda-moonshot-cutlass-kernel-kimi-linear?utm_source=openai))

你可以搜：

```text
fla-org flash-linear-attention
```

或者：

```bash
git clone https://github.com/fla-org/flash-linear-attention
```

重点看目录通常包括：

```text
fla/ops/
fla/modules/
fla/layers/
```

学习顺序建议：

```text
gla -> delta_rule -> gated_delta_rule -> kimi_delta
```

如果你主要想理解 **KDA / GDA / GDN / DeltaNet**，这个仓库优先级最高。

---

## 2. Triton 官方 Fused Attention Tutorial：学习 FlashAttention kernel 骨架

### 仓库/文档：**triton-lang/triton**

Triton 官方教程里有 **Fused Attention**，实现的是 FlashAttention v2 风格算法。官方文档明确说明这是 Triton 实现的 FlashAttention v2 algorithm。([triton-lang.org](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html?utm_source=openai))

建议先把这个 kernel 读透，因为后面很多变体都围绕它修改：

- online softmax
- block tiling
- causal mask
- \(QK^\top\) 分块
- \(P V\) 分块累加
- SRAM/HBM 访存优化
- forward / backward 拆分

搜索：

```text
triton fused attention tutorial
```

重点文件/文档：

```text
python/tutorials/06-fused-attention.py
```

这个是学习 **FlashAttention 类算子实现** 的最佳入口。

---

## 3. Dao-AILab/flash-attention：工业级 FlashAttention / GQA / MQA / KV cache

### 推荐仓库：**Dao-AILab/flash-attention**

这是 FlashAttention 官方仓库。虽然核心高性能实现主要是 CUDA/C++，但它是理解现代 attention 工程接口的标准参考。

重点看：

- FlashAttention-2
- FlashAttention-3
- causal / varlen attention
- GQA / MQA
- sliding window
- rotary embedding
- KV cache
- paged KV cache
- inference decode kernel 接口

如果你要理解“市面上的 attention 算子”，这个仓库是基准。NVIDIA cuDNN frontend 文档也说明其 attention backend 使用 FlashAttention-2 算法。([docs.nvidia.com](https://docs.nvidia.com/deeplearning/cudnn/v1.9.0/operations/Attention.html?utm_source=openai))

适合解决这些问题：

```text
为什么 GQA/MQA 能降低 KV cache？
为什么 prefill 和 decode kernel 不一样？
为什么 varlen attention 需要 cu_seqlens？
为什么 paged attention 要 block table？
```

---

## 4. vLLM：学习 Paged Attention / Triton 推理 backend

### 推荐仓库：**vllm-project/vllm**

如果你关心 LLM 推理，那么必须看 **vLLM 的 paged attention**。

vLLM 2026 年的 Triton backend 文章专门解释了为什么 Triton 适合 vLLM，并深入讲了高性能 paged attention kernel 的实现；它还比较了 prefill、decode、mixed workload 下多个 Triton paged attention kernel 变体。([vllm-project.github.io](https://vllm-project.github.io/2026/03/04/vllm-triton-backend-deep-dive.html?utm_source=openai))

重点理解：

- paged KV cache
- block table
- sequence slot mapping
- decode attention
- prefill vs decode
- continuous batching
- Triton attention backend
- KV cache memory layout

搜索：

```text
vllm triton attention backend
```

建议看：

```text
vllm/attention/
vllm/platforms/
vllm/worker/
```

以及 Triton kernel 相关目录。

---

## 5. FlashInfer：学习推理 attention kernel 接口设计

### 推荐仓库：**flashinfer-ai/flashinfer**

FlashInfer 是推理服务里很重要的 attention kernel 库，支持：

- paged KV cache
- batch decode
- batch prefill
- MLA
- MQA/GQA
- sampling kernels
- cascade attention

它不完全是 Triton 实现，很多是 CUDA/CUTLASS，但它的 API 设计很值得参考。你如果想学习“attention 算子怎么被服务框架调用”，FlashInfer 非常有价值。

适合搭配 vLLM 看。

---

## 6. Native Sparse Attention / NSA：学习 DeepSeek 稀疏注意力

### 推荐仓库方向

可以搜：

```text
native sparse attention triton
DeepSeek Native Sparse Attention implementation
lucidrains native-sparse-attention-pytorch triton
```

`lucidrains/native-sparse-attention-pytorch` 有 Triton kernel 讲解，DeepWiki 页面提到它有优化的 Triton CUDA kernel，并使用 query head groups 来提升并行性。([deepwiki.com](https://deepwiki.com/lucidrains/native-sparse-attention-pytorch/2.2-triton-kernel-implementation?utm_source=openai))

NSA 的核心思想一般包括：

- compressed attention
- selected attention
- sliding window attention
- block sparse layout
- top-k block selection
- routing / selection kernel
- sparse attention computation kernel

NSA 和 DSA 都属于你应该重点看的 **稀疏 attention** 路线。

---

## 7. Dynamic Sparse Attention / DSA：学习动态稀疏选择

### 推荐仓库：**epfml/dynamic-sparse-flash-attention**

这个方向可以搜：

```text
epfml dynamic-sparse-flash-attention
Dynamic Sparse FlashAttention Triton
Fast Attention Over Long Sequences With Dynamic Sparse Flash Attention
```

搜索结果里论文 PDF 提到其自定义 Triton kernels 的实现细节，并给出了仓库链接 `epfml/dynamic-sparse-flash-attention`。([proceedings.nips.cc](https://proceedings.nips.cc/paper_files/paper/2023/file/bc222e8153a49c1b30a1b8ba96b35117-Paper-Conference.pdf?utm_source=openai))

DSA 的核心不是简单 fixed sparse mask，而是：

1. 根据 query/key 动态选择重要 block；
2. 只对选中的 block 计算 attention；
3. 保持 FlashAttention 风格的分块 softmax；
4. 尽量减少 gather/scatter 和非连续访存带来的损失。

你需要重点关注：

```text
block selection
top-k routing
sparse block table
masked flash attention
load balancing
```

注意：DSA 这个缩写也可能指 Dynamic Self-Attention，早期 NLP 里有不同含义；现代 LLM kernel 场景下，你更可能指的是 Dynamic Sparse Attention。([emergentmind.com](https://www.emergentmind.com/topics/dynamic-sparse-attention-dsa?utm_source=openai))

---

## 8. DashAttention：新近的自适应稀疏层次注意力

### 方向：DashAttention

DashAttention 是 2026 年的新工作，全称是 **Differentiable and Adaptive Sparse Hierarchical Attention**。论文摘要称其提供了 GPU-aware Triton 实现，并在推理时相对 FlashAttention-3 获得加速。([arxiv.org](https://arxiv.org/abs/2605.18753?utm_source=openai))

它适合在学完 DSA/NSA 后看，因为它也是：

- adaptive sparsity
- hierarchy
- differentiable sparse selection
- Triton kernel

搜索：

```text
DashAttention Triton GitHub
```

---

## 9. Kimi Linear / FlashKDA：学习 KDA 的高性能演进

### 方向：KDA / FlashKDA / Kimi Linear

KDA = **Kimi Delta Attention**，是 Moonshot AI Kimi Linear 里的关键线性注意力机制。FlashKDA 是 Moonshot 后续开源的 CUTLASS kernel，文章中提到它是针对 KDA 的高性能 CUDA/CUTLASS 实现，并且拿 FLA 的 Triton 路径作为参考/baseline。([nerdleveltech.com](https://nerdleveltech.com/flashkda-moonshot-cutlass-kernel-kimi-linear?utm_source=openai))

学习顺序建议：

1. 先看 FLA 里的 Triton KDA/DeltaNet 实现；
2. 再看 Kimi Linear 论文/代码；
3. 最后看 FlashKDA/CUTLASS kernel。

原因是：KDA 这类算子不像 standard attention 那样是 \(QK^\top V\)，而更像状态空间/递推：

\[
S_t = f(S_{t-1}, k_t, v_t, \beta_t, g_t)
\]

\[
o_t = q_t^\top S_t
\]

高性能实现重点变成：

- chunkwise scan
- associative scan
- recurrent forward
- parallel training kernel
- decay/gate fusion
- state update fusion
- backward 的反向 scan

---

# 如果你的目标是“系统学习”，推荐路线如下

## 第一阶段：标准 Attention 与 FlashAttention

先学：

1. vanilla attention
2. online softmax
3. FlashAttention v1
4. FlashAttention v2
5. causal / non-causal
6. varlen attention

对应代码：

```text
triton-lang/triton tutorials/06-fused-attention.py
Dao-AILab/flash-attention
```

你需要掌握这个公式：

\[
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

以及 online softmax 分块更新：

\[
m_i^{new} = \max(m_i, m_{ij})
\]

\[
l_i^{new} = e^{m_i - m_i^{new}} l_i + e^{m_{ij} - m_i^{new}} l_{ij}
\]

\[
O_i^{new} =
\frac{
e^{m_i - m_i^{new}} l_i O_i +
e^{m_{ij} - m_i^{new}} P_{ij} V_j
}{
l_i^{new}
}
\]

读懂这个后，FlashAttention 系列就通了。

---

## 第二阶段：GQA / MQA / MLA / KV cache

看：

```text
Dao-AILab/flash-attention
vllm-project/vllm
flashinfer-ai/flashinfer
```

重点理解：

| 算子 | 核心变化 |
|---|---|
| MHA | 每个 query head 有自己的 K/V head |
| MQA | 所有 query head 共享一个 K/V head |
| GQA | 多个 query head 共享一组 K/V head |
| MLA | KV 被压缩到 latent 表示，再还原或融合计算 |
| Paged Attention | KV cache 按页/块管理 |

GQA/MQA 的核心是减少 KV cache：

\[
H_q > H_{kv}
\]

比如：

```text
num_q_heads = 32
num_kv_heads = 8
group_size = 4
```

那么每 4 个 query heads 共享一组 KV。

---

## 第三阶段：Sparse Attention / DSA / NSA

看：

```text
epfml/dynamic-sparse-flash-attention
lucidrains/native-sparse-attention-pytorch
DeepSeek NSA 相关实现
DashAttention
```

重点不是 softmax 本身，而是：

```text
哪些 block 要算？
block index 怎么存？
top-k selection 怎么做？
稀疏 mask 怎么和 FlashAttention 融合？
不同 query 的稀疏度不一样时如何 load balance？
```

典型结构：

\[
O_i =
\operatorname{softmax}
\left(
Q_i K_{\mathcal{S}(i)}^\top
\right)
V_{\mathcal{S}(i)}
\]

其中 \(\mathcal{S}(i)\) 是动态选择的 key/value block 集合。

---

## 第四阶段：Linear Attention / GLA / DeltaNet / KDA / GDA

看：

```text
fla-org/flash-linear-attention
Kimi Linear / FlashKDA
Qwen GDN / FlashQLA 相关实现
```

这类 attention 的核心是把 quadratic attention 改写成 recurrent/state update：

\[
S_t = \lambda_t S_{t-1} + k_t^\top v_t
\]

\[
o_t = q_t S_t
\]

或者 Delta-rule 风格：

\[
S_t = S_{t-1} + \beta_t k_t^\top (v_t - k_t S_{t-1})
\]

KDA/Gated DeltaNet 会加 gate、decay、chunkwise 并行算法。

高性能实现关键词：

```text
chunkwise
recurrent
parallel scan
WY representation
state passing
fused gate
fused decay
custom backward
```

---

# 我建议你重点 clone 这几个

```bash
# 1. Linear attention / KDA / GLA / DeltaNet
git clone https://github.com/fla-org/flash-linear-attention

# 2. FlashAttention 官方实现
git clone https://github.com/Dao-AILab/flash-attention

# 3. Triton 官方教程
git clone https://github.com/triton-lang/triton

# 4. vLLM paged attention
git clone https://github.com/vllm-project/vllm

# 5. FlashInfer 推理 attention kernel
git clone https://github.com/flashinfer-ai/flashinfer

# 6. Dynamic Sparse FlashAttention
git clone https://github.com/epfml/dynamic-sparse-flash-attention

# 7. Native Sparse Attention 学习实现
git clone https://github.com/lucidrains/native-sparse-attention-pytorch
```

---

# 最推荐的阅读顺序

如果你是为了“学会自己写 attention Triton 算子”，我建议这样读：

## Step 1：读 Triton 官方 FlashAttention

```text
triton/python/tutorials/06-fused-attention.py
```

目标：

- 能手写 forward kernel
- 理解 block \(Q\)、block \(K/V\)
- 理解 online softmax
- 理解 causal mask

---

## Step 2：读 FlashAttention 官方接口

```text
Dao-AILab/flash-attention
```

目标：

- 理解真实工程里 attention API 怎么设计
- 理解 varlen、GQA、KV cache
- 理解 prefill/decode 区别

---

## Step 3：读 vLLM paged attention

目标：

- 理解推理时 attention 不是简单 \(QK^\top V\)
- 理解 KV cache layout 比算术本身更重要
- 理解 block table 和 paged memory

---

## Step 4：读 FLA

重点：

```text
fla/ops/gla
fla/ops/delta_rule
fla/ops/gated_delta_rule
fla/ops/kimi_delta
```

目标：

- 理解 linear attention 和 recurrent kernel
- 理解 chunkwise training
- 理解 state passing
- 理解 backward scan

---

## Step 5：读 DSA/NSA

目标：

- 理解稀疏 block 选择
- 理解 sparse FlashAttention
- 理解动态稀疏的 kernel 难点

---

# 各类 attention 的实现难点对比

| 类型 | 算法重点 | Kernel 难点 |
|---|---|---|
| FlashAttention | online softmax | SRAM/HBM tiling |
| GQA/MQA | KV head 共享 | head mapping / KV cache layout |
| Paged Attention | KV cache 分页 | block table / irregular load |
| Sliding Window | 局部 attention | boundary mask |
| DSA | 动态选 block | top-k + sparse block load |
| NSA | 压缩 + 选择 + 窗口 | 多分支融合 |
| MLA | latent KV 压缩 | projection 与 attention fusion |
| GLA | gate + linear recurrence | parallel scan |
| DeltaNet | delta-rule update | chunkwise state update |
| KDA | Kimi delta recurrence | fused chunkwise kernel |
| GDA/GDN | grouped/gated delta | gate-aware backward |

---

# 最短结论

如果你只想先找一个最核心的仓库开始，我推荐：

```text
fla-org/flash-linear-attention
```

因为它最贴近你提到的 **KDA / GDA / DeltaNet / GLA** 这一批新 attention 算子。

如果你要同时覆盖传统 attention、推理 attention 和稀疏 attention，那就按这个组合：

```text
triton-lang/triton
Dao-AILab/flash-attention
fla-org/flash-linear-attention
vllm-project/vllm
epfml/dynamic-sparse-flash-attention
lucidrains/native-sparse-attention-pytorch
```

学习主线可以概括为：

```text
FlashAttention
→ GQA/MQA/KV Cache
→ Paged Attention
→ Sparse Attention / DSA / NSA
→ Linear Attention / GLA / DeltaNet / KDA / GDA
```

这样基本就能覆盖 2024–2026 年主流 LLM attention kernel 的主要方向。

---
Learn more:
1. [FlashKDA: Moonshot's CUTLASS Kernel for Kimi Linear (2026) | Nerd Level Tech](https://nerdleveltech.com/flashkda-moonshot-cutlass-kernel-kimi-linear?utm_source=openai)
2. [Fused Attention — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html?utm_source=openai)
3. [Attention — NVIDIA cuDNN Frontend](https://docs.nvidia.com/deeplearning/cudnn/v1.9.0/operations/Attention.html?utm_source=openai)
4. [vLLM Triton Attention Backend Deep Dive | vLLM Blog](https://vllm-project.github.io/2026/03/04/vllm-triton-backend-deep-dive.html?utm_source=openai)
5. [Triton Kernel Implementation | lucidrains/native-sparse-attention-pytorch | DeepWiki](https://deepwiki.com/lucidrains/native-sparse-attention-pytorch/2.2-triton-kernel-implementation?utm_source=openai)
6. [Fast Attention Over Long Sequences With Dynamic](https://proceedings.nips.cc/paper_files/paper/2023/file/bc222e8153a49c1b30a1b8ba96b35117-Paper-Conference.pdf?utm_source=openai)
7. [Dynamic Sparse Attention (DSA)](https://www.emergentmind.com/topics/dynamic-sparse-attention-dsa?utm_source=openai)
8. [DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention](https://arxiv.org/abs/2605.18753?utm_source=openai)