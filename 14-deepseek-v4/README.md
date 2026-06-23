# 14 · DeepSeek V4 Hybrid Attention（CSA + HCA）

> DeepSeek V4（2026-04）的长上下文注意力：**两种压缩注意力交替**。它是第 08 章 **NSA** 的进化——核心仍是
> "先压缩 KV、再稀疏地用"，但压缩换成**学习式 softmax 池化**、选块换成 **ReLU lightning indexer**。
>
> 本章按约定**自写简要版讲机制 + 指向来源**（DSv4 不在 fla，生产 kernel 在 vLLM/官方，见 [`SOURCES.md`](./SOURCES.md)），
> 用纯 PyTorch 把 **compress → index → attend** 三步讲清楚，不提取生产 kernel。

---

## 1. 一句话：把 KV 压缩到极致，能选就选、不能选就全用

长上下文的瓶颈是 KV cache 和 $O(S^2)$ 注意力。DeepSeek V4 的思路：**把相邻 token 压成压缩块**，再在压缩
后的短序列上做注意力。两种配置交替使用：

| | 压缩比 m | 选块 | 复杂度 | 角色 |
|---|---|---|---|---|
| **CSA**（Compressed Sparse Attention） | 4× | lightning indexer 选 top-k=1024 | 稀疏 | 保留细粒度 |
| **HCA**（Heavily Compressed Attention） | 128× | 不选（稠密） | 压缩到够短，稠密也便宜 | 极致省 KV |

V4-Pro 61 层 ≈ 30 CSA + 31 HCA 交替。1M token 上下文下，单 token 推理 FLOP 仅 V3.2 的 **27%**、KV cache 仅 **10%**。

> 两者其实是**同一个**「压缩注意力」$\text{CompAttn}(m,k)$ 的特例：CSA 是 $(m{=}4,\,k{=}1024)$、
> HCA 是 $(m{=}128,\,k{=}\text{all})$。代码里就是一个 `compressed_attention(...)` 配不同参数。

---

## 2. 三步机制

### 步骤 1 · 学习式 softmax 压缩（`softmax_compress`）

每 $m$ 个 token 压成 1 个压缩块。**不是** NSA 的均值/卷积池化，而是**每个特征维度独立**在 $m$ 个位置上做
softmax 加权：

$$C_i=\sum_{j\in\text{block }i} \text{softmax}_j(Z_{\cdot,d})\odot x_j,\qquad x\in\{K,V\}.$$

权重 $Z$ 由一个学习投影给出，所以池化方式是学到的（能逼近"取最大""取均值"或任意软选择）。压缩后的
$C$ **同时充当 K 和 V**——每块只缓存一个向量，这是省 KV cache 的关键。

### 步骤 2 · Lightning indexer 选块（`lightning_indexer`，仅 CSA）

给每个 query 选 top-$k$ 个最相关的压缩块：

$$I_{t,s}=\sum_h w_{t,h}\,\text{ReLU}\!\bigl(q^I_{t,h}\cdot K^{IComp}_s\bigr),\qquad \text{选 top-}k\text{ 个 }s.$$

注意是 **ReLU 不是 softmax**——这是排序信号、不是概率分布，生产实现用 FP4 精度算，极省。块级 causal：
query $t$ 只能选完全落在它之前的块（$s\cdot m+m-1<t$）。

### 步骤 3 · 压缩块上的 MQA（`compressed_attention`）

在选中的压缩块上做 shared-KV 的多查询注意力（HCA 不选块，对所有可见压缩块稠密注意力）：

$$o_t=\text{Softmax}\!\Bigl(\tfrac{q_t K_{\mathcal S(t)}^\top}{\sqrt d}\Bigr)V_{\mathcal S(t)},\qquad K{=}V{=}C.$$

---

## 3. 与第 08 章 NSA 的关系

```
NSA（第 08 章）   =  压缩分支  +  选块分支  +  sliding window（局部）
DSv4 CSA          =  压缩分支  +  选块分支         （压缩→softmax池化，选块→ReLU indexer）
DSv4 HCA          =  压缩到 128×，稠密，不选块
                     （DSv4 没有 sliding window，省全靠压缩 + 稀疏索引）
```

NSA 用三分支覆盖"全局粗看 + 重点细看 + 邻近精看"；DSv4 把它简化成"压缩 + 按需选块"，并把压缩和选块都做得
更激进（学习式压缩、FP4 indexer、128× 压缩），从而在 1M 上下文上把 FLOP/KV 砍到极低。

---

## 4. 测试与运行

```bash
pytest 14-deepseek-v4/test_deepseek_v4.py -v   # 压缩缩长 / softmax 加权 / indexer top-k+causal / 形状 / causal / CSA(top_k=全)≡HCA
jupyter notebook 14-deepseek-v4/deepseek_v4.ipynb
```

> 学习路径：先读 §2 的三步机制 + `deepseek_v4.py`，把 compress→index→attend 跑通；再读 §3 对照第 08 章 NSA，
> 看清 DSv4 是怎么把"压缩稀疏"推到极致的。**这是简化的机制版**——FP4/FP8 量化、dual-stream 重叠压缩、KV
> cache 工程等生产细节见 `SOURCES.md` 指向的 vLLM/官方实现。

**上一章** ← 13-gated-deltanet（线性线终点）；本章回到**稀疏压缩线**（08 NSA → 14 DSv4），是其最新形态。
