# 06 · Multi-head Latent Attention（MLA）

> GQA 让多个 query 头共享 K/V 头，把 KV cache 砍了几倍。MLA 更狠——它把整组 K/V **压缩成一个
> 低维 latent 向量**，推理时只缓存这个 latent。KV cache 直接比 MHA 小 **50 多倍**。
> 这是 DeepSeek-V2/V3 的招牌。本章用纯 PyTorch 讲透机制（[为什么不提取 kernel](./SOURCES.md)）。

---

## 1. 思路：别存 K/V，存一个能"解压"出 K/V 的 latent

标准注意力把每个 token 投影成每个 head 的 K、V 存起来。MLA 反其道而行：

1. **压缩（down-projection）**：把 token 投影成一个低维 latent $c^{KV}$（如 512 维），**只缓存它**；
2. **解压（up-projection）**：需要时用 $W^{UK}, W^{UV}$ 把 $c^{KV}$ 重建回每个 head 的 K、V。

$$c^{KV}_t = W^{DKV} h_t \quad(\text{缓存这个}), \qquad k^C_t = W^{UK} c^{KV}_t,\ \ v_t = W^{UV} c^{KV}_t$$

一个 512 维的 latent，顶替了原来 128 个头 × 各 256 维的 K/V。这就是省显存的来源。

---

## 2. 拦路虎：RoPE 没法直接压缩 → decoupled RoPE

有个麻烦：旋转位置编码 RoPE 会把维度"缠"在一起，一旦压缩再解压，位置信息就乱了。
DeepSeek 的解法是把 query/key 劈成**两股**：

| 股 | 维度 | 谁来管 |
|---|---|---|
| **content（内容流）** | `qk_nope_head_dim` | 走压缩-解压，不带位置 |
| **position（位置流）** | `qk_rope_head_dim` | 一个小小的、**不压缩**的 RoPE 分量，专管位置 |

注意力分数 = 内容流点积 + 位置流点积，两股**只在算分数时相遇**。位置流的 key 是**所有 head 共享**
的一个小向量（也缓存），所以代价极小。`mla.py` 里 query 拆成 `q_nope / q_rope`，key 拆成
`k_nope`（从 latent 重建）`/ k_rope`（共享、带 RoPE）。

---

## 3. 杀手锏：weight absorption（推理时根本不重建 K/V）

朴素做法每步都要把 latent 解压成完整 K/V，再算注意力。MLA 发现这步可以**用矩阵结合律消掉**：

$$q^C \cdot k^C = (W^{UQ}c^Q)\cdot(W^{UK}c^{KV}) = \underbrace{(q^C W^{UK})}_{\text{把 }W^{UK}\text{ 吸进 query}} \cdot\, c^{KV}$$

也就是说，把上投影矩阵 $W^{UK}$ **吸收进 query**，query 就能**直接和缓存的 latent 算分数**，
压根不用重建 $k^C$。输出端同理，把 $W^{UV}$ 吸进输出投影。于是推理时：

- 缓存只有 latent + 共享 RoPE key；
- 注意力直接在 latent 维度上算，K/V **从不被材料化**。

`mla.py` 提供两条**数学等价**的路径：`forward_naive`（重建，作 ground truth）与 `forward_absorb`
（吸收，推理用）。`test_mla.py` 验证二者逐位等价（double 下差 ~1e-7，纯浮点累积）。

---

## 4. 数字说话：KV cache 比 MHA 小 57×（`python bench.py`）

按 DeepSeek-V2 规模（128 头, head_dim=128, kv_lora_rank=512, qk_rope=64）：

| 方案 | 每 token 缓存 | 相对 MHA |
|---|---:|---:|
| MHA（128 KV 头） | 32768 个值 | 1× |
| GQA（8 KV 头） | 2048 个值 | 16× |
| **MLA（latent 512+64）** | **576 个值** | **57×** |

放到整模型（60 层, batch=32, 8192 上下文, fp16）：

| 方案 | KV cache |
|---|---:|
| MHA | 960 GB |
| GQA | 60 GB |
| **MLA** | **17 GB** |

MLA 把 KV cache 从"几乎不可能"压到"一张卡装得下"——这正是 DeepSeek 能低成本服务长上下文的底气。

---

## 5. 文件

| 文件 | 作用 |
|---|---|
| `mla.py` | `MLA` 模块：`forward_naive`（重建）/ `forward_absorb`（PyTorch 吸收）/ `forward_absorb_triton`（latent 注意力走 triton kernel）三条等价路径、RMSNorm、decoupled RoPE |
| `mla_triton.py` | **提取**自 lightllm 的 MLA prefill triton kernel（absorb 格式，latent 维度直接计算） |
| `test_mla.py` | 三条路径互相等价（fp32/fp16/double 多维度）、KV cache 远小于 MHA |
| `bench.py` | KV cache 对比（MHA/GQA/MLA）、naive/absorb 前向延迟 |
| `mla.ipynb` | tutorial：从"GQA 还不够省"出发，理解压缩、解耦 RoPE、absorb 与真实 kernel |

```bash
pytest 06-mla/ -v        # 8 个用例
python 06-mla/bench.py
```

**能力边界**：注意力计算有两种真实实现——手写 PyTorch（naive，作 ground truth）与提取自 lightllm 的
MLA **prefill** triton kernel（absorb，在 latent 上算）。decode 的 flash-decoding（分阶段 split-K）
更复杂，留作延伸；追求极致速度可用 DeepSeek 的 FlashMLA（CUDA）。

**上一章** ← [05-paged-attention](../05-paged-attention/) ·
**推理线告一段落** → 接下来可走稀疏线（NSA/DSA）或线性线（GLA/DeltaNet/KDA）
