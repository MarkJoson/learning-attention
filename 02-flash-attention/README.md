# 02 · FlashAttention v2（Triton Kernel 精读）

> 把 [01 章](../01-vanilla-attention/) 的 online softmax **融合进单个 GPU kernel**。
> 本章的 kernel **拷贝自 triton 官方教程**（非自写），我们负责精读、测试、基准与讲解。
> kernel 来源、license、改动见 [`SOURCES.md`](./SOURCES.md)。

---

## 1. FlashAttention 要解决什么

01 章的结论：online softmax 算法能把显存从 $O(S^2)$ 降到 $O(S)$，但纯 PyTorch 分块
反而更慢（Python 循环 + 反复读写 HBM）。FlashAttention 的贡献是把整个 online softmax
**融合进一个 kernel**：$QK^\top$、softmax、$PV$ 全在片上 SRAM 完成，中间矩阵 $S,P$
**从不写回 HBM**。这样既省显存又省带宽。

| | 标准实现 | FlashAttention |
|---|---|---|
| 中间矩阵 $S,P$ | 写回 HBM，$O(S^2)$ 显存 | 只在 SRAM，$O(S)$ 显存 |
| HBM 访问 | 多次读写 $S\times S$ | 只读 Q/K/V、只写 O |
| softmax | 全局 | online（分块增量） |

---

## 2. Kernel 结构（`flash_triton.py`）

### 前向：`_attn_fwd` + `_attn_fwd_inner`

`_attn_fwd` 是被 `@triton.autotune` + `@triton.jit` 修饰的前向 kernel。grid 的每个 program 负责
**一个 query 块（BLOCK_M 行）× 一个 (batch, head)**：

```
m_i = -inf            # running max，每个 query 行一个     ┐
l_i = 1.0             # running sum（分母）                ├ 对应 01 章 online softmax 的三个量
acc = 0               # running output（未归一化）         ┘
q = load(Q block)     # 一次性载入 SRAM，整个内循环常驻
```

内循环 `_attn_fwd_inner` 遍历 K/V 块，逐块更新这三个量 —— 与 01 章 `online_softmax_attention`
**逐行对应**：

```python
qk = tl.dot(q, k)                       # S_ij = Q·Kᵀ
m_ij = tl.maximum(m_i, tl.max(qk, 1))   # 新的 running max
p   = tl.math.exp2(qk - m_ij)           # 注意是 exp2，不是 exp（见下）
alpha = tl.math.exp2(m_i - m_ij)        # 修正因子 α
l_i = l_i * alpha + tl.sum(p, 1)        # 更新分母
acc = acc * alpha + tl.dot(p, v)        # 更新输出
m_i = m_ij
```

**为什么用 `exp2` 不用 `exp`**：GPU 硬件有快速的 `exp2`（2 的幂）指令。把缩放因子预乘
$1/\ln 2 = 1.4427$（`qk_scale *= 1.44269504`），就能用 $2^{x/\ln2}=e^{x}$ 等价替换，省一条指令。

**尾声（epilogue）**：

```python
m_i += tl.math.log2(l_i)   # 保存 logsumexp（= m + log l），反向要用
acc = acc / l_i            # 最终归一化
store(O); store(M=m_i)     # O 是输出，M 是每行的 logsumexp
```

### causal 的两阶段技巧

causal 时 `_attn_fwd` 把内循环拆成两段（`STAGE` 参数）：

- **off-band（对角线下方）**：整块都在 query 之前，**无需逐元素 mask**，全算；
- **on-band（对角线块）**：query 与 key 部分重叠，**需要 mask** 掉上三角。

这样大部分块走无 mask 的快路径，只有对角线块付 mask 代价 —— 这是 01 章"跳过上三角块"
优化的 kernel 级精细版。non-causal 时只有一个阶段、全程无 mask。

### 反向：`_attn_bwd_preprocess` + `_attn_bwd`

- `_attn_bwd_preprocess` 先算 $\delta = \mathrm{rowsum}(O\odot dO)$；
- `_attn_bwd` 用保存的 logsumexp $M$ 重算注意力权重（不存它，重算更划算），再求 $dQ,dK,dV$。

---

## 3. RTX 4090 适配（真实工程经验）

原教程 config 是为 A100/H100（192–228KB 共享内存）调的，直接搬到 4090（**~99KB** 共享内存）
会踩三个坑。**kernel 计算逻辑一字未改**，只调资源/启动参数（细节见 `flash_triton.py` 文件头与 `SOURCES.md`）：

| 坑 | 现象 | 适配 |
|---|---|---|
| 前向 autotune | `num_stages=7` 在 D=128 超出共享内存 | 候选收窄为 `[2,3,4]` |
| 前向 autotune 不跳过 OOM config | 直接抛 `OutOfResources` 而非跳过 | 新增 `_prune_4090_smem` 按 head_dim 裁剪 |
| 反向 kernel | 硬编码 `NUM_STAGES=5`，D=128 超限 | D≥128 时降为 2 |

实测校准的前向安全集（D=128，共享内存上限决定）：

| num_stages | BLOCK_N=64 | BLOCK_N=128 |
|---|:---:|:---:|
| 2 | ✅ | ✅ |
| 3 | ✅ | ❌ |
| 4 | ❌ | ❌ |

> **基准里的一个坑**：Triton autotune 首次触发时会异步 benchmark 多个 config。若与交错的
> SDPA/naive kernel 同时跑，会在大 seqlen 下因抢显存触发**异步**非法访存（`CUDA_LAUNCH_BLOCKING=1`
> 即消失）。`bench.py` 因此分两阶段：先在显存干净时跑完所有 autotune，再正式测量。

---

## 4. 性能（`python bench.py`，RTX 4090, B=2 H=16 causal fp16）

**head_dim = 64**

| seqlen | flash TFLOP/s | flash 显存 | SDPA TFLOP/s | vs naive |
|---:|---:|---:|---:|---:|
| 1024 | 78.5 | 272 MB | 54.1 | 28.9× |
| 2048 | 113.1 | 296 MB | 90.9 | 40.7× |
| 4096 | 134.9 | 329 MB | 120.9 | 48.8× |
| 8192 | 146.4 | 393 MB | 140.7 | naive OOM |

**head_dim = 128**

| seqlen | flash TFLOP/s | flash 显存 | SDPA TFLOP/s | vs naive |
|---:|---:|---:|---:|---:|
| 1024 | 73.6 | 296 MB | 82.3 | 14.5× |
| 4096 | 125.6 | 393 MB | 139.8 | 24.0× |
| 8192 | 139.9 | 521 MB | 150.9 | naive OOM |

读这两张表：

1. **显存随 seqlen 线性增长**（393→521 MB），naive 是平方增长、S=8192 直接 OOM；
2. **TFLOP/s 随序列变长而升高**：序列越长，kernel 内计算/访存比越高，越接近峰值；
3. **与 PyTorch SDPA 同量级**：都是融合 kernel，D=64 时 triton 教程版略快，D=128 时
   SDPA（cuDNN/flash 后端）略快 —— 教程 kernel 已是"接近生产"的水准。

---

## 5. 已知限制（继承自教程 kernel）

- **backward 仅支持 `causal=True`**（用原教程自带 reference 复现确认，non-causal 反向数值不对）；
- 仅标准 **MHA**：`Hq == Hkv`（不支持 GQA → 03 章）、`Sq == Sk`；
- `head_dim ∈ {16,32,64,128,256}`；backward 要求 `seqlen` 为 128 的倍数。

这些边界恰好说明：教程 kernel 聚焦最常用的 causal MHA，把 GQA / 变长 / 非整除等留给工程库
（FlashAttention 官方 CUDA 实现、本仓库后续章节）处理。

---

## 文件

| 文件 | 作用 |
|---|---|
| `flash_triton.py` | **拷贝**的 Triton kernel（FlashAttention v2）+ 4090 适配 + 中文注释 |
| `flash.py` | 调用封装（默认 scale、输入校验），不含 kernel 逻辑 |
| `test_flash.py` | 数值正确性：fwd/bwd vs naive / SDPA（含 D=128 适配验证） |
| `bench.py` | flash vs SDPA vs naive 的延迟 / 显存 / TFLOP-s |
| `SOURCES.md` | 外部来源、license、改动与能力边界登记 |

```bash
pytest 02-flash-attention/ -v        # 12 个用例
python 02-flash-attention/bench.py
```

**上一章** ← [01-vanilla-attention](../01-vanilla-attention/)（online softmax 算法）
**下一章** → 03-gqa-mqa（KV head 共享，本章 kernel 不支持的 GQA）
