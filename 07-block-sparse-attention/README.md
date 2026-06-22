# 07 · 块稀疏注意力（稀疏线入门）

> 标准注意力让每个 query 看**所有** key，但其实大部分注意力权重都集中在少数地方。
> 块稀疏的想法很直接：把 key 切成块，每个 query 块只挑**最相关的 top-k 个 key 块**来算，其余不看。
> 这是 DeepSeek **NSA** 等方法的核心地基。本章全部自写 PyTorch（[为什么](./SOURCES.md)），
> 为下一章读真实稀疏 kernel 打底。

---

## 1. 动机：注意力是稀疏的

FlashAttention 已经把 full 注意力做到了又快又省显存，但计算量仍是 $O(S^2)$——每个 query 都要
和每个 key 算一遍。可实际上，一个 token 真正"在意"的往往只是少数几段上下文。那就别全算了，
**只算重要的那几块**。

---

## 2. 三步：分块 → 选块 → 稀疏算

**① 分块 + pooled 代表**：把 key（和 query）按块取均值，得到每块的一个"代表向量"。

**② 块级重要性 + top-k 选块**：用代表向量两两打分，得到 (query 块 × key 块) 的重要性矩阵；
每个 query 块从中选出最相关的 **top-k 个** key 块。

$$\text{importance}_{ij} = \frac{\bar q_i \cdot \bar k_j}{\sqrt d}, \qquad
\mathcal{S}(i) = \operatorname{top\text{-}k}_j\ \text{importance}_{ij}$$

**③ 稀疏计算**：只对选中的块 $\mathcal{S}(i)$ 算注意力。计算量从 $O(nb)$ 个块降到 $O(\text{top-k})$ 个块。

`block_sparse.py` 里 `select_topk_blocks` 做①②，`block_sparse_attention` 做③。

> causal 细节：query 块 $i$ 只能选 $\le i$ 的 key 块。若块预算 top-k 超过可选块数，topk 会顺带选到
> 几个"未来块"，但它们在注意力里会被 causal 掩码全部排除，不影响结果。

---

## 3. 两条等价路径：看得见 vs 算得省

| 实现 | 做法 | 用途 |
|---|---|---|
| `block_sparse_reference` | 把块选择展开成 token 级 mask，再做一次 **full** 注意力 | ground truth；直观"看见"哪些块被选、哪些被屏蔽 |
| `block_sparse_attention` | **gather** 出选中的块，只对它们算注意力 | 真正省计算 |

两者数值一致（`test_block_sparse.py`，19 个用例，含"全选退化为 full""gather≡reference"）。

---

## 4. 性能：稀疏省了多少（`python bench.py`）

RTX 4090, S=2048, block_size=64（nb=32 块）, B=4 H=8 causal fp16：

| top_k | 稀疏度 | gather 延迟 | vs 朴素 full | vs SDPA full |
|---:|---:|---:|---:|---:|
| 1 | 3% | 0.34 ms | 18× | 0.6× |
| 2 | 6% | 0.58 ms | 11× | 0.3× |
| 4 | 12% | 1.43 ms | 4.3× | 0.1× |
| 32 | 100% | 10.5 ms | 0.6× | — |

读这张表：

- **少算块 = 快**：只看 2/32 块（6%），就比朴素 full 快 11×；
- **但纯 PyTorch 的 gather 干不过融合 kernel**：和 SDPA（0.19 ms）比，gather 的 gather/scatter
  开销明显——即便只算 3% 的块也没追上。

**结论**：稀疏要真正变成加速，必须把"选块 + 稀疏算"**融进一个 Triton kernel**，而不是用 PyTorch
gather。这样的 kernel 正是下面的深度优化版（§5，复用 08 NSA）。

---

## 5. 深度优化版：复用 08 NSA 的真实 triton kernel（`block_sparse_triton.py`）

上面说"稀疏要真正加速，得把选块 + 稀疏算融进一个 Triton kernel"。这样的 kernel 长什么样？其实
07 的"动态 top-k 块稀疏"**正是 08 NSA 的 selected 分支**。开源生态里没有独立、干净、4090 可跑的
"动态块稀疏 triton kernel"（NSA 那种与压缩 / 滑窗耦合；MoBA 用 flash_attn；triton 旧 `blocksparse`
ops 已移除）——所以本章深度优化版直接**复用 08 已提取并验证的** `nsa_triton.native_sparse_attend`。

**语义升级：对角块必看 + top-k 历史块。** 真实 kernel 的块稀疏比简要版的"纯 top-k"多一条规矩：
当前块（对角块）**总是**算。这是动态块稀疏的通用工程选择（当前块几乎总最相关），NSA 用专门的
Part1 处理对角块、selected 只选互补的历史块。复用时把对角块从选块里排除、交给 kernel 自动算。

| | 简要版 `block_sparse.py` | 深度优化版 `block_sparse_triton.py` |
|---|---|---|
| 选块语义 | 纯 top-k（causal） | 对角块必看 + top-k 历史块 |
| 算法 | PyTorch gather / mask | 复用 08 NSA selected triton kernel |
| 省算 | gather 省算（但有 gather 开销） | kernel 内融合"选块 + flash" |
| 验证 | 全选 ≡ full | 全历史 ≡ full causal、kernel ≡ 匹配语义参考 |

> `test_block_sparse_triton.py`：全历史 → full causal（锚点）+ triton ≡「对角 + top-k 历史」参考（10 用例）。

---

## 6. 对照：MoBA —— 另一条动态块稀疏路线（`moba_naive.py` / `moba_efficient.py`）

Kimi 的 **MoBA**（Mixture of Block Attention）是与 NSA 同期的动态块稀疏。本章拷贝它作对照：

- **机制**（`moba_naive.py`，纯 PyTorch，**可跑**）：KV 按 chunk 分块、块均值算 gate、每个 query
  选 top-k chunk；**当前块强制选中**（`gate[i块,i块]=inf`）—— 和深度优化版"对角块必看"一个道理。
- **高效做法**（`moba_efficient.py`，依赖 flash_attn，**仅阅读**）：MoBA 高效版**没有自己的 triton
  kernel**，而是把稀疏拆成 self-attn + moba-attn 两路，用 **flash_attn 的 varlen 接口 + 数据重排**
  算，再用在线 softmax 合并两路输出。

**两条落地路线对照**：

| | 08 NSA（07 深度优化版复用） | MoBA |
|---|---|---|
| 高效实现 | **自写**稀疏 triton kernel（选块 gather 融进 flash） | **借** flash_attn varlen + 数据重排 |
| 当前块 | 对角块必看（kernel Part1） | 当前块必选（gate=inf） |
| 依赖 | triton / einx（本仓库已跑通 fwd+bwd） | flash-attn==2.6.3（本环境未装） |

> 都"动态选块"，但工程路线不同：自写 kernel（NSA）极致但难写；借 flash_attn（MoBA）省力但受限于
> flash 的接口。理解这个对照，就理解了"动态稀疏注意力落地"的两种主流做法。

---

## 7. 文件

| 文件 | 作用 |
|---|---|
| `block_sparse.py` | 简要版：`select_topk_blocks` / `block_sparse_reference` / `block_sparse_attention` |
| `block_sparse_triton.py` | 深度优化版：复用 08 NSA selected kernel（对角块 + top-k 历史块） |
| `moba_naive.py` | MoBA 对照：纯 PyTorch，可跑（动态 chunk 选块 + 当前块必选） |
| `moba_efficient.py` | MoBA 高效版：flash_attn varlen 编排，依赖 flash-attn，仅阅读 |
| `test_block_sparse.py` | 简要版：全选 ≡ full、gather ≡ reference、causal 选块（19 用例） |
| `test_block_sparse_triton.py` | 深度优化版：全历史 ≡ full causal、kernel ≡ 参考（10 用例） |
| `test_moba.py` | MoBA naive 可跑 + 全选 ≡ full causal（3 用例） |
| `bench.py` | 稀疏度与计算量缩减；gather vs 朴素 / SDPA |
| `block_sparse.ipynb` | tutorial：从"注意力是稀疏的"出发，可视化选块、理解稀疏计算 |

```bash
pytest 07-block-sparse-attention/ -v
python 07-block-sparse-attention/bench.py
```

**上一章** ← [06-mla](../06-mla/) ·
**下一章** → 08-native-sparse-attention（NSA：压缩 + 选择 + 滑窗三分支，读真实稀疏 Triton kernel）
