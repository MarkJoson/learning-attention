# 08 · Native Sparse Attention（NSA）

> DeepSeek 的"原生稀疏注意力"。一句话：**用廉价的"块压缩"粗看全局、并顺手产出选块路由，再用
> "top-k 选块"在重要的少数块里恢复细节，加一条滑窗兜底局部**——三分支由门控合并，稀疏模式是
> *学出来的*、且*对齐 GPU 硬件*（按块而非按 token 选，访存连续）。
>
> 本章有**两个版本**：
> - `nsa.py` —— 自写纯 PyTorch，mask 版三分支，把机制讲透（ground truth）。
> - `nsa_triton.py` —— **完整提取** lucidrains 的 1987 行真实稀疏 triton kernel，本文逐段拆解它
>   怎么把这套设计落到硬件（来源见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. 稀疏注意力的"鸡生蛋"难题，与 NSA 的破局

稀疏的目标是把 full attention 的 O(S²) 降下来 —— **只算重要的那些 KV**。但马上撞墙：

> 要判断 query 和某个 KV 重不重要，得先算它俩的注意力分数；要判断**所有** KV，就是 full
> attention，O(S²) 原地复活。**选择的前提是打分，而打分的代价 ≈ 不稀疏。**

NSA 的解法是**两级筛**：

1. **块压缩（compressed）= 廉价粗筛 + 免费路由器。** 把 KV 按块（b 个 token）压成 1 个压缩 token，
   S → S/b 个；query attend 这 S/b 个，代价 O(S²/b)，便宜 b 倍。**关键副产品**：query 对每个压缩
   块的注意力分数 = 这个块有多重要 —— 这张"块重要性表"是算压缩输出时**顺手就出来的**。
   （`nsa.py:57` 的 mean 压缩；`nsa.py:60` 的 `sim` 一身二职：既喂 compressed 分支的 softmax，
   又作为 `blk_sim` 返回给选块。）
2. **top-k 选块（selected）= 精筛恢复细节。** 拿上一步的块分数，每个 query 选 top-k 个最重要的块
   （`nsa.py:76` 的 `topk`），**只对这 k 个块的原始（未压缩）token** 做精细 attention。压缩是有损的
   （b→1 的平均抹掉了块内细节），选块把细节**只在重要的少数块**补回来。选谁完全由压缩分支免费
   提供 —— **压缩分支一身二职：全局视野 + 选块路由**。这就是为什么二者必须成对出现。
3. **滑窗（sliding）= 保底局部 + 训练分流。** 固定 attend 最近 window 个 token（`nsa.py:94`）。最近
   的 token 几乎总相关，不值得"选"，直接固定覆盖更省；而且单列一条局部分支，能避免梯度诱导压缩 /
   选择去拟合局部模式、污染它们本该学的全局 / 检索能力。

```
query ─┬─→ 压缩: attend S/b 个压缩块 ─→ 粗全局输出 + 块重要性分数 ┐
       │                                          (分数流向选择) ↓
       ├─→ 选择: 用分数选 top-k 块, attend 块内原始 token ─→ 细节输出
       └─→ 滑窗: attend 最近 window 个 token ──────────→ 局部输出
                                          门控加权融合(nsa.py:111) → 输出
```

---

## 2. 三分支与门控（`nsa.py`）

| 分支 | 看什么 | 代价 | 在 `nsa.py` |
|---|---|---|---|
| **compressed** | 全局粗粒度（S/b 个压缩 token） | O(S²/b) | `compressed_branch` |
| **selected** | 重点细粒度（top-k 个原始块） | O(S·k·b) | `selected_branch`（复用压缩分数选块）|
| **sliding** | 局部（最近 window） | O(S·w) | `sliding_branch` |

三分支输出由一个从 query 算出的**门控**（每个 head 三个权重，softmax 归一）加权合并
（`nsa.py:111`）。让模型自己决定每个 token 更信哪条。

---

## 3. 为什么按"块"选 → 硬件对齐（引出 kernel）

NSA 标题里的 **Hardware-Aligned** 就在这里：选"块"而非单 token，gather 的是**连续内存段**，GPU 上
连续访存、能套 FlashAttention 式的分块在线 softmax。若按单 token 选，gather 离散、访存碎片化，GPU
上反而慢。**块粒度就是为了让稀疏在 GPU 上真的快** —— 这正是 `nsa_triton.py` 那 1987 行在做的事。

简要版 `nsa.py` 用 mask 实现选块（算了全部分数再 mask 掉没选的，**不省计算，只为讲清机制**）；
真正省算、省访存，要靠下面的 triton kernel：**只 gather 选中的块、只算它们**。

---

## 4. 两个版本对照

| | `nsa.py`（简要版） | `nsa_triton.py`（深度优化版） |
|---|---|---|
| 实现 | 自写纯 PyTorch，三分支 mask 版 | 完整提取 lucidrains 真实 triton kernel |
| 目的 | 讲透"压缩+选块+滑窗+门控"机制 | 讲透"per-query 选块怎么在 GPU 上算得快" |
| 选块 | 算全部分数再 mask（不省算） | **只 gather 选中块**（真省算 + 省访存） |
| 覆盖 | 三分支 + 门控（完整架构） | selected + sliding 的稀疏 flash（最重的一段）|
| fwd/bwd | fwd（教学） | fwd + bwd（生产级，atomic 累加梯度） |

> 注意分工：那 1987 行**只算 selected + sliding**；"块压缩、算块分数、top-k 选块、门控合并"是
> host 端 PyTorch（`SparseAttention` Module / `nsa.py`）。kernel 的入口 `native_sparse_attend` 接收
> **已经算好的** `selected_block_indices`，只负责把它们算得飞快。

---

## 5. 拆解 `nsa_triton.py`（1987 行）

整份文件的骨架（行号为本仓库拷贝后的 `nsa_triton.py`）：

```
forward_kernel_causal_and_sparse  (108-520)  ← 前向主 kernel：local + sparse
forward_kernel                    (529-633)  ← wrapper：用 grid 第三维分派 sliding / selected
native_sparse_attn_forward        (634-730)  ← host：reshape、切 grid、起 kernel
backward_*                        (731-1680) ← 反向：preprocess + 两个 col-block kernel
native_sparse_attn_backward       (1681-1858)← host：起反向 kernel
class NSA(Function)               (1859-1962)← autograd 封装（forward/backward）
native_sparse_attend              (1973-2022)← 对外入口
```

### 5.1 forward host：grid 怎么切（`native_sparse_attn_forward`, 634-730）

```python
grid = lambda META: (
    triton.cdiv(seqlen_q, META["BLOCK"]),   # ① query 分块（每块 BLOCK=16 个 query）
    batch * kv_heads,                        # ② batch × kv_heads ← 注意是 kv_heads！
    (2 if return_sliding_window_out else 1)  # ③ 是否单独算 sliding 输出
)
```

**最关键的设计在第 ② 维用 `kv_heads` 而不是 query heads**：一个 program 负责一个 KV head，把这个
KV head 对应的**整组 query head 一次性加载**（GQA：`QUERY_HEAD_GROUPS = nheads // kv_heads` 个 query
head 共享一份 KV）。于是 **KV 只从显存读一次，被组内多个 query head 复用** —— 这正是 GQA 的省带宽
收益，在 kernel 层面兑现。`q` 因此是三维 `(BLOCK, QUERY_HEAD_GROUPS, HEADDIM)`，在线 softmax 的
`m_i / lse_i / acc_o` 也都带 query-head-group 维（`forward_kernel...:175-198`）。

约束：`block_size` 必须是 16 的倍数（`num_blocks_per_sel = block_size // 16`，一个选择块拆成若干
16-子块）；`dim ≤ 128`；只支持 fp16 / bf16。

### 5.2 forward kernel：local + sparse 汇入**同一个**在线 softmax

`forward_kernel_causal_and_sparse` 分两段，但共享一套滚动统计量 `m_i / lse_i / acc_o`（flash 的
在线 softmax），所以两段的贡献被正确地合并归一：

**Part 1 — 块对角 + 滑窗（227-361）：连续访问的"local"部分。**
处理当前 query 块**自己所在的对角块**（块内 causal），以及（`SLIDING` 时）相邻的前一块。这部分
key 位置是连续的，直接按 `offs_n` 加载、标准 flash 累加。causal_mask 保证不看未来（306）；`SLIDING`
时再叠加窗口约束 `(offs_m - offs_n) <= SEL_BLOCK`（308-312）。

**Part 2 — per-query 选块 gather（363-482）：NSA 的灵魂。**
这是"稀疏"真正发生的地方：

```python
# 把"每个 query 选的第几个块"翻译成原始序列里的实际 key 位置（400-403）
blocks_offs_n = block_indices[:, None] * (BLOCK * NUM_BLOCKS_PER_SEL) \
              + tl.arange(0, BLOCK)[None, :] + (off_blocks_per_sel * BLOCK)
# 用它当指针偏移，直接 gather 每个 query 各自选中的 K/V（423, 458）
k_block = tl.load(block_k_ptrs, mask=blocks_offs_n[:, :, None] < seqlen_k, other=0.)
```

不是固定稀疏模式，是 **query-dependent 的动态选块**：`block_indices` 来自 host 端 top-k，每个 query
gather 自己那几块。外层循环 `NUM_SEL_KV_BLOCKS`（选了几块），内层 `NUM_BLOCKS_PER_SEL`（一块拆成
几个 16-子块）。`block_masks`（392）把无效选块（早期 token 选不满 / padding）置 `-inf`（442）。

### 5.3 三个让它"在 GPU 上真的快"的技巧

1. **动态 gather 索引（5.2 Part 2）** —— 把抽象的"块号"算成显存偏移，让每个 query 只触碰自己选中
   的块，省算又省访存。
2. **GQA 凑 16 维（379-382 + heuristic 526）。** triton 的 `tl.dot` 要求 M 维 ≥ 16，但一组 query
   head 数 `QUERY_HEAD_GROUPS` 常 < 16（如 4）。于是把它**广播复制** `QUERY_EXPAND_DIM = 16 //
   QUERY_HEAD_GROUPS` 次凑满 16 去做矩阵乘，算完再 `tl.reduce(..., reduce_avg)` 把复制维平均还原
   （438-439, 472-474）。用一次"凑整"把 GQA 的 KV 复用塞进 MMA 的硬件约束。
3. **sliding 与 selected 分两个 program、各自 lse（wrapper 577-586）。** grid 第三维 `program_id(2)`：
   `==0` 只算 sliding（`num_sel_kv_blocks=0`）写 `SlidingOut/SlidingLse`，`==1` 算 selected 写
   `Out/Lse`。两分支各自独立在线 softmax、各自输出 lse，host 层再用门控把它们合并 —— kernel 不掺和
   门控，职责干净。

收尾：用 `lse` 对 `acc_o` 做最终归一化（486-487），写回 `o` 和 `lse`（489-519）。

### 5.4 backward：per-query gather 的梯度必须 atomic 累加

反向的难点恰恰来自 Part 2 的动态 gather：**多个不同的 query 可能选中同一个 KV 块**，于是该块的
`dk/dv` 会收到来自多个 query 的梯度贡献，必须**累加**而非覆盖。kernel 的处理：

- `NSA.backward`（1909-1960）把 `dq/dk/dv` 用 **fp32 零初始化**（1935-1937），再用
  `tl.atomic_add(..., sem='relaxed')` 累加（`dk/dv` 见 812-823，`dq` 见 1153 / 1431）。
- `backward_preprocess_do_o_dot`（731）先算 `delta = rowsum(do ∘ o)` —— flash 反向的标准预处理。
- 反向也分 sparse / causal 两段，对应 `backward_kernel_one_col_block_sparse`（826）与
  `backward_kernel_one_col_block_causal`（1170），与前向的 Part 2 / Part 1 对称。

### 5.5 对外入口（`native_sparse_attend`, 1973-2022）

```python
out = native_sparse_attend(
    fq, fk, fv,                 # 原始（未压缩）q/k/v，fine 分支用
    block_size,                 # 选择块大小（16 的倍数）
    selected_block_indices,     # 每个 query 选中的块号（host 端 top-k 的结果）
    fmask,                      # 选块有效掩码
    include_block_causal=True,  # causal
)
```

`sel_heads` 可按 query head 或 kv head 给选块；按 query head 选时把 KV `repeat` 到 q_heads（1996）。
`NSA.apply` 是 autograd Function，前向转 fp16 算、保存反向所需张量。

---

## 6. 测试与运行

```bash
# 简要版三分支机制（纯 PyTorch，4 用例）
pytest 08-native-sparse-attention/test_nsa.py -v

# 深度优化版 kernel（需 pip install native-sparse-attention-pytorch）
#   ① 拷贝忠实：本地拷贝 == 库原版（atol=0，证明零改动）
#   ② 端到端：替换库 triton 入口 vs 库 PyTorch 路径，fwd+bwd 对齐
pytest 08-native-sparse-attention/test_nsa_triton.py -v

# 三分支"信息覆盖"与延迟基准
python 08-native-sparse-attention/bench.py
```

- 简要版 `nsa.py`：零额外依赖，讲机制。
- 深度优化版 `nsa_triton.py`：去依赖后仅需 `torch/triton/einx/einops` 即可 `import` 运行；
  其**输入准备**（块压缩 / 选块）目前借 lucidrains 的 `SparseAttention` 完成，故测试依赖该包。

> 学习路径建议：先读 `nsa.py` + 本文 §1–§3 弄懂**为什么这么设计**，再对照本文 §5 逐段读
> `nsa_triton.py` 弄懂**怎么把它在 GPU 上算快**。
