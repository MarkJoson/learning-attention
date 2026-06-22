# 05 · Paged Attention（KV cache 的分页管理）

> 前几章都在优化注意力**怎么算**。这一章不一样——paged attention 优化的是 KV cache
> **在显存里怎么摆**。它把操作系统的"虚拟内存分页"搬到了 KV cache 上，是 vLLM 的招牌。
> 本章 kernel **提取自 lightllm**（见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. 问题：序列还没生成完，显存怎么分配？

推理时 KV cache 随着生成逐 token 增长，可你**事先不知道**一个请求最终会生成多长。
传统做法只能按"模型支持的最大长度"给每个请求预留一整块连续显存：

```
请求A（实际用 300）  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  预留 8192
请求B（实际用 1500） [████████████░░░░░░░░░░░░░░░░░░░░]  预留 8192
                      ↑ 实心是真用到的，空心全是浪费
```

参差不齐的真实负载下，这种预留会浪费大量显存，也限制了能同时跑多少请求。

---

## 2. 解法：分页 + block table（抄操作系统的作业）

操作系统早就解决过一模一样的问题——虚拟内存。paged attention 照搬：

1. 把 KV cache 切成固定大小的物理 **block（页）**，丢进一个统一的池子；
2. 每个序列维护一张 **block table**，记录"我的 token 存在哪些物理 block"；
3. 序列在物理上**不必连续**，需要了就从池子里取一个空闲 block——**按需分配**。

于是显存几乎零浪费，序列长度也不必预知。`paged.py` 的 `PagedKVCache` 就是这套机制的最小实现
（block table + free list + 按需分配 + 交错分配）。

---

## 3. kernel 的灵魂：一次间接寻址

paged 的注意力计算和普通注意力**没有区别**（还是 QK^T·softmax·V）。唯一的不同是：
KV 散落在物理池各处，读之前要先查 block table 把"逻辑位置"翻译成"物理 slot"。
`paged_decode_triton.py`（提取自 lightllm）里就是这两步：

```python
kv_loc = Req_to_tokens[req_idx, logical_pos]   # 逻辑第 i 个 token → 物理 slot
k      = K[kv_loc]                              # 用物理 slot 去读 KV
```

`Req_to_tokens` 是 block table 展开到 token 级的映射。正因为有这层间接，物理 slot 哪怕**完全乱序**，
结果依然正确（`test_paged.py` 专门用乱序/交错分配验证了这点）。

---

## 4. 价值：显存利用率（`python bench.py`）

64 个请求、长度 128~2048、模型 max_len=8192、block_size=16：

| 方案 | KV cache 占用 | 说明 |
|---|---:|---|
| 传统预分配（每请求按 max_len） | 2048 MB | 大量空预留 |
| **paged（按需分配 block）** | **274 MB** | 省 **7×** |
| 实际所需 | 272 MB | paged 利用率 **99%** |

paged 几乎把碎片榨干（利用率 99%）。同样的显存能跑下多得多的并发请求——这正是 vLLM 高吞吐的基石。
decode kernel 本身也很快：64 序列 × 2048 历史，一步 decode 仅 **0.63 ms**。

---

## 5. 文件

| 文件 | 作用 |
|---|---|
| `paged_decode_triton.py` | **提取**自 lightllm 的 paged decode kernel（间接寻址、GQA） |
| `paged.py` | `PagedKVCache`（分页内存管理）+ `paged_decode_attention` 封装 |
| `test_paged.py` | 正确性：含物理乱序、多序列交错分配、内存记账 |
| `bench.py` | 显存利用率（vs 传统预分配）与 decode 延迟 |
| `paged.ipynb` | tutorial：从"显存怎么分"出发，理解分页与间接寻址 |

```bash
pytest 05-paged-attention/ -v       # 11 个用例
python 05-paged-attention/bench.py
```

**说明**：这里 `PagedKVCache` 用 Python 管理 block table（讲清机制），真实框架（vLLM）的分配器
是高度优化的 C++/CUDA，但**核心思想完全一致**。kernel 仅 decode；prefill 见 03。

**上一章** ← [04-sliding-window](../04-sliding-window/) ·
**基础线到此完成** → 接下来可走稀疏线（NSA/DSA）或线性线（GLA/DeltaNet/KDA）
