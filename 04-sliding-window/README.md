# 04 · Sliding Window Attention（滑动窗口）

> 让每个 token 只看**最近 W 个** token，而不是全部历史。Mistral、Qwen 等都在用。
> 本章**不写新 kernel**——03 提取自 vLLM 的 kernel 本就自带滑窗，我们把它点亮。

---

## 1. 一个朴素的反问：真要看全部历史吗？

标准因果注意力里，第 10000 个 token 要和前面 9999 个全部算一遍。可语言有很强的**局部性**——
大部分依赖都在邻近几百上千个 token 内。那么：只看最近 $W$ 个，会怎样？

这就是滑动窗口：query $i$ 只关注 key $\in (i-W,\ i]$（最近 $W$ 个，含自己）。

```
全注意力(causal)          滑动窗口 (W=3)
q4: k0 k1 k2 k3 k4        q4:       k2 k3 k4
q3: k0 k1 k2 k3           q3:    k1 k2 k3
q2: k0 k1 k2              q2: k0 k1 k2
```

代价是单层看不到更远的历史，但**多层堆叠后感受野会线性扩大**（第 $L$ 层能间接看到 $L\times W$ 之外），
所以实际效果很好。

---

## 2. 实现：一个窗口掩码（注意 off-by-one）

滑窗在数学上就是在因果掩码之外，再屏蔽掉"太老"的 key：

$$\text{可见} \iff i - W < j \le i$$

`common.naive_attention(window=W)` 就是这么做参考的。而 03 的 kernel 用 `SLIDING_WINDOW_Q`
表示"key 不得早于 query 之前 `SLIDING_WINDOW_Q` 个位置"，含 `SLIDING_WINDOW_Q + 1` 个位置。
所以"看最近 $W$ 个"要传 `SLIDING_WINDOW = W - 1`——`sliding.py` 已替你换算：

```python
sliding_window_attention(q, k, v, window_size=W)   # 内部传 sliding_window = W-1
```

> 边界：kernel 以 `SLIDING_WINDOW > 0` 来启用滑窗，所以无法表达 `W=1`（只看自己），
> `sliding.py` 要求 `window_size >= 2`。

---

## 3. 真正的价值：decode 的 KV cache 被「钉死」

滑窗最大的意义在**推理 decode**：既然每个新 token 只看最近 $W$ 个，那更早的 K/V 永远不会再被用到，
可以**直接丢弃**。于是 KV cache **固定为 $W$，不再随上下文增长**：

`python bench.py`（单序列, Hkv=2, D=128, fp16, W=4096）：

| 上下文长度 | full KV cache | 滑窗 KV cache | 省 |
|---:|---:|---:|---:|
| 4 K | 4 MB | 4 MB | 1× |
| 64 K | 64 MB | 4 MB | 16× |
| 256 K | 256 MB | 4 MB | 64× |
| 1 M | 1024 MB | 4 MB | **256×** |

上下文越长，省得越多。这是固定显存下吞下超长上下文的关键。

---

## 4. 诚实：这个实现**不**加速 prefill

本章 kernel 用**掩码**实现滑窗——窗口外的 key 块照样读进来算、再屏蔽，循环范围没缩小。
所以 prefill 延迟和 full causal 相近：

| seqlen | full causal | 滑窗(W=4096) |
|---:|---:|---:|
| 4096 | 0.76 ms | 0.77 ms |
| 8192 | 2.41 ms | 2.59 ms |
| 16384 | 8.81 ms | 9.50 ms |

要在 prefill 也省算力，得让 kernel **主动跳过**窗口外的整块（block-sparse）——那是后面
稀疏注意力章节的主题。这里先把"滑窗的概念 + decode 价值 + 这个实现的边界"讲清楚。

---

## 5. 文件

| 文件 | 作用 |
|---|---|
| `sliding.py` | 薄封装：`window_size` → kernel 的 `SLIDING_WINDOW`（复用 03 的 kernel） |
| `test_sliding.py` | 滑窗 vs 朴素带窗参考；GQA+滑窗；窗口≥序列退化为 full；window=1 边界 |
| `bench.py` | decode KV cache（滑窗固定）与 prefill 延迟（与 full 相近） |
| `sliding.ipynb` | tutorial：从"真要看全部历史吗"出发，理解窗口掩码与 decode 价值 |

```bash
pytest 04-sliding-window/ -v        # 15 个用例
python 04-sliding-window/bench.py
```

**上一章** ← [03-gqa-mqa](../03-gqa-mqa/) ·
**下一章** → 05-paged-attention（推理 KV cache 的分页管理）
