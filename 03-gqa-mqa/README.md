# 03 · GQA / MQA（KV head 共享）

> 02 的 kernel 要求每个 query 头都配一个 KV 头（标准 MHA）。可推理时，KV cache 是显存大头——
> 让多个 query 头**共享**同一组 KV 头，就能把它成倍砍下来。这就是 GQA / MQA，几乎所有现代大模型都在用。
> 本章 kernel **提取自 vLLM**（见 [`SOURCES.md`](./SOURCES.md)）。

---

## 1. MHA → GQA → MQA：一根滑杆上的三个点

设 query 头数 $H_q$、KV 头数 $H_{kv}$，分组大小 $g = H_q / H_{kv}$：

| | $H_{kv}$ | 含义 |
|---|---|---|
| **MHA**（多头） | $=H_q$ | 每个 query 头独享一组 K/V |
| **GQA**（分组） | $1 < H_{kv} < H_q$ | 每 $g$ 个 query 头共享一组 K/V |
| **MQA**（多查询） | $=1$ | 所有 query 头共享同一组 K/V |

```
        MHA                  GQA (g=2)              MQA
   q0 q1 q2 q3            q0 q1  q2 q3           q0 q1 q2 q3
   │  │  │  │             └┬─┘   └┬─┘             └──┬──┬──┘
   k0 k1 k2 k3             k0     k1                  k0
```

为什么这么做？因为推理时要把历史所有 token 的 K/V 缓存下来（KV cache），它的大小正比于 $H_{kv}$。
$H_{kv}$ 越小，KV cache 越小，能塞下的 batch / 上下文就越长——而模型质量几乎不掉。

---

## 2. 关键洞察：同一个 kernel，只差一句 head 映射

GQA **不是新算法**。它和 MHA 跑的是同一套注意力，唯一区别是：query 头 $h$ 该去读哪组 K/V？

$$\text{kv\_head} = \left\lfloor h \,/\, g \right\rfloor$$

在 `gqa_triton.py` 里就是这一行（kernel 内，原样来自 vLLM）：

```python
cur_kv_head = cur_head // kv_group_num
```

就这一句，让 $g$ 个连续的 query 头都去读同一个 KV 头。**KV 在显存里只存 $H_{kv}$ 份，从不复制**——
这才是省 KV cache 的真正实现方式（而不是把 KV 复制成 $H_q$ 份再当普通 MHA 算，那样并不省）。

> 对照：`common.repeat_kv` 是"物理复制"版，数学等价、训练常用，但推理时复制了 KV 就白省了。
> 本章 kernel 走的是"索引映射"版，这正是推理框架的做法。

---

## 3. varlen：真实推理的数据长这样

这个 kernel 用的是 **varlen packed** 布局：把一个 batch 里长度不一的序列**首尾相接**成
`(总 token 数, head, head_dim)`，再用 `b_start_loc` / `b_seq_len` 标出每条序列的边界。

```
序列A(300) 序列B(500) 序列C(128)
[==========|================|====]   ← 一条大张量，没有 padding 浪费
 ^0         ^300             ^800     b_start_loc=[0,300,800]
```

真实推理里序列长度参差不齐，padding 到等长会浪费大量算力，varlen 是标准做法。
`gqa.py` 同时提供 varlen 原生接口和便捷的 `(B,H,S,D)` 接口（内部转换）。

---

## 4. 性能：省的是 KV cache，不是速度（`python bench.py`）

RTX 4090, $H_q$=32, D=128, seqlen=4096, batch=8, causal fp16：

| 模式 | $H_{kv}$ | KV cache | vs MHA | 注意力延迟 |
|---|---:|---:|---:|---:|
| MHA | 32 | 512 MB | 1× | 12.2 ms |
| GQA | 8 | 128 MB | 4× | 9.5 ms |
| GQA | 4 | 64 MB | 8× | 9.2 ms |
| MQA | 1 | 16 MB | **32×** | 9.0 ms |

读这张表：

- **KV cache 随 $H_{kv}$ 线性缩小**：MQA 比 MHA 省 32×（512→16 MB）；
- **延迟几乎不变、甚至略降**：query 头数没变，注意力计算量基本一样，省 KV 几乎是"免费"的；
- 这就是 Llama-2/3、Mistral、Qwen 等清一色用 GQA 的原因。

---

## 5. 文件

| 文件 | 作用 |
|---|---|
| `gqa_triton.py` | **提取**自 vLLM 的 prefill attention kernel（原生 GQA/MQA、causal、sliding window、varlen） |
| `gqa.py` | 封装：varlen 原生接口 + `(B,H,S,D)` 便捷接口 |
| `test_gqa.py` | MHA/GQA/MQA × causal × D=64/128、varlen 不等长、对照 SDPA enable_gqa |
| `bench.py` | KV cache 显存与延迟随 $H_{kv}$ 的变化 |
| `gqa.ipynb` | tutorial：从"KV cache 太大"出发，一步步理解 head 共享与映射 |

```bash
pytest 03-gqa-mqa/ -v        # 23 个用例
python 03-gqa-mqa/bench.py
```

**能力边界**：本章 kernel 仅 forward（推理 prefill 场景）。需要反向的训练场景可用 02 的 kernel（MHA）
或 PyTorch SDPA（`enable_gqa=True`）。

**上一章** ← [02-flash-attention](../02-flash-attention/) ·
**下一章** → 04-sliding-window（复用本章 kernel 自带的滑动窗口能力）
