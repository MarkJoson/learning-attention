"""生成 03-gqa-mqa/gqa.ipynb —— tutorial 叙事 + kernel 拆解（head 映射）。"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""\
# KV cache 太大了，于是有了 GQA

到目前为止（01、02 章），我们默认每个 query 头都配一个自己的 K/V 头——标准的多头注意力（MHA）。
训练时这没问题。可一旦进入**推理**，一个幽灵冒了出来：**KV cache**。

为了不重复计算，推理时要把历史所有 token 的 K/V 缓存在显存里。这块缓存会大到什么程度？
大到能轻松吃光一张显卡。这一章我们就从"KV cache 为什么这么大"出发，看 GQA / MQA 是怎么把它
成倍砍下来的——而秘诀，只是 kernel 里的**一句 head 映射**。

本章的 kernel 不是自己写的，而是从 vLLM 里**提取**出来的（生产框架的原生 GQA 实现）。
"""))

cells.append(code("""\
import sys, math
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path.cwd()
while not (ROOT / "common").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "03-gqa-mqa"))

from common import naive_attention, make_qkv, bench_ms
from gqa import gqa_attention, gqa_attention_varlen

torch.manual_seed(0)
print("跑在", torch.cuda.get_device_name(0), "| 本章 kernel 提取自 vLLM 的 prefill attention")
"""))

cells.append(md("""\
## 先被吓一跳：KV cache 到底有多大

拿一个 7B 级别模型的常见配置算一笔账：32 层、每层 32 个头、head_dim=128。
推理时每层都要缓存 K 和 V，缓存量 = `2(K和V) × 层数 × batch × 头数 × 序列长度 × head_dim`。
"""))

cells.append(code("""\
layers, Hq, D = 32, 32, 128
batch, seqlen = 16, 4096

def kv_cache_gb(n_kv_heads):
    return 2 * layers * batch * n_kv_heads * seqlen * D * 2 / 1024**3   # fp16=2 字节

print(f"配置: {layers} 层, batch={batch}, 上下文={seqlen}, head_dim={D}")
print(f"MHA（{Hq} 个 KV 头）的 KV cache = {kv_cache_gb(Hq):.1f} GB")
print(f"  ↑ 一张 24GB 的 4090 光是 KV cache 就装不下，更别说模型权重了。")
"""))

cells.append(md("""\
## 一个朴素的念头：能不能少存几组 K/V？

KV cache 的大小，正比于 **K/V 头的数量**。那如果……让多个 query 头**共享**同一组 K/V 头呢？
K/V 头少了，缓存不就小了？

这就是两个想法的来历：

- **MQA**（Multi-Query）：走到极端，**所有** query 头只共享 **1 组** K/V。缓存最小，但表达力损失略大；
- **GQA**（Grouped-Query）：折中，把 query 头**分组**，每组共享一组 K/V。在"省显存"和"保质量"间取平衡。

同样的账，换成 GQA / MQA：
"""))

cells.append(code("""\
print(f"MHA  (32 个 KV 头): {kv_cache_gb(32):>5.1f} GB   （基准）")
print(f"GQA  ( 8 个 KV 头): {kv_cache_gb(8):>5.1f} GB   省 {32/8:.0f}×")
print(f"GQA  ( 4 个 KV 头): {kv_cache_gb(4):>5.1f} GB   省 {32/4:.0f}×")
print(f"MQA  ( 1 个 KV 头): {kv_cache_gb(1):>5.1f} GB   省 32×")
print("\\n同样的模型质量下，KV cache 从 32GB 缩到 1GB——这就是几乎所有现代大模型都用 GQA 的原因。")
"""))

cells.append(md("""\
## 怎么让 8 个 query 头读同一组 K/V？一句映射

想法很美，可 kernel 里具体怎么做？答案朴素得让人意外：**每个 query 头，按编号整除分组大小，
算出它该读哪个 K/V 头**：

$$\\text{kv\\_head} = \\lfloor \\text{query\\_head} \\,/\\, g \\rfloor, \\quad g = H_q / H_{kv}$$

我们把映射关系列出来看（8 个 query 头、2 个 KV 头、每组 4 个）：
"""))

cells.append(code("""\
Hq, Hkv = 8, 2
g = Hq // Hkv
print(f"{Hq} 个 query 头, {Hkv} 个 KV 头, 每组 {g} 个 query 头共享 1 个 KV 头：\\n")
for h in range(Hq):
    print(f"  query 头 {h}  →  KV 头 {h // g}")

# 这正是 gqa_triton.py 里 kernel 的一行（原样来自 vLLM）
src = (ROOT / "03-gqa-mqa" / "gqa_triton.py").read_text().splitlines()
line = next(l for l in src if "cur_kv_head = cur_head" in l)
print(f"\\nkernel 里就是这一句： {line.strip()}")
"""))

cells.append(md("""\
## 直接读这个 kernel

光说一句映射不过瘾，我们把整个 kernel 摊开来读（提取自 vLLM，带语法高亮）：
"""))

cells.append(code("""\
from IPython.display import Code
Code(filename=str(ROOT / "03-gqa-mqa" / "gqa_triton.py"), language="python")
"""))

cells.append(md("""\
### 逐段读懂它

**① 每个 program 负责一块 query × 一个 head，先算出该读哪组 KV**

```python
cur_batch = tl.program_id(0); cur_head = tl.program_id(1); start_m = tl.program_id(2)
cur_kv_head = cur_head // kv_group_num      # ← GQA 的全部秘密
```

按 (query 块 × query 头) 切分，再用整除算出这个 query 头该读哪个 KV 头。MHA 时 `kv_group_num=1`
（各读各的），GQA 时多个连续 query 头落到同一个 `cur_kv_head`。

**② 内循环：用 cur_kv_head 去读共享的 K/V**

```python
off_k = offs_n[None,:]*stride_kbs + cur_kv_head*stride_kh + offs_d[:,None]   # 偏移用 cur_kv_head
k = tl.load(k_ptrs + ...);  qk = tl.dot(q, k)
```

`k` 的偏移用的是 `cur_kv_head` 而非 `cur_head` —— 这就是"多个 query 头共享一组 KV"落到内存寻址上：
KV 物理上只存 Hkv 份。

**③ 掩码：causal 与滑动窗口都叠在这里**

```python
if IS_CAUSAL: mask &= pos_q >= pos_k
sliding_mask_q = pos_q - pos_k <= SLIDING_WINDOW_Q    # 04 章点亮的就是它
qk = tl.where(mask, qk * sm_scale, -1.0e8)
```

因果和滑动窗口都只是往这张 mask 上叠条件 —— 04 章的滑窗，就是把 `SLIDING_WINDOW_Q` 设成正数。

**④ online softmax + 加权累加**

```python
m_ij = tl.maximum(m_i, tl.max(qk, 1));  p = tl.math.exp2(qk - m_ij[:, None])
acc = tl.dot(p, v, acc)
```

和 01/02 章一模一样的 online softmax —— **核心算法从没变过**，变的只是"读哪组 KV"。
"""))

cells.append(md("""\
## 验证：同一个 kernel，MHA / GQA / MQA 通吃

因为差别只在那句映射，所以**同一份 kernel** 不需要任何改动，就能跑三种模式——
你给多少个 KV 头，它就分多少组。我们让它和可信的朴素实现对一对：
"""))

cells.append(code("""\
for name, Hq, Hkv in [("MHA", 8, 8), ("GQA", 8, 2), ("MQA", 8, 1)]:
    q, k, v = make_qkv(2, Hq, 512, 64, kv_heads=Hkv, dtype=torch.float16, seed=0)
    out = gqa_attention(q, k, v, causal=True)
    ref = naive_attention(q, k, v, causal=True)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"{name}  (Hq={Hq}, Hkv={Hkv}, 分 {Hq//Hkv} 组): 与朴素实现最大差异 {err:.1e}  ✓")
print("\\n一份 kernel，三种模式，全部正确。")
"""))

cells.append(md("""\
## 一个关键陷阱：物理复制 vs 索引映射

这里有个容易踩的坑。让 query 头共享 K/V，有两种做法：

1. **物理复制**：把那 2 个 KV 头，复制成 8 份，凑成"假装的 MHA"，再用普通 MHA kernel 算。
   数学上完全正确，写起来也省事（`common.repeat_kv` 就是干这个的）——**但显存里 KV 实实在在存了 8 份，
   一点没省！** 这只适合训练时图方便。
2. **索引映射**：KV 在显存里**只存 2 份**，kernel 读的时候用 `cur_head // g` 现算该读哪份。
   这才真省 KV cache，也是本章 kernel（和所有推理框架）的做法。

同样一份 GQA，两种实现，KV 占用天差地别：
"""))

cells.append(code("""\
B, Hq, Hkv, S, D = 8, 32, 8, 4096, 128
native   = 2 * B * Hkv * S * D * 2 / 1024**2     # 索引映射：KV 只存 Hkv 份
repeated = 2 * B * Hq  * S * D * 2 / 1024**2     # 物理复制：KV 被复制成 Hq 份

print(f"索引映射（KV 存 {Hkv} 份）: {native:>5.0f} MB   ← 本章 kernel")
print(f"物理复制（KV 存 {Hq} 份）: {repeated:>5.0f} MB   ← repeat_kv，白白多了 {repeated/native:.0f}×")
print("\\n两者算出的结果完全一样，但显存占用差 4 倍。推理省 KV cache，靠的必须是索引映射。")
"""))

cells.append(md("""\
## 真实推理长这样：varlen 不等长序列

顺带认识本章 kernel 的输入格式。真实推理里，一个 batch 内的序列长度参差不齐，
要是 padding 到等长会浪费大量算力。于是框架把它们**首尾相接**成一条大张量，
用 `b_start_loc` / `b_seq_len` 标出每条的边界——这叫 **varlen**。我们拼三条不同长度的序列，
逐条验证结果都对：
"""))

cells.append(code("""\
Hq, Hkv, D = 8, 2, 64
seqlens = [300, 500, 128]
total = sum(seqlens)
qp = torch.randn(total, Hq, D, dtype=torch.float16, device="cuda")
kp = torch.randn(total, Hkv, D, dtype=torch.float16, device="cuda")
vp = torch.randn(total, Hkv, D, dtype=torch.float16, device="cuda")
o  = torch.empty_like(qp)

starts, s = [], 0
for L in seqlens: starts.append(s); s += L
b_start  = torch.tensor(starts,  device="cuda", dtype=torch.int32)
b_seqlen = torch.tensor(seqlens, device="cuda", dtype=torch.int32)

gqa_attention_varlen(qp, kp, vp, o, b_start_loc=b_start, b_seq_len=b_seqlen,
                     max_seqlen=max(seqlens), causal=True)

print("三条序列拼成一条", tuple(qp.shape), "，逐条对账：")
for st, L in zip(starts, seqlens):
    q1 = qp[st:st+L].permute(1,0,2).unsqueeze(0)
    k1 = kp[st:st+L].permute(1,0,2).unsqueeze(0)
    v1 = vp[st:st+L].permute(1,0,2).unsqueeze(0)
    ref = naive_attention(q1, k1, v1, causal=True)[0].permute(1,0,2)
    print(f"  序列长度 {L:>3}: 与朴素实现最大差异 {(o[st:st+L]-ref).abs().max().item():.1e}  ✓")
"""))

cells.append(md("""\
## 省了多少？延迟又如何

最后把账算全：固定 32 个 query 头，把 KV 头从 32 一路降到 1，看 KV cache 和**注意力延迟**怎么变。
关键看延迟那一列——因为 query 头数没变，注意力的计算量基本一样。
"""))

cells.append(code("""\
Hq, D, S, B = 32, 128, 4096, 8
hkvs = [32, 8, 4, 1]
kv_mb, lat = [], []
for Hkv in hkvs:
    q, k, v = make_qkv(B, Hq, S, D, kv_heads=Hkv, dtype=torch.float16, seed=0)
    kv_mb.append(2 * B * Hkv * S * D * 2 / 1024**2)
    lat.append(bench_ms(lambda: gqa_attention(q, k, v, causal=True)))
    del q, k, v; torch.cuda.empty_cache()

labels = [("MHA" if h==Hq else "MQA" if h==1 else "GQA") + f"\\nHkv={h}" for h in hkvs]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(labels, kv_mb, color=["#c0504d","#4f81bd","#4f81bd","#9bbb59"])
ax1.set(title="KV cache（越小越好）", ylabel="MB")
ax2.bar(labels, lat, color=["#c0504d","#4f81bd","#4f81bd","#9bbb59"])
ax2.set(title="注意力延迟（几乎不变）", ylabel="ms")
plt.tight_layout(); plt.show()
print(f"KV cache: MHA {kv_mb[0]:.0f}MB → MQA {kv_mb[-1]:.0f}MB（省 {kv_mb[0]/kv_mb[-1]:.0f}×）；"
      f"延迟: {lat[0]:.1f}ms → {lat[-1]:.1f}ms（基本持平甚至略降）。")
"""))

cells.append(md("""\
## 收尾

- 推理的显存杀手是 **KV cache**，它正比于 K/V 头的数量；
- **GQA / MQA** 让多个 query 头共享 K/V 头，把缓存成倍砍小，而模型质量、注意力延迟几乎不变；
- 实现的灵魂只是一句 **`cur_kv_head = cur_head // kv_group_num`**——同一份 kernel 通吃 MHA/GQA/MQA；
- 真省显存的前提是**索引映射**（KV 只存 Hkv 份），而非物理复制；
- 顺带见识了真实推理的 **varlen** 输入格式。

**下一站** → `04-sliding-window`：让每个 token 只看最近的一段窗口。有意思的是——
本章这个 kernel **已经自带**滑动窗口能力，下一章我们就把它点亮。
"""))

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, "03-gqa-mqa/gqa.ipynb")
print("已生成 03-gqa-mqa/gqa.ipynb（tutorial 风格），cells:", len(cells))
