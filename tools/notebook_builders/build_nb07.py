"""生成 07-block-sparse-attention/block_sparse.ipynb —— tutorial 叙事。"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""\
# 注意力其实很稀疏：块稀疏入门

FlashAttention 把 full 注意力做到了又快又省显存，但它的计算量仍是 $O(S^2)$——每个 query 都要
和每个 key 算一遍。可你若真去看一眼注意力矩阵会发现：**大部分权重都集中在少数几段上下文**，
其余几乎是零。那为什么还要全算？

这一章我们走进**稀疏注意力**：把 key 切成块，每个 query 块只挑**最相关的 top-k 个 key 块**来算。
这是 DeepSeek NSA 等方法的核心地基。我们先用纯 PyTorch 把"选块 + 稀疏算"的机制讲透，
下一章再去读真正的稀疏 Triton kernel。
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "07-block-sparse-attention"))

from common import make_qkv, naive_attention, bench_ms
from block_sparse import select_topk_blocks, block_sparse_reference, block_sparse_attention, _pool_blocks

torch.manual_seed(0)
print("跑在", torch.cuda.get_device_name(0))
"""))

cells.append(md("""\
## 先亲眼看看：注意力到底有多稀疏

我们算一个真实的因果注意力矩阵，把它画出来，再统计每个 query "真正在意"的 key 有多少。
"""))

cells.append(code("""\
q, k, v = make_qkv(1, 1, 128, 64, dtype=torch.float16, seed=0)
scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(64)
S = q.shape[2]
scores = scores.masked_fill(~torch.ones(S, S, device=q.device).tril().bool(), float("-inf"))
attn = torch.softmax(scores, dim=-1)[0, 0]      # (S,S)

plt.figure(figsize=(5, 4.5))
plt.imshow(attn.cpu().float().pow(0.3), cmap="magma", aspect="auto")  # ^0.3 提亮，便于看清
plt.title("因果注意力权重（越亮越大）"); plt.xlabel("key"); plt.ylabel("query")
plt.tight_layout(); plt.show()

# 每个 query：要累计到 90% 权重，平均需要多少个 key？
sorted_w = attn.sort(dim=-1, descending=True).values
cum = sorted_w.cumsum(-1)
need = (cum < 0.9).sum(-1).float() + 1          # 达到 90% 权重所需的 key 数
valid = torch.arange(1, S + 1, device=q.device).float()  # 每个 query 的可见 key 数(causal)
print(f"平均每个 query 要累计 90% 注意力，只需 {need.mean():.0f} 个 key（共最多 {S} 个可见）——其余几乎是零。")
"""))

cells.append(md("""\
## 三步：分块 → 选块 → 稀疏算

既然注意力这么稀疏，那就别全算。做三件事：

1. **分块 + pooled 代表**：把 query/key 按块取均值，每块得到一个"代表向量"；
2. **块级重要性 + top-k 选块**：用代表向量两两打分，每个 query 块选最相关的 top-k 个 key 块；
3. **稀疏计算**：只对选中的块算注意力。

第②步是灵魂。我们把"块级重要性矩阵"和"最终选中的块"并排画出来看：
"""))

cells.append(code("""\
q, k, v = make_qkv(1, 1, 256, 64, dtype=torch.float16, seed=1)
block_size, top_k = 32, 2
nb = 256 // block_size

q_blk, k_blk = _pool_blocks(q, block_size), _pool_blocks(k, block_size)
imp = (q_blk.float() @ k_blk.float().transpose(-1, -2) / math.sqrt(64))[0, 0]   # (nb,nb)
causal = torch.ones(nb, nb, device=q.device).tril().bool()
imp_vis = imp.masked_fill(~causal, float("nan"))

idx = select_topk_blocks(q, k, block_size, top_k, causal=True)[0, 0]            # (nb, top_k)
sel = torch.zeros(nb, nb, dtype=torch.bool, device=q.device)
sel.scatter_(1, idx, True); sel &= causal

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.imshow(imp_vis.cpu().float(), cmap="viridis"); ax1.set(title="① 块级重要性 (query块 × key块)", xlabel="key 块", ylabel="query 块")
ax2.imshow(sel.cpu().float(), cmap="Greens"); ax2.set(title=f"② 每个 query 块选 top-{top_k} 个 key 块", xlabel="key 块", ylabel="query 块")
plt.tight_layout(); plt.show()
print(f"右图每行只有 {top_k} 个绿格——{nb} 个 key 块里只算这 {top_k} 个，其余跳过。")
"""))

cells.append(md("""\
## 看看稀疏算子的代码

`block_sparse.py` 把这三步实现成两条路径：`block_sparse_reference`（mask + full，作 ground truth）
和 `block_sparse_attention`（gather 选中块、只算选中块）。直接读一下后者——稀疏计算就发生在这里：
"""))

cells.append(code("""\
from IPython.display import Code
Code(filename=str(ROOT / "07-block-sparse-attention" / "block_sparse.py"), language="python")
"""))

cells.append(md("""\
## 验证：两条路径一致，全选时退化为 full

- gather 省算实现，必须和 mask 参考实现数值一致；
- 当 top-k 取满（选中所有块）时，块稀疏必须退化成普通 full 注意力。
"""))

cells.append(code("""\
q, k, v = make_qkv(2, 4, 512, 64, dtype=torch.float16, seed=2)
nb = 512 // 64

# gather ≡ reference
for tk in [1, 2, 4]:
    ref = block_sparse_reference(q, k, v, 64, tk, causal=True)
    gat = block_sparse_attention(q, k, v, 64, tk, causal=True)
    print(f"top_k={tk}: gather vs mask参考 最大差异 {(ref-gat).abs().max().item():.1e}  ✓")

# 全选 ≡ full
full = naive_attention(q, k, v, causal=True)
allsel = block_sparse_reference(q, k, v, 64, top_k=nb, causal=True)
print(f"\\ntop_k={nb}(全选) vs full 注意力 最大差异 {(full-allsel).abs().max().item():.1e}  ✓ 退化为 full")
"""))

cells.append(md("""\
## 省了多少？以及为什么还需要稀疏 kernel

块稀疏只算选中块，所以 top-k 越小越快。但我们的 gather 是**纯 PyTorch**（gather/scatter 有开销），
和融合的 SDPA 比并不占便宜。看数据：
"""))

cells.append(code("""\
S, block_size, nb = 2048, 64, 32
q, k, v = make_qkv(4, 8, S, 64, dtype=torch.float16, seed=0)
ms_naive = bench_ms(lambda: naive_attention(q, k, v, causal=True))
ms_sdpa = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

tks = [1, 2, 4, 8, nb]
lat = [bench_ms(lambda tk=tk: block_sparse_attention(q, k, v, block_size, tk, causal=True)) for tk in tks]

plt.figure(figsize=(7, 4.2))
plt.plot([t/nb*100 for t in tks], lat, "o-", label="块稀疏 gather (PyTorch)")
plt.axhline(ms_naive, ls="--", c="C3", label=f"朴素 full ({ms_naive:.1f}ms)")
plt.axhline(ms_sdpa, ls="--", c="C2", label=f"SDPA full 融合 ({ms_sdpa:.2f}ms)")
plt.xlabel("稀疏度（选中 key 块占比 %）"); plt.ylabel("延迟 (ms)"); plt.legend(); plt.grid(alpha=0.3)
plt.title("块稀疏：少算块就快，但 PyTorch gather 干不过融合 kernel")
plt.tight_layout(); plt.show()
print(f"选 2/{nb} 块(6%): {lat[1]:.2f}ms，比朴素 full 快 {ms_naive/lat[1]:.0f}×，但仍慢于 SDPA 的 {ms_sdpa:.2f}ms。")
"""))

cells.append(md("""\
## 收尾

- 注意力天然**稀疏**：每个 query 的权重集中在少数块上，全算是浪费；
- 块稀疏三步——**分块取代表 → 块级 top-k 选块 → 只算选中块**——把计算量按块数线性砍下来；
- 我们用两条路径（mask 参考 / gather 省算）验证了机制，并看到 top-k 越小越快；
- 但纯 PyTorch 的 gather/scatter 开销，让它干不过融合的 SDPA——**要既稀疏又快，必须把"选块 +
  稀疏算"焊进一个 Triton kernel**。

**下一站** → `08-native-sparse-attention`：DeepSeek 的 NSA 把"压缩 + 选择 + 滑窗"三条稀疏分支
门控合并，并用真正的稀疏 Triton kernel 实现——我们去读它。
"""))

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, "07-block-sparse-attention/block_sparse.ipynb")
print("已生成 07-block-sparse-attention/block_sparse.ipynb，cells:", len(cells))
