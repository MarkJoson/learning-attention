# 10-linear-attention · 外部来源登记

本章有自写简要版（linear attention 三形式 + GLA reference）+ 深度优化版（**解耦自 fla** 的 GLA
chunk-parallel triton kernel）。汇总见仓库根 [`NOTICE`](../NOTICE)。

## 深度优化版：解耦自 fla-org/flash-linear-attention 的 GLA kernel

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/fla-org/flash-linear-attention |
| commit | `0b27f7b` |
| license | MIT (Copyright fla-org) |
| 取用文件 | `ops/gla/chunk.py`→`_fla_gla_chunk.py`；`ops/common/chunk_h.py`→`_fla_chunk_h.py`；`ops/utils/cumsum.py`→`_fla_cumsum.py`；`ops/gla/naive.py`→`gla_naive.py` |

### 为什么是它，以及怎么脱离 fla

fla 是 linear attention / GLA / DeltaNet / RWKV 等的权威库。但它的 GLA chunk kernel **深度耦合 fla
框架**：依赖 `chunk_h`（块间状态传递）、`cumsum`（门控累积）、autotune 磁盘缓存、GPU shared-mem 适配。
不像 08 NSA 那样自包含。

本仓库的处理（采用"**拷贝核心 kernel + 薄适配层脱离 fla**"）：**完整拷贝核心三件套 kernel
（计算逻辑一字未改），只把对 fla 框架的 import 改指向本地薄适配层 `_fla_compat.py`**，使其脱离 fla
包独立运行。关键发现是——kernel 真正用到的框架符号都是"工具/适配"类，可薄薄复现：

| fla 符号 | 薄适配做法（`_fla_compat.py`，本仓库自写） |
|---|---|
| `exp2` / `RCP_LN2` | libdevice `exp2` + 常量 `1/ln2` |
| `fla_cache_autotune` | 退化为标准 `triton.autotune`（签名一致，放弃磁盘缓存，不影响计算） |
| `check_shared_mem` | 用 `torch` 查 GPU shared memory 选 autotune block 候选；不精确也由 triton.autotune 跳过 OOM 兜底 |
| `input_guard` | contiguous 装饰器 |
| `prepare_chunk_indices/offsets` | 变长（cu_seqlens / sequence packing）的分块索引，**从 fla 拷贝纯 torch 实现，支持变长** |

`test_gla_triton.py::test_faithful_vs_fla` 验证"本地解耦 kernel == fla 原版"（`atol=1e-3`，近 bitwise），
证明解耦没改任何计算。

### 三件套分工

- `_fla_gla_chunk.py`：**chunk-parallel 主体** —— 块内并行 attention + 块间状态合并 + 门控；
- `_fla_chunk_h.py`：**块间递归状态** H 的前向/反向（`chunk_fwd_h` / `chunk_bwd_dh`，带门控衰减）；
- `_fla_cumsum.py`：门控的**块内累积** `chunk_local_cumsum`（log decay 的 prefix-sum）。

## 本仓库自写

- `linear.py`：linear attention 的 parallel / recurrent / chunked 三等价形式 + GLA recurrent（ground truth）。
- `gla_triton.py`：深度优化版入口（封装解耦 kernel，提供 `[B,H,T,D]` layout 便捷接口）。
- `_fla_compat.py`：薄适配层（复现 fla 框架辅助符号，使 kernel 脱离 fla）。
- `test_linear.py` / `test_gla_triton.py` / `bench.py` / `linear.ipynb`。

## 算法来源

- **Linear Attention**：Katharopoulos et al. 2020,《Transformers are RNNs》（kernel feature map + 结合律）。
- **GLA**：Yang et al. 2023,《Gated Linear Attention Transformers with Hardware-Efficient Training》。
