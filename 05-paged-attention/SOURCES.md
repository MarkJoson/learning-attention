# 05-paged-attention · 外部来源登记

本目录的 Triton kernel 提取自外部权威实现（汇总见仓库根 [`NOTICE`](../NOTICE)）。

## `paged_decode_triton.py`

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/ModelTC/lightllm |
| 文件 | `lightllm/common/basemodel/triton_kernel/att/decode_att/gqa/gqa_decode_flashattention_nopad.py` |
| commit | `9ae6a5d4886312bd827295f3cb0de231639f0c77` |
| License | Apache-2.0 |

### 为什么选它

paged attention 的高效 decode kernel，vLLM 版是 791 行的 flash-decoding（split-K + 多段
reduce），深度优化、可读性差，对"理解 paged 机制"是噪音。lightllm 的这个版本只有 **163 行、
依赖干净（torch/triton/math）、原生 GQA**，而且把 paged 的灵魂——**间接寻址**——表达得极其清楚：

```python
kv_loc = Req_to_tokens[req_idx, logical_pos]   # 逻辑位置 → 物理 slot
k      = K[kv_loc]                              # 用物理 slot 去读 KV
```

`Req_to_tokens` 就是 block table 的 "page size = 1"（token 级）形式。

### 本仓库改动

**无**——kernel 依赖本就只有 `torch/triton/math`，无框架内部耦合，原样拷贝即可（仅在文件头
加了来源标注）。

### 自写部分

- `paged.py` 的 `PagedKVCache`：用 block 级 block table + free list 演示分页内存管理
  （按需分配、交错分配），并把 block table 展开成 kernel 需要的 `Req_to_tokens`。

### 场景

decode（每序列 1 个 query token），GQA。prefill 用 03 的 kernel。
