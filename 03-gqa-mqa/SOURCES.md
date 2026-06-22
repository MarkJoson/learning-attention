# 03-gqa-mqa · 外部来源登记

本目录的 Triton kernel 提取自外部权威实现（汇总见仓库根 [`NOTICE`](../NOTICE)）。

## `gqa_triton.py`

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/vllm-project/vllm |
| 文件 | `vllm/v1/attention/ops/triton_prefill_attention.py` |
| commit | `435f82d61a1eddb84854ca59a008a8e4d97ab439` |
| License | Apache-2.0 |
| 来源链 | **LightLLM → SGLang → vLLM**（vLLM 改编自 SGLang，SGLang 改编自 LightLLM，原始 license 头在文件内保留） |

### 为什么选它

GQA 没有"独立的新算法"——它和 MHA 共用注意力 kernel，只在读 K/V 时多一步 head 映射。
三大候选里，triton 官方教程和 Dao-AILab 的 triton 实现都假设 `Hq == Hkv`（GQA 调用直接报错），
只有生产框架（vLLM/SGLang）原生支持。vLLM 的 `unified_attention` 深度耦合 paged KV cache（1100+ 行、
多个内部依赖），而这个 **prefill attention（253 行）依赖极少、原生 GQA、还自带 sliding window**，
是最干净的可提取对象。

### 本仓库改动（**不涉及 kernel 计算逻辑**）

只去除 3 个 vLLM 内部依赖，`_fwd_kernel` 与 `context_attention_fwd` 一字未改：

1. `from vllm.platforms import current_platform` → 去除；`get_block_size` 改用 `torch.cuda.get_device_capability` 判断算力；
2. `from vllm.triton_utils import tl, triton` → `import triton` / `import triton.language as tl`；
3. `from vllm.utils.math_utils import RCP_LN2` → 内联常量 `1.4426950408889634`。

### 能力

- **原生 GQA / MQA / MHA**：`kv_group_num = Hq // Hkv`，kernel 内 `cur_kv_head = cur_head // kv_group_num`；
- **causal**、**sliding window**（04 章会复用同一 kernel）、**varlen**（不等长序列拼接，真实推理格式）；
- 仅 **forward**（prefill / 推理场景，不含 backward）。
