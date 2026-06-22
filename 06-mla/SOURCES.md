# 06-mla · 外部来源登记

本章既有自写 PyTorch 参考，也提取了一份真实的 MLA triton kernel（汇总见仓库根 [`NOTICE`](../NOTICE)）。

## `mla_triton.py`（提取的 MLA prefill kernel）

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/ModelTC/lightllm |
| 文件 | `lightllm/models/deepseek2/triton_kernel/context_flashattention_nopad.py` |
| commit | `9ae6a5d4886312bd827295f3cb0de231639f0c77` |
| License | Apache-2.0 |
| 取用 | `_fwd_kernel_no_prompt_cache` + `context_attention_fwd_no_prompt_cache`（MLA prefill，无 prefix cache 版） |

### 为什么是它

MLA 的注意力是**非对称**的（absorb 后 score 在 `kv_lora+rope` 维度、value 在 `kv_lora` 维度），
标准 FlashAttention/SDPA 用不了，所以需要专门 kernel。DeepSeek 的 FlashMLA 是 CUDA，vLLM 的 MLA
backend 深度耦合框架——只有 lightllm 这份 **deepseek2 的 MLA triton kernel** 既是纯 triton、
依赖又少（去掉 1 个 `is_tesla` 即自包含），而且接口正好是 **absorb 格式**：

```python
context_attention_fwd_no_prompt_cache(q_nope, q_rope, kv_nope, kv_rope, o, ...)
#   score = q_nope·kv_nope + q_rope·kv_rope,   out = attn·kv_nope
```

正好对接 `mla.py` 里 absorb 算出的中间量（q 投到 latent、c_kv 当 content-key 兼 value）。

### 本仓库改动（**不改 kernel 计算逻辑**）

1. 去除 `from lightllm.utils.device_utils import is_tesla`，BLOCK 选择改用 `torch.cuda.get_device_capability`；
2. 针对 RTX 4090 共享内存收紧大 head 维度时的 BLOCK（`kv_lora_rank>=256` 时调小，避免 OutOfResources）。

## 算法来源

DeepSeek-V2 / V3 技术报告（Multi-head Latent Attention）。配置量级参考 DeepSeek-V2。

## 本仓库自写

- `mla.py`：`MLA` 模块。三条等价路径——`forward_naive`（重建 K/V，ground truth）、
  `forward_absorb`（PyTorch 在 latent 上算）、`forward_absorb_triton`（latent 注意力用上面的
  triton kernel）。含 RMSNorm、decoupled RoPE、weight absorption。三者数值一致（见 `test_mla.py`）。
