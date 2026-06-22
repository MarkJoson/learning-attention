# 02-flash-attention · 外部来源登记

本目录的 Triton kernel 拷贝自外部权威实现，登记如下（汇总见仓库根 [`NOTICE`](../NOTICE)）。

## `flash_triton.py`

| 项 | 值 |
|---|---|
| 来源仓库 | https://github.com/triton-lang/triton |
| 文件 | `python/tutorials/06-fused-attention.py` |
| tag / commit | `v3.1.0` / `cf34004b8a67d290a962da166f5aa2fc66751326` |
| License | MIT（Copyright 2018-2020 Philippe Tillet; 2020-2022 OpenAI） |
| 算法 | FlashAttention v2（Tri Dao），kernel 由 OpenAI kernel team 编写 |

### 为什么用 v3.1.0 而非最新版

triton 主分支最新的 `06-fused-attention.py` 已演进到带 TMA `TensorDescriptor`、
warp-specialization、FP8、Blackwell 特化的版本（775 行）。实测它在 RTX 4090 (sm_89)
上因依赖 device-side tensor descriptor 而无法运行；且大量硬件特化代码对**学习** FlashAttention
核心算法是噪音。v3.1.0 是纯指针、无 TMA 的经典 FlashAttention v2 实现（640 行），
专注 online-softmax + 分块，且在消费级 GPU 上久经验证，更适合精读。

### 本仓库改动（均**不涉及 kernel 计算逻辑**）

1. 删除原文件自带的 `test_op` / `bench_flash_attention`（改用本目录 `test_flash.py` / `bench.py`）；
2. autotune 的 `num_stages` 候选 `[3,4,7]` → `[2,3,4]`（4090 共享内存适配）；
3. forward autotune 新增按 `head_dim` 估算共享内存的 `_prune_4090_smem`（实测校准）；
4. backward 的 `NUM_STAGES` 在 `head_dim>=128` 时由 5 降为 2；
5. 文件头与关键函数补充中文学习注释。

### 继承自原 kernel 的能力边界

- **backward 仅支持 `causal=True`**：原教程 benchmark 亦只测 causal 反向；用其自带
  reference 已复现确认 non-causal 反向数值不正确。
- 仅标准 **MHA**：要求 `Hq == Hkv`（不支持 GQA，见 03 章）、`Sq == Sk`；
- `head_dim ∈ {16,32,64,128,256}`；backward 要求 `seqlen` 为 128 的倍数。
