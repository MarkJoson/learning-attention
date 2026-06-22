# 04-sliding-window · 外部来源登记

本章**不引入新 kernel**，而是复用 03 提取自 vLLM 的 kernel——它本就自带滑动窗口能力。

## kernel 复用

| 项 | 值 |
|---|---|
| 复用文件 | [`../03-gqa-mqa/gqa_triton.py`](../03-gqa-mqa/gqa_triton.py)（提取自 vLLM prefill attention） |
| 点亮能力 | kernel 内 `SLIDING_WINDOW_Q/K` 参数与对应的窗口掩码 |
| 来源详情 | 见 [`../03-gqa-mqa/SOURCES.md`](../03-gqa-mqa/SOURCES.md)（vLLM, commit `435f82d`, Apache-2.0，来源链 LightLLM→SGLang→vLLM） |

## 本章自写

- `sliding.py`：薄封装，把"看最近 window_size 个 token"翻译成 kernel 的
  `SLIDING_WINDOW = window_size - 1`（处理 off-by-one；kernel 以 `SLIDING_WINDOW>0` 启用滑窗，
  故要求 `window_size >= 2`）。
- `common.naive_attention` 新增 `window` 参数（朴素带窗参考）。

## 实现性质（重要）

kernel 用**掩码**实现滑窗：窗口外的 key 块仍被读入计算再屏蔽，循环范围未缩小。因此：

- ✅ 数值正确（与朴素带窗参考一致）；
- ✅ decode 时 KV cache 可只保留最近 W 个 token（滑窗的真正价值）；
- ❌ **不加速 prefill**（窗口外的块没被跳过）。要在 prefill 也省算力，需 block-sparse 地
  跳过窗口外整块——属于后续稀疏注意力章节。
