# learning-attention

> 一个**收集 + 精读**现代高效 Attention 实现的学习仓库。
> 从 GitHub 上各类高性能 Attention 开源实现入手，逐个拆解变体的**算法原理**与**kernel 实现细节**，
> 每个变体配套：参考实现 + 优化算子 + 测试用例 + 基准 + 可运行的 Jupyter notebook。

本仓库的定位不是"再造一个 attention 库"，而是把散落在各处的优秀实现**集中、对照、讲透**，
让你既能看懂数学，也能看懂工程上"为什么这么写 kernel"。

---

## 核心原则

1. **Triton / CUDA kernel 一律从外部权威开源实现拷贝**，不自己手写。
   每份拷贝来的 kernel 都会在文件头标注**来源仓库、commit、原始 license**，
   并在该变体目录的 `SOURCES.md` 中登记。汇总见根目录 [`NOTICE`](./NOTICE)。
   > 这样做的理由：kernel 是经过社区大规模验证的"标准答案"，拿权威实现来精读，
   > 比自己手写更可靠，也更贴合"学习真实生产代码"的目标。

2. **参考实现、测试、基准、文档、notebook 由本仓库自写。**
   每个变体都有一份刻意写得直白的纯 PyTorch 参考实现（`common/reference.py` 或本地 `reference.py`），
   作为数值 **ground truth**；优化实现（拷贝来的 kernel）必须在数值上与之对齐才算正确。

3. **先用 `.py` 跑通功能与性能，再沉淀成 notebook。**
   每个变体先有 `test_*.py`（pytest 验证数值正确）和 `bench.py`（基准），
   全部验证通过后，再写 `*.ipynb` 做分步讲解与可视化呈现。

---

## 学习路线图

主线（参考 [`lecture.md`](./lecture.md)）：

```
基础线   vanilla → online softmax → FlashAttention v1/v2 → GQA/MQA → sliding window → paged
稀疏线   Native Sparse Attention (NSA) → Dynamic Sparse (DSA) → DashAttention
线性线   Linear Attention → GLA → DeltaNet → Gated DeltaNet → KDA
推理线   KV cache → MLA → Paged Attention → FlashInfer 风格接口
```

### 变体清单与状态

| # | 变体 | 主线 | 优化实现来源 | 状态 |
|---|---|---|---|---|
| 01 | [vanilla-attention](./01-vanilla-attention/) | 基础 | 自写参考（naive + online softmax） | ✅ 完成 |
| 02 | [flash-attention](./02-flash-attention/) | 基础 | triton-lang/triton（FlashAttention v2 kernel） | ✅ 完成 |
| 03 | [gqa-mqa](./03-gqa-mqa/) | 基础 | vLLM（prefill attention，原生 GQA/MQA） | ✅ 完成 |
| 04 | [sliding-window](./04-sliding-window/) | 基础 | vLLM（复用 03 kernel，自带滑窗） | ✅ 完成 |
| 05 | [paged-attention](./05-paged-attention/) | 推理 | lightllm（paged decode，间接寻址） | ✅ 完成 |
| 06 | [mla](./06-mla/) | 推理 | lightllm MLA prefill + 自写参考（absorb） | ✅ 完成 |
| 07 | [block-sparse-attention](./07-block-sparse-attention/) | 稀疏 | 自写简要版 + 复用 08 NSA kernel + MoBA 对照 | ✅ 完成 |
| 08 | [native-sparse-attention](./08-native-sparse-attention/) | 稀疏 | 自写三分支 + lucidrains NSA triton kernel（完整提取） | ✅ 完成 |
| 09 | dynamic-sparse-attention | 稀疏 | epfml DSA | 📋 计划 |
| 10 | [linear-attention](./10-linear-attention/) | 线性 | 自写三形式 + 解耦 fla GLA chunk kernel | ✅ 完成 |
| 11 | [deltanet](./11-deltanet/) | 线性 | 自写 delta rule + 完整解耦 fla DeltaNet kernel | ✅ 完成 |
| 12 | [kda](./12-kda/) | 线性 | 自写 gated delta + 完整解耦 fla KDA kernel（Kimi Linear） | ✅ 完成 |

> 状态：✅ 完成 · 🚧 进行中 · 📋 计划。按"先做样板、验收后批量推进"的方式逐步补齐。

---

## 每个变体目录的统一结构

```
NN-variant-name/
├── README.md          # 文档：数学原理、算法推导、kernel 结构讲解、参考来源
├── reference.py       # 纯 PyTorch 参考实现（ground truth，可读优先）；通用部分复用 common/
├── <variant>.py       # 优化实现：拷贝来的 Triton kernel + 调用胶水（文件头标注来源）
├── SOURCES.md         # 该变体所有外部拷贝代码的来源、commit、license 登记
├── test_<variant>.py  # pytest：优化实现 vs 参考实现 / SDPA 的数值一致性（fwd + bwd）
├── bench.py           # 基准：延迟 / 显存 / TFLOP-s 对比
└── <variant>.ipynb    # 最终呈现：分步讲解 + 每步测试输入与示范输出 + 可视化
```

公共工具在 [`common/`](./common/)：

- `common/reference.py` — 标准注意力朴素参考实现 + GQA `repeat_kv`
- `common/testing.py`   — `assert_close`（带误差报告）、`make_qkv`（构造测试张量）
- `common/benchmark.py` — `bench_ms`、`attention_flops`、`tflops`、`peak_memory_mb`

---

## 如何使用

```bash
# 1) 安装依赖（已在 RTX 4090 + CUDA 12 + PyTorch 2.x 验证）
pip install -r requirements.txt

# 2) 跑某个变体的测试（数值正确性）
pytest 02-flash-attention/ -v

# 3) 跑基准（性能数字）
python 02-flash-attention/bench.py

# 4) 打开 notebook 逐步学习
jupyter lab 02-flash-attention/flash_attention.ipynb
```

跑全部测试：

```bash
pytest -v
```

---

## 致谢

本仓库精读并拷贝了多个优秀开源项目的 kernel 实现，全部来源与 license 登记在 [`NOTICE`](./NOTICE)
及各变体的 `SOURCES.md` 中。向这些项目的作者致谢 —— 没有它们就没有这份学习材料。

环境：Python 3.12 · PyTorch 2.13 (cu132) · Triton 3.7 · NVIDIA RTX 4090。
