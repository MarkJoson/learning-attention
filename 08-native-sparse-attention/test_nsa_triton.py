"""08-native-sparse-attention 深度优化版（拷贝的 lucidrains NSA triton kernel）正确性测试。

两层校验：
  1. 忠实性 test_faithful_vs_library：捕获库 forward 实际传给 native_sparse_attend 的真实
     输入，用本地拷贝 nsa_triton 重算，结果与库原版 bitwise 相等 —— 证明"完整拷贝 + 仅去
     tensor_typing 依赖"没有改动任何计算逻辑。
  2. 端到端 test_end_to_end_fwd_bwd：把库的 triton 入口替换为本地拷贝，跑完整 NSA（块压缩 /
     top-k 选块 / 三分支门控），与库的纯 PyTorch 路径对齐 forward + backward —— 证明在真实
     NSA pipeline 里我们的 kernel 正确。

依赖 lucidrains native-sparse-attention-pytorch（pip 安装）；未安装则整文件 skip。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("native_sparse_attention_pytorch")
import native_sparse_attention_pytorch.triton_native_sparse_attention as lib_t  # noqa: E402
from native_sparse_attention_pytorch import SparseAttention  # noqa: E402

import nsa_triton as ours  # noqa: E402  本地完整拷贝（去依赖后）

# monkeypatch 之前先备份库原版入口，供忠实性对比
_LIB_NATIVE_SPARSE_ATTEND = lib_t.native_sparse_attend


def build(use_triton: bool, seed: int = 0) -> SparseAttention:
    torch.manual_seed(seed)
    model = SparseAttention(
        dim=256, dim_head=64, heads=4,
        sliding_window_size=64,
        compress_block_size=32, compress_block_sliding_stride=32,
        selection_block_size=32, num_selected_blocks=4,
        causal=True, use_triton_kernel=use_triton,
    ).cuda()
    return model.eval()


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(2, 512, 256, device="cuda")


def test_faithful_vs_library(monkeypatch, x):
    """拷贝忠实：本地拷贝重算 == 库原版（同输入，atol=0）。"""
    captured = {}

    def spy(*args, **kwargs):
        out = _LIB_NATIVE_SPARSE_ATTEND(*args, **kwargs)
        captured["args"], captured["kwargs"], captured["lib_out"] = args, kwargs, out
        return out

    monkeypatch.setattr(lib_t, "native_sparse_attend", spy)
    build(use_triton=True)(x)  # 触发一次 forward，捕获库实际传入的真实输入

    assert "lib_out" in captured, "forward 未调用 native_sparse_attend（输入构造需触发选块分支）"
    ours_out = ours.native_sparse_attend(*captured["args"], **captured["kwargs"])
    torch.testing.assert_close(ours_out, captured["lib_out"], rtol=0, atol=0)


def test_end_to_end_fwd_bwd(monkeypatch, x):
    """端到端：库 triton 入口替换为本地拷贝，完整 NSA 对齐库 PyTorch 路径（fwd+bwd）。"""
    monkeypatch.setattr(lib_t, "native_sparse_attend", ours.native_sparse_attend)

    m_tri = build(use_triton=True, seed=0)
    m_ref = build(use_triton=False, seed=0)
    m_ref.load_state_dict(m_tri.state_dict())  # 两路径权重严格一致

    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    o_tri = m_tri(x1)
    o_ref = m_ref(x2)
    torch.testing.assert_close(o_tri, o_ref, rtol=2e-2, atol=2e-2)

    o_tri.sum().backward()
    o_ref.sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=2e-2, atol=2e-2)
