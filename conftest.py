"""pytest 根配置。

把仓库根加入 sys.path，让各变体目录下的 test_*.py 能直接 `from common import ...`，
并提供一个 `cuda` 自动跳过：没有 GPU 时整库测试自动 skip 而不是 error。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    """没有 CUDA 时，给所有用例打上 skip 标记（本仓库的 kernel 基本都需要 GPU）。"""
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="需要 CUDA GPU")
    for item in items:
        item.add_marker(skip_cuda)
