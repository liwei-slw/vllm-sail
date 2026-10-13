# SPDX-License-Identifier: Apache-2.0
"""为 ACEXT 补齐 SAIL SDK 的 cuBLAS 动态链接依赖。"""

from __future__ import annotations

import ctypes
from functools import cache
from pathlib import Path
from types import ModuleType

from vllm_sail import envs


@cache
def _load_sdk_cublas(sdk: str) -> ctypes.CDLL:
    # 仅加载指定 SDK 的兼容库，并保留句柄；不从系统 CUDA 路径搜索。
    library = Path(sdk) / "CUDA_SDK/lib64/libcublas.so"
    return ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def load_acext() -> ModuleType:
    """仅对已确认的缺失符号补齐依赖，其他导入和 ABI 错误原样抛出。"""
    try:
        import acext
    except OSError as exc:
        # Torch 会包装 dlopen 异常，根因保存在 __cause__ 中。
        cause = exc.__cause__ or exc
        if not isinstance(cause, OSError) or not str(cause).endswith(
            ": undefined symbol: cublasGemmEx"
        ):
            raise
        # hggcrt3 Torch 直接链接 libacblas；旧 ACEXT 未声明 cuBLAS 依赖。
        # ACEXT 修正链接关系后首次 import 即成功，不再进入此兼容分支。
        _load_sdk_cublas(envs.PPU_SDK)
        import acext
    return acext
