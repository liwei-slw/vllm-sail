# SPDX-License-Identifier: Apache-2.0
"""Keep Qwen3.5 INT8 graph replay disabled by default on Zhenwu M890."""

from __future__ import annotations

import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


class GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


@pytest.fixture
def platform(monkeypatch):
    class NvmlCudaPlatform:
        capability = (8, 9)

        @classmethod
        def is_device_capability(cls, value):
            return cls.capability == value

    modules = {
        "torch": NS(backends=NS(cuda=NS(enable_cudnn_sdp=lambda value: None))),
        "vllm.logger": NS(init_logger=lambda name: Mock()),
        "vllm.platforms.cuda": NS(NvmlCudaPlatform=NvmlCudaPlatform),
        "vllm.platforms.interface": NS(DeviceCapability=tuple, PlatformEnum=NS(CUDA=0)),
        "vllm.v1.attention.backends.registry": NS(AttentionBackendEnum=object),
        "vllm.config.compilation": NS(CUDAGraphMode=GraphMode),
        "vllm_sail.ops": types.ModuleType("vllm_sail.ops"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / "vllm_sail/platform.py"
    spec = importlib.util.spec_from_file_location("_ppu_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PPUPlatform


def config(mode=GraphMode.PIECEWISE):
    quant = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "W8A8": {
                "weights": {"num_bits": 8, "type": "int"},
                "input_activations": {"num_bits": 8, "type": "int"},
            }
        },
    }
    return NS(
        model_config=NS(
            hf_config=NS(quantization_config=quant),
            hf_text_config=NS(model_type="qwen3_5_moe_text"),
            enforce_eager=False,
        ),
        compilation_config=NS(cudagraph_mode=mode, custom_ops=[]),
        kernel_config=NS(moe_backend="auto"),
        attention_config=NS(flash_attn_max_num_splits_for_cuda_graph=None),
    )


@pytest.mark.parametrize(
    "mode", [None, GraphMode.PIECEWISE, GraphMode.FULL_AND_PIECEWISE]
)
def test_qwen_w8a8_disables_graphs_for_prefill_and_batched_decode(platform, mode):
    cfg = config(mode)
    platform.apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_mode is GraphMode.NONE
    platform.apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_mode is GraphMode.NONE
    assert cfg.compilation_config.custom_ops.count("+quant_fp8") == 1
    assert cfg.compilation_config.custom_ops.count("+sparse_attn_indexer") == 1


@pytest.mark.parametrize(
    "mode", [GraphMode.NONE, GraphMode.FULL_DECODE_ONLY, GraphMode.FULL]
)
def test_other_graph_modes_are_preserved(platform, mode):
    cfg = config(mode)
    platform.apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_mode is mode


@pytest.mark.parametrize(
    "case", ["other_model", "bf16", "fp8", "w8a16", "810e", "eager", "no_model"]
)
def test_other_model_and_execution_configs_are_preserved(platform, case):
    cfg = config()
    if case == "other_model":
        cfg.model_config.hf_text_config.model_type = "deepseek_v4"
    elif case == "bf16":
        cfg.model_config.hf_config.quantization_config = None
    elif case == "fp8":
        group = cfg.model_config.hf_config.quantization_config["config_groups"]["W8A8"]
        group["weights"]["type"] = "float"
        group["input_activations"]["type"] = "float"
    elif case == "w8a16":
        cfg.model_config.hf_config.quantization_config["config_groups"]["W8A8"][
            "input_activations"
        ] = None
    elif case == "810e":
        platform.capability = (8, 0)
    elif case == "eager":
        cfg.model_config.enforce_eager = True
    elif case == "no_model":
        cfg.model_config = None
    platform.apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_mode is GraphMode.PIECEWISE
