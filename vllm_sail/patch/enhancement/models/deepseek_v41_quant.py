# SPDX-License-Identifier: Apache-2.0
"""Preserve PPU channelwise scale names in V4.1 target and draft loaders."""

import sys

from vllm.platforms import current_platform

from vllm_sail.patch.utils import PATCH_MARKER, patch

_MODULE = "vllm.models.deepseek_v41.nvidia.model"


@patch(
    _MODULE,
    "_linear_scale_param_name",
    reason="PPU channelwise FP8 dense layers register weight_scale for V4.1 target and draft loaders.",
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="The V4.1 scale-name helper queries the linear quantization method's parameter contract.",
)
def _linear_scale_param_name(vllm_config, expert_dtype):
    if current_platform.is_ppu() and getattr(
        vllm_config.quant_config, "fp8_channelwise_layers", None
    ):
        return "weight_scale"
    original = getattr(_linear_scale_param_name, PATCH_MARKER)[
        f"{_MODULE}._linear_scale_param_name"
    ]
    return original(vllm_config, expert_dtype)


# The VL wrapper and DSpark loader import the helper by value.
def _rebind_scale_aliases():
    original = getattr(_linear_scale_param_name, PATCH_MARKER)[
        f"{_MODULE}._linear_scale_param_name"
    ]
    for name in (
        "vllm.models.deepseek_v41.nvidia.vl_model",
        "vllm.models.deepseek_v41.nvidia.dspark",
    ):
        module = sys.modules.get(name)
        if (
            module is not None
            and getattr(module, "_linear_scale_param_name", None) is original
        ):
            patch(
                name,
                "_linear_scale_param_name",
                reason="Preloaded V4.1 target and draft consumers must retain PPU channelwise scale names.",
                affected_versions=">=0.31.0,<0.32.0",
                remove_when="V4.1 loaders resolve the scale-name helper through its provider module.",
            )(_linear_scale_param_name)


_rebind_scale_aliases()
