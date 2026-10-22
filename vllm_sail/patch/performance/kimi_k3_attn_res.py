# SPDX-License-Identifier: Apache-2.0
"""PPU patch: tuned Kimi-K3 AttnRes Triton fallback.

``vllm/models/kimi_k3/nvidia/ops/attn_res.py`` ships two implementations: a
native CUDA kernel that is SM100-only, and a Triton kernel used everywhere else.
``csrc/upstream/excluded_ops.toml`` excludes ``kimi_k3_attn_res`` from the PPU
build, so on PPU the Triton kernel *is* the production path and its launch
configuration and memory traffic decide what the operator costs.

Three functionally equivalent changes:

1. ``_attn_res_kernel`` seeds the online softmax with the bf16-rounded prefix
   instead of re-reading it as the last source of the loop. That value is
   already live in the register tile, so the reload, the per-tile
   ``tl.where`` pointer select, the ``tl.debug_barrier`` that keeps it alive and
   one whole loop iteration disappear. With only the prefix source the softmax
   is exactly one, so the seed is also the answer.
2. The output RMSNorm's softmax denominator cancels algebraically against the
   normalisation, so the fused epilogue folds it into the epsilon term
   (``eps * denom**2``) instead of dividing the mixed vector first.
3. ``attn_res`` takes ``(BLOCK_L, num_warps, num_stages)`` from
   ``attn_res_triton_config``, a table measured on GB300 over 42
   ``(num_tokens, num_blocks)`` shapes, instead of one hardcoded config with
   ``num_stages=2``. Decode is latency bound on the per-row reduction chain and
   wants wide CTAs with a deeper prefetch pipeline; prefill is DRAM bound and
   only needs enough CTAs resident, so narrow CTAs win. Source tiling only pays
   off once there are several sources to fill the tile.

``attn_res_triton_config`` is installed as a new module attribute so that the
benchmark in ``benchmarks/kernels/`` and any future caller share one table.

Measured end to end on GB300: 1.12x geometric mean over the pre-tuning
heuristic, 1.38x on the worst decode batch, 1.16-1.69x over the original
upstream configuration.
"""

from __future__ import annotations

from importlib import import_module

import torch
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.31.0,<0.32.0"

# ``ops/__init__.py`` re-exports the function under the module's own name, so the
# submodule cannot be reached with ``from ...ops.attn_res import ...``.
attn_res_ops = import_module("vllm.models.kimi_k3.nvidia.ops.attn_res")

# Captured before patching so the wrapper can delegate off PPU.
_upstream_attn_res = attn_res_ops.attn_res


@patch(
    "vllm.models.kimi_k3.nvidia.ops.attn_res",
    "attn_res_triton_config",
    allow_missing=True,
    reason=(
        "PPU always runs the AttnRes Triton fallback because the native kernel "
        "is SM100-only and excluded from the PPU build, and upstream hardcodes "
        "one launch configuration with num_stages=2. Install the launch table "
        "measured on GB300 so the wrapper and the benchmark share it."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream picks the AttnRes Triton launch configuration from a measured "
        "(num_tokens, num_blocks) table instead of its fixed prefill/decode "
        "pair."
    ),
)
def attn_res_triton_config(num_tokens: int, num_blocks: int) -> tuple[int, int, int]:
    """Pick ``(BLOCK_L, num_warps, num_stages)`` for the Triton fallback."""
    if num_blocks <= 1:
        if num_tokens < 128:
            return 1, 16, 3
        if 256 <= num_tokens < 4096:
            return 1, 8, 2
        return 1, 4, 2
    if num_tokens < 64:
        return 4, 16, 2
    if num_tokens < 128:
        return 4, 8, 2
    return 1, 8, 2


@patch(
    "vllm.models.kimi_k3.nvidia.ops.attn_res",
    "_attn_res_kernel",
    reason=(
        "Triton is the AttnRes path on PPU. Verbatim upstream kernel with the "
        "prefix seeded into the online softmax instead of being re-read as the "
        "last source, and the output RMSNorm's softmax denominator folded into "
        "its epsilon term. Both remove memory traffic and a reduction pass "
        "without changing the result."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream seeds the AttnRes online softmax with the updated prefix and "
        "folds the output RMSNorm denominator itself."
    ),
)
@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    updated_prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + row_idx * stride_delta_m + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix += delta
        # Match the BF16 prefix-add result before using it as a residual source.
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    if WRITE_BLOCK:
        tl.store(
            blocks_ptr
            + row_idx * stride_block_m
            + block_write_idx * stride_block_r
            + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    # PPU MODIFICATION: begin — the prefix is one of the AttnRes sources and its
    # bf16-rounded value is already live in `updated_prefix`, so it seeds the
    # online softmax directly instead of being re-read as the last source of the
    # loop. That removes the reload, the `tl.where` pointer select and the
    # `tl.debug_barrier` that keeps the prefix tile alive across the loop. With
    # only the prefix source the AttnRes softmax is exactly one, i.e. this seed
    # is the result.
    mixed = updated_prefix
    denominator = tl.full((), 1.0, tl.float32)
    if num_blocks > 0:
        input_qk_weight = tl.load(
            norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32) * tl.load(
            qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        max_logit = tl.sum(mixed * input_qk_weight, axis=0) * tl.rsqrt(
            tl.sum(mixed * mixed, axis=0) * (1.0 / hidden_size) + eps
        )

        # The loop only covers the block sources now, so it never straddles the
        # prefix and needs no mask-based source selection.
        for source_start in range(0, num_blocks, BLOCK_L):
            source_offsets = source_start + tl.arange(0, BLOCK_L)
            source_mask = source_offsets < num_blocks
            values = tl.load(
                blocks_ptr
                + row_idx * stride_block_m
                + source_offsets[:, None] * stride_block_r
                + d_offsets[None, :],
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            # PPU MODIFICATION: end
            reciprocal_std = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
            scores = tl.where(source_mask, logits, -float("inf"))

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp(max_logit - new_max_logit)
            block_scales = tl.exp(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
            mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
            max_logit = new_max_logit

    # PPU MODIFICATION: begin — upstream divides here (`mixed /= denominator;
    # output = mixed`) because the un-normalised denominator is about to be
    # discarded. Keeping `mixed` un-divided lets the fused output RMSNorm fold
    # the denominator into its epsilon term instead:
    # (mixed / denom) * rsqrt(mean((mixed / denom)^2) + eps)
    #   = mixed * rsqrt(mean(mixed^2) + eps * denom^2).
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if APPLY_OUTPUT_NORM:
        output_reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps * denominator * denominator
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        output = mixed * output_reciprocal_std * output_norm_weight
    elif num_blocks == 0:
        output = mixed
    else:
        output = mixed / denominator
    # PPU MODIFICATION: end

    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


# Two targets share one replacement: the defining module, and ``nvidia/model.py``
# which binds the function by value at import time (`from ...ops import attn_res`)
# and would otherwise keep launching the upstream wrapper.
#
# The model target is the inner decorator so it installs *first*. If a not yet
# imported ``model.py`` were imported after the defining module was patched, its
# binding would already be the marked replacement and the double-patch guard
# would reject this patch.
@patch(
    "vllm.models.kimi_k3.nvidia.ops.attn_res",
    "attn_res",
    reason=(
        "PPU runs the AttnRes Triton fallback, whose launch configuration "
        "upstream hardcodes as one prefill/decode pair with num_stages=2. Take "
        "it from attn_res_triton_config instead; the native SM100 branch is "
        "kept and non-PPU platforms delegate to upstream unchanged."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "attn_res itself selects the tuned AttnRes Triton launch configuration, "
        "or the launch table stops beating upstream's fixed pair on PPU."
    ),
)
@patch(
    "vllm.models.kimi_k3.nvidia.model",
    "attn_res",
    reason=(
        "nvidia/model.py imports attn_res by value, so replacing the defining "
        "module alone would leave the decoder layers calling the upstream "
        "wrapper. Point the module global at the replacement too; without this "
        "the tuned kernel is silently bypassed whenever model.py was imported "
        "before the patches were installed."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "nvidia/model.py resolves attn_res through the module attribute at call "
        "time, or upstream ships the tuned launch configuration."
    ),
)
def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    # PPU MODIFICATION: begin — off PPU, upstream (including its SM100 native
    # kernel) is unchanged, so delegate instead of copying it.
    if not current_platform.is_ppu():
        return _upstream_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            block_write_idx,
            eps,
            output_norm_eps,
        )
    # PPU MODIFICATION: end
    num_tokens, hidden_size = prefix.shape
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    # The native kernel covers every Kimi-K3 AttnRes variant on dense SM100
    # inputs. The op is only compiled under CUDA >= 13, so a device check alone
    # is not enough to know it exists.
    if (
        hidden_size == 7168
        and prefix.stride(0) == hidden_size
        and (delta is None or delta.stride(0) == hidden_size)
        and 0 <= num_blocks <= 8
        and num_blocks <= blocks.shape[1]
        and -1 <= block_write_idx < blocks.shape[1]
        and current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "kimi_k3_attn_res")
    ):
        return ops.kimi_k3_attn_res(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            num_blocks,
            block_write_idx,
            eps,
            output_norm_eps,
        )
    output = prefix.new_empty(prefix.shape)
    # PPU MODIFICATION: begin — upstream hardcodes
    # `block_l, num_warps = (1, 4) | (4, 8)` and `num_stages=2`. Take all three
    # from the measured table; the Triton warmup calls this wrapper, so it warms
    # exactly the decode configuration the server will use.
    block_l, num_warps, num_stages = attn_res_triton_config(num_tokens, num_blocks)
    # PPU MODIFICATION: end
    _attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        num_blocks,
        hidden_size,
        block_write_idx,
        eps,
        output_norm_eps,
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        # PPU MODIFICATION: num_stages is part of the tuned table (upstream: 2).
        num_stages=num_stages,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return output
