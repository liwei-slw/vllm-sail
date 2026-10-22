# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric oracle for the patched Kimi-K3 AttnRes Triton fallback.

Adapted from ``vllm/tests/models/kimi_k3/test_attn_res.py``. The reference
computation, the tolerances and the exact prefix/blocks write-back assertions
are upstream's; only the plumbing and the shape table differ, because this copy
has to exercise the implementation PPU actually ships:

* ``vllm_sail.register_out_of_tree()`` runs before ``attn_res`` is bound, so the
  imported callable is the patched wrapper from
  ``vllm_sail/patch/performance/kimi_k3_attn_res.py``. Upstream imports
  ``from ...nvidia.ops import attn_res``, which resolves the *package*
  re-export; the patch replaces the attribute on the ``...ops.attn_res``
  module, so the submodule is resolved through the import system here.
* The ROCm<10 and SM100 native cases are dropped: this suite runs on PPU only,
  where ``kimi_k3_attn_res`` is excluded from the build and nothing can take
  the native path.
* The shape table adds longer sequences so that every branch of
  ``attn_res_triton_config`` is covered. ``test_patch_is_installed`` fails
  loudly rather than silently re-testing upstream.
"""

import importlib

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

attn_res_module = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.attn_res")
attn_res = attn_res_module.attn_res

pytestmark = pytest.mark.ppu

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5


def _randn_with_row_padding(*shape: int, padding: int = 0) -> torch.Tensor:
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return storage[..., : shape[-1]]


def _reference(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if delta is not None:
        prefix = prefix + delta
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (HIDDEN_SIZE,), norm_weight, EPS)
    probs = (keys @ qk_weight).softmax(dim=-1)
    output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
    if output_norm_weight is not None:
        output = F.rms_norm(output, (HIDDEN_SIZE,), output_norm_weight, EPS)
    return output, prefix


def test_patch_is_installed() -> None:
    """Assert the patched objects are what this module imported.

    Without this guard a rename or a missed target would leave the whole file
    validating upstream's kernel while still passing.
    """
    from vllm.models.kimi_k3.nvidia import model as kimi_k3_model

    from vllm_sail.patch.utils import PATCH_MARKER

    assert getattr(attn_res, PATCH_MARKER, {}), (
        "vllm.models.kimi_k3.nvidia.ops.attn_res.attn_res carries no patch marker"
    )
    assert getattr(kimi_k3_model.attn_res, PATCH_MARKER, {}), (
        "nvidia/model.py still binds the upstream attn_res by value"
    )
    assert attn_res_module.attn_res_triton_config(1, 0) == (1, 16, 3)


@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "row_padding", "write_block", "has_delta"),
    [
        pytest.param(1, 0, 0, True, False, id="empty"),
        pytest.param(1, 0, 0, True, True, id="empty-add"),
        pytest.param(17, 5, 7, True, False, id="write"),
        pytest.param(17, 5, 7, True, True, id="write-add"),
        pytest.param(3, 8, 0, False, False, id="full"),
        pytest.param(3, 8, 0, False, True, id="full-add"),
        # Branches of attn_res_triton_config the upstream table never reaches
        # on a device without the native op.
        pytest.param(100, 4, 0, False, True, id="decode-100"),
        pytest.param(320, 1, 0, False, True, id="decode-320"),
        pytest.param(320, 8, 0, False, True, id="prefill-320"),
        pytest.param(4096, 1, 0, False, True, id="long-4096"),
    ],
)
def test_attn_res(
    num_tokens: int,
    num_blocks: int,
    row_padding: int,
    write_block: bool,
    has_delta: bool,
):
    prefix = _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=row_padding)
    delta = (
        _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=row_padding)
        if has_delta
        else None
    )
    blocks = _randn_with_row_padding(
        num_tokens, MAX_BLOCKS, HIDDEN_SIZE, padding=row_padding
    )
    norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    original_blocks = blocks.clone()
    expected, expected_prefix = _reference(
        prefix.clone(),
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )
    block_write_idx = num_blocks if write_block else -1

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        EPS,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    if write_block:
        original_blocks[:, block_write_idx].copy_(expected_prefix)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("num_blocks", range(MAX_BLOCKS + 1))
def test_attn_res_block_counts(num_blocks: int):
    prefix = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn(
        1, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = torch.ones_like(norm_weight)
    expected, _ = _reference(
        prefix.clone(),
        None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )

    actual = attn_res(
        prefix,
        None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        -1,
        EPS,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


def test_attn_res_without_output_norm():
    prefix = torch.randn(7, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(prefix)
    blocks = torch.randn(
        7, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    expected, _ = _reference(
        prefix.clone(), delta, blocks, norm_weight, qk_weight, None, MAX_BLOCKS
    )

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        None,
        MAX_BLOCKS,
        -1,
        EPS,
        0.0,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
