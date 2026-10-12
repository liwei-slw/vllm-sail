# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""PPU fused MTP parity against the unfused Triton recurrence and gated norm.

Adapted from vLLM v0.31.0 tests/kernels/fla/test_fused_gdn_post_conv.py.
Covers BF16/FP32 state, SiLU/sigmoid output gates, ragged/padded batches,
accepted-token rewind, TP head ratios and the eight-token boundary.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU", allow_module_level=True)
pytest.importorskip("vllm")
import vllm_sail

vllm_sail.register_out_of_tree()
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

from vllm_sail.attention.gdn_decode import has_fused_gdn_kernel

pytestmark = pytest.mark.ppu
if not current_platform.is_ppu():
    pytest.skip("requires the PPU platform", allow_module_level=True)


@pytest.fixture(autouse=True)
def native_kernel_and_independent_reference(monkeypatch):
    assert has_fused_gdn_kernel(), (
        "Build/install the native GDN MTP implementation before running parity"
    )
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")


@pytest.mark.parametrize(
    "head_ratio,tp_size,query_lengths,state_dtype,norm_dtype",
    [
        pytest.param(8, 16, (4, 4), torch.bfloat16, torch.bfloat16, id="tp16-bf16"),
        pytest.param(8, 4, (4, 4), torch.float32, torch.float32, id="tp4-fp32"),
        pytest.param(8, 16, (4, 2, 0), torch.bfloat16, torch.float32, id="tp16-ragged"),
        pytest.param(8, 4, (4, 2, 0), torch.float32, torch.bfloat16, id="tp4-ragged"),
        pytest.param(8, 16, (8,), torch.float32, torch.bfloat16, id="tp16-max"),
        pytest.param(8, 4, (8,), torch.bfloat16, torch.float32, id="tp4-max"),
        pytest.param(
            1,
            1,
            (4, 4),
            torch.float32,
            torch.bfloat16,
            id="ratio1-tp1-fp32",
        ),
        pytest.param(
            2,
            1,
            (4, 2, 0),
            torch.bfloat16,
            torch.bfloat16,
            id="ratio2-tp1-ragged-bf16",
        ),
        pytest.param(
            2,
            4,
            (8,),
            torch.float32,
            torch.float32,
            id="ratio2-tp4-max-fp32",
        ),
        pytest.param(
            3,
            2,
            (4, 2, 0),
            torch.float32,
            torch.float32,
            id="ratio3-tp2-ragged-fp32",
        ),
        pytest.param(
            4,
            4,
            (8,),
            torch.float32,
            torch.bfloat16,
            id="ratio4-tp4-max-fp32",
        ),
    ],
)
@pytest.mark.parametrize("output_gate_activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_head_ratios(
    head_ratio: int,
    tp_size: int,
    query_lengths: tuple[int, ...],
    state_dtype: torch.dtype,
    norm_dtype: torch.dtype,
    output_gate_activation: str,
) -> None:
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("fused GDN decode MTP requires compute capability 8.0+")
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        pytest.skip("fused GDN decode MTP op is not built")

    torch.manual_seed(0)
    device = "cuda"
    H = 16 // tp_size
    HV = head_ratio * H
    K = V = 128
    num_reqs = len(query_lengths)
    state_width = max(query_lengths)
    num_tokens = sum(query_lengths)
    num_slots = num_reqs * state_width + 1
    scale = K**-0.5
    eps = 1e-6

    mixed_qkv = torch.randn(
        num_tokens,
        2 * H * K + HV * V,
        dtype=torch.bfloat16,
        device=device,
    )
    query, key, value = torch.split(
        mixed_qkv,
        [H * K, H * K, HV * V],
        dim=-1,
    )
    query = query.view(1, num_tokens, H, K)
    key = key.view(1, num_tokens, H, K)
    value = value.view(1, num_tokens, HV, V)
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    assert not a.is_contiguous()
    assert not b.is_contiguous()
    A_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    output_gate = torch.randn(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(V, dtype=norm_dtype, device=device)
    state_ref = (
        0.01 * torch.randn(num_slots, HV, V, K, dtype=torch.float32, device=device)
    ).to(state_dtype)
    state_actual = state_ref.clone()
    state_indices = torch.arange(1, num_slots, dtype=torch.int32, device=device).view(
        num_reqs, state_width
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    num_accepted_tokens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    if query_lengths[-1] == 0:
        state_indices[-1].zero_()

    for step, accepted_tokens in enumerate((1, min(2, state_width), state_width)):
        num_accepted_tokens.fill_(accepted_tokens)
        if query_lengths[-1] == 0:
            num_accepted_tokens[-1] = 1
        raw_ref, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=state_ref,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        expected = rmsnorm_fn(
            raw_ref.squeeze(0),
            norm_weight,
            None,
            z=output_gate,
            eps=eps,
            norm_before_gate=True,
            activation=output_gate_activation,
        )
        actual = ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            state=state_actual,
            output_gate=output_gate,
            norm_weight=norm_weight,
            out=torch.empty_like(output_gate),
            scale=scale,
            norm_eps=eps,
            output_gate_activation=output_gate_activation,
        )

        output_error = (actual.float() - expected.float()).norm()
        output_relative_l2 = output_error / expected.float().norm().clamp_min(1e-20)
        assert output_relative_l2 < 5e-4, (
            f"MTP output relative L2 mismatch at step {step}: "
            f"{output_relative_l2.item():.6g}"
        )

    torch.testing.assert_close(state_actual, state_ref, atol=3e-2, rtol=3e-2)
