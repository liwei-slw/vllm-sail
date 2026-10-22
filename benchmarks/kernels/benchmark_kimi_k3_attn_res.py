# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Kimi-K3 AttnRes kernels.

On PPU the Triton fallback is the production path: the native kernel is
SM100-only and ``kimi_k3_attn_res`` is excluded from the PPU build, so
``vllm_sail.patch.performance.kimi_k3_attn_res`` replaces both the kernel and the
launch heuristic. This tool measures it per ``(num_blocks, num_tokens)`` shape.

The benchmark uses a zero delta so repeated in-place launches remain stable.
Without ``--triton-config`` the launch configuration comes from the patched
``attn_res_triton_config`` heuristic, so the numbers are what the model runs;
pass it to sweep the kernel instead.

Record the PPU card, SAIL SDK and PyTorch versions with any result, and keep raw
measurement data outside version control (see ``benchmarks/README.md``).

Example:

.. code-block:: console

    python benchmarks/kernels/benchmark_kimi_k3_attn_res.py --backend triton \
      --output /tmp/attn_res.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from importlib import import_module
from pathlib import Path

import torch
import vllm._custom_ops  # noqa: F401 - loads torch.ops._C
from vllm.platforms import current_platform
from vllm.triton_utils import triton

# Importing a patch module installs its patches; a plain script never runs
# vLLM's general-plugin hook. Importing the performance package installs this
# category (this patch plus the FLA autotune additions) and leaves every other
# category alone, which vllm_sail.patch.install() would not do.
from vllm_sail.patch.performance import kimi_k3_attn_res  # noqa: F401

# ``ops/__init__.py`` re-exports the function under the module's own name, so the
# submodule itself has to be resolved through the import system.
attn_res_ops = import_module("vllm.models.kimi_k3.nvidia.ops.attn_res")

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5


@dataclass
class Inputs:
    prefix: torch.Tensor
    delta: torch.Tensor
    blocks: torch.Tensor
    norm_weight: torch.Tensor
    qk_weight: torch.Tensor
    output_norm_weight: torch.Tensor


@dataclass
class Result:
    num_tokens: int
    num_blocks: int
    backend: str
    median_us: float
    p20_us: float
    p80_us: float


def _parse_triton_config(value: str) -> tuple[int, int, int]:
    try:
        block_l, num_warps, num_stages = (int(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected BLOCK_L,NUM_WARPS,NUM_STAGES, got {value!r}"
        ) from None
    if min(block_l, num_warps, num_stages) < 1:
        raise argparse.ArgumentTypeError("Triton config values must be >= 1")
    return block_l, num_warps, num_stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--num-blocks", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--backend",
        choices=("both", "triton", "cuda"),
        default="both",
        help=(
            "Backend(s) to benchmark. 'cuda' requires the native SM100 kernel; "
            "'both' falls back to Triton when it is unavailable. PPU only ever "
            "has Triton."
        ),
    )
    parser.add_argument(
        "--triton-config",
        type=_parse_triton_config,
        default=None,
        metavar="BLOCK_L,NUM_WARPS,NUM_STAGES",
        help=(
            "Override the Triton launch config for every shape instead of using "
            "the patched attn_res_triton_config heuristic. Use this to sweep "
            "the kernel."
        ),
    )
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_inputs(num_tokens: int) -> Inputs:
    prefix = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn(
        num_tokens,
        MAX_BLOCKS,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    return Inputs(
        prefix=prefix,
        delta=torch.zeros_like(prefix),
        blocks=blocks,
        norm_weight=norm_weight,
        qk_weight=qk_weight,
        output_norm_weight=output_norm_weight,
    )


def launch_triton(
    inputs: Inputs,
    output: torch.Tensor,
    num_blocks: int,
    config: tuple[int, int, int] | None = None,
) -> None:
    num_tokens = inputs.prefix.shape[0]
    block_l, num_warps, num_stages = config or attn_res_ops.attn_res_triton_config(
        num_tokens, num_blocks
    )
    attn_res_ops._attn_res_kernel[(num_tokens,)](
        inputs.prefix,
        inputs.delta,
        inputs.blocks,
        inputs.norm_weight,
        inputs.qk_weight,
        inputs.output_norm_weight,
        output,
        inputs.prefix.stride(0),
        inputs.delta.stride(0),
        inputs.blocks.stride(0),
        inputs.blocks.stride(1),
        output.stride(0),
        num_blocks,
        HIDDEN_SIZE,
        -1,
        EPS,
        EPS,
        HAS_DELTA=True,
        WRITE_BLOCK=False,
        APPLY_OUTPUT_NORM=True,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(HIDDEN_SIZE),
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )


def launch_cuda(inputs: Inputs, output: torch.Tensor, num_blocks: int) -> None:
    torch.ops._C.kimi_k3_attn_res(
        inputs.prefix,
        inputs.delta,
        inputs.blocks,
        inputs.norm_weight,
        inputs.qk_weight,
        inputs.output_norm_weight,
        output,
        num_blocks,
        -1,
        EPS,
        EPS,
    )


def benchmark(
    function: Callable[[], None], warmup_ms: int, repeat_ms: int
) -> tuple[float, float, float]:
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=repeat_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return median_ms * 1000, p20_ms * 1000, p80_ms * 1000


def validate(inputs: Inputs, num_blocks: int) -> None:
    outputs = [torch.empty_like(inputs.prefix) for _ in range(2)]
    launch_triton(inputs, outputs[0], num_blocks)
    launch_cuda(inputs, outputs[1], num_blocks)
    torch.accelerator.synchronize()
    torch.testing.assert_close(outputs[1], outputs[0], atol=8e-2, rtol=3e-2)


def main() -> None:
    args = parse_args()
    if any(not 0 <= blocks <= MAX_BLOCKS for blocks in args.num_blocks):
        raise SystemExit(f"--num-blocks values must be in [0, {MAX_BLOCKS}]")

    native_unavailable = torch.cuda.get_device_capability()[0] != 10 or not hasattr(
        getattr(torch.ops, "_C", None), "kimi_k3_attn_res"
    )
    if args.backend == "cuda" and native_unavailable:
        raise SystemExit(
            "The kimi_k3_attn_res backend requires the SM100 family and a "
            "build that contains the op"
        )
    if args.backend == "both" and native_unavailable:
        print(
            "native AttnRes kernel unavailable; benchmarking the Triton fallback only",
            flush=True,
        )
        args.backend = "triton"
    run_triton = args.backend != "cuda"
    run_cuda = args.backend != "triton"
    if args.triton_config is not None and run_triton:
        print(
            f"triton config override: BLOCK_L={args.triton_config[0]} "
            f"num_warps={args.triton_config[1]} "
            f"num_stages={args.triton_config[2]}",
            flush=True,
        )

    torch.manual_seed(0)
    if run_cuda:
        check_inputs = make_inputs(5)
        for num_blocks in args.num_blocks:
            validate(check_inputs, num_blocks)
        print("correctness: all selected block counts passed", flush=True)

    results: list[Result] = []
    for num_blocks in args.num_blocks:
        for num_tokens in args.num_tokens:
            inputs = make_inputs(num_tokens)
            output = torch.empty_like(inputs.prefix)
            candidates: dict[str, Callable[[], None]] = {}
            if run_triton:
                candidates["vllm_triton"] = partial(
                    launch_triton, inputs, output, num_blocks, args.triton_config
                )
            if run_cuda:
                candidates["vllm_cuda"] = partial(
                    launch_cuda, inputs, output, num_blocks
                )
            row: dict[str, float] = {}
            for backend, candidate in candidates.items():
                median_us, p20_us, p80_us = benchmark(
                    candidate, args.warmup_ms, args.repeat_ms
                )
                results.append(
                    Result(
                        num_tokens=num_tokens,
                        num_blocks=num_blocks,
                        backend=backend,
                        median_us=median_us,
                        p20_us=p20_us,
                        p80_us=p80_us,
                    )
                )
                row[backend] = median_us
            timings = "  ".join(
                f"{name.removeprefix('vllm_')}={median_us:9.2f} us"
                for name, median_us in row.items()
            )
            print(
                f"blocks={num_blocks} tokens={num_tokens:5d}  {timings}",
                flush=True,
            )

    if args.output is not None:
        metadata = {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "warmup_ms": args.warmup_ms,
            "repeat_ms": args.repeat_ms,
            "backend": args.backend,
            "triton_config": args.triton_config,
        }
        args.output.write_text(
            json.dumps(
                {"metadata": metadata, "results": [asdict(item) for item in results]},
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
