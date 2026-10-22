# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kimi-K3 AttnRes performance patch.

Two tiers, following the suite's established split:

* Bare CPU (no torch, no vLLM): the patch module is loaded against stub modules
  — the same trick as ``test_fla_autotune_cache.py`` — so the patches really
  install and the Triton launch arguments can be inspected without a device.
  The launch table, the off-PPU delegation, the by-value redirect of
  ``nvidia/model.py`` and the copied kernel body are all asserted here.
* With vLLM: the same patches must land on the real ``attn_res`` module with
  their markers, and re-application must raise.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[2]
    / "vllm_sail"
    / "patch"
    / "performance"
    / "kimi_k3_attn_res.py"
)
OPS_MODULE = "vllm.models.kimi_k3.nvidia.ops.attn_res"
MODEL_MODULE = "vllm.models.kimi_k3.nvidia.model"
HIDDEN_SIZE = 7168
EPS = 1e-5

_ATTN_RES_TARGETS = {
    f"{OPS_MODULE}._attn_res_kernel",
    f"{OPS_MODULE}.attn_res_triton_config",
    f"{OPS_MODULE}.attn_res",
    f"{MODEL_MODULE}.attn_res",
}


class FakeTensor:
    """Contiguous-shape stand-in exposing only what the wrapper touches."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        strides: list[int] = []
        running = 1
        for size in reversed(self.shape):
            strides.append(running)
            running *= size
        self._strides = tuple(reversed(strides))

    def stride(self, dim: int) -> int:
        return self._strides[dim]

    def new_empty(self, shape: tuple[int, ...]) -> FakeTensor:
        return FakeTensor(shape)


class FakeKernel:
    """``@triton.jit`` stand-in that records launches instead of compiling."""

    def __init__(self, function: object, launches: list) -> None:
        self.function = function
        self.launches = launches

    def __getitem__(self, grid: tuple[int, ...]):
        def launch(*args: object, **kwargs: object) -> None:
            self.launches.append((grid, args, kwargs))

        return launch


@pytest.fixture
def attn_res_patch(monkeypatch: pytest.MonkeyPatch, patch_utils_module):
    """Load the patch module against stub vLLM/torch/triton modules."""
    launches: list = []
    platform_state = {"ppu": True}

    def upstream_attn_res(*args: object, **kwargs: object) -> str:
        return "upstream"

    custom_ops = types.ModuleType("vllm._custom_ops")
    custom_ops.kimi_k3_attn_res = lambda *args: "native"

    platform = types.SimpleNamespace(
        is_ppu=lambda: platform_state["ppu"],
        is_device_capability_family=lambda family: False,
        is_arch_support_pdl=lambda: False,
    )
    triton = types.SimpleNamespace(
        jit=lambda function: FakeKernel(function, launches),
        next_power_of_2=lambda size: 1 << (size - 1).bit_length(),
    )

    # ``ops/__init__.py`` re-exports the function under the module's own name.
    ops = types.ModuleType(OPS_MODULE)
    ops._attn_res_kernel = FakeKernel(None, launches)
    ops.attn_res = upstream_attn_res
    model = types.ModuleType(MODEL_MODULE)
    model.attn_res = upstream_attn_res

    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.ops = types.SimpleNamespace(_C=types.SimpleNamespace())

    vllm = types.ModuleType("vllm")
    vllm._custom_ops = custom_ops
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.tl = types.SimpleNamespace(constexpr=object)
    triton_utils.triton = triton

    for name, module in {
        "torch": torch,
        "vllm": vllm,
        "vllm._custom_ops": custom_ops,
        "vllm.platforms": types.SimpleNamespace(current_platform=platform),
        "vllm.triton_utils": triton_utils,
        OPS_MODULE: ops,
        MODEL_MODULE: model,
        "vllm_sail.patch.utils": patch_utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_test_kimi_k3_attn_res", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return types.SimpleNamespace(
        module=module,
        ops=ops,
        model=model,
        platform_state=platform_state,
        launches=launches,
        records=patch_utils_module.PATCH_REGISTRY,
    )


def _launch(
    attn_res_patch, num_tokens: int, num_blocks: int, *, expect_launch: bool = True
):
    patch = attn_res_patch
    prefix = FakeTensor((num_tokens, HIDDEN_SIZE))
    blocks = FakeTensor((num_tokens, 8, HIDDEN_SIZE))
    before = len(patch.launches)
    result = patch.module.attn_res(
        prefix,
        None,
        blocks,
        FakeTensor((HIDDEN_SIZE,)),
        FakeTensor((HIDDEN_SIZE,)),
        FakeTensor((HIDDEN_SIZE,)),
        num_blocks,
        -1,
        EPS,
        EPS,
    )
    if not expect_launch:
        assert len(patch.launches) == before
        return result, None, None
    assert len(patch.launches) == before + 1
    grid, _, kwargs = patch.launches[-1]
    return result, grid, kwargs


# ---------------------------------------------------------------------------
# Launch table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "expected"),
    [
        # Several sources: a 4-source tile pays for itself in decode.
        (1, 2, (4, 16, 2)),
        (4, 4, (4, 16, 2)),
        (32, 8, (4, 16, 2)),
        (63, 8, (4, 16, 2)),
        (64, 2, (4, 8, 2)),
        (127, 8, (4, 8, 2)),
        (128, 2, (1, 8, 2)),
        (8192, 8, (1, 8, 2)),
        # A single source: the tile stays 1 and decode goes wide and deep.
        (1, 0, (1, 16, 3)),
        (1, 1, (1, 16, 3)),
        (127, 1, (1, 16, 3)),
        (128, 1, (1, 4, 2)),
        (255, 0, (1, 4, 2)),
        (256, 1, (1, 8, 2)),
        (4095, 1, (1, 8, 2)),
        (4096, 0, (1, 4, 2)),
        (8192, 1, (1, 4, 2)),
    ],
)
def test_launch_table_matches_the_measured_optimum(
    attn_res_patch, num_tokens: int, num_blocks: int, expected: tuple[int, int, int]
) -> None:
    config = attn_res_patch.module.attn_res_triton_config(num_tokens, num_blocks)
    assert config == expected


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------
def test_every_target_is_installed_with_complete_metadata(attn_res_patch) -> None:
    patch = attn_res_patch
    assert {record.target for record in patch.records} == _ATTN_RES_TARGETS
    for record in patch.records:
        assert record.kind == "module", record.target
        assert record.reason.strip(), record.target
        assert record.remove_when.strip(), record.target
        assert record.affected_versions == ">=0.31.0,<0.32.0", record.target
    additive = {r.target: r for r in patch.records}[
        f"{OPS_MODULE}.attn_res_triton_config"
    ]
    assert additive.was_missing is True


def test_replacements_land_on_their_targets(attn_res_patch) -> None:
    patch = attn_res_patch
    assert patch.ops._attn_res_kernel is patch.module._attn_res_kernel
    assert patch.ops.attn_res is patch.module.attn_res
    assert patch.model.attn_res is patch.module.attn_res
    assert patch.ops.attn_res_triton_config is patch.module.attn_res_triton_config


def test_kernel_and_wrapper_record_both_targets(attn_res_patch) -> None:
    patch = attn_res_patch
    markers = getattr(patch.module.attn_res, "__vllm_sail_patch__", {})
    assert set(markers) == {f"{OPS_MODULE}.attn_res", f"{MODEL_MODULE}.attn_res"}
    assert markers[f"{OPS_MODULE}.attn_res"] is not None
    assert markers[f"{MODEL_MODULE}.attn_res"] is not None
    kernel_markers = getattr(patch.module._attn_res_kernel, "__vllm_sail_patch__", {})
    assert set(kernel_markers) == {f"{OPS_MODULE}._attn_res_kernel"}


def test_reapplication_is_rejected(attn_res_patch) -> None:
    patch = attn_res_patch
    with pytest.raises(RuntimeError, match="already patched"):

        @patch.module.patch(
            OPS_MODULE,
            "attn_res",
            reason="test",
            affected_versions=">=0.31.0,<0.32.0",
            remove_when="never; test-only",
        )
        def attn_res(*args: object, **kwargs: object) -> None:
            raise AssertionError("must not install")


# ---------------------------------------------------------------------------
# Wrapper behaviour
# ---------------------------------------------------------------------------
def test_non_ppu_delegates_to_upstream(attn_res_patch) -> None:
    patch = attn_res_patch
    patch.platform_state["ppu"] = False
    result, _, _ = _launch(patch, num_tokens=4, num_blocks=8, expect_launch=False)
    assert result == "upstream"
    # Nothing reached the Triton launch.
    assert not patch.launches


@pytest.mark.parametrize(
    ("num_tokens", "num_blocks"),
    [(1, 2), (8, 4), (64, 8), (128, 8), (8192, 8), (1, 1), (512, 0)],
)
def test_ppu_launch_uses_the_tuned_config(
    attn_res_patch, num_tokens: int, num_blocks: int
) -> None:
    patch = attn_res_patch
    expected = patch.module.attn_res_triton_config(num_tokens, num_blocks)
    _, grid, kwargs = _launch(patch, num_tokens, num_blocks)
    assert grid == (num_tokens,)
    assert (kwargs["BLOCK_L"], kwargs["num_warps"], kwargs["num_stages"]) == expected
    assert kwargs["BLOCK_D"] == 8192
    assert kwargs["HAS_DELTA"] is False
    assert kwargs["WRITE_BLOCK"] is False
    assert kwargs["APPLY_OUTPUT_NORM"] is True
    assert kwargs["launch_pdl"] is False


def test_decode_uses_a_deeper_pipeline_than_upstream(attn_res_patch) -> None:
    """Upstream hardcodes num_stages=2; the single-source decode path needs 3."""
    _, _, kwargs = _launch(attn_res_patch, num_tokens=1, num_blocks=1)
    assert kwargs["num_stages"] == 3


# ---------------------------------------------------------------------------
# Copied kernel body
# ---------------------------------------------------------------------------
def test_kernel_body_keeps_the_fork_changes() -> None:
    """A refresh of the copied body from upstream must keep both optimisations.

    The body is a verbatim copy of upstream by design, so an upgrade re-copy is
    the most likely way for the fork's changes to disappear silently.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_attn_res_kernel"
    )
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
    }
    assert "tl.debug_barrier" not in calls

    seeded = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mixed"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "updated_prefix"
    ]
    assert seeded, "the prefix no longer seeds the online softmax"
    assert "output_norm_eps * denominator * denominator" in ast.unparse(kernel)
    assert "num_sources = num_blocks + 1" not in ast.unparse(kernel)


def test_triton_kernel_patch_is_the_outermost_decorator() -> None:
    """``patch`` must install the fully decorated Triton object, not the raw one."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_attn_res_kernel"
    )
    decorators = [
        ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
        for decorator in kernel.decorator_list
    ]
    assert decorators[0] == "patch"
    assert decorators[1:] == ["triton.jit"]


def test_patch_module_has_no_import_time_platform_access() -> None:
    """``current_platform`` may be imported eagerly but only *used* lazily.

    The object resolves the active platform on attribute access, so touching it
    at import time would pin the module to whatever device was initialised
    first.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    eager = [
        ast.unparse(node)
        for statement in tree.body
        if not isinstance(statement, ast.FunctionDef)
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "current_platform"
    ]
    assert not eager, f"the patch module resolves the platform at import time: {eager}"


# ---------------------------------------------------------------------------
# Everything below needs a real vLLM to patch against.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def installed_attn_res_patch():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    from vllm_sail.patch.utils import PATCH_REGISTRY

    existing = [
        record for record in PATCH_REGISTRY if record.target in _ATTN_RES_TARGETS
    ]
    if len(existing) == len(_ATTN_RES_TARGETS):
        return existing

    import vllm_sail.patch.performance.kimi_k3_attn_res  # noqa: F401

    records = [
        record for record in PATCH_REGISTRY if record.target in _ATTN_RES_TARGETS
    ]
    assert records, "no AttnRes patch records were installed"
    return records


def test_real_targets_are_patched(installed_attn_res_patch) -> None:
    from vllm_sail.patch.utils import PATCH_MARKER

    ops = importlib.import_module(OPS_MODULE)
    model = importlib.import_module(MODEL_MODULE)

    assert {record.target for record in installed_attn_res_patch} == _ATTN_RES_TARGETS
    assert ops.attn_res_triton_config(1, 4) == (4, 16, 2)
    assert getattr(ops._attn_res_kernel, PATCH_MARKER, None) is not None
    assert ops.attn_res is model.attn_res
    markers = getattr(ops.attn_res, PATCH_MARKER, None)
    assert markers is not None
    assert set(markers) == {f"{OPS_MODULE}.attn_res", f"{MODEL_MODULE}.attn_res"}


def test_real_patch_reapplication_raises(installed_attn_res_patch) -> None:
    from vllm_sail.patch.utils import patch

    with pytest.raises(RuntimeError, match="already patched"):

        @patch(
            OPS_MODULE,
            "attn_res",
            reason="test",
            affected_versions=">=0.31.0,<0.32.0",
            remove_when="never; test-only",
        )
        def attn_res(*args: object, **kwargs: object) -> None:
            raise AssertionError("must not install")
