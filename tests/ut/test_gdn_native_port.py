# SPDX-License-Identifier: Apache-2.0
"""The fused GDN MTP op must be compiled and registered, never stubbed."""

from pathlib import Path

from vllm_sail.native import manifest, stubs

ROOT = Path(__file__).parents[2]
OP = "fused_gdn_decode_post_conv_mtp"


def test_native_mtp_is_in_the_compiled_corpus_and_cuda_bindings():
    corpus = manifest.load()
    assert OP in corpus.ops(corpus.compiled())
    assert OP not in {op.name for op in stubs.load_schemas()}
    bindings = (ROOT / "csrc/upstream/libtorch_stable/torch_bindings.cpp").read_text()
    assert f'ops.impl("{OP}",' in bindings
    assert f"TORCH_BOX(&{OP})" in bindings


def test_hggc_copy_path_has_no_ptx_and_retains_block_synchronization():
    source = (
        ROOT / "csrc/upstream/libtorch_stable/gdn/fused_gdn_decode_kernel.cu"
    ).read_text()
    start = source.index("#if defined(__HGGCCC__)")
    end = source.index("#else", start)
    hggc_copy = source[start:end]
    assert "reinterpret_cast<int4*>" in hggc_copy
    assert "asm volatile" not in hggc_copy
    # Every stage waits before the whole block reads the completed tile.
    assert "cp_async_wait_all();\n    __syncthreads();" in source
