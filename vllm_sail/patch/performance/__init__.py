# SPDX-License-Identifier: Apache-2.0
"""PPU performance patches.

Optimisations that PPU needs on top of upstream's code paths. Unlike the
`enhancement` category, removing these does not break correctness — it costs
performance.

Currently contains `fla` (flash-linear-attention): the fork's cache_results
autotune additions and the Qwen3-Next fused GDN decode path; and
`kimi_k3_attn_res`: the tuned Kimi-K3 AttnRes Triton fallback, which is the
production path on PPU because the native kernel is SM100-only.
"""

from vllm_sail.patch.performance import fla, kimi_k3_attn_res  # noqa: F401
