# SPDX-License-Identifier: Apache-2.0
"""V4.1 integration contracts; CPU tests never import torch or vLLM."""

from types import SimpleNamespace

import pytest

from tests.support.source import function
from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules
load_patch = harness.load_patch


@pytest.fixture
def quant_module(modules):
    calls = []

    class Upstream:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @classmethod
        def override_quantization_method(cls, *args):
            calls.append(("recognize", args))
            return "upstream-method"

        @classmethod
        def from_config(cls, config):
            calls.append(("config", config))
            return cls()

        def get_quant_method(self, layer, prefix):
            calls.append(("dispatch", layer, prefix))
            return "upstream-kernel"

    modules("vllm.models.deepseek_v41.quant_config", DeepseekV4FP8Config=Upstream)
    modules("vllm.model_executor.layers.fused_moe", RoutedExperts=type("MoE", (), {}))
    modules(
        "vllm.model_executor.layers.linear",
        LinearBase=type("Linear", (), {}),
        UnquantizedLinearMethod=object,
    )
    modules(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        is_layer_skipped=lambda **kwargs: False,
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    return load_patch("vllm_sail/models/deepseek_v41/quant_config.py"), calls


def test_standard_quantization_delegates_without_rewriting_config(quant_module):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    checkpoint = {"quant_method": "fp8", "weight_block_size": [32, 32]}
    hf = SimpleNamespace(model_type="deepseek_v41")
    assert cls.override_quantization_method(checkpoint, None, hf) == "deepseek_v41_fp8"
    config = cls.from_config(checkpoint)
    layer = object()
    assert config.get_quant_method(layer, "model.ffn") == "upstream-kernel"
    assert calls == [
        ("recognize", (checkpoint, None, hf)),
        ("config", checkpoint),
        ("dispatch", layer, "model.ffn"),
    ]
    assert calls[1][1] is checkpoint


@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"]),
        ("deepseek_v41_text", []),
        # SpeculativeConfig verifies this temporary config before rewriting
        # it to deepseek_v41 / DSparkV41DraftModel.
        ("deepseek_mtp", ["DeepseekV41ForCausalLM"]),
    ],
)
@pytest.mark.parametrize(
    "ignored_key", ["ignore", "ignored_layers", "modules_to_not_convert"]
)
def test_qlean_metadata_is_translated_without_editing_checkpoint(
    quant_module, model_type, architectures, ignored_key
):
    module, calls = quant_module
    cls = module.DeepseekV41FP8Config
    cls.__bases__[0].override_quantization_method = classmethod(lambda *args: None)
    checkpoint = {
        "quant_method": "mxfp4",
        "expert_dtype": "mxfp4",
        "fp8_channelwise_layers": ["layers.0.attn.wo_a"],
        ignored_key: ["layers.0.attn.compressor.wkv"],
    }
    assert (
        cls.override_quantization_method(
            checkpoint,
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        == "deepseek_v41_fp8"
    )
    result = cls.from_config(checkpoint)
    translated = calls[-1][1]
    assert translated["quant_method"] == "fp8"
    assert translated["activation_scheme"] == "dynamic"
    assert translated["ignored_layers"] == checkpoint[ignored_key]
    assert result._checkpoint_ignored_layers == checkpoint[ignored_key]
    assert result.fp8_channelwise_layers == checkpoint["fp8_channelwise_layers"]
    assert (
        checkpoint["quant_method"] == "mxfp4" and "activation_scheme" not in checkpoint
    )


@pytest.mark.parametrize("quant_method", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "ppu,model_type,architectures",
    [
        (False, "deepseek_v41", ["DeepseekV41ForCausalLM"]),
        (True, "deepseek_v4", ["DeepseekV4ForCausalLM"]),
        (True, "deepseek_v4_text", []),
        (True, "deepseek_mtp", ["DeepSeekV4MTPModel"]),
        (True, "deepseek_mtp", ["DeepSeekMTPModel"]),
        (True, "other", []),
    ],
)
def test_v41_quant_override_leaves_other_models_and_platforms_untouched(
    quant_module, quant_method, ppu, model_type, architectures
):
    module, calls = quant_module
    module.current_platform.is_ppu = lambda: ppu
    assert (
        module.DeepseekV41FP8Config.override_quantization_method(
            {
                "quant_method": quant_method,
                "fp8_channelwise_layers": ["layers.0.attn.wq_a"],
            },
            None,
            SimpleNamespace(model_type=model_type, architectures=architectures),
        )
        is None
    )
    assert calls == []


@pytest.mark.parametrize("ppu", [False, True])
def test_quant_registration_is_lazy_idempotent_and_preserves_v4(modules, ppu):
    calls = []
    modules("vllm.logger", init_logger=lambda _: SimpleNamespace(debug=lambda *a: None))
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    modules(
        "vllm.model_executor.layers.quantization",
        QUANTIZATION_METHODS=[],
        register_quantization_config=lambda name: lambda cls: calls.append((name, cls)),
    )
    mixed = type("Mixed", (), {"get_name": staticmethod(lambda: "mixed_precision_w4")})
    v4 = type("V4", (), {})
    v41 = type("V41", (), {"get_name": staticmethod(lambda: "deepseek_v41_fp8")})
    modules(
        "vllm_sail.model_executor.layers.quantization.mixed_precision_w4",
        MixedPrecisionW4Config=mixed,
    )
    modules("vllm_sail.models.deepseek_v4.quant_config", DeepseekV4FP8Config=v4)
    modules("vllm_sail.models.deepseek_v41.quant_config", DeepseekV41FP8Config=v41)
    registry = load_patch("vllm_sail/registry/quant_config/__init__.py")
    assert calls == []
    registry.register()
    registry.register()
    assert calls == [("mixed_precision_w4", mixed)] + (
        [("deepseek_v4_fp8", v4), ("deepseek_v41_fp8", v41)] if ppu else []
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("quant_method", ["fp8", "deepseek_v4_fp8", "mxfp4"])
@pytest.mark.parametrize(
    "model_type,architectures,expected",
    [
        ("deepseek_v41", ["DeepseekV41ForCausalLM"], "deepseek_v41_fp8"),
        ("deepseek_v41_text", [], "deepseek_v41_fp8"),
        ("deepseek_mtp", ["DeepseekV41ForCausalLM"], "deepseek_v41_fp8"),
        ("deepseek_v4", ["DeepseekV4ForCausalLM"], "deepseek_v4_fp8"),
    ],
)
def test_upstream_model_config_selects_model_specific_quantization(
    quant_module,
    modules,
    upstream_source_root,
    quant_method,
    model_type,
    architectures,
    expected,
):
    """Exercise upstream recognition and override ordering, not a reimplemented selector."""
    from typing import Literal, cast, get_args

    module, _ = quant_module
    cls = module.DeepseekV41FP8Config
    upstream = modules("vllm.models.deepseek_v41.quant_config").DeepseekV4FP8Config
    assert cls.__bases__ == (upstream,)
    upstream.override_quantization_method = classmethod(
        function(
            upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
            "DeepseekV4FP8Config.override_quantization_method",
            {},
        )
    )

    class UpstreamV4(upstream):
        pass

    UpstreamV4.override_quantization_method = classmethod(
        function(
            upstream_source_root / "vllm/models/deepseek_v4/quant_config.py",
            "DeepseekV4FP8Config.override_quantization_method",
            {},
        )
    )
    modules("vllm.models.deepseek_v4.quant_config", DeepseekV4FP8Config=UpstreamV4)
    v4 = load_patch("vllm_sail/models/deepseek_v4/quant_config.py").DeepseekV4FP8Config
    assert v4.__bases__ == (UpstreamV4,)
    platform = module.current_platform
    platform.supported_quantization = ["fp8", "mxfp4", "deepseek_v4_fp8"]
    verified = []
    platform.verify_quantization = verified.append
    registry = {}
    names = list(platform.supported_quantization)
    register = function(
        upstream_source_root / "vllm/model_executor/layers/quantization/__init__.py",
        "register_quantization_config",
        {
            "QUANTIZATION_METHODS": names,
            "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG": registry,
            "QuantizationConfig": upstream,
            "current_platform": platform,
        },
    )
    register(cls.get_name())(cls)
    assert cls.get_name() in platform.supported_quantization
    registry["deepseek_v4_fp8"] = v4
    no_override = SimpleNamespace(override_quantization_method=lambda *a, **kw: None)
    quant = SimpleNamespace(
        QUANTIZATION_METHODS=names,
        QuantizationMethods=Literal["fp8", "mxfp4", "deepseek_v4_fp8"],
        DEPRECATED_QUANTIZATION_METHODS=[],
        get_quantization_config=lambda name: registry.get(name, no_override),
    )
    verify = function(
        upstream_source_root / "vllm/config/model.py",
        "ModelConfig._verify_quantization",
        {
            "me_quant": quant,
            "get_args": get_args,
            "cast": cast,
            "current_platform": platform,
        },
    )
    checkpoint = {"quant_method": quant_method}
    if quant_method == "mxfp4":
        checkpoint["fp8_channelwise_layers"] = ["layers.0.attn.wq_a"]
    config = SimpleNamespace(
        quantization=None,
        model_arch_config=SimpleNamespace(quantization_config=checkpoint),
        hf_config=SimpleNamespace(model_type=model_type, architectures=architectures),
    )
    verify(config)
    assert config.quantization == expected
    assert verified == [expected]
    assert registry[config.quantization] is (cls if expected == cls.get_name() else v4)
    assert checkpoint["quant_method"] == quant_method


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "model_type",
    ["deepseek_v4", "deepseek_v4_text", "deepseek_v41", "deepseek_v41_text"],
)
def test_upstream_recognizes_target_and_text_configs(upstream_source_root, model_type):
    recognize = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.override_quantization_method",
        {},
    )
    assert (
        recognize(
            object,
            {"quant_method": "fp8"},
            None,
            SimpleNamespace(model_type=model_type),
        )
        == "deepseek_v4_fp8"
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
def test_upstream_mxfp4_dispatch_preserves_target_and_draft_moe(
    upstream_source_root, modules, experts, topk
):
    class RoutedExperts:
        moe_config = SimpleNamespace(num_experts=experts, experts_per_token=topk)

    expected = object()
    modules(
        "vllm.model_executor.layers.quantization.mxfp4",
        Mxfp4MoEMethod=lambda cfg: expected,
    )
    dispatch = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.get_quant_method",
        {
            "LinearBase": type("Linear", (), {}),
            "RoutedExperts": RoutedExperts,
            "is_layer_skipped": lambda **kwargs: False,
            "Mxfp4MoEMethod": lambda cfg: (expected, cfg),
        },
    )
    config = SimpleNamespace(
        ignored_layers=[],
        packed_modules_mapping={},
        expert_dtype="fp4",
        moe_quant_algo="",
    )
    layer = RoutedExperts()
    assert dispatch(config, layer, "model.ffn.experts") == (expected, layer.moe_config)


def test_v41_draft_inherits_its_own_loader_and_execution(modules):
    class Upstream:
        load_weights = object()
        forward = object()

    class SupportsQuant:
        pass

    mapper_args = []
    mapper = object()

    def make_mapper(*args):
        mapper_args.append(args)
        return mapper

    modules("vllm.model_executor.models.interfaces", SupportsQuant=SupportsQuant)
    modules(
        "vllm.models.deepseek_v41.nvidia.dspark", DSparkDeepseekV4ForCausalLM=Upstream
    )
    modules(
        "vllm.models.deepseek_v41.nvidia.model",
        _make_deepseek_v4_weights_mapper=make_mapper,
    )
    draft = load_patch(
        "vllm_sail/models/deepseek_v41/dspark.py"
    ).DSparkDeepseekV41ForCausalLM
    assert issubclass(draft, SupportsQuant)
    assert (
        draft.load_weights is Upstream.load_weights
        and draft.forward is Upstream.forward
    )
    assert draft.hf_to_vllm_mapper is mapper
    assert mapper_args == [("fp4", "weight_scale")]
    assert draft.packed_modules_mapping["fused_wqa_wkv"] == ["wq_a", "wkv"]


@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("channelwise", [False, True])
def test_scale_patch_rebinds_preloaded_loaders_without_attention(
    modules, ppu, channelwise
):
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: ppu))
    calls = []

    def original(config, expert_dtype):
        calls.append((config, expert_dtype))
        return "weight_scale_inv"

    provider = modules(
        "vllm.models.deepseek_v41.nvidia.model", _linear_scale_param_name=original
    )
    consumers = [
        modules(name, _linear_scale_param_name=original)
        for name in (
            "vllm.models.deepseek_v41.nvidia.dspark",
            "vllm.models.deepseek_v41.nvidia.vl_model",
        )
    ]
    path = "vllm_sail/patch/enhancement/models/deepseek_v41_quant.py"
    replacement = load_patch(path)
    assert provider._linear_scale_param_name is replacement._linear_scale_param_name
    marker = getattr(provider._linear_scale_param_name, replacement.PATCH_MARKER)
    assert marker[f"{provider.__name__}._linear_scale_param_name"] is original
    config = SimpleNamespace(
        quant_config=SimpleNamespace(
            fp8_channelwise_layers=["attn.wo_a"] if channelwise else []
        )
    )
    for consumer in consumers:
        assert consumer._linear_scale_param_name is provider._linear_scale_param_name
        assert consumer._linear_scale_param_name(config, "fp4") == (
            "weight_scale" if ppu and channelwise else "weight_scale_inv"
        )
    assert calls == ([] if ppu and channelwise else [(config, "fp4")] * 2)
    replacement._rebind_scale_aliases()

    def other(*args):
        return "other-owner"

    consumers[0]._linear_scale_param_name = other
    replacement._rebind_scale_aliases()
    assert consumers[0]._linear_scale_param_name is other
    with pytest.raises(RuntimeError, match="already patched"):
        load_patch(path)


@pytest.mark.upstream_source
@pytest.mark.parametrize("expert_dtype,block", [("fp4", [32, 32]), ("fp8", [128, 128])])
def test_scale_patch_preserves_upstream_non_channelwise_contract(
    modules, upstream_source_root, expert_dtype, block
):
    original = function(
        upstream_source_root / "vllm/models/deepseek_v41/nvidia/model.py",
        "_linear_scale_param_name",
        {},
    )
    modules("vllm.platforms", current_platform=SimpleNamespace(is_ppu=lambda: True))
    provider = modules(
        "vllm.models.deepseek_v41.nvidia.model", _linear_scale_param_name=original
    )
    load_patch("vllm_sail/patch/enhancement/models/deepseek_v41_quant.py")
    config = SimpleNamespace(quant_config=SimpleNamespace(weight_block_size=block))
    assert provider._linear_scale_param_name(config, expert_dtype) == original(
        config, expert_dtype
    )


@pytest.mark.upstream_source
def test_upstream_quark_recognition_is_not_overridden(
    quant_module, upstream_source_root
):
    module, _ = quant_module
    cls = module.DeepseekV41FP8Config
    cls.__bases__[0].override_quantization_method = classmethod(
        function(
            upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
            "DeepseekV4FP8Config.override_quantization_method",
            {},
        )
    )
    checkpoint = {
        "quant_method": "quark",
        "global_quant_config": {
            "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32}
        },
    }
    assert (
        cls.override_quantization_method(
            checkpoint, None, SimpleNamespace(model_type="deepseek_v41")
        )
        is None
    )


@pytest.mark.upstream_source
def test_standard_mxfp8_uses_upstream_quant_spec(
    quant_module, modules, upstream_source_root
):
    module, _ = quant_module
    cls = module.DeepseekV41FP8Config
    linear = modules("vllm.model_executor.layers.linear").LinearBase
    modules(
        "vllm.model_executor.layers.quantization.modelopt",
        CkptCtx=lambda **kwargs: kwargs,
        ModelOptLinearMethod=lambda spec, ctx: (spec, ctx),
    )
    cls.__bases__[0].get_quant_method = function(
        upstream_source_root / "vllm/models/deepseek_v41/quant_config.py",
        "DeepseekV4FP8Config.get_quant_method",
        {
            "LinearBase": linear,
            "is_layer_skipped": lambda **kwargs: False,
            "QuantSpec": lambda **kwargs: kwargs,
            "kMxfp8Static": "static-mxfp8",
            "kMxfp8Dynamic": "dynamic-mxfp8",
        },
    )
    config = cls(
        weight_block_size=[32, 32],
        is_scale_e8m0=True,
        ignored_layers=[],
        ignored_layers_match_mode="exact",
    )
    assert config.get_quant_method(linear(), "model.layers.0.attn.wq_b") == (
        {"weight": "static-mxfp8", "activation": "dynamic-mxfp8"},
        {"scale_block_size": (32, 32)},
    )
