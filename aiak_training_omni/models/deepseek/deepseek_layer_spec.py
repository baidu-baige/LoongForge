"""Deepseek layer spec."""
from typing import Tuple, Optional

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import TransformerLayer, get_transformer_layer_offset
from megatron.core.transformer.multi_latent_attention import MLASelfAttention, MLASelfAttentionSubmodules

from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.enums import Fp8Recipe

from aiak_training_omni.models.deepseek.transformer import DeepSeekTransformerConfig
from aiak_training_omni.models.deepseek.transformer.moe_layer import MoELayer
from aiak_training_omni.models.deepseek.transformer.mtp_transformer_layer import MultiTokenPredLayerDeepSeekSubmodules

from aiak_training_omni.models.dispatch import multiacc_modules


def _get_deepseek_layer_with_te_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = True,
    qk_layernorm: Optional[bool] = False,
) -> ModuleSpec:
    """Get the transformer layer spec for deepseek"""

    mlp = _get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=MultiTokenPredLayerDeepSeekSubmodules(
            input_layernorm=multiacc_modules.TENorm,
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=multiacc_modules.TEColumnParallelLinear,
                    linear_q_down_proj=multiacc_modules.TEColumnParallelLinear,
                    linear_q_up_proj=(
                        multiacc_modules.TELayerNormColumnParallelLinear
                        if qk_layernorm else multiacc_modules.TEColumnParallelLinear
                    ),
                    linear_kv_down_proj=multiacc_modules.TEColumnParallelLinear,
                    linear_kv_up_proj=(
                        multiacc_modules.TELayerNormColumnParallelLinear
                        if qk_layernorm else multiacc_modules.TEColumnParallelLinear
                    ),
                    core_attention=multiacc_modules.DotProductAttention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    kv_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=multiacc_modules.TENorm,
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
            # For MTP
            eh_proj=multiacc_modules.TELinear,
            enorm=multiacc_modules.TENorm,
            hnorm=multiacc_modules.TENorm,
            output_layernorm=multiacc_modules.TENorm,
        )
    )


def _get_mlp_module_spec(
    num_experts: int=None,
    moe_grouped_gemm: bool=False
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    if num_experts is None:
        # Dense MLP w/ TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=multiacc_modules.TEColumnParallelLinear,
                linear_fc2=multiacc_modules.TERowParallelLinear,
                bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
            ),
        )

    # moe mlp
    if moe_grouped_gemm:
        # use TEGroupedLinear
        assert multiacc_modules.TEColumnParallelGroupedLinear is not None
        expert_module = TEGroupedMLP
        linear_fc1 = multiacc_modules.TEColumnParallelGroupedLinear
        linear_fc2 = multiacc_modules.TERowParallelGroupedLinear
        
    else:
        expert_module = SequentialMLP
        linear_fc1 = multiacc_modules.TEColumnParallelLinear
        linear_fc2 = multiacc_modules.TERowParallelLinear

    # share expert
    shared_linear_fc1 = multiacc_modules.TEColumnParallelLinear
    shared_linear_fc2 = multiacc_modules.TERowParallelLinear

    return ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            shared_experts=ModuleSpec(
                module=SharedExpertMLP,
                params={"gate": False},
                submodules=MLPSubmodules(
                    linear_fc1=shared_linear_fc1,
                    linear_fc2=shared_linear_fc2,
                    bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
                )
            ),
            experts=ModuleSpec(
                module=expert_module,
                submodules=MLPSubmodules(
                    linear_fc1=linear_fc1,
                    linear_fc2=linear_fc2,
                    bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
                )
            )
        )
    )


def get_deepseek_decoder_block_and_mtp_spec(
    config: DeepSeekTransformerConfig,
) -> Tuple[TransformerBlockSubmodules, Optional[ModuleSpec]]:
    """Get the deepseek decoder block and multi-token prediction layer spec."""
    assert config.num_moe_experts > 0, "Only support MOE when using DeepSeek"
    assert config.multi_latent_attention, "Only support multi-latent attention when using DeepSeek"

    block_spec = None
    mtp_spec = None

    # Layer specs.
    dense_layer_spec = _get_deepseek_layer_with_te_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
    )

    moe_layer_spec = _get_deepseek_layer_with_te_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
    )

    # In FP8 training, replace `linear_q_down_proj` and `linear_kv_down_proj`
    # with TELinear to support tensor parallelism
    if config.fp8 and config.fp8_recipe == Fp8Recipe.blockwise:
        dense_layer_spec.submodules.self_attention.submodules.linear_q_down_proj = multiacc_modules.TELinear
        dense_layer_spec.submodules.self_attention.submodules.linear_kv_down_proj = multiacc_modules.TELinear
        moe_layer_spec.submodules.self_attention.submodules.linear_q_down_proj = multiacc_modules.TELinear
        moe_layer_spec.submodules.self_attention.submodules.linear_kv_down_proj = multiacc_modules.TELinear
    
    if config.num_nextn_predict_layers > 0:
        # copy for mtp
        mtp_spec = moe_layer_spec
    
    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=multiacc_modules.TENorm, # TODO: Whether the Local Norm should be compatible
    )

    return block_spec, mtp_spec
