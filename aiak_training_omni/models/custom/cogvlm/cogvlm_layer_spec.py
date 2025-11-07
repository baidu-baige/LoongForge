"""cogvlm layer spec."""

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.core.transformer.attention import (
    SelfAttentionSubmodules,
)
from megatron.core.extensions.transformer_engine import (
    TENorm,
    TELinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from aiak_training_omni.models.custom.common.local_norm import LocalNorm

from .cogvlm_attention import CogvlmSelfAttention
from .cogvlm_transformer_layer import (
    TransformerLayerCogvlm,
    TransformerLayerCogvlmSubmodules,
)
from .cogvlm_mlp import CogvlmMlp
from .linear import VisionExpertLinear, VisionExpertLinearSubmodules, get_expert_mask
from .adapter import AdapterSubmodules
from .rotary_pos_embedding import apply_rotary_pos_emb_with_position_ids


def get_vision_layer_with_spec() -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    mlp = ModuleSpec(
        module=CogvlmMlp,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )

    return ModuleSpec(
        module=TransformerLayerCogvlm,
        submodules=TransformerLayerCogvlmSubmodules(
            self_attention=ModuleSpec(
                module=CogvlmSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    apply_rotary_fn=apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_self_attn_layernorm=TENorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=(TENorm),
        ),
    )


def get_adapeter_layer_with_spec() -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return AdapterSubmodules(
        linear_proj=TELinear,
        layernorm=LocalNorm,
        mlp=ModuleSpec(
            module=CogvlmMlp,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        ),
    )


def get_language_layer_with_spec(
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """

    # Dense MLP w/ or w/o TE modules.
    linear_fc1 = ModuleSpec(
        module=VisionExpertLinear,
        submodules=VisionExpertLinearSubmodules(
            language_linear=TEColumnParallelLinear,
            vision_linear=TEColumnParallelLinear,
            apply_mask_fn=get_expert_mask,
        ),
    )
    linear_fc2 = ModuleSpec(
        module=VisionExpertLinear,
        submodules=VisionExpertLinearSubmodules(
            language_linear=TERowParallelLinear,
            vision_linear=TERowParallelLinear,
            apply_mask_fn=get_expert_mask,
        ),
    )
    mlp = ModuleSpec(
        module=CogvlmMlp,
        submodules=MLPSubmodules(
            linear_fc1=linear_fc1,
            linear_fc2=linear_fc2,
        ),
    )

    return ModuleSpec(
        module=TransformerLayerCogvlm,
        submodules=TransformerLayerCogvlmSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=CogvlmSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ModuleSpec(
                        module=VisionExpertLinear,
                        params={"add_vision_bias": True},
                        submodules=VisionExpertLinearSubmodules(
                            language_linear=TEColumnParallelLinear,
                            vision_linear=TEColumnParallelLinear,
                            apply_mask_fn=get_expert_mask,
                        ),
                    ),
                    core_attention=TEDotProductAttention,
                    linear_proj=ModuleSpec(
                        module=VisionExpertLinear,
                        submodules=VisionExpertLinearSubmodules(
                            language_linear=TERowParallelLinear,
                            vision_linear=TERowParallelLinear,
                            apply_mask_fn=get_expert_mask,
                        ),
                    ),
                    apply_rotary_fn=apply_rotary_pos_emb_with_position_ids,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )
