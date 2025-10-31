"""STDiT3 layer spec."""

from .fused_bias_dropout import get_bias_dropout_add

from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)

from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from .stdit3_layer import (
    LlamaRMSNorm,
    STDiT3Layer,
    STDiT3Block,
    STDiT3BlockSubmodules,
    STDiT3LayerSubmodules,
)
from aiak_training_omni.models.stdit.stdit_attention import UlyssesSelfAttention, UlyssesCrossAttention

from aiak_training_omni.models.custom.common.local_norm import LocalNorm


def get_stdit3_layer_with_te_spec() -> ModuleSpec:
    """
    Use this spec to use lower level Transformer Engine modules (required for fp8 training)
    """
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    
    return ModuleSpec(
        module=STDiT3Layer,
        submodules=STDiT3LayerSubmodules(
            # Spatial Block
            spatial_block=ModuleSpec(
                module=STDiT3Block,
                submodules=STDiT3BlockSubmodules(
                    # Input LayerNorm
                    input_layernorm=ModuleSpec(
                        module=LocalNorm,
                        params={"elementwise_affine": False},
                    ),
                    # Self Attention
                    self_attention=ModuleSpec(
                        module=UlyssesSelfAttention,
                        params={"attn_mask_type": AttnMaskType.padding, "ulysses_gather_idx": 1},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TEColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                            apply_rotary_fn=apply_rotary_pos_emb,
                            q_layernorm=ModuleSpec(
                                module=LlamaRMSNorm,
                            ),
                            k_layernorm=ModuleSpec(
                                module=LlamaRMSNorm,
                            ),
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,

                    # Cross attention
                    cross_attention=ModuleSpec(
                        module=UlyssesCrossAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=CrossAttentionSubmodules(
                            linear_q=TEColumnParallelLinear,
                            linear_kv=TEColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                            apply_rotary_fn=apply_rotary_pos_emb,
                        ),
                    ),
                    cross_attn_bda=get_bias_dropout_add,

                    # MLP
                    pre_mlp_layernorm=ModuleSpec(
                        module=LocalNorm,
                        params={"elementwise_affine": False},
                    ),
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            # Temporal Block
            temporal_block=ModuleSpec(
                module=STDiT3Block,
                submodules=STDiT3BlockSubmodules(
                    # Input LayerNorm
                    input_layernorm=ModuleSpec(
                        module=LocalNorm,
                        params={"elementwise_affine": False},
                    ),
                    # Self Attention
                    self_attention=ModuleSpec(
                        module=UlyssesSelfAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TEColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                            apply_rotary_fn=apply_rotary_pos_emb,
                            q_layernorm=ModuleSpec(
                                module=LlamaRMSNorm,
                            ),
                            k_layernorm=ModuleSpec(
                                module=LlamaRMSNorm,
                            ),
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,

                    # Cross attention
                    cross_attention=ModuleSpec(
                        module=UlyssesCrossAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=CrossAttentionSubmodules(
                            linear_q=TEColumnParallelLinear,
                            linear_kv=TEColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                            apply_rotary_fn=apply_rotary_pos_emb,
                        ),
                    ),
                    cross_attn_bda=get_bias_dropout_add,

                    # MLP
                    pre_mlp_layernorm=ModuleSpec(
                        module=LocalNorm,
                        params={"elementwise_affine": False},
                    ),
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                )
            )
        )
    )
