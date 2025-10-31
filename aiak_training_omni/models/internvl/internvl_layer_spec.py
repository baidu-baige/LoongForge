"""Intern CLIP/SIGLIP Model layer spec."""

import torch
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from ..internlm.internlm_layer_spec import get_internlm_layer_with_te_spec
from ..qwen.qwen_layer_spec import get_qwen_layer_with_te_spec
from .intern_vision_attention import InternSelfAttention, SelfAttentionSubmodules
from .intern_vision_transformer_layer import TransformerLayerIntern, TransformerLayerInternVisionSubmodules
from .adapter import AdapterSubmodules, Adapter
from .internvl_config import InternVisionConfig
from aiak_training_omni.models.dispatch import multiacc_modules

from aiak_training_omni.utils import is_te_min_version

def get_vision_layer_with_te_spec(config: InternVisionConfig) -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""

    from .intern_vision_attention import InternViTTEDotProductAttention, InternViTRMSNorm

    qk_norm = multiacc_modules.TENorm if is_te_min_version("1.9.0") else multiacc_modules.LocalNorm
    if config.vision_type == "vit_6b":
        qk_layernorm = InternViTRMSNorm
        core_attention = InternViTTEDotProductAttention
    elif config.vision_type == "vit_300m":
        qk_layernorm = qk_norm if config.qk_layernorm else IdentityOp
        core_attention = multiacc_modules.DotProductAttention
    else:
        raise NotImplementedError

    return ModuleSpec(
        module=TransformerLayerIntern,
        submodules=TransformerLayerInternVisionSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=InternSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=multiacc_modules.TELayerNormColumnParallelLinear,
                    core_attention=core_attention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    q_layernorm=qk_layernorm,
                    k_layernorm=qk_layernorm,
                    apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
                )),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=multiacc_modules.TELayerNormColumnParallelLinear,
                    linear_fc2=multiacc_modules.TERowParallelLinear,
                    bias_activation_func_impl=multiacc_modules.bias_activation_func_impl),
            ),
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
    )


def get_adapeter_layer_with_te_spec() -> AdapterSubmodules:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""

    return AdapterSubmodules(
        layernorm=multiacc_modules.TENorm,
        linear_fc1=torch.nn.Linear,
        linear_fc2=torch.nn.Linear,
    )

get_language_layer_spec_func = {
    "internvl2.5-8b": get_internlm_layer_with_te_spec,
    "internvl2.5-26b": get_internlm_layer_with_te_spec,
    "internvl2.5-38b": get_qwen_layer_with_te_spec,
    "internvl2.5-78b": get_qwen_layer_with_te_spec,
    "internvl3-8b": get_qwen_layer_with_te_spec,
    "internvl3-14b": get_qwen_layer_with_te_spec,
    "internvl3-38b": get_qwen_layer_with_te_spec,
    "internvl3-78b": get_qwen_layer_with_te_spec,
    "internvl3.5-1b": get_qwen_layer_with_te_spec,
    "internvl3.5-2b": get_qwen_layer_with_te_spec,
    "internvl3.5-4b": get_qwen_layer_with_te_spec,
    "internvl3.5-8b": get_qwen_layer_with_te_spec,
    "internvl3.5-14b": get_qwen_layer_with_te_spec,
    "internvl3.5-38b": get_qwen_layer_with_te_spec,
    "internvl3.5-30b-a3b": get_qwen_layer_with_te_spec,
    "internvl3.5-241b-a28b": get_qwen_layer_with_te_spec,
}

def get_language_layer_with_te_spec(
    model_name,
    config: TransformerConfig
) -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    if model_name and model_name not in get_language_layer_spec_func:
        raise Exception(f'missing {model_name} layer_spec_func,'
                        f'current support model:{get_language_layer_spec_func.keys()}')
    return get_language_layer_spec_func[model_name](config)

