"""register STDiT model with different config"""

from dataclasses import dataclass

from aiak_training_omni.utils.constants import VideoLanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class STDiTConfig:
    """configuration for STDiT model
    
    The fields need to be consistent with the definitions in args
    """
    latent_in_channels: int
    latent_out_channels: int
    latent_patch_size: tuple
    latent_space_scale: float
    latent_time_scale: float
    num_layers: int
    hidden_size: int
    caption_channels: int
    num_attention_heads: int
    group_query_attention: bool = False
    num_query_groups: int = 1
    position_embedding_type: str = "learned_absolute"
    rotary_interleaved: bool = False
    normalization: str = "LayerNorm"
    swiglu: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    qk_layernorm: bool = False
    untie_embeddings_and_output_weights: bool = True
    add_position_embedding: bool = True
    attention_softmax_in_fp32: bool = True


@register_model_config(model_family=VideoLanguageModelFamilies.STDIT, model_arch="STDiT-XL/2")
def STDiT_XL_2():
    """STDiT-XL/2"""
    return STDiTConfig(
        num_layers=28,
        hidden_size=1152,
        caption_channels=4096,
        num_attention_heads=16,
        latent_in_channels=4,
        latent_out_channels=8,
        latent_patch_size=(1, 2, 2),
        latent_space_scale=0.5,
        latent_time_scale=1.0,
    )
