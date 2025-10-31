"""register STDiT model with different config"""

from dataclasses import dataclass

from aiak_training_omni.utils.constants import VideoLanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class WanConfig:
    """configuration for Wan model

    The fields need to be consistent with the definitions in args
    """

    latent_in_channels: int
    latent_out_channels: int
    latent_patch_size: tuple
    latent_space_scale: float
    latent_time_scale: float
    num_layers: int
    hidden_size: int
    # kv_channels: int
    ffn_hidden_size: int
    caption_channels: int
    num_attention_heads: int
    norm_epsilon: float = 1e-06
    group_query_attention: bool = False
    num_query_groups: int = 1
    position_embedding_type: str = "learned_absolute"
    rotary_interleaved: bool = False
    normalization: str = "RMSNorm"

    swiglu: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = True
    add_position_embedding: bool = True
    attention_softmax_in_fp32: bool = True


@register_model_config(
    model_family=VideoLanguageModelFamilies.WAN2_1_T2V, model_arch="WAN2_1_T2V"
)
def WAN2_1_T2V():
    """Configuration for WAN2.1 text-to-video model."""
    return WanConfig(
        num_layers=30,
        hidden_size=1536,
        ffn_hidden_size=8960,
        norm_epsilon=1e-6,
        caption_channels=4096,
        num_attention_heads=40,
        latent_in_channels=4,
        latent_out_channels=8,
        latent_patch_size=(1, 2, 2),
        latent_space_scale=0.5,
        latent_time_scale=1.0,
    )


@register_model_config(
    model_family=VideoLanguageModelFamilies.WAN2_1_I2V, model_arch="WAN2_1_I2V"
)
def WAN2_1_I2V():
    """Configuration for WAN2.1 image-to-video model."""
    return WanConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13824,
        norm_epsilon=1e-6,
        caption_channels=4096,
        num_attention_heads=40,
        latent_in_channels=4,
        latent_out_channels=8,
        latent_patch_size=(1, 2, 2),
        latent_space_scale=0.5,
        latent_time_scale=1.0,
    )
