# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image model config."""

from dataclasses import dataclass, field

from loongforge.models.common import BaseModelStditConfig
from loongforge.models.factory import register_model_config
from loongforge.utils.constants import CustomModelFamilies


@dataclass
class QwenImageConfig(BaseModelStditConfig):
    """Configuration for Qwen-Image-Edit DiT."""

    num_layers: int = 60
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 24
    kv_channels: int = 128
    model_type: str = "qwen_image"

    text_dim: int = 3584
    latent_channels: int = 16
    patch_size: int = 2
    patch_dim: int = 64
    time_dim: int = 256
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    rope_theta: int = 10000
    scale_rope: bool = True
    use_layer3d_rope: bool = False
    # Enable the fused Triton interleaved-RoPE kernel (reuses wan's kernel).
    # Can also be toggled at runtime via env QWEN_IMAGE_USE_FUSED_ROPE=1.
    use_fused_qwen_image_rope: bool = True
    use_additional_t_cond: bool = False
    qwen_image_zero_cond_t: bool = False
    norm_epsilon: float = 1e-6

    group_query_attention: bool = False
    num_query_groups: int = 24
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    normalization: str = "LayerNorm"
    swiglu: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    qk_layernorm: bool = False
    untie_embeddings_and_output_weights: bool = True
    attention_softmax_in_fp32: bool = True

    # BaseModelStditConfig compatibility. Qwen-Image is image-only but these
    # fields are consumed by common Megatron argument/config plumbing.
    latent_in_channels: int = 16
    latent_out_channels: int = 16
    latent_patch_size: tuple = (1, 2, 2)
    latent_space_scale: float = 0.5
    latent_time_scale: float = 1.0
    num_latent_frames: int = 1
    max_latent_height: int = 1024
    max_latent_width: int = 1024


@register_model_config(
    model_family=CustomModelFamilies.QWEN_IMAGE,
    model_arch="qwen-image-edit-2511",
)
def qwen_image_edit_2511_config():
    """Register Qwen-Image-Edit-2511 in the model factory."""
    return QwenImageConfig()
